"""Train the transformer on POP909 token sequences.

    uv run python -m secondchair.train --name baseline

Training examples are windows of block_size tokens starting on a Bar (or BOS) token, so each one
begins the way bar-by-bar generation will. Every training window is transposed by a random -5..+6
semitones, bass pitches and chord roots together: twelve keys' worth of data, and a model that
treats harmony the same in every key. Validation windows are fixed and left untransposed.

Each run writes runs/<name>/: config.json, log.csv (one row per evaluation), best.pt (lowest
validation loss), last.pt and loss.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from secondchair import plots
from secondchair.chords import NO_CHORD, PITCH_CLASS_NAMES
from secondchair.dataset import DEFAULT_OUT, read_dataset
from secondchair.model import GPT, GPTConfig
from secondchair.tokenizer import BAR, BOS, RemiTokenizer

RUNS = Path(__file__).resolve().parent.parent / "runs"
TRANSPOSITIONS = tuple(range(-5, 7))  # twelve keys
TOKEN_KINDS = ("Bar", "Position", "Chord", "Pitch", "Velocity", "Duration")  # BOS/EOS/PAD count as "other"


@dataclass(frozen=True)
class TrainConfig:
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    batch_size: int = 64
    max_steps: int = 4000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_windows: int = 512
    transpose: bool = True
    seed: int = 1337

    def gpt_config(self, vocab_size: int) -> GPTConfig:
        return GPTConfig(vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd, self.dropout)


# --------------------------------------------------------------------------- data
def load_tokenizer(vocab_path: Path) -> RemiTokenizer:
    """The tokenizer a dataset was built with; refuses a vocabulary that has since drifted."""
    meta = json.loads(vocab_path.read_text(encoding="utf-8"))
    tokenizer = RemiTokenizer(meta["beat_res"], meta["max_bar_beats"], meta["n_velocities"])
    if list(tokenizer.vocab) != meta["vocab"]:
        raise ValueError(f"{vocab_path} no longer matches the tokenizer's vocabulary; rebuild the dataset")
    return tokenizer


def split_ids(dataset_path: Path, tokenizer: RemiTokenizer, split: str) -> np.ndarray:
    """Token ids of every song in `split`, concatenated in song order."""
    songs = [s for s in read_dataset(dataset_path) if s.split == split]
    if not songs:
        raise ValueError(f"no songs in split {split!r} of {dataset_path}")
    return np.concatenate([np.asarray(tokenizer.to_ids(s.tokens), dtype=np.int64) for s in songs])


def window_starts(ids: np.ndarray, tokenizer: RemiTokenizer, block_size: int) -> np.ndarray:
    """Positions of Bar and BOS tokens that leave room for a window plus its last target."""
    bar, bos = tokenizer.to_ids([BAR, BOS])
    starts = np.flatnonzero((ids == bar) | (ids == bos))
    return starts[starts + block_size + 1 <= len(ids)]


def transposition_table(tokenizer: RemiTokenizer, shifts: Sequence[int] = TRANSPOSITIONS) -> np.ndarray:
    """table[i, id] is token `id` transposed by shifts[i] semitones; only pitches and chord roots move.

    Pitches are clamped to 0..127, which POP909 basslines (24..96) never reach.
    """
    index = {token: i for i, token in enumerate(tokenizer.vocab)}
    table = np.tile(np.arange(len(tokenizer.vocab)), (len(shifts), 1))
    for row, shift in enumerate(shifts):
        for i, token in enumerate(tokenizer.vocab):
            kind, _, value = token.partition("_")
            if kind == "Pitch":
                table[row, i] = index[f"Pitch_{min(max(int(value) + shift, 0), 127)}"]
            elif kind == "Chord" and value != NO_CHORD:
                root, quality = value.split(":")
                moved = PITCH_CLASS_NAMES[(PITCH_CLASS_NAMES.index(root) + shift) % 12]
                table[row, i] = index[f"Chord_{moved}:{quality}"]
    return table


def token_kinds(tokenizer: RemiTokenizer) -> np.ndarray:
    """For each vocabulary id, its index in TOKEN_KINDS (len(TOKEN_KINDS) for BOS/EOS/PAD)."""
    names = [token.partition("_")[0] for token in tokenizer.vocab]
    return np.array([TOKEN_KINDS.index(n) if n in TOKEN_KINDS else len(TOKEN_KINDS) for n in names])


class WindowSampler:
    """Random training windows, optionally transposed, drawn on `device`."""

    def __init__(
        self,
        ids: np.ndarray,
        starts: np.ndarray,
        block_size: int,
        table: np.ndarray | None,
        device: torch.device,
        seed: int,
    ) -> None:
        if len(starts) == 0:
            raise ValueError("no window fits: the split is shorter than block_size")
        self.ids = torch.as_tensor(ids, device=device)
        self.starts = torch.as_tensor(starts, device=device)
        self.offsets = torch.arange(block_size + 1, device=device)
        self.table = None if table is None else torch.as_tensor(table, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs and targets (targets are the inputs shifted by one token)."""
        device = self.ids.device
        pick = torch.randint(len(self.starts), (batch_size,), generator=self.generator, device=device)
        windows = self.ids[self.starts[pick, None] + self.offsets]
        if self.table is not None:
            shift = torch.randint(len(self.table), (batch_size, 1), generator=self.generator, device=device)
            windows = self.table[shift, windows]
        return windows[:, :-1], windows[:, 1:]


def eval_windows(ids: np.ndarray, starts: np.ndarray, block_size: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Up to `n` evenly spread, untransposed windows: the same ones at every evaluation."""
    chosen = starts[np.unique(np.linspace(0, len(starts) - 1, min(n, len(starts))).round().astype(int))]
    windows = ids[chosen[:, None] + np.arange(block_size + 1)]
    return windows[:, :-1], windows[:, 1:]


# --------------------------------------------------------------------------- training
def _autocast(device: torch.device) -> torch.autocast:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model: GPT, x: torch.Tensor, y: torch.Tensor, kinds: torch.Tensor, batch_size: int) -> dict[str, float]:
    """Mean loss over every target token, plus loss and top-1 accuracy per token kind."""
    model.eval()
    device = x.device
    loss_sum, hits, count = (torch.zeros(len(TOKEN_KINDS) + 1, device=device) for _ in range(3))
    for i in range(0, len(x), batch_size):
        with _autocast(device):
            logits, _ = model(x[i : i + batch_size])
        target = y[i : i + batch_size].reshape(-1)
        logits = logits.float().reshape(-1, logits.size(-1))
        kind = kinds[target]
        loss_sum.index_add_(0, kind, F.cross_entropy(logits, target, reduction="none"))
        hits.index_add_(0, kind, (logits.argmax(-1) == target).float())
        count.index_add_(0, kind, torch.ones_like(target, dtype=torch.float))
    model.train()
    metrics = {"val_loss": (loss_sum.sum() / count.sum()).item()}
    for i, kind in enumerate(TOKEN_KINDS):
        if count[i] > 0:
            metrics[f"val_loss_{kind}"] = (loss_sum[i] / count[i]).item()
            metrics[f"val_acc_{kind}"] = (hits[i] / count[i]).item()
    return metrics


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay from lr to min_lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps), 1.0)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def make_optimizer(model: GPT, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (not biases or LayerNorm gains)."""
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.99), fused=device.type == "cuda")


def save_checkpoint(path: Path, model: GPT, cfg: TrainConfig, tokenizer: RemiTokenizer, metrics: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "gpt_config": asdict(model.cfg),
            "train_config": asdict(cfg),
            "tokenizer": {"beat_res": tokenizer.beat_res, "max_bar_beats": tokenizer.max_bar_beats,
                          "n_velocities": tokenizer.n_velocities},
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[GPT, RemiTokenizer, dict]:
    """Model (in eval mode), its tokenizer and the saved metrics."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**checkpoint["gpt_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, RemiTokenizer(**checkpoint["tokenizer"]), checkpoint["metrics"]


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(
    cfg: TrainConfig,
    run_dir: Path,
    dataset_dir: Path = DEFAULT_OUT,
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    """Train from scratch into a new `run_dir`; return the evaluation row of the best checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=False)
    dev = pick_device(device)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")

    tokenizer = load_tokenizer(dataset_dir / "vocab.json")
    train_ids, val_ids = (split_ids(dataset_dir / "pop909.jsonl", tokenizer, s) for s in ("train", "val"))
    table = transposition_table(tokenizer) if cfg.transpose else None
    sampler = WindowSampler(train_ids, window_starts(train_ids, tokenizer, cfg.block_size), cfg.block_size,
                            table, dev, cfg.seed)
    vx, vy = (torch.as_tensor(a, device=dev) for a in
              eval_windows(val_ids, window_starts(val_ids, tokenizer, cfg.block_size), cfg.block_size, cfg.eval_windows))
    kinds = torch.as_tensor(token_kinds(tokenizer), device=dev)

    model = GPT(cfg.gpt_config(len(tokenizer.vocab))).to(dev)
    optimizer = make_optimizer(model, cfg, dev)
    summary = {"train": asdict(cfg), "device": str(dev), "params": model.num_params(),
               "train_tokens": len(train_ids), "val_tokens": len(val_ids), "val_windows": len(vx)}
    (run_dir / "config.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"{model.num_params():,} parameters on {dev}; {len(train_ids):,} train / {len(val_ids):,} val tokens")

    rows: list[dict[str, float]] = []
    best: dict[str, float] = {"val_loss": math.inf}
    loss_sum, loss_count = torch.zeros((), device=dev), 0
    started = time.perf_counter()
    for step in range(cfg.max_steps + 1):
        lr = learning_rate(step, cfg)
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            row = {"step": step, "lr": lr,
                   "train_loss": (loss_sum / loss_count).item() if loss_count else math.nan,
                   **evaluate(model, vx, vy, kinds, cfg.batch_size),
                   "elapsed_s": time.perf_counter() - started}
            rows.append(row)
            _write_log(rows, run_dir / "log.csv")
            if row["val_loss"] < best["val_loss"]:
                best = row
                save_checkpoint(run_dir / "best.pt", model, cfg, tokenizer, row)
            log(f"step {step:5d}  train {row['train_loss']:.3f}  val {row['val_loss']:.3f}  "
                f"chord acc {row['val_acc_Chord']:.1%}  pitch acc {row['val_acc_Pitch']:.1%}  "
                f"{row['elapsed_s']:.0f}s")
            loss_sum, loss_count = torch.zeros((), device=dev), 0
        if step == cfg.max_steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = sampler.sample(cfg.batch_size)
        with _autocast(dev):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        loss_sum += loss.detach()
        loss_count += 1

    save_checkpoint(run_dir / "last.pt", model, cfg, tokenizer, rows[-1])
    plot_losses(rows, run_dir / "loss.png")
    return best


def _write_log(rows: Sequence[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_losses(rows: Sequence[dict[str, float]], path: Path) -> Path:
    """Training and validation loss against step, from the first evaluation after step 0.

    The untrained loss (about ln(vocab size)) would flatten everything after it, so it is quoted in
    the title instead of plotted.
    """
    title = f"Loss (next-token cross-entropy; untrained: {rows[0]['val_loss']:.2f})"
    fig, ax = plots.chart(title, "step", "loss", size=(8, 4))
    trained = [r for r in rows if not math.isnan(r["train_loss"])]
    ax.plot([r["step"] for r in trained], [r["train_loss"] for r in trained],
            color=plots.SERIES[0], linewidth=2, label="train")
    ax.plot([r["step"] for r in trained], [r["val_loss"] for r in trained], color=plots.SERIES[1], linewidth=2,
            label="validation")
    ax.legend(frameon=False, labelcolor=plots.INK)
    return plots.save(fig, path)


# --------------------------------------------------------------------------- command line
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the bass/chord transformer on POP909 tokens.")
    parser.add_argument("--name", default=None, help="run folder under runs/ (default: a timestamp)")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUT, help="folder with pop909.jsonl and vocab.json")
    for f in fields(TrainConfig):
        if isinstance(f.default, bool):
            parser.add_argument(f"--no-{f.name}", dest=f.name, action="store_false", help=f"disable {f.name}")
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = parser.parse_args(argv)

    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    run_dir = RUNS / (args.name or datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    best = train(cfg, run_dir, args.dataset, args.device)
    print(f"best: step {best['step']}, val loss {best['val_loss']:.3f} -> {run_dir}")


if __name__ == "__main__":
    main()
