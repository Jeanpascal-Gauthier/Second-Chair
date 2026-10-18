"""Safer guesses: when the next chord isn't known yet, play a bass note that fits whatever is likely.

Live, the bass has to sound on a downbeat before the chord recognizer has heard the new chord, so the
model's chord is a guess. Playing the root of its single best guess clashes whenever that guess is
wrong. Instead, every chord is weighed by the model's probability and the bass takes the pitch class
that fits them best on average: a note shared by the likely chords (the root of one, the fifth or
third of another) sounds right either way.

    uv run python -m secondchair.safety runs/baseline/best.pt     # scorecard: best guess vs safe guess
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from secondchair.chords import CHORD_CLASSES, NO_CHORD, PITCH_CLASS_NAMES
from secondchair.dataset import DEFAULT_OUT, read_dataset
from secondchair.model import GPT
from secondchair.tokenizer import BAR, RemiTokenizer
from secondchair.train import load_checkpoint, pick_device

# How well a bass note sits under a chord, by its role in the chord; anything else clashes (0).
ROLE_FIT = {"root": 1.0, "fifth": 0.8, "third": 0.6}


def chord_root(chord: str) -> int | None:
    return None if chord == NO_CHORD else PITCH_CLASS_NAMES.index(chord.split(":")[0])


def fit(pitch_class: int, chord: str) -> float:
    """How well a bass note of `pitch_class` fits under `chord` (0 = clash; nothing fits N)."""
    root = chord_root(chord)
    if root is None:
        return 0.0
    third = 4 if chord.endswith(":maj") else 3
    roles = {root: "root", (root + 7) % 12: "fifth", (root + third) % 12: "third"}
    return ROLE_FIT.get(roles.get(pitch_class % 12, ""), 0.0)


FIT_TABLE = np.array([[fit(pc, chord) for pc in range(12)] for chord in CHORD_CLASSES])  # chords x 12


def chord_probabilities(logits: torch.Tensor | np.ndarray, tokenizer: RemiTokenizer) -> np.ndarray:
    """The model's next-token logits turned into probabilities over CHORD_CLASSES alone."""
    ids = tokenizer.to_ids([f"Chord_{chord}" for chord in CHORD_CLASSES])
    scores = np.asarray(logits.float().cpu() if isinstance(logits, torch.Tensor) else logits, dtype=float)[ids]
    weights = np.exp(scores - scores.max())
    return weights / weights.sum()


def expected_fit(probs: np.ndarray) -> np.ndarray:
    """For each of the 12 pitch classes, its fit averaged over the chord probabilities."""
    return probs @ FIT_TABLE


def safest_pitch_class(probs: np.ndarray) -> int:
    """The pitch class that fits the likely chords best on average."""
    return int(np.argmax(expected_fit(probs)))


def likeliest_root(probs: np.ndarray) -> int:
    """Root of the single most likely chord other than N: the naive guess."""
    best = max(range(1, len(CHORD_CLASSES)), key=lambda i: probs[i])
    return chord_root(CHORD_CLASSES[best])


# --------------------------------------------------------------------------- scorecard
@dataclass(frozen=True)
class Decision:
    """One moment where the bass must sound before the chord is heard."""

    kind: str  # "chord continues", "new change" or "repeated change"
    truth: str  # the chord actually played
    naive: int  # pitch class of the best guess's root
    safe: int  # pitch class chosen by expected fit
    confidence: float  # probability of the model's top chord


@torch.no_grad()
def decisions(model: GPT, tokenizer: RemiTokenizer, songs: Sequence[Sequence[str]], batch_size: int = 32) -> list[Decision]:
    """A Decision at every chord token of every song, predicted from the song so far.

    Each prediction sees as much of the song as fits the model, starting at a bar line as in
    training. "Repeated change" means the song made the same chord-to-chord move earlier.
    """
    device = next(model.parameters()).device
    block = model.cfg.block_size
    jobs: dict[tuple[int, int], list[int]] = defaultdict(list)  # (song, window start) -> chord token indices
    for s, tokens in enumerate(songs):
        bar_starts = [i for i, t in enumerate(tokens) if t == BAR]
        for i, token in enumerate(tokens):
            if token.startswith("Chord_") and bar_starts and i > bar_starts[0]:
                start = next(b for b in bar_starts if i - b <= block)
                jobs[(s, start)].append(i)

    probs: dict[tuple[int, int], np.ndarray] = {}
    keys = sorted(jobs)
    for k in range(0, len(keys), batch_size):
        batch = keys[k : k + batch_size]
        windows = [tokenizer.to_ids(songs[s][start : start + block]) for s, start in batch]
        width = max(map(len, windows))
        x = torch.tensor([w + [0] * (width - len(w)) for w in windows], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(x)
        for row, (s, start) in enumerate(batch):
            for i in jobs[(s, start)]:
                probs[(s, i)] = chord_probabilities(logits[row, i - 1 - start], tokenizer)

    out = []
    for s, tokens in enumerate(songs):
        current, seen = None, set()
        for i, token in enumerate(tokens):
            if not token.startswith("Chord_"):
                continue
            truth = token[len("Chord_"):]
            if (s, i) in probs and truth != NO_CHORD and current is not None:
                kind = ("chord continues" if truth == current else
                        "repeated change" if (current, truth) in seen else "new change")
                p = probs[(s, i)]
                out.append(Decision(kind, truth, likeliest_root(p), safest_pitch_class(p), float(p.max())))
            if current is not None and truth != current:
                seen.add((current, truth))
            current = truth
    return out


def scorecard(found: Sequence[Decision]) -> dict[str, dict[str, float]]:
    """Per kind of decision (plus all, and the unsure ones): how each policy's note fits the real chord."""
    groups: dict[str, list[Decision]] = defaultdict(list)
    for d in found:
        groups[d.kind].append(d)
        groups["all"].append(d)
        if d.confidence < 0.5:
            groups["model unsure (<50%)"].append(d)
    table = {}
    for name, items in groups.items():
        row = {"count": float(len(items))}
        for policy in ("naive", "safe"):
            fits = np.array([fit(getattr(d, policy), d.truth) for d in items])
            row[f"{policy} root"] = float(np.mean([getattr(d, policy) == chord_root(d.truth) for d in items]))
            row[f"{policy} clash"] = float(np.mean(fits == 0))
        table[name] = row
    return table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare best-guess and safe-guess bass notes at chord changes.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device(args.device))
    songs = [s.tokens for s in read_dataset(DEFAULT_OUT / "pop909.jsonl") if s.split == args.split]
    table = scorecard(decisions(model, tokenizer, songs))
    print(f"{'':22s} {'count':>6s}   {'on the root':^19s}   {'clashes':^19s}")
    print(f"{'':22s} {'':6s}   {'best':>8s} {'safe':>8s}     {'best':>8s} {'safe':>8s}")
    for name in ("chord continues", "repeated change", "new change", "model unsure (<50%)", "all"):
        r = table[name]
        print(f"{name:22s} {int(r['count']):6d}   {r['naive root']:8.1%} {r['safe root']:8.1%}     "
              f"{r['naive clash']:8.1%} {r['safe clash']:8.1%}")


if __name__ == "__main__":
    main()
