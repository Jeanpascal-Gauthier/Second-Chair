from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from secondchair.chords import PITCH_CLASS_NAMES, ChordEvent
from secondchair.dataset import SongTokens, write_dataset
from secondchair.grid import Grid
from secondchair.model import GPT
from secondchair.notes import Note
from secondchair.tokenizer import RemiTokenizer
from secondchair.train import (
    TOKEN_KINDS,
    TRANSPOSITIONS,
    TrainConfig,
    WindowSampler,
    eval_windows,
    evaluate,
    learning_rate,
    load_checkpoint,
    load_tokenizer,
    token_kinds,
    train,
    transposition_table,
    window_starts,
)

TOK = RemiTokenizer()
ID = {token: i for i, token in enumerate(TOK.vocab)}
TABLE = transposition_table(TOK)
TINY = TrainConfig(block_size=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0, batch_size=8, max_steps=60,
                   warmup_steps=5, eval_every=30, eval_windows=16, lr=3e-3)


def synthetic_song(offset: int, n_bars: int = 16) -> tuple[str, ...]:
    """I-IV-V-vi shifted by `offset` semitones, with the root on beats 1 and 3 of every bar."""
    grid = Grid.constant_tempo(bpm=120)  # half a second per beat, two seconds per bar
    notes, chords = [], []
    for bar in range(n_bars):
        degree, quality = ((0, "maj"), (5, "maj"), (7, "maj"), (9, "min"))[bar % 4]
        root = (degree + offset) % 12
        chords.append(ChordEvent(bar, 0, f"{PITCH_CLASS_NAMES[root]}:{quality}"))
        notes += [Note(2.0 * bar, 2.0 * bar + 1.0, 36 + root, 90), Note(2.0 * bar + 1.0, 2.0 * bar + 2.0, 36 + root, 70)]
    return TOK.encode(notes, chords, grid).tokens


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    songs = [SongTokens(f"{k:03d}", "val" if k % 4 == 0 else "train", 0, synthetic_song(k)) for k in range(12)]
    write_dataset(songs, TOK, tmp_path / "tokens")
    return tmp_path / "tokens"


def ids_of(tokens: list[str]) -> np.ndarray:
    return np.array([ID[t] for t in tokens])


# --------------------------------------------------------------------------- transposition
def test_transposition_moves_pitches_and_chord_roots_only() -> None:
    up2 = TABLE[TRANSPOSITIONS.index(2)]
    assert TOK.vocab[up2[ID["Pitch_40"]]] == "Pitch_42"
    assert TOK.vocab[up2[ID["Chord_B:maj"]]] == "Chord_C#:maj"
    assert TOK.vocab[TABLE[TRANSPOSITIONS.index(-5)][ID["Chord_Bb:min"]]] == "Chord_F:min"
    for token in ("Chord_N", "Bar", "Position_8", "Velocity_3", "Duration_4", "BOS", "EOS", "PAD"):
        assert up2[ID[token]] == ID[token]


def test_transposition_zero_is_identity_and_shifts_undo_each_other() -> None:
    assert (TABLE[TRANSPOSITIONS.index(0)] == np.arange(len(TOK.vocab))).all()
    up, down = TABLE[TRANSPOSITIONS.index(5)], TABLE[TRANSPOSITIONS.index(-5)]
    in_range = [i for i, t in enumerate(TOK.vocab) if not t.startswith("Pitch_") or 5 <= int(t[6:]) <= 122]
    assert all(down[up[i]] == i for i in in_range)


def test_twelve_keys() -> None:
    assert len(TRANSPOSITIONS) == 12 and len({k % 12 for k in TRANSPOSITIONS}) == 12


# --------------------------------------------------------------------------- windows
def test_window_starts_are_bars_or_bos_with_room_for_targets() -> None:
    ids = ids_of(["BOS", "Bar", "Position_0", "Chord_N", "Bar", "Position_0", "Chord_N", "EOS"])
    assert window_starts(ids, TOK, block_size=4).tolist() == [0, 1]  # the Bar at 4 lacks 4 + 1 tokens


def test_sampler_targets_are_the_inputs_shifted() -> None:
    ids = ids_of(list(synthetic_song(0)))
    sampler = WindowSampler(ids, window_starts(ids, TOK, 16), 16, TABLE, torch.device("cpu"), seed=0)
    x, y = sampler.sample(32)
    assert x.shape == y.shape == (32, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_sampler_transposes_each_window_by_one_amount() -> None:
    ids = ids_of(list(synthetic_song(0)))
    starts = window_starts(ids, TOK, 16)
    originals = ids[starts[:, None] + np.arange(16)]
    sampler = WindowSampler(ids, starts, 16, TABLE, torch.device("cpu"), seed=0)
    shifts_seen = set()
    for row in sampler.sample(64)[0].numpy():
        matches = {k for k, shifted in zip(TRANSPOSITIONS, TABLE) for w in originals if (shifted[w] == row).all()}
        assert matches, "a sampled window is not a real window moved by a single amount"
        shifts_seen |= matches
    assert len(shifts_seen) > 1  # windows are moved by different amounts


def test_eval_windows_are_fixed() -> None:
    ids = ids_of(list(synthetic_song(0)))
    starts = window_starts(ids, TOK, 16)
    a, b = eval_windows(ids, starts, 16, 5), eval_windows(ids, starts, 16, 5)
    assert all((p == q).all() for p, q in zip(a, b))
    assert a[0].shape == (5, 16)


# --------------------------------------------------------------------------- schedule & metrics
def test_learning_rate_warms_up_then_decays_to_min() -> None:
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    warmup = [learning_rate(s, cfg) for s in range(10)]
    decay = [learning_rate(s, cfg) for s in range(10, 101)]
    assert warmup == sorted(warmup) and warmup[-1] == pytest.approx(1e-3)
    assert decay == sorted(decay, reverse=True)
    assert decay[-1] == pytest.approx(1e-4)


def test_evaluate_reports_loss_and_accuracy_per_kind() -> None:
    ids = ids_of(list(synthetic_song(0)))
    x, y = (torch.as_tensor(a) for a in eval_windows(ids, window_starts(ids, TOK, 16), 16, 8))
    model = GPT(TINY.gpt_config(len(TOK.vocab)))
    metrics = evaluate(model, x, y, torch.as_tensor(token_kinds(TOK)), batch_size=4)
    assert metrics["val_loss"] > 0
    for kind in TOKEN_KINDS:
        assert 0.0 <= metrics[f"val_acc_{kind}"] <= 1.0
    assert model.training  # evaluate leaves the model in training mode


def test_load_tokenizer_refuses_a_drifted_vocabulary(tmp_path: Path) -> None:
    meta = {"beat_res": 4, "max_bar_beats": 8, "n_velocities": 16, "vocab": list(TOK.vocab)}
    (tmp_path / "vocab.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="rebuild the dataset"):
        load_tokenizer(tmp_path / "vocab.json")


# --------------------------------------------------------------------------- end to end
def test_tiny_training_run(dataset_dir: Path, tmp_path: Path) -> None:
    best = train(TINY, tmp_path / "run", dataset_dir, device="cpu", log=lambda _: None)
    run = tmp_path / "run"
    assert {p.name for p in run.iterdir()} == {"config.json", "log.csv", "best.pt", "last.pt", "loss.png"}

    with (run / "log.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == [0, 30, 60]
    assert float(rows[-1]["val_loss"]) < float(rows[0]["val_loss"]) / 2

    model, tokenizer, metrics = load_checkpoint(run / "best.pt")
    assert tokenizer == TOK and metrics["val_loss"] == pytest.approx(best["val_loss"])
    assert not model.training
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(ids_of(["BOS", "Bar", "Position_0"]))[None])
    assert logits.shape == (1, 3, len(TOK.vocab))


def test_training_refuses_to_overwrite_a_run(dataset_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        train(TINY, tmp_path / "run", dataset_dir, device="cpu", log=lambda _: None)
