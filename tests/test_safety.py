from __future__ import annotations

import numpy as np
import pytest
import torch

from secondchair.chords import CHORD_CLASSES, ChordEvent
from secondchair.grid import Grid
from secondchair.model import GPT, GPTConfig
from secondchair.notes import Note
from secondchair.safety import (
    FIT_TABLE,
    chord_probabilities,
    decisions,
    expected_fit,
    fit,
    likeliest_root,
    safest_pitch_class,
    scorecard,
)
from secondchair.tokenizer import RemiTokenizer

TOK = RemiTokenizer()
C, E, G, A = 0, 4, 7, 9


def probs(**weights: float) -> np.ndarray:
    """Chord probabilities from keyword weights, e.g. probs(C_maj=0.5, G_maj=0.5)."""
    p = np.zeros(len(CHORD_CLASSES))
    for name, w in weights.items():
        p[CHORD_CLASSES.index(name.replace("_", ":"))] = w
    return p


def test_fit_by_role_in_the_chord() -> None:
    assert [fit(pc, "C:maj") for pc in (C, G, E, A)] == [1.0, 0.8, 0.6, 0.0]
    assert [fit(pc, "A:min") for pc in (A, E, C, G)] == [1.0, 0.8, 0.6, 0.0]
    assert fit(C + 12, "C:maj") == 1.0  # octave doesn't matter
    assert all(fit(pc, "N") == 0.0 for pc in range(12))


def test_fit_table_rows_follow_chord_classes() -> None:
    assert FIT_TABLE.shape == (25, 12)
    assert FIT_TABLE[CHORD_CLASSES.index("G:maj")].tolist() == [fit(pc, "G:maj") for pc in range(12)]


def test_a_certain_chord_gets_its_root() -> None:
    assert safest_pitch_class(probs(C_maj=1.0)) == C


def test_torn_between_two_chords_picks_the_shared_note() -> None:
    assert safest_pitch_class(probs(C_maj=0.5, G_maj=0.5)) == G  # root of G, fifth of C
    assert safest_pitch_class(probs(C_maj=0.5, A_min=0.5)) == C  # root of C, third of A minor
    assert expected_fit(probs(C_maj=0.5, G_maj=0.5))[G] == pytest.approx(0.9)


def test_likeliest_root_skips_no_chord() -> None:
    assert likeliest_root(probs(N=0.6, F_maj=0.3, G_maj=0.1)) == 5


def test_chord_probabilities_use_only_chord_logits() -> None:
    logits = torch.zeros(len(TOK.vocab))
    logits[TOK.to_ids(["Pitch_40"])[0]] = 50.0  # a strong non-chord token changes nothing
    logits[TOK.to_ids(["Chord_D:min"])[0]] = 2.0
    p = chord_probabilities(logits, TOK)
    assert p.sum() == pytest.approx(1.0)
    assert CHORD_CLASSES[int(np.argmax(p))] == "D:min"


def test_scorecard_from_a_model() -> None:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(TOK.vocab), block_size=64, n_layer=1, n_head=2, n_embd=32)).eval()
    grid = Grid.constant_tempo(bpm=120)
    cycle = ["C:maj", "G:maj", "A:min", "F:maj"]
    chords = [ChordEvent(bar, 0, cycle[bar % 4]) for bar in range(12)]
    notes = [Note(2.0 * bar, 2.0 * bar + 2.0, 48, 80) for bar in range(12)]
    song = TOK.encode(notes, chords, grid).tokens

    found = decisions(model, TOK, [song, song])
    assert {d.kind for d in found} == {"new change", "repeated change"}  # every bar changes chord
    table = scorecard(found)
    assert table["all"]["count"] == len(found) == 2 * 11  # the first chord of a song has no history
    for row in table.values():
        assert all(0.0 <= row[k] <= 1.0 for k in row if k != "count")
