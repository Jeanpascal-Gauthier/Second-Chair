from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pytest
import torch

from secondchair.analysis import (
    SAMPLE_RATE,
    BeatFeatures,
    _fill_bars,
    analyze,
    chord_templates,
    grid_from_beats,
    label_beats,
    recognize_chords,
)
from secondchair.chords import CHORD_CLASSES

C_MAJOR, A_MINOR = [36, 60, 64, 67], [45, 57, 60, 64]  # bass note + triad


def tones(pitches: list[int], seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return sum(0.2 * np.sin(2 * np.pi * librosa.midi_to_hz(p) * t) for p in pitches)


# --------------------------------------------------------------------------- bars
@pytest.mark.parametrize(
    ("marks", "meter", "expected"),
    [
        ([0, 4, 12, 16], 4, [0, 4, 8, 12, 16]),  # one missed downbeat restored
        ([0, 4, 10, 14], 4, [0, 4, 10, 14]),  # a real 6-beat bar is kept
        ([0, 13], 4, [0, 4, 8, 12, 13]),  # too long for the tokenizer: cut into bars
        ([0, 6, 9], 3, [0, 3, 6, 9]),
    ],
)
def test_fill_bars(marks: list[int], meter: int, expected: list[int]) -> None:
    assert _fill_bars(marks, meter) == expected


def test_grid_from_beats_snaps_downbeats_and_restores_missed_ones() -> None:
    beats = np.arange(20) * 0.5
    grid = grid_from_beats(beats, np.array([0.52, 2.49, 6.51, 30.0]))  # beat 9 missed; 30 s is no beat
    assert grid.downbeats == (1, 5, 9, 13)
    assert grid.beats_per_bar == 4


def test_grid_from_beats_finds_triple_meter() -> None:
    beats = np.arange(16) * 0.5
    assert grid_from_beats(beats, beats[[0, 3, 6, 9, 12]]).beats_per_bar == 3


def test_grid_from_beats_without_downbeats_starts_bars_on_the_first_beat() -> None:
    grid = grid_from_beats(np.arange(8) * 0.5, np.array([]))
    assert (grid.downbeats, grid.beats_per_bar) == ((0,), 4)


def test_grid_needs_two_beats() -> None:
    with pytest.raises(ValueError, match="two beats"):
        grid_from_beats(np.array([1.0]), np.array([1.0]))


# --------------------------------------------------------------------------- chords
def test_templates_are_the_24_triads() -> None:
    names, templates = chord_templates()
    assert sorted(names) == sorted(CHORD_CLASSES[1:])
    assert np.allclose(np.linalg.norm(templates, axis=1), 1.0)


def test_label_beats_from_clean_features() -> None:
    _, templates = chord_templates()
    columns = [templates[0]] * 3 + [templates[19]] * 3  # C:maj then A:min
    bass = np.zeros((12, 7))
    bass[0, :3], bass[9, 3:6] = 1.0, 1.0
    features = BeatFeatures(np.array(columns + [np.ones(12)]).T, bass, np.array([1.0] * 6 + [0.0]))
    assert label_beats(features) == ["C:maj"] * 3 + ["A:min"] * 3 + ["N"]  # the silent beat is N


def test_recognize_chords_on_synthesized_audio() -> None:
    y = np.concatenate([tones(C_MAJOR, 2.0), tones(A_MINOR, 2.0)])
    segments = recognize_chords(y, SAMPLE_RATE, np.arange(8) * 0.5)
    assert [s.label for s in segments] == ["C:maj", "A:min"]
    assert segments[1].start == pytest.approx(2.0)
    assert segments[-1].end == pytest.approx(4.0, abs=0.01)


# --------------------------------------------------------------------------- whole file
def test_analyze_a_rendered_groove(groove_wav: Path, beat_tracker: None) -> None:
    """Eight bars of C-G-Am-F at 120 bpm: each bar lasts two seconds."""
    result = analyze(groove_wav, device="cuda" if torch.cuda.is_available() else "cpu")
    assert 60 / float(np.median(np.diff(result.beats))) == pytest.approx(120, rel=0.03)
    assert result.grid.beats_per_bar == 4
    bar_starts = 2.0 * np.arange(8)
    assert np.mean([np.min(np.abs(result.downbeats - t)) < 0.07 for t in bar_starts]) >= 0.75

    def label_at(t: float) -> str:
        return next((c.label for c in reversed(result.chords) if c.start <= t), "N")

    expected = ["C:maj", "G:maj", "A:min", "F:maj"]
    assert np.mean([label_at(t + 1.0) == expected[bar % 4] for bar, t in enumerate(bar_starts)]) >= 0.75
