"""Audio analysis: a recording's beats, bars and chords, as the Grid and chord events the model uses.

The audio front-end, run offline on a whole file:
- beats and downbeats: beat_this (CPJKU's pretrained tracker; weights download on first use);
- chords: librosa chroma of the harmonic part of the signal, averaged per beat and scored against
  the 24 major/minor triads, then smoothed with a Viterbi pass so the label doesn't flicker from
  beat to beat. Silent beats are N.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import librosa
import numpy as np

from secondchair.chords import NO_CHORD, PITCH_CLASS_NAMES, ChordSegment, merge_repeats
from secondchair.grid import Grid

SAMPLE_RATE = 22050
HOP = 512
MAX_BAR_BEATS = 8  # longest bar the tokenizer can express


@dataclass(frozen=True)
class Analysis:
    duration: float  # seconds
    beats: np.ndarray  # seconds
    downbeats: np.ndarray  # seconds, as tracked (the grid may add missed ones)
    grid: Grid
    chords: list[ChordSegment]  # labels from CHORD_CLASSES, merged, covering the first beat to the end


# --------------------------------------------------------------------------- beats -> grid
@cache
def _beat_tracker(device: str):  # noqa: ANN202 - beat_this's Audio2Beats
    from beat_this.inference import Audio2Beats

    return Audio2Beats(checkpoint_path="final0", device=device)


def track_beats(y: np.ndarray, sr: int, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times (seconds) of a mono signal."""
    beats, downbeats = _beat_tracker(device)(y, sr)
    return np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float)


def grid_from_beats(beats: np.ndarray, downbeats: np.ndarray, beat_res: int = 4) -> Grid:
    """Grid whose bars start at the tracked downbeats.

    Each downbeat snaps to its nearest beat. The commonest bar length becomes the meter, a gap of
    exactly two or more bars gets the missed downbeats back, and a gap too long for the tokenizer
    is cut into bars of the commonest length.
    """
    if len(beats) < 2:
        raise ValueError("need at least two beats to build a grid")
    tolerance = 0.25 * float(np.median(np.diff(beats)))
    marks = sorted({int(np.argmin(np.abs(beats - d))) for d in downbeats if np.min(np.abs(beats - d)) <= tolerance})
    gaps = np.diff(marks)
    meter = statistics.mode(gaps.tolist()) if len(gaps) else 4
    return Grid(beats, _fill_bars(marks or [0], meter), beats_per_bar=meter, beat_res=beat_res)


def _fill_bars(marks: Sequence[int], meter: int) -> list[int]:
    """Downbeat indices with missed downbeats restored and over-long bars cut to fit the tokenizer."""
    filled = [marks[0]]
    for start, end in zip(marks, marks[1:]):
        gap = end - start
        if gap % meter == 0 or gap > MAX_BAR_BEATS:  # whole bars missed, or too long to keep as one
            filled += range(start + meter, end, meter)
        filled.append(end)
    return filled


# --------------------------------------------------------------------------- chords
def chord_templates() -> tuple[list[str], np.ndarray]:
    """Names and unit-length chroma templates of the 24 major and minor triads."""
    names, rows = [], []
    for root, name in enumerate(PITCH_CLASS_NAMES):
        for quality, third in (("maj", 4), ("min", 3)):
            row = np.zeros(12)
            row[[root, (root + third) % 12, (root + 7) % 12]] = 1.0
            names.append(f"{name}:{quality}")
            rows.append(row / np.linalg.norm(row))
    return names, np.array(rows)


@dataclass(frozen=True)
class BeatFeatures:
    """Per-beat audio features (columns are beats)."""

    chroma: np.ndarray  # 12 x beats, full range
    bass: np.ndarray  # 12 x beats, the three octaves from C1 only
    energy: np.ndarray  # RMS per beat


def beat_features(y: np.ndarray, sr: int, bounds: np.ndarray) -> BeatFeatures:
    """Chroma of the harmonic signal (all of it, and its bass register) and RMS, averaged per beat."""
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=HOP)
    bass = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("C1"), n_octaves=3)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    frames = np.clip(librosa.time_to_frames(bounds, sr=sr, hop_length=HOP), 0, chroma.shape[1] - 1)
    spans = [(a, max(b, a + 1)) for a, b in zip(frames[:-1], frames[1:])]

    def per_beat(matrix: np.ndarray) -> np.ndarray:
        return np.stack([matrix[:, a:b].mean(axis=1) for a, b in spans], axis=1)

    return BeatFeatures(per_beat(chroma), per_beat(bass), np.array([rms[a:b].mean() for a, b in spans]))


def label_beats(
    features: BeatFeatures,
    sharpness: float = 20.0,
    stay: float = 0.5,
    bass_weight: float = 0.5,
    silence_db: float = -40.0,
) -> list[str]:
    """One label per beat: N where the beat is silent, otherwise the best-matching triad, smoothed.

    A triad scores its similarity to the beat's chroma plus `bass_weight` times how much of the bass
    register's energy sits on its root. `sharpness` and `stay` (the Viterbi self-transition
    probability) were chosen on rendered POP909 validation songs.
    """
    names, templates = chord_templates()
    chroma, energy = features.chroma, features.energy
    unit = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-9)
    bass_share = features.bass / np.maximum(features.bass.sum(axis=0, keepdims=True), 1e-9)
    roots = np.repeat(np.arange(12), 2)  # templates alternate maj/min per root
    scores = np.exp(sharpness * (templates @ unit + bass_weight * bass_share[roots]))
    silent = librosa.amplitude_to_db(energy, ref=np.max(energy) if energy.size else 1.0) < silence_db
    posterior = np.vstack([np.where(silent, 1e-6, scores / scores.sum(axis=0)), np.where(silent, 1.0, 1e-6)])
    posterior /= posterior.sum(axis=0)
    path = librosa.sequence.viterbi_discriminative(posterior, librosa.sequence.transition_loop(len(posterior), stay))
    return [(names + [NO_CHORD])[state] for state in path]


def recognize_chords(y: np.ndarray, sr: int, beats: np.ndarray) -> list[ChordSegment]:
    """Beat-synchronous chord segments from the first beat to the end of the signal."""
    end = max(len(y) / sr, beats[-1] + 1e-3)
    bounds = np.append(beats, end)
    labels = label_beats(beat_features(y, sr, bounds))
    return merge_repeats(ChordSegment(float(a), float(b), label) for a, b, label in zip(bounds[:-1], bounds[1:], labels))


# --------------------------------------------------------------------------- whole file
def analyze(path: Path, device: str = "cpu", beat_res: int = 4) -> Analysis:
    """Beats, bars and chords of an audio file (wav, mp3, flac, ...), decoded once for both."""
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    beats, downbeats = track_beats(y, sr, device)
    if len(beats) < 2:
        raise ValueError(f"found no steady beat in {path}")
    return Analysis(len(y) / sr, beats, downbeats, grid_from_beats(beats, downbeats, beat_res),
                    recognize_chords(y, sr, beats))
