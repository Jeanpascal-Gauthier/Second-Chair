"""Chords: POP909's annotations, the coarse chord vocabulary, and grid-level chord events.

POP909 labels are Harte-style strings ("C#:maj7/5", "Bb:sus4(b7)", "N" for no chord). Every
segment boundary falls on an annotated beat, and segments are contiguous. POP909 does not merge
adjacent segments with the same label; `merge_repeats` does.

A real-time chord recognizer can only report something coarse, so training uses the same coarse
vocabulary: 12 roots x {maj, min} plus N (`CHORD_CLASSES`). `collapse` maps any label onto it via
the explicit `QUALITY_TO_CLASS` table; inversions and extensions are dropped.

`ChordEvent(bar, position, chord)` is the interface between chord recognition and generation: a
chord taking effect at a grid point.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from secondchair.grid import Grid

NO_CHORD = "N"
_CONTIGUITY_TOLERANCE = 1e-6  # seconds; POP909 boundaries are written with 6 decimals

PITCH_CLASS_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_LETTER_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Every chord quality in POP909, mapped to the class a real-time recognizer can tell apart.
QUALITY_TO_CLASS: dict[str, str] = {
    "maj": "maj", "maj7": "maj", "7": "maj", "maj6": "maj", "aug": "maj", "sus2": "maj", "sus4": "maj",
    "min": "min", "min7": "min", "min6": "min", "minmaj7": "min", "dim": "min", "hdim7": "min", "dim7": "min",
}
CHORD_CLASSES: tuple[str, ...] = (
    NO_CHORD, *(f"{root}:{quality}" for root in PITCH_CLASS_NAMES for quality in ("maj", "min"))
)
_LABEL = re.compile(r"^(?P<root>[A-G][#b]*):(?P<quality>[^(/]+)(?:\((?P<ext>[^)]*)\))?(?:/(?P<bass>.+))?$")


@dataclass(frozen=True)
class ChordSegment:
    start: float  # seconds
    end: float  # seconds
    label: str

    def __post_init__(self) -> None:
        if not self.end > self.start:
            raise ValueError(f"chord segment must end after it starts: {self}")


def read_chord_midi(path: str | Path) -> list[ChordSegment]:
    """Parse a POP909 chord_midi.txt (`start end label` per line, tab-separated)."""
    segments: list[ChordSegment] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"{path}:{lineno}: expected 3 columns, got {len(fields)}")
        segments.append(ChordSegment(float(fields[0]), float(fields[1]), fields[2]))
    return segments


def merge_repeats(segments: Iterable[ChordSegment]) -> list[ChordSegment]:
    """Merge contiguous segments that carry the same label."""
    merged: list[ChordSegment] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if prev and prev.label == seg.label and abs(seg.start - prev.end) <= _CONTIGUITY_TOLERANCE:
            merged[-1] = replace(prev, end=seg.end)
        else:
            merged.append(seg)
    return merged


# --------------------------------------------------------------------------- coarse vocabulary
def collapse(label: str) -> str:
    """Map a chord label onto CHORD_CLASSES, e.g. "F#:maj7/5" -> "F#:maj", "Db:hdim7" -> "C#:min".

    Idempotent on CHORD_CLASSES. Raises ValueError for labels it cannot classify.
    """
    if label == NO_CHORD:
        return NO_CHORD
    match = _LABEL.match(label)
    if not match:
        raise ValueError(f"unrecognised chord label {label!r}")
    quality = QUALITY_TO_CLASS.get(match["quality"])
    if quality is None:
        raise ValueError(f"no class for chord quality {match['quality']!r} in {label!r}")
    return f"{PITCH_CLASS_NAMES[_root_pitch_class(match['root'])]}:{quality}"


def _root_pitch_class(root: str) -> int:
    return (_LETTER_PITCH_CLASS[root[0]] + root.count("#") - root.count("b")) % 12


# --------------------------------------------------------------------------- grid-level events
@dataclass(frozen=True, order=True)
class ChordEvent:
    """A chord from CHORD_CLASSES taking effect at grid point (bar, position)."""

    bar: int
    position: int
    chord: str

    def __post_init__(self) -> None:
        if self.chord not in CHORD_CLASSES:
            raise ValueError(f"chord must be one of CHORD_CLASSES, got {self.chord!r}")
        if self.position < 0:
            raise ValueError(f"position must be non-negative: {self}")


def dedupe_changes(events: Iterable[ChordEvent]) -> list[ChordEvent]:
    """Order events by grid point and keep only real changes.

    Before any change the chord is N, so an event repeating the chord in effect (a leading N
    included) is dropped; of two events on the same grid point, the later one in `events` wins.
    """
    changes: list[ChordEvent] = []
    for event in sorted(events, key=lambda e: (e.bar, e.position)):
        if changes and (changes[-1].bar, changes[-1].position) == (event.bar, event.position):
            changes.pop()
        if event.chord != (changes[-1].chord if changes else NO_CHORD):
            changes.append(event)
    return changes


def chord_events(segments: Iterable[ChordSegment], grid: Grid) -> list[ChordEvent]:
    """Collapsed chord changes on `grid` from labelled segments (e.g. a POP909 chord file)."""
    return dedupe_changes(ChordEvent(*grid.quantize(seg.start), collapse(seg.label)) for seg in segments)
