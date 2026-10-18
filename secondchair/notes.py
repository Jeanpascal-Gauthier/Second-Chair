"""Plain note events in seconds, converted to and from symusic only at the MIDI boundary.

Inside the pipeline (bass extraction, tokenization, scheduling) notes are immutable `Note`s in
seconds; the Grid turns those seconds into bars and positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from symusic import Note as SymusicNote
from symusic import Score, Track


@dataclass(frozen=True, order=True)
class Note:
    start: float  # seconds
    end: float  # seconds
    pitch: int
    velocity: int

    def __post_init__(self) -> None:
        if not self.end > self.start:
            raise ValueError(f"note must end after it starts: {self}")
        if not 0 <= self.pitch <= 127 or not 1 <= self.velocity <= 127:
            raise ValueError(f"pitch must be 0..127 and velocity 1..127: {self}")

    @property
    def duration(self) -> float:
        return self.end - self.start


def cut_overlaps(line: Sequence[Note]) -> list[Note]:
    """One voice: end each note no later than the next one starts (`line` ordered by start)."""
    return [replace(n, end=min(n.end, nxt.start)) for n, nxt in zip(line, line[1:])] + list(line[-1:])


def playable(notes: Sequence[Note]) -> list[Note]:
    """Notes MIDI can hold: those ending by time 0 dropped, the rest starting no earlier than 0."""
    return [replace(n, start=max(n.start, 0.0)) for n in notes if n.end > 0]


def track_notes(score: Score, name: str) -> list[Note]:
    """Notes of the track called `name`, in seconds, sorted by (start, pitch).

    Zero-length notes are dropped: they never sound.
    """
    matches = [t for t in score.to("second").tracks if t.name == name]
    if len(matches) != 1:
        names = [t.name for t in score.tracks]
        raise KeyError(f"expected exactly one track named {name!r}, found {len(matches)} in {names}")
    return sorted(
        Note(n.time, n.time + n.duration, n.pitch, n.velocity)
        for n in matches[0].notes
        if n.duration > 0
    )


def to_track(notes: Sequence[Note], name: str, program: int = 0) -> Track:
    """A seconds-based symusic Track holding `notes` with the given General MIDI program."""
    track = Track(name=name, program=program, is_drum=False, ttype="second")
    for n in notes:
        track.notes.append(SymusicNote(n.start, n.duration, n.pitch, n.velocity, ttype="second"))
    return track
