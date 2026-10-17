"""Beat-locked metrical grid: map between seconds and (bar, position).

A `Grid` is built from annotated beat times plus the beat indices where bars start. Times between
beats are linearly interpolated, so tempo may drift freely; nothing assumes a constant tempo or a
fixed number of ticks per beat. Outside the annotated beats the grid extrapolates at the first /
last beat interval.

Conventions
-----------
- Bar 0 starts at the first annotated downbeat. Material before it (pickups) falls in negative
  bars laid out backwards at the nominal bar length, so an upbeat on beat 4 is the last beat of
  bar -1, and a note before the first annotated beat still has a metrical position.
- Bars between annotated downbeats keep their annotated length (POP909 has irregular 2-, 3-, 5-
  and 6-beat bars). From the last annotated downbeat onwards, bars have the nominal length.
- A bar of L beats has positions 0 .. L * beat_res - 1. Bars never share a position: a time that
  rounds onto a barline belongs to position 0 of the next bar.

POP909
------
beat_midi.txt has three space-separated columns: beat time (s), a flag marking POP909's measures
(every `num_beats_per_measure` beats per index.xlsx, i.e. 2 or 3), and a flag marking pairs of
those measures (every 4 or 6 beats). `Grid.from_pop909` takes bars from the pair column for duple
songs (4-beat bars) and from the measure column for triple songs (3-beat bars). The MIDI files'
own time signatures and barlines are never used: they are unreliable, and the MIDI tick grid is
offset from the annotated beats by a fraction of a beat.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BeatAnnotations:
    """Contents of a POP909 beat_midi.txt; flag columns are stored as beat indices."""

    times: tuple[float, ...]
    measure_starts: tuple[int, ...]  # column 2: every 2 (or 3) beats
    double_measure_starts: tuple[int, ...]  # column 3: every 4 (or 6) beats


def read_beat_midi(path: str | Path) -> BeatAnnotations:
    """Parse a POP909 beat_midi.txt (whitespace-separated `time flag flag` lines)."""
    times: list[float] = []
    measure_starts: list[int] = []
    double_measure_starts: list[int] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"{path}:{lineno}: expected 3 columns, got {len(fields)}")
        index = len(times)
        times.append(float(fields[0]))
        if _parse_flag(fields[1], path, lineno):
            measure_starts.append(index)
        if _parse_flag(fields[2], path, lineno):
            double_measure_starts.append(index)
    return BeatAnnotations(tuple(times), tuple(measure_starts), tuple(double_measure_starts))


def _parse_flag(text: str, path: str | Path, lineno: int) -> bool:
    value = float(text)
    if value not in (0.0, 1.0):
        raise ValueError(f"{path}:{lineno}: downbeat flag must be 0 or 1, got {text!r}")
    return value == 1.0


@dataclass(frozen=True, repr=False)
class Grid:
    """Bidirectional map between seconds and (bar, position) on an annotated beat grid.

    beat_times:    strictly increasing beat times in seconds (at least two).
    downbeats:     strictly increasing indices into `beat_times` where bars start (at least one).
    beats_per_bar: nominal bar length, used only for bars before the first / after the last
                   annotated downbeat.
    beat_res:      grid positions per beat (4 = sixteenths, 12 also covers triplets).
    """

    beat_times: Sequence[float]
    downbeats: Sequence[int]
    beats_per_bar: int
    beat_res: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "beat_times", tuple(float(t) for t in self.beat_times))
        object.__setattr__(self, "downbeats", tuple(int(d) for d in self.downbeats))
        _validate(self.beat_times, self.downbeats, self.beats_per_bar, self.beat_res)

    @classmethod
    def from_pop909(cls, beat_file: str | Path, beats_per_measure: int, beat_res: int = 4) -> Grid:
        """Grid for a POP909 song from its beat_midi.txt and index.xlsx num_beats_per_measure."""
        ann = read_beat_midi(beat_file)
        if beats_per_measure == 2:
            return cls(ann.times, ann.double_measure_starts, 4, beat_res)
        if beats_per_measure == 3:
            return cls(ann.times, ann.measure_starts, 3, beat_res)
        raise ValueError(f"num_beats_per_measure must be 2 or 3, got {beats_per_measure}")

    @classmethod
    def constant_tempo(
        cls, bpm: float, beats_per_bar: int = 4, beat_res: int = 4, start: float = 0.0
    ) -> Grid:
        """Steady tempo with bar 0 starting at `start` seconds, extending indefinitely."""
        return cls((start, start + 60.0 / bpm), (0,), beats_per_bar, beat_res)

    def __repr__(self) -> str:
        return (
            f"Grid({len(self.beat_times)} beats {self.beat_times[0]:.3f}..{self.beat_times[-1]:.3f}s, "
            f"{len(self.downbeats)} annotated bars, beats_per_bar={self.beats_per_bar}, "
            f"beat_res={self.beat_res})"
        )

    # ------------------------------------------------------------------ beats <-> seconds
    def beat_at(self, seconds: float) -> float:
        """Fractional beat index at `seconds`; 0.0 is the first annotated beat."""
        t = self.beat_times
        last = len(t) - 1
        if seconds < t[0]:
            return (seconds - t[0]) / (t[1] - t[0])
        if seconds >= t[last]:
            return last + (seconds - t[last]) / (t[last] - t[last - 1])
        j = bisect_right(t, seconds) - 1
        return j + (seconds - t[j]) / (t[j + 1] - t[j])

    def time_at(self, beat: float) -> float:
        """Seconds at fractional beat index `beat` (inverse of `beat_at`)."""
        t = self.beat_times
        last = len(t) - 1
        if beat < 0:
            return t[0] + beat * (t[1] - t[0])
        if beat >= last:
            return t[last] + (beat - last) * (t[last] - t[last - 1])
        j = int(beat)
        return t[j] + (beat - j) * (t[j + 1] - t[j])

    # ------------------------------------------------------------------ bars
    def bar_start_beat(self, bar: int) -> int:
        """Beat index where `bar` starts (negative for pickup bars before the first beat)."""
        d, n = self.downbeats, self.beats_per_bar
        if bar < 0:
            return d[0] + bar * n
        if bar < len(d):
            return d[bar]
        return d[-1] + (bar - len(d) + 1) * n

    def beats_in_bar(self, bar: int) -> int:
        return self.bar_start_beat(bar + 1) - self.bar_start_beat(bar)

    def positions_in_bar(self, bar: int) -> int:
        return self.beats_in_bar(bar) * self.beat_res

    def bar_at_step(self, step: int) -> int:
        """Bar containing grid step `step` (= beat index * beat_res)."""
        d, r, n = self.downbeats, self.beat_res, self.beats_per_bar
        if step < d[0] * r:
            return (step - d[0] * r) // (n * r)
        if step >= d[-1] * r:
            return len(d) - 1 + (step - d[-1] * r) // (n * r)
        return bisect_right(d, step // r) - 1

    # ------------------------------------------------------------------ public mapping
    def quantize(self, seconds: float) -> tuple[int, int]:
        """Nearest grid point to `seconds` as (bar, position); halfway rounds later."""
        step = math.floor(self.beat_at(seconds) * self.beat_res + 0.5)
        bar = self.bar_at_step(step)
        return bar, step - self.bar_start_beat(bar) * self.beat_res

    def to_seconds(self, bar: int, position: int) -> float:
        """Time of grid point (bar, position); position must lie inside the bar."""
        if not 0 <= position < self.positions_in_bar(bar):
            raise ValueError(
                f"position {position} outside bar {bar} (0..{self.positions_in_bar(bar) - 1})"
            )
        return self.time_at(self.bar_start_beat(bar) + position / self.beat_res)


def _validate(
    beat_times: tuple[float, ...], downbeats: tuple[int, ...], beats_per_bar: int, beat_res: int
) -> None:
    if len(beat_times) < 2:
        raise ValueError("need at least two beat times")
    if not all(math.isfinite(t) for t in beat_times):
        raise ValueError("beat times must be finite")
    if any(b <= a for a, b in zip(beat_times, beat_times[1:])):
        raise ValueError("beat times must be strictly increasing")
    if not downbeats:
        raise ValueError("need at least one downbeat")
    if any(b <= a for a, b in zip(downbeats, downbeats[1:])):
        raise ValueError("downbeats must be strictly increasing")
    if downbeats[0] < 0 or downbeats[-1] >= len(beat_times):
        raise ValueError(f"downbeats must index into the {len(beat_times)} beat times")
    if beats_per_bar < 1 or beat_res < 1:
        raise ValueError("beats_per_bar and beat_res must be positive")
