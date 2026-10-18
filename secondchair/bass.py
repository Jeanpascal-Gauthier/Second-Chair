"""Derive a bassline from POP909's PIANO track.

POP909 has no bass track (only MELODY, BRIDGE and PIANO), so the bassline is extracted from the
piano. `extract_bass(score, strategy)` accepts any of the strategies below; each returns a single
voice of notes in seconds, in time order, keeping the piano's own (performed) onsets and
velocities:

- `LowestSounding`:      at each grid step, the lowest sounding piano note, taken when struck.
- `BelowThreshold`:      the piano's notes below a pitch threshold (its left-hand register).
- `LowestAtChordChange`: one note per chord, the lowest note sounding at the change, held to the
                         next change.

Shared rules:
- "Sounding" uses written note lengths; the sustain pedal is ignored.
- Timing decisions are made on the grid: onsets and releases are rounded to the nearest grid
  step, so notes struck on the same step count as simultaneous (the lowest wins), and a note
  released a hair after the next one starts does not hide it.
- One voice: each bass note is cut off where the next one starts.

Compare the strategies by ear with `uv run python -m secondchair.bass 001`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from symusic import Score

from secondchair.chords import NO_CHORD, ChordSegment, merge_repeats, read_chord_midi
from secondchair.grid import Grid
from secondchair.notes import Note, to_track, track_notes
from secondchair.pop909 import pop909_root, read_index, song_dir
from secondchair.render import render

BassStrategy = Callable[[Sequence[Note]], list[Note]]

SOURCE_TRACK = "PIANO"
BASS_TRACK = "BASS"
BASS_PROGRAM = 33  # General MIDI "Electric Bass (finger)"
DEFAULT_THRESHOLD = 55  # G3
STRATEGY_NAMES = ("lowest", "threshold", "chord")


def extract_bass(score: Score, strategy: BassStrategy, source: str = SOURCE_TRACK) -> list[Note]:
    """Bassline derived from the `source` track of `score` by `strategy`."""
    return strategy(track_notes(score, source))


@dataclass(frozen=True)
class LowestSounding:
    """At each grid step, the lowest sounding piano note, taken on the step where it is struck.

    A note joins the bass if no lower note is sounding on the step where it starts. A note struck
    earlier never re-enters the bass when a lower note is released, so the line contains no
    attacks the pianist didn't play: when the lowest voice rests, the bass rests.
    """

    grid: Grid

    def __call__(self, piano: Sequence[Note]) -> list[Note]:
        spans = _step_spans(self.grid, piano)
        lowest = _lowest_pitch_per_step(piano, spans)
        return _one_voice(self.grid, [n for n, (on, _) in zip(piano, spans) if n.pitch == lowest[on]])


@dataclass(frozen=True)
class BelowThreshold:
    """The piano's notes below `threshold` (its left-hand register) reduced to one voice.

    On each step where such notes are struck the lowest is taken; it lasts until it is released or
    the next one is struck, whichever comes first.
    """

    grid: Grid
    threshold: int = DEFAULT_THRESHOLD

    def __call__(self, piano: Sequence[Note]) -> list[Note]:
        return _one_voice(self.grid, [n for n in piano if n.pitch < self.threshold])


@dataclass(frozen=True)
class LowestAtChordChange:
    """One note per chord: the lowest piano note sounding at each chord change, held to the next.

    Identical neighbouring chord labels are merged first. A note struck on the change's grid step
    keeps its own onset; a note held across the change (a pedal point) is restruck at the change.
    "N" (no chord) segments, and changes where the piano is silent, get no note.
    """

    grid: Grid
    chords: Sequence[ChordSegment]

    def __call__(self, piano: Sequence[Note]) -> list[Note]:
        spans = _step_spans(self.grid, piano)
        line: list[Note] = []
        for seg in merge_repeats(self.chords):
            if seg.label == NO_CHORD:
                continue
            step = self.grid.step_at(seg.start)
            sounding = [(n, on) for n, (on, off) in zip(piano, spans) if on <= step < off]
            if not sounding:
                continue
            # lowest pitch wins; on a tie, prefer the note struck on the change
            note, on = min(sounding, key=lambda item: (item[0].pitch, item[1] != step))
            start = note.start if on == step else seg.start
            line.append(Note(start, seg.end, note.pitch, note.velocity))
        return _cut_overlaps(line)


def make_strategy(
    name: str, grid: Grid, chords: Sequence[ChordSegment], threshold: int = DEFAULT_THRESHOLD
) -> BassStrategy:
    """Strategy by its short name (see STRATEGY_NAMES)."""
    if name == "lowest":
        return LowestSounding(grid)
    if name == "threshold":
        return BelowThreshold(grid, threshold)
    if name == "chord":
        return LowestAtChordChange(grid, chords)
    raise ValueError(f"unknown bass strategy {name!r}; choose from {STRATEGY_NAMES}")


# --------------------------------------------------------------------------- shared rules
def _step_spans(grid: Grid, notes: Sequence[Note]) -> list[tuple[int, int]]:
    """Grid steps each note occupies: [onset step, release step), at least one step long."""
    spans = []
    for n in notes:
        on = grid.step_at(n.start)
        spans.append((on, max(grid.step_at(n.end), on + 1)))
    return spans


def _lowest_pitch_per_step(notes: Sequence[Note], spans: Sequence[tuple[int, int]]) -> dict[int, int]:
    lowest: dict[int, int] = {}
    for n, (on, off) in zip(notes, spans):
        for step in range(on, off):
            if n.pitch < lowest.get(step, 128):
                lowest[step] = n.pitch
    return lowest


def _one_voice(grid: Grid, notes: Iterable[Note]) -> list[Note]:
    """Keep the lowest note struck on each grid step (longest on a tie), cut at the next onset."""
    chosen: dict[int, Note] = {}
    for n in notes:
        step = grid.step_at(n.start)
        best = chosen.get(step)
        if best is None or (n.pitch, -n.duration) < (best.pitch, -best.duration):
            chosen[step] = n
    return _cut_overlaps([chosen[step] for step in sorted(chosen)])


def _cut_overlaps(line: Sequence[Note]) -> list[Note]:
    """End each note no later than the next one starts; `line` must be ordered by start."""
    return [replace(n, end=min(n.end, nxt.start)) for n, nxt in zip(line, line[1:])] + list(line[-1:])


# --------------------------------------------------------------------------- listening
def bass_score(
    score: Score, bass: Sequence[Note], *, arrangement: bool = True, program: int = BASS_PROGRAM
) -> Score:
    """Copy of `score` (in seconds) with `bass` added as a BASS track.

    With `arrangement=False` the original tracks are dropped; the tempo map is kept either way.
    """
    out = score.to("second")
    if not arrangement:
        out.tracks.clear()
    out.tracks.append(to_track(bass, BASS_TRACK, program))
    return out


def render_comparison(
    song_id: str,
    out_dir: Path,
    strategies: Sequence[str] = STRATEGY_NAMES,
    threshold: int = DEFAULT_THRESHOLD,
    beat_res: int = 4,
    root: Path | None = None,
) -> dict[str, list[Note]]:
    """Render the original song plus, per strategy, the bass with the arrangement and solo.

    Writes original.wav, <strategy>.mid/.wav and <strategy>_solo.mid/.wav into `out_dir` and
    returns each strategy's bassline.
    """
    root = root or pop909_root()
    folder = song_dir(song_id, root)
    grid = Grid.from_pop909(folder / "beat_midi.txt", read_index(root)[song_id].beats_per_measure, beat_res)
    chords = read_chord_midi(folder / "chord_midi.txt")
    score = Score(str(folder / f"{song_id}.mid"), ttype="second")

    out_dir.mkdir(parents=True, exist_ok=True)
    render(folder / f"{song_id}.mid", out_dir / "original.wav")
    lines: dict[str, list[Note]] = {}
    for name in strategies:
        lines[name] = extract_bass(score, make_strategy(name, grid, chords, threshold))
        for suffix, arrangement in (("", True), ("_solo", False)):
            midi = out_dir / f"{name}{suffix}.mid"
            bass_score(score, lines[name], arrangement=arrangement).dump_midi(str(midi))
            render(midi, midi.with_suffix(".wav"))
    return lines


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render each bass extraction strategy for a POP909 song.")
    parser.add_argument("song", help="POP909 song id, e.g. 001")
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, action="append", help="repeatable; default: all")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="for 'threshold' (MIDI pitch)")
    parser.add_argument("--beat-res", type=int, default=4, help="grid positions per beat")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: renders/<song>)")
    args = parser.parse_args(argv)

    song = args.song.zfill(3)
    out_dir = args.out or Path("renders") / song
    lines = render_comparison(song, out_dir, args.strategy or STRATEGY_NAMES, args.threshold, args.beat_res)
    for name, line in lines.items():
        pitches = [n.pitch for n in line]
        span = f"pitches {min(pitches)}-{max(pitches)}" if pitches else "empty"
        print(f"{name:>9}: {len(line):4d} notes, {span}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
