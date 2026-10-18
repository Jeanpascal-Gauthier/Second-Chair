"""REMI tokenization of a bassline and its chords on a beat-locked grid.

One bar of tokens:

    Bar Position_0 Chord_<c> [Pitch_<p> Velocity_<v> Duration_<d>]...
        Position_<q> [Chord_<c>] [Pitch_<p> Velocity_<v> Duration_<d>]...

- Bar and Position place events on the metrical grid: a position counts grid steps (1/beat_res
  of a beat) from the bar's downbeat. Notes carry a Duration in grid steps; there are no
  time-shift or note-off tokens.
- Every bar opens with Position_0 and the chord in effect there, restated even when unchanged, so
  each bar stands alone; any other Chord token marks a change. At one position the chord comes
  first, then the notes in pitch order.
- Notes are split at barlines (restruck on each downbeat), so no note outlasts its bar and a
  Duration never exceeds the bar length.
- Velocities are bucketed into `n_velocities` equal-width levels.
- Chords use the coarse vocabulary `secondchair.chords.CHORD_CLASSES`.

Bar lengths are not tokenized: decoding takes them from the Grid. A song's sequence starts at its
first bar holding a note or a chord other than N; `TokenSequence.first_bar` records which grid bar
that is, so decoding can lay the tokens back onto the song's grid.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from symusic import Score, Tempo, TextMeta

from secondchair.bass import BASS_PROGRAM, BASS_TRACK
from secondchair.chords import CHORD_CLASSES, NO_CHORD, ChordEvent, collapse, dedupe_changes
from secondchair.grid import Grid
from secondchair.notes import Note, to_track, track_notes

PAD, BOS, EOS, BAR = "PAD", "BOS", "EOS", "Bar"
MIDI_TICKS_PER_QUARTER = 480
MIDI_QPM = 120.0  # tempo of MIDI files written by `to_midi`; times are kept in seconds regardless


@dataclass(frozen=True)
class TokenSequence:
    tokens: tuple[str, ...]
    first_bar: int  # grid bar of the first Bar token


@dataclass(frozen=True, order=True)
class _Placed:
    """A note on the grid: steps from its bar's downbeat, pitch, length in steps, velocity level."""

    bar: int
    position: int
    pitch: int
    duration: int
    level: int


@dataclass(frozen=True)
class RemiTokenizer:
    beat_res: int = 4  # grid steps per beat; must match the Grid
    max_bar_beats: int = 8  # longest bar the vocabulary can express
    n_velocities: int = 8

    def __post_init__(self) -> None:
        if self.beat_res < 1 or self.max_bar_beats < 1:
            raise ValueError("beat_res and max_bar_beats must be positive")
        if not 1 <= self.n_velocities <= 64:
            raise ValueError("n_velocities must be between 1 and 64")

    # ------------------------------------------------------------------ vocabulary
    @property
    def max_bar_steps(self) -> int:
        return self.max_bar_beats * self.beat_res

    @cached_property
    def vocab(self) -> tuple[str, ...]:
        steps = self.max_bar_steps
        return (
            PAD, BOS, EOS, BAR,
            *(f"Position_{p}" for p in range(steps)),
            *(f"Chord_{c}" for c in CHORD_CLASSES),
            *(f"Pitch_{p}" for p in range(128)),
            *(f"Velocity_{v}" for v in range(self.n_velocities)),
            *(f"Duration_{d}" for d in range(1, steps + 1)),
        )

    @cached_property
    def _ids(self) -> dict[str, int]:
        return {token: i for i, token in enumerate(self.vocab)}

    def to_ids(self, tokens: Sequence[str]) -> list[int]:
        unknown = [t for t in tokens if t not in self._ids]
        if unknown:
            raise ValueError(f"tokens not in the vocabulary: {unknown[:5]}")
        return [self._ids[t] for t in tokens]

    def to_tokens(self, ids: Sequence[int]) -> list[str]:
        bad = [i for i in ids if not 0 <= i < len(self.vocab)]
        if bad:
            raise ValueError(f"ids outside the vocabulary: {bad[:5]}")
        return [self.vocab[i] for i in ids]

    # ------------------------------------------------------------------ velocity
    def velocity_level(self, velocity: int) -> int:
        return min(velocity * self.n_velocities // 128, self.n_velocities - 1)

    def level_velocity(self, level: int) -> int:
        """Velocity a level decodes to: the middle of its range."""
        return max(1, round((level + 0.5) * 128 / self.n_velocities))

    # ------------------------------------------------------------------ encoding
    def encode(
        self, notes: Sequence[Note], chords: Sequence[ChordEvent], grid: Grid, bars: range | None = None
    ) -> TokenSequence:
        """Tokens for `notes` (seconds) and chord events on `grid`.

        `bars` defaults to the span from the first to the last bar holding a note or a chord other
        than N; notes starting outside `bars` are left out.
        """
        self._check_grid(grid)
        placed = [self._place(note, grid) for note in split_at_barlines(notes, grid)]
        changes = dedupe_changes(chords)
        if bars is None:
            bars = _content_bars(placed, changes)
        by_bar: dict[int, list[_Placed]] = defaultdict(list)
        for p in sorted(placed):
            by_bar[p.bar].append(p)
        tokens = [BOS]
        for bar in bars:
            tokens += self._encode_bar(bar, by_bar[bar], changes, grid)
        tokens.append(EOS)
        return TokenSequence(tuple(tokens), bars.start)

    def _place(self, note: Note, grid: Grid) -> _Placed:
        on = grid.step_at(note.start)
        off = max(grid.step_at(note.end), on + 1)
        bar = grid.bar_at_step(on)
        position = on - grid.bar_start_beat(bar) * self.beat_res
        return _Placed(bar, position, note.pitch, off - on, self.velocity_level(note.velocity))

    def _encode_bar(
        self, bar: int, placed: Sequence[_Placed], changes: Sequence[ChordEvent], grid: Grid
    ) -> list[str]:
        n_steps = grid.positions_in_bar(bar)
        if n_steps > self.max_bar_steps:
            raise ValueError(
                f"bar {bar} has {grid.beats_in_bar(bar)} beats; the tokenizer supports at most {self.max_bar_beats}"
            )
        chords_at = {c.position: c.chord for c in changes if c.bar == bar}
        if any(position >= n_steps for position in chords_at):
            raise ValueError(f"chord event outside bar {bar} ({n_steps} positions)")
        chords_at.setdefault(0, _chord_in_effect(changes, bar))
        notes_at: dict[int, list[_Placed]] = defaultdict(list)
        for p in placed:
            notes_at[p.position].append(p)

        tokens = [BAR]
        for position in sorted(chords_at.keys() | notes_at.keys()):
            tokens.append(f"Position_{position}")
            if position in chords_at:
                tokens.append(f"Chord_{chords_at[position]}")
            for p in notes_at[position]:
                tokens += [f"Pitch_{p.pitch}", f"Velocity_{p.level}", f"Duration_{p.duration}"]
        return tokens

    # ------------------------------------------------------------------ decoding
    def decode(
        self, tokens: Sequence[str], grid: Grid, first_bar: int = 0
    ) -> tuple[list[Note], list[ChordEvent]]:
        """Notes (seconds) and chord changes, laying the first Bar token on grid bar `first_bar`.

        Raises ValueError, naming the offending token, for sequences that are not well formed.
        """
        self._check_grid(grid)
        body = self._body(tokens)
        notes: list[Note] = []
        chords: list[ChordEvent] = []
        bar, position, n_steps = first_bar - 1, None, 0
        i = 0
        while i < len(body):
            token = body[i]
            kind, _, value = token.partition("_")
            where = f"token {i + 1} ({token})"
            if token == BAR:
                bar, position, n_steps = bar + 1, None, grid.positions_in_bar(bar + 1)
            elif kind == "Position":
                if bar < first_bar:
                    raise ValueError(f"{where}: Position before the first Bar")
                if position is not None and int(value) <= position:
                    raise ValueError(f"{where}: positions must increase within a bar")
                if int(value) >= n_steps:
                    raise ValueError(f"{where}: bar {bar} has only {n_steps} positions")
                position = int(value)
            elif kind == "Chord":
                if i == 0 or not body[i - 1].startswith("Position_"):
                    raise ValueError(f"{where}: a Chord must directly follow a Position")
                chords.append(ChordEvent(bar, position, value))
            elif kind == "Pitch":
                notes.append(self._decode_note(body, i, bar, position, n_steps, grid))
                i += 2
            else:
                raise ValueError(f"{where}: unexpected token")
            i += 1
        return notes, dedupe_changes(chords)

    def _decode_note(
        self, body: Sequence[str], i: int, bar: int, position: int | None, n_steps: int, grid: Grid
    ) -> Note:
        where = f"token {i + 1} ({body[i]})"
        if position is None:
            raise ValueError(f"{where}: a note needs a Position first")
        follow = list(body[i + 1 : i + 3])
        if len(follow) < 2 or not follow[0].startswith("Velocity_") or not follow[1].startswith("Duration_"):
            raise ValueError(f"{where}: Pitch must be followed by Velocity and Duration")
        steps = int(follow[1].partition("_")[2])
        if position + steps > n_steps:
            raise ValueError(f"{where}: note runs past the end of bar {bar}")
        first = grid.bar_start_beat(bar) * self.beat_res + position
        return Note(
            _step_time(grid, first),
            _step_time(grid, first + steps),
            int(body[i].partition("_")[2]),
            self.level_velocity(int(follow[0].partition("_")[2])),
        )

    def _body(self, tokens: Sequence[str]) -> list[str]:
        """Tokens between BOS and EOS, after checking the frame (BOS ... EOS PAD*)."""
        tokens = list(tokens)
        self.to_ids(tokens)
        if not tokens or tokens[0] != BOS:
            raise ValueError("sequence must start with BOS")
        if EOS not in tokens:
            raise ValueError("sequence must end with EOS")
        end = tokens.index(EOS)
        if any(t != PAD for t in tokens[end + 1 :]):
            raise ValueError("only PAD may follow EOS")
        body = tokens[1:end]
        if any(t in (PAD, BOS) for t in body):
            raise ValueError("PAD or BOS inside the sequence")
        return body

    def _check_grid(self, grid: Grid) -> None:
        if grid.beat_res != self.beat_res:
            raise ValueError(f"grid beat_res {grid.beat_res} does not match tokenizer beat_res {self.beat_res}")


# --------------------------------------------------------------------------- notes on the grid
def split_at_barlines(notes: Sequence[Note], grid: Grid) -> list[Note]:
    """Split notes sounding across a barline into one note per bar, restruck on each downbeat.

    Crossing is judged on the grid: a note whose release rounds onto the barline does not cross
    it. Idempotent; returns notes sorted by start.
    """
    pieces: list[Note] = []
    for note in notes:
        on = grid.step_at(note.start)
        off = max(grid.step_at(note.end), on + 1)
        bar, start = grid.bar_at_step(on), note.start
        while (barline := grid.bar_start_beat(bar + 1) * grid.beat_res) < off:
            downbeat = _step_time(grid, barline)
            pieces.append(Note(start, downbeat, note.pitch, note.velocity))
            start, bar = downbeat, bar + 1
        pieces.append(Note(start, note.end, note.pitch, note.velocity))
    return sorted(pieces)


def _step_time(grid: Grid, step: int) -> float:
    return grid.time_at(step / grid.beat_res)


def _content_bars(placed: Sequence[_Placed], changes: Sequence[ChordEvent]) -> range:
    bars = [p.bar for p in placed] + [c.bar for c in changes if c.chord != NO_CHORD]
    return range(min(bars), max(bars) + 1) if bars else range(0)


def _chord_in_effect(changes: Sequence[ChordEvent], bar: int) -> str:
    """Chord sounding at the downbeat of `bar` from changes before it (N before any change)."""
    earlier = [c for c in changes if c.bar < bar]
    return earlier[-1].chord if earlier else NO_CHORD


# --------------------------------------------------------------------------- MIDI
def to_midi(notes: Sequence[Note], chords: Sequence[ChordEvent], grid: Grid, program: int = BASS_PROGRAM) -> Score:
    """Seconds-based Score holding `notes` as a BASS track and each chord as a marker naming it.

    MIDI cannot hold events before time 0, so those raise ValueError instead of being garbled.
    """
    marker_times = [grid.to_seconds(event.bar, event.position) for event in chords]
    earliest = min([n.start for n in notes] + marker_times, default=0.0)
    if earliest < 0:
        raise ValueError(f"MIDI cannot hold events before time 0; the earliest is at {earliest:.3f}s")
    score = Score(MIDI_TICKS_PER_QUARTER)
    score.tempos.append(Tempo(time=0, qpm=MIDI_QPM))
    score = score.to("second")
    score.tracks.append(to_track(notes, BASS_TRACK, program))
    for event, time in zip(chords, marker_times):
        score.markers.append(TextMeta(time, event.chord, ttype="second"))
    return score


def from_midi(score: Score, grid: Grid, track: str = BASS_TRACK) -> tuple[list[Note], list[ChordEvent]]:
    """Notes of `track` and the chord events in the markers of a Score shaped like `to_midi`'s."""
    seconds = score.to("second")
    chords = [ChordEvent(*grid.quantize(marker.time), collapse(marker.text)) for marker in seconds.markers]
    return track_notes(seconds, track), chords
