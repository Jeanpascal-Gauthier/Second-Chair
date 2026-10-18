from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from symusic import Score

from secondchair.bass import LowestAtChordChange, extract_bass
from secondchair.chords import ChordEvent, chord_events, read_chord_midi
from secondchair.grid import Grid
from secondchair.notes import Note
from secondchair.pop909 import DEFAULT_ROOT, read_index
from secondchair.tokenizer import (
    BOS,
    EOS,
    PAD,
    RemiTokenizer,
    TokenSequence,
    from_midi,
    split_at_barlines,
    to_midi,
)

TOK = RemiTokenizer()
GRID = Grid.constant_tempo(bpm=60, beats_per_bar=4)  # one beat per second, 16 steps per bar

# A drifting, irregular grid: bars of 4, 2, 6 beats then 4, with three beats of pickup.
DRIFT_TIMES = [0.0, 0.5, 1.1, 1.6, 2.0, 2.5, 3.0, 3.6, 4.1, 4.5, 5.0, 5.4, 5.9, 6.5, 7.0, 7.4, 7.9, 8.3, 8.8]
DRIFT_DOWNBEATS = [3, 7, 9, 15]

needs_pop909 = pytest.mark.skipif(not DEFAULT_ROOT.is_dir(), reason="POP909 not extracted to data/")
REAL_SONGS = ["001", "012", "034", "102", "418", "806"]  # duple, irregular bars, triple, ..., triplet feel


def n(start: float, end: float, pitch: int, velocity: int = 80) -> Note:
    return Note(start, end, pitch, velocity)


def through_midi_file(score: Score, path: Path) -> Score:
    score.dump_midi(str(path))
    return Score(str(path))


@cache
def pop909_song(song: str, beat_res: int = 4) -> tuple[Grid, list[Note], list[ChordEvent]]:
    """Grid, `chord`-strategy bassline and chord events of a POP909 song."""
    folder = DEFAULT_ROOT / song
    grid = Grid.from_pop909(folder / "beat_midi.txt", _index()[song].beats_per_measure, beat_res)
    segments = read_chord_midi(folder / "chord_midi.txt")
    bass = extract_bass(Score(str(folder / f"{song}.mid"), ttype="second"), LowestAtChordChange(grid, segments))
    return grid, bass, chord_events(segments, grid)


@cache
def _index():  # noqa: ANN202
    return read_index(DEFAULT_ROOT)


# --------------------------------------------------------------------------- vocabulary
def test_vocabulary_layout() -> None:
    assert len(TOK.vocab) == len(set(TOK.vocab)) == 4 + 32 + 25 + 128 + 8 + 32
    assert TOK.vocab[:4] == (PAD, "BOS", "EOS", "Bar")
    assert TOK.to_ids([PAD]) == [0]
    assert "Position_31" in TOK.vocab and "Position_32" not in TOK.vocab
    assert "Duration_32" in TOK.vocab and "Duration_0" not in TOK.vocab


def test_vocabulary_scales_with_beat_res() -> None:
    tok12 = RemiTokenizer(beat_res=12)
    assert tok12.max_bar_steps == 96
    assert "Position_95" in tok12.vocab and "Duration_96" in tok12.vocab


def test_ids_round_trip_and_reject_unknowns() -> None:
    tokens = ["BOS", "Bar", "Position_0", "Chord_C:maj", "Pitch_36", "Velocity_5", "Duration_16", "EOS"]
    assert TOK.to_tokens(TOK.to_ids(tokens)) == tokens
    with pytest.raises(ValueError, match="not in the vocabulary"):
        TOK.to_ids(["Position_99"])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        TOK.to_tokens([-1])


@pytest.mark.parametrize("kwargs", [{"beat_res": 0}, {"max_bar_beats": 0}, {"n_velocities": 0}, {"n_velocities": 65}])
def test_tokenizer_rejects_bad_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        RemiTokenizer(**kwargs)


# --------------------------------------------------------------------------- velocity levels
def test_velocity_levels_are_sixteen_wide() -> None:
    assert [TOK.velocity_level(v) for v in (1, 15, 16, 31, 64, 112, 127)] == [0, 0, 1, 1, 4, 7, 7]
    assert [TOK.level_velocity(level) for level in range(8)] == [8, 24, 40, 56, 72, 88, 104, 120]


@pytest.mark.parametrize("n_velocities", [1, 8, 16, 32, 64])
def test_every_level_decodes_inside_itself(n_velocities: int) -> None:
    tok = RemiTokenizer(n_velocities=n_velocities)
    assert all(tok.velocity_level(tok.level_velocity(level)) == level for level in range(n_velocities))
    worst = max(abs(v - tok.level_velocity(tok.velocity_level(v))) for v in range(1, 128))
    assert worst <= 64 / n_velocities


# --------------------------------------------------------------------------- barline splitting
def test_split_leaves_notes_inside_a_bar_alone() -> None:
    assert split_at_barlines([n(0, 2, 40)], GRID) == [n(0, 2, 40)]


def test_split_restrikes_on_each_downbeat() -> None:
    assert split_at_barlines([n(3, 5, 40)], GRID) == [n(3, 4, 40), n(4, 5, 40)]
    assert split_at_barlines([n(3, 9, 40, 99)], GRID) == [n(3, 4, 40, 99), n(4, 8, 40, 99), n(8, 9, 40, 99)]


def test_split_ignores_a_release_that_rounds_onto_the_barline() -> None:
    assert split_at_barlines([n(2, 4.1, 40)], GRID) == [n(2, 4.1, 40)]


def test_split_follows_irregular_bars_and_is_idempotent() -> None:
    grid = Grid(DRIFT_TIMES, DRIFT_DOWNBEATS, beats_per_bar=4)
    pieces = split_at_barlines([n(DRIFT_TIMES[1], DRIFT_TIMES[12], 40)], grid)
    assert [p.start for p in pieces] == [DRIFT_TIMES[k] for k in (1, 3, 7, 9)]  # bars -1, 0, 1 (2 beats), 2
    assert split_at_barlines(pieces, grid) == pieces


# --------------------------------------------------------------------------- encoding layout
def test_encode_example_bar() -> None:
    seq = TOK.encode([n(0, 2, 47, 65), n(2, 4, 49, 97)], [ChordEvent(0, 0, "B:maj"), ChordEvent(0, 8, "C#:maj")], GRID)
    assert seq == TokenSequence(
        (BOS, "Bar", "Position_0", "Chord_B:maj", "Pitch_47", "Velocity_4", "Duration_8",
         "Position_8", "Chord_C#:maj", "Pitch_49", "Velocity_6", "Duration_8", EOS),
        first_bar=0,
    )


def test_encode_restates_the_chord_at_every_bar() -> None:
    seq = TOK.encode([n(0, 1, 40), n(8, 9, 43)], [ChordEvent(0, 0, "C:maj")], GRID)
    assert seq.tokens == (
        BOS,
        "Bar", "Position_0", "Chord_C:maj", "Pitch_40", "Velocity_5", "Duration_4",
        "Bar", "Position_0", "Chord_C:maj",
        "Bar", "Position_0", "Chord_C:maj", "Pitch_43", "Velocity_5", "Duration_4",
        EOS,
    )


def test_encode_states_no_chord_when_none_is_known() -> None:
    assert TOK.encode([n(0, 1, 40)], [], GRID).tokens[1:4] == ("Bar", "Position_0", "Chord_N")


def test_encode_chord_change_without_a_note() -> None:
    seq = TOK.encode([n(0, 1, 36)], [ChordEvent(0, 0, "C:maj"), ChordEvent(0, 12, "G:maj")], GRID)
    assert seq.tokens[-3:] == ("Position_12", "Chord_G:maj", EOS)


def test_encode_simultaneous_notes_in_pitch_order() -> None:
    tokens = TOK.encode([n(0, 1, 48, 30), n(0, 2, 36, 100)], [], GRID).tokens
    assert tokens[4:10] == ("Pitch_36", "Velocity_6", "Duration_8", "Pitch_48", "Velocity_1", "Duration_4")


def test_encode_held_note_is_restruck_in_each_bar() -> None:
    tokens = TOK.encode([n(2, 10, 40)], [ChordEvent(0, 0, "C:maj")], GRID).tokens
    durations = [t for t in tokens if t.startswith("Duration_")]
    assert durations == ["Duration_8", "Duration_16", "Duration_8"]


def test_encode_spans_bars_from_first_to_last_content() -> None:
    chords = [ChordEvent(0, 0, "N"), ChordEvent(2, 0, "C:maj"), ChordEvent(5, 0, "N")]
    seq = TOK.encode([n(12, 13, 40)], chords, GRID)  # note in bar 3; trailing N is not content
    assert seq.first_bar == 2
    assert seq.tokens.count("Bar") == 2


def test_encode_explicit_bars() -> None:
    seq = TOK.encode([n(12, 13, 40)], [ChordEvent(0, 0, "A:min")], GRID, bars=range(0, 2))
    assert seq.first_bar == 0
    assert seq.tokens == (BOS, "Bar", "Position_0", "Chord_A:min", "Bar", "Position_0", "Chord_A:min", EOS)


def test_encode_nothing() -> None:
    assert TOK.encode([], [ChordEvent(0, 0, "N")], GRID) == TokenSequence((BOS, EOS), first_bar=0)


def test_encode_rejects_bars_longer_than_the_vocabulary() -> None:
    grid = Grid([0.5 * k for k in range(20)], [0, 9], beats_per_bar=4)  # bar 0 has 9 beats
    with pytest.raises(ValueError, match="at most 8"):
        TOK.encode([n(0.0, 0.5, 40)], [], grid)


def test_encode_rejects_a_grid_with_another_resolution() -> None:
    with pytest.raises(ValueError, match="beat_res"):
        TOK.encode([n(0, 1, 40)], [], Grid.constant_tempo(bpm=60, beat_res=12))


# --------------------------------------------------------------------------- decoding
def test_decode_lands_notes_on_the_grid() -> None:
    seq = TOK.encode([n(0.05, 1.9, 40, 70), n(4.1, 6, 43, 100)], [ChordEvent(0, 0, "C:maj"), ChordEvent(1, 0, "C:maj")], GRID)
    notes, chords = TOK.decode(seq.tokens, GRID, seq.first_bar)
    assert notes == [n(0, 2, 40, 72), n(4, 6, 43, 104)]
    assert chords == [ChordEvent(0, 0, "C:maj")]  # the restatement in bar 1 is not a change


def test_decode_places_the_first_bar_where_asked() -> None:
    seq = TOK.encode([n(0, 1, 40)], [ChordEvent(0, 0, "C:maj")], GRID)
    notes, chords = TOK.decode(seq.tokens, GRID, first_bar=2)
    assert notes == [n(8, 9, 40, 88)]
    assert chords == [ChordEvent(2, 0, "C:maj")]


def test_decode_accepts_padding_after_eos() -> None:
    tokens = [BOS, "Bar", "Position_0", "Chord_A:min", EOS, PAD, PAD]
    assert TOK.decode(tokens, GRID) == ([], [ChordEvent(0, 0, "A:min")])


def test_decode_reports_no_change_for_a_leading_no_chord() -> None:
    assert TOK.decode([BOS, "Bar", "Position_0", "Chord_N", EOS], GRID) == ([], [])


def test_to_midi_refuses_events_before_time_zero() -> None:
    grid = Grid.constant_tempo(bpm=60, start=0.5)  # bar -1 starts at -3.5 s
    with pytest.raises(ValueError, match="before time 0"):
        to_midi([], [ChordEvent(-1, 0, "C:maj")], grid)
    with pytest.raises(ValueError, match="before time 0"):
        to_midi([n(-0.25, 1, 40)], [], grid)


@pytest.mark.parametrize(
    ("tokens", "match"),
    [
        (["Bar", EOS], "start with BOS"),
        ([BOS, "Bar"], "end with EOS"),
        ([BOS, EOS, "Bar"], "only PAD may follow EOS"),
        ([BOS, PAD, EOS], "inside the sequence"),
        ([BOS, "Position_99", EOS], "not in the vocabulary"),
        ([BOS, "Position_0", EOS], "before the first Bar"),
        ([BOS, "Bar", "Position_4", "Position_2", EOS], "must increase"),
        ([BOS, "Bar", "Position_20", EOS], "only 16 positions"),
        ([BOS, "Bar", "Chord_C:maj", EOS], "directly follow a Position"),
        ([BOS, "Bar", "Position_0", "Chord_C:maj", "Chord_G:maj", EOS], "directly follow a Position"),
        ([BOS, "Bar", "Pitch_40", "Velocity_3", "Duration_8", EOS], "needs a Position"),
        ([BOS, "Bar", "Position_0", "Pitch_40", "Duration_4", EOS], "followed by Velocity and Duration"),
        ([BOS, "Bar", "Position_12", "Pitch_40", "Velocity_3", "Duration_8", EOS], "past the end of bar 0"),
        ([BOS, "Bar", "Velocity_3", EOS], "unexpected token"),
    ],
)
def test_decode_rejects_malformed_sequences(tokens: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TOK.decode(tokens, GRID)


# --------------------------------------------------------------------------- 1. bit-exact
def assert_bit_exact(tok: RemiTokenizer, seq: TokenSequence, grid: Grid, tmp_path: Path) -> None:
    """tokens -> MIDI file -> tokens must be the identity. Hard assert, no tolerance."""
    notes, chords = tok.decode(seq.tokens, grid, seq.first_bar)
    score = through_midi_file(to_midi(notes, chords, grid), tmp_path / "decoded.mid")
    read_notes, read_chords = from_midi(score, grid)
    assert read_chords == chords  # every chord survives the MIDI file on its grid point
    again = tok.encode(read_notes, read_chords, grid)
    assert again.first_bar == seq.first_bar
    assert again.tokens == seq.tokens


@pytest.mark.parametrize("beat_res", [4, 12])
def test_bit_exact_synthetic(beat_res: int, tmp_path: Path) -> None:
    grid = Grid(DRIFT_TIMES, DRIFT_DOWNBEATS, beats_per_bar=4, beat_res=beat_res)
    t = grid.time_at
    notes = [
        n(t(1), t(3), 31, 5),  # pickup bar
        n(t(4), t(10), 40, 20),  # held across a 2-beat bar into a 6-beat bar
        n(t(9), t(10.5), 36, 40), n(t(9), t(9.25), 43, 64), n(t(9), t(12), 48, 127),  # polyphony
        n(t(12.03), t(13.6), 45, 100),  # performed slightly late
        n(t(20), t(21.5), 50, 90),  # beyond the annotated beats
    ]
    chords = [ChordEvent(-1, 2 * beat_res, "G:maj"), ChordEvent(0, 0, "C:maj"), ChordEvent(1, beat_res, "A:min"),
              ChordEvent(2, 0, "N"), ChordEvent(2, 3 * beat_res, "F:maj"), ChordEvent(4, 0, "G:maj")]
    tok = RemiTokenizer(beat_res=beat_res)
    seq = tok.encode(notes, chords, grid)
    assert seq.first_bar == -1
    assert_bit_exact(tok, seq, grid, tmp_path)


@needs_pop909
@pytest.mark.parametrize("song", REAL_SONGS)
@pytest.mark.parametrize("beat_res", [4, 12])
def test_bit_exact_real_songs(song: str, beat_res: int, tmp_path: Path) -> None:
    grid, bass, chords = pop909_song(song, beat_res)
    tok = RemiTokenizer(beat_res=beat_res)
    assert_bit_exact(tok, tok.encode(bass, chords, grid), grid, tmp_path)


# --------------------------------------------------------------------------- 2. budgeted
@needs_pop909
@pytest.mark.parametrize("song", REAL_SONGS)
def test_budgeted_midi_round_trip(song: str, tmp_path: Path) -> None:
    """MIDI -> tokens -> MIDI: same notes and pitches; timing and velocity within the grid budget.

    The reference is the source MIDI's bassline as the tokenizer reads it, split at barlines (the
    representation the tokens encode). It must come from the MIDI file itself: tick rounding can
    move an onset sitting on a rounding boundary to the step before a barline, which adds a split.
    """
    grid, bass, chords = pop909_song(song)
    source = through_midi_file(to_midi(bass, chords, grid), tmp_path / "source.mid")
    before, before_chords = from_midi(source, grid)
    reference = split_at_barlines(before, grid)

    seq = TOK.encode(before, before_chords, grid)
    decoded = through_midi_file(to_midi(*TOK.decode(seq.tokens, grid, seq.first_bar), grid), tmp_path / "decoded.mid")
    after, after_chords = from_midi(decoded, grid)

    assert len(after) == len(reference)
    assert [x.pitch for x in after] == [x.pitch for x in reference]
    assert TOK.encode(after, after_chords, grid) == seq  # chords within the encoded bars survive too
    before = reference
    for a, b in zip(after, before, strict=True):
        # the decoded note sits on the grid step nearest the original, within MIDI tick rounding
        assert grid.step_at(a.start) == grid.step_at(b.start)
        assert TOK.velocity_level(a.velocity) == TOK.velocity_level(b.velocity)

    onset = np.array([abs(a.start - b.start) for a, b in zip(after, before)]) * 1000
    duration = np.array([abs(a.duration - b.duration) for a, b in zip(after, before)]) * 1000
    velocity = np.array([abs(a.velocity - b.velocity) for a, b in zip(after, before)])
    assert velocity.max() <= 8
    print(
        f"\n{song}: {len(after)} notes | onset ms p50/p95/max {np.percentile(onset, 50):.1f}/"
        f"{np.percentile(onset, 95):.1f}/{onset.max():.1f} | duration ms {np.percentile(duration, 50):.1f}/"
        f"{np.percentile(duration, 95):.1f}/{duration.max():.1f} | velocity {np.percentile(velocity, 50):.0f}/"
        f"{np.percentile(velocity, 95):.0f}/{velocity.max():.0f}"
    )
