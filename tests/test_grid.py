from __future__ import annotations

from pathlib import Path

import pytest
from symusic import Score

from secondchair.grid import Grid, read_beat_midi

POP909 = Path(__file__).resolve().parents[1] / "data" / "POP909"
needs_pop909 = pytest.mark.skipif(not POP909.is_dir(), reason="POP909 not extracted to data/")


def write_beat_file(path: Path, times: list[float], col2: set[int], col3: set[int]) -> Path:
    """Write a beat file exactly as POP909 does: space-separated, CRLF, no trailing newline."""
    lines = [f"{t!r} {float(i in col2)} {float(i in col3)}" for i, t in enumerate(times)]
    path.write_bytes("\r\n".join(lines).encode("ascii"))
    return path


def round_trip_all(grid: Grid, bars: range) -> None:
    for bar in bars:
        for position in range(grid.positions_in_bar(bar)):
            assert grid.quantize(grid.to_seconds(bar, position)) == (bar, position)


# A drifting, irregular grid: beat intervals vary, bars are 4, 2, 6, 4 beats, and the first
# downbeat is the fourth beat (three beats of pickup).
DRIFT_TIMES = [0.0, 0.5, 1.1, 1.6, 2.0, 2.5, 3.0, 3.6, 4.1, 4.5, 5.0, 5.4, 5.9, 6.5, 7.0, 7.4, 7.9, 8.3, 8.8]
DRIFT_DOWNBEATS = [3, 7, 9, 15]


@pytest.fixture
def drift() -> Grid:
    return Grid(DRIFT_TIMES, DRIFT_DOWNBEATS, beats_per_bar=4, beat_res=4)


# --------------------------------------------------------------------------- beat file parsing
def test_read_beat_midi_parses_pop909_format(tmp_path: Path) -> None:
    path = write_beat_file(tmp_path / "beat_midi.txt", [0.055, 0.722, 1.389, 2.055, 2.722], {0, 2, 4}, {0, 4})
    ann = read_beat_midi(path)
    assert ann.times == (0.055, 0.722, 1.389, 2.055, 2.722)
    assert ann.measure_starts == (0, 2, 4)
    assert ann.double_measure_starts == (0, 4)


def test_read_beat_midi_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "beat_midi.txt"
    path.write_text("0.5 1.0 1.0\n1.0 0.0\n")
    with pytest.raises(ValueError, match=":2: expected 3 columns"):
        read_beat_midi(path)


def test_read_beat_midi_rejects_non_binary_flags(tmp_path: Path) -> None:
    path = tmp_path / "beat_midi.txt"
    path.write_text("0.5 1.0 2.0\n")
    with pytest.raises(ValueError, match="flag must be 0 or 1"):
        read_beat_midi(path)


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    ("times", "downbeats", "match"),
    [
        ([0.0], [0], "two beat times"),
        ([0.0, 0.5, 0.5], [0], "strictly increasing"),
        ([0.0, float("nan")], [0], "finite"),
        ([0.0, 0.5], [], "at least one downbeat"),
        ([0.0, 0.5, 1.0], [1, 1], "downbeats must be strictly increasing"),
        ([0.0, 0.5, 1.0], [3], "index into"),
        ([0.0, 0.5, 1.0], [-1], "index into"),
    ],
)
def test_grid_rejects_bad_annotations(times: list[float], downbeats: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Grid(times, downbeats, beats_per_bar=4)


@pytest.mark.parametrize(("beats_per_bar", "beat_res"), [(0, 4), (4, 0)])
def test_grid_rejects_non_positive_sizes(beats_per_bar: int, beat_res: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        Grid([0.0, 0.5], [0], beats_per_bar=beats_per_bar, beat_res=beat_res)


# --------------------------------------------------------------------------- constant tempo
def test_constant_tempo_grid() -> None:
    grid = Grid.constant_tempo(bpm=120, beats_per_bar=4, beat_res=4, start=1.0)
    assert grid.to_seconds(0, 0) == 1.0
    assert grid.to_seconds(0, 1) == pytest.approx(1.125)
    assert grid.to_seconds(3, 8) == pytest.approx(1.0 + 3 * 2.0 + 1.0)
    assert grid.quantize(1.0 + 3 * 2.0 + 1.0) == (3, 8)
    assert grid.quantize(0.0) == (-1, 8)  # half a bar before bar 0


def test_beat_res_12_represents_triplets_and_sixteenths() -> None:
    grid = Grid.constant_tempo(bpm=60, beat_res=12)
    assert grid.quantize(1 / 3) == (0, 4)  # eighth-note triplet
    assert grid.quantize(0.25) == (0, 3)  # sixteenth
    assert grid.positions_in_bar(0) == 48


# --------------------------------------------------------------------------- drift & irregular bars
def test_quantize_interpolates_within_each_beat(drift: Grid) -> None:
    # beat 4 -> 5 spans 2.0 .. 2.5 s; 2.125 s is a quarter of the way through beat 4.
    assert drift.beat_at(2.125) == pytest.approx(4.25)
    assert drift.quantize(2.125) == (0, 5)
    # beat 7 -> 8 spans 3.6 .. 4.1 s (bar 1 starts at beat 7).
    assert drift.quantize(3.85) == (1, 2)


def test_irregular_bars_keep_annotated_lengths(drift: Grid) -> None:
    assert [drift.beats_in_bar(bar) for bar in range(-2, 5)] == [4, 4, 4, 2, 6, 4, 4]
    assert drift.to_seconds(2, 0) == DRIFT_TIMES[9]
    assert drift.quantize(DRIFT_TIMES[14]) == (2, 20)  # 5 beats into the 6-beat bar


def test_round_trip_covers_pickups_irregular_bars_and_extrapolation(drift: Grid) -> None:
    round_trip_all(drift, range(-3, 7))


def test_round_trip_beat_res_12(drift: Grid) -> None:
    grid = Grid(DRIFT_TIMES, DRIFT_DOWNBEATS, beats_per_bar=4, beat_res=12)
    round_trip_all(grid, range(-2, 6))


def test_bar_starts_are_annotated_downbeats(drift: Grid) -> None:
    for bar, beat in enumerate(DRIFT_DOWNBEATS):
        assert drift.to_seconds(bar, 0) == DRIFT_TIMES[beat]


def test_rounding_to_nearest_step(drift: Grid) -> None:
    step = (DRIFT_TIMES[4] - DRIFT_TIMES[3]) / 4  # one sixteenth inside beat 3 (bar 0, beat 0)
    start = DRIFT_TIMES[3]
    assert drift.quantize(start + 0.49 * step) == (0, 0)
    assert drift.quantize(start + 0.51 * step) == (0, 1)
    assert drift.quantize(start + 0.5 * step) == (0, 1)  # halfway rounds later


def test_note_just_before_barline_belongs_to_next_bar(drift: Grid) -> None:
    assert drift.quantize(DRIFT_TIMES[7] - 0.01) == (1, 0)


def test_to_seconds_rejects_positions_outside_bar(drift: Grid) -> None:
    with pytest.raises(ValueError, match="outside bar 1"):
        drift.to_seconds(1, 8)  # bar 1 has 2 beats = 8 positions
    with pytest.raises(ValueError, match="outside bar 0"):
        drift.to_seconds(0, -1)


# --------------------------------------------------------------------------- pickups & extrapolation
def test_pickup_beats_land_at_end_of_bar_minus_one(drift: Grid) -> None:
    # First downbeat is beat 3, so beats 0, 1, 2 are beats 2, 3, 4 of bar -1.
    assert [drift.quantize(DRIFT_TIMES[k]) for k in range(3)] == [(-1, 4), (-1, 8), (-1, 12)]


def test_notes_before_first_annotated_beat_use_first_interval(drift: Grid) -> None:
    first_interval = DRIFT_TIMES[1] - DRIFT_TIMES[0]
    assert drift.quantize(-first_interval) == (-1, 0)
    assert drift.quantize(-first_interval / 2) == (-1, 2)
    assert drift.to_seconds(-2, 0) == pytest.approx(-5 * first_interval)


def test_long_pickup_spans_several_bars() -> None:
    grid = Grid([0.5 * k for k in range(12)], [6], beats_per_bar=4)
    assert grid.quantize(0.0) == (-2, 8)
    assert grid.quantize(1.0) == (-1, 0)


def test_bars_after_last_downbeat_are_nominal_and_extrapolated(drift: Grid) -> None:
    last_interval = DRIFT_TIMES[-1] - DRIFT_TIMES[-2]
    assert drift.beats_in_bar(3) == drift.beats_in_bar(4) == 4
    # bar 4 starts at beat 19, one interval past the last annotated beat (18).
    assert drift.to_seconds(4, 0) == pytest.approx(DRIFT_TIMES[-1] + last_interval)
    assert drift.quantize(DRIFT_TIMES[-1] + 2.5 * last_interval) == (4, 6)


# --------------------------------------------------------------------------- POP909 entry point
def test_from_pop909_duple_uses_four_beat_column(tmp_path: Path) -> None:
    path = write_beat_file(tmp_path / "beat_midi.txt", [0.5 * k for k in range(10)], {1, 3, 5, 7, 9}, {1, 5, 9})
    grid = Grid.from_pop909(path, beats_per_measure=2)
    assert (grid.downbeats, grid.beats_per_bar) == ((1, 5, 9), 4)


def test_from_pop909_triple_uses_measure_column(tmp_path: Path) -> None:
    path = write_beat_file(tmp_path / "beat_midi.txt", [0.5 * k for k in range(10)], {1, 4, 7}, {1, 7})
    grid = Grid.from_pop909(path, beats_per_measure=3, beat_res=12)
    assert (grid.downbeats, grid.beats_per_bar, grid.beat_res) == ((1, 4, 7), 3, 12)


def test_from_pop909_rejects_other_meters(tmp_path: Path) -> None:
    path = write_beat_file(tmp_path / "beat_midi.txt", [0.0, 0.5], {0}, {0})
    with pytest.raises(ValueError, match="2 or 3"):
        Grid.from_pop909(path, beats_per_measure=4)


# --------------------------------------------------------------------------- real POP909 songs
# song id -> num_beats_per_measure from index.xlsx. Picked for their quirks: 001 plain duple,
# 012 irregular 2-beat bars, 034 triple with a pickup beat, 102 triple whose two flag columns
# disagree, 418 annotated beats off the MIDI tick grid, 806 triplet feel.
REAL_SONGS = {"001": 2, "012": 2, "034": 3, "102": 3, "418": 2, "806": 2}


@needs_pop909
@pytest.mark.parametrize("song", sorted(REAL_SONGS))
@pytest.mark.parametrize("beat_res", [4, 12])
def test_real_song_round_trips_every_grid_point(song: str, beat_res: int) -> None:
    grid = Grid.from_pop909(POP909 / song / "beat_midi.txt", REAL_SONGS[song], beat_res)
    first = grid.bar_at_step(-grid.beat_res)  # include one beat of pre-roll
    last = grid.bar_at_step((len(grid.beat_times) + 1) * grid.beat_res)
    round_trip_all(grid, range(first, last + 1))


@needs_pop909
@pytest.mark.parametrize("song", sorted(REAL_SONGS))
def test_real_song_annotated_beats_land_on_beat_positions(song: str) -> None:
    grid = Grid.from_pop909(POP909 / song / "beat_midi.txt", REAL_SONGS[song])
    for k, t in enumerate(grid.beat_times):
        bar, position = grid.quantize(t)
        assert position % grid.beat_res == 0
        assert grid.bar_start_beat(bar) + position // grid.beat_res == k
        assert grid.to_seconds(bar, position) == pytest.approx(t, abs=1e-9)


@needs_pop909
def test_real_irregular_bars_follow_annotation() -> None:
    ann = read_beat_midi(POP909 / "012" / "beat_midi.txt")
    grid = Grid.from_pop909(POP909 / "012" / "beat_midi.txt", 2)
    marks = ann.double_measure_starts
    assert [grid.beats_in_bar(k) for k in range(len(marks) - 1)] == [b - a for a, b in zip(marks, marks[1:])]
    assert {grid.beats_in_bar(k) for k in range(len(marks) - 1)} == {2, 4}


@needs_pop909
def test_real_triple_song_has_three_beat_bars_and_pickup() -> None:
    grid = Grid.from_pop909(POP909 / "034" / "beat_midi.txt", 3)
    assert grid.downbeats[0] == 1
    assert grid.quantize(grid.beat_times[0]) == (-1, 8)  # beat 3 of a 3/4 pickup bar
    assert grid.beats_in_bar(0) == 3


@needs_pop909
def test_real_onsets_sit_on_grid() -> None:
    """Song 001's melody is fully quantized in the MIDI and its piano almost fully (6 of 985
    notes are played freely). Using the MIDI's own tick grid instead of the annotated beats would
    put every onset ~55 ms off."""
    grid = Grid.from_pop909(POP909 / "001" / "beat_midi.txt", 2)
    tracks = {t.name: t for t in Score(str(POP909 / "001" / "001.mid"), ttype="second").tracks}

    def errors(name: str) -> list[float]:
        return [abs(grid.to_seconds(*grid.quantize(n.time)) - n.time) for n in tracks[name].notes]

    assert max(errors("MELODY")) < 0.002
    piano = errors("PIANO")
    assert sum(e < 0.002 for e in piano) / len(piano) > 0.99
    assert grid.quantize(tracks["PIANO"].notes[0].time) == (1, 0)  # piano enters on bar 1's downbeat
