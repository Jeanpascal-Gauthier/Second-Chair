from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from symusic import Score, Tempo

from secondchair import bass as bs
from secondchair.bass import (
    BelowThreshold,
    LowestAtChordChange,
    LowestSounding,
    bass_score,
    extract_bass,
    make_strategy,
)
from secondchair.chords import ChordSegment, read_chord_midi
from secondchair.grid import Grid
from secondchair.notes import Note, to_track, track_notes
from secondchair.pop909 import DEFAULT_ROOT, SongInfo, read_index
from secondchair.render import soundfont_path

GRID = Grid.constant_tempo(bpm=60, beat_res=4)  # one beat per second; grid steps of 0.25 s

needs_pop909 = pytest.mark.skipif(not DEFAULT_ROOT.is_dir(), reason="POP909 not extracted to data/")
needs_fluidsynth = pytest.mark.skipif(
    shutil.which("fluidsynth") is None or not soundfont_path().is_file(),
    reason="fluidsynth or the soundfont is not installed",
)


def n(start: float, end: float, pitch: int, velocity: int = 80) -> Note:
    return Note(start, end, pitch, velocity)


def chords(*rows: tuple[float, float, str]) -> list[ChordSegment]:
    return [ChordSegment(s, e, label) for s, e, label in rows]


def score_with(tracks: dict[str, list[Note]], qpm: float = 120.0) -> Score:
    score = Score(480)
    score.tempos.append(Tempo(time=0, qpm=qpm))
    score = score.to("second")
    for name, notes in tracks.items():
        score.tracks.append(to_track(notes, name))
    return score


# --------------------------------------------------------------------------- LowestSounding
def test_lowest_takes_lowest_of_simultaneous_notes() -> None:
    assert LowestSounding(GRID)([n(0, 1, 48), n(0, 1, 36), n(0, 1, 52)]) == [n(0, 1, 36)]


def test_lowest_held_low_note_hides_higher_onsets() -> None:
    assert LowestSounding(GRID)([n(0, 2, 36), n(1, 2, 43)]) == [n(0, 2, 36)]


def test_lowest_new_lower_note_takes_over() -> None:
    assert LowestSounding(GRID)([n(0, 2, 40), n(1, 2, 36)]) == [n(0, 1, 40), n(1, 2, 36)]


def test_lowest_never_invents_attacks_for_held_notes() -> None:
    # When C2 ends, the held E3 above it is the lowest sounding note, but it was not struck then:
    # the bass rests instead of starting a note the pianist never played.
    assert LowestSounding(GRID)([n(0, 1, 36), n(0, 3, 52)]) == [n(0, 1, 36)]


def test_lowest_ignores_small_legato_overlap() -> None:
    # C2 is released 50 ms after G2 starts, less than half a grid step.
    assert LowestSounding(GRID)([n(0, 1.05, 36), n(1, 2, 43)]) == [n(0, 1, 36), n(1, 2, 43)]


def test_lowest_follows_the_lowest_voice_up_when_the_left_hand_rests() -> None:
    assert LowestSounding(GRID)([n(0, 1, 36), n(2, 3, 64)]) == [n(0, 1, 36), n(2, 3, 64)]


# --------------------------------------------------------------------------- BelowThreshold
def test_threshold_keeps_only_notes_below_threshold() -> None:
    assert BelowThreshold(GRID)([n(0, 1, 54), n(1, 2, 55), n(2, 3, 60)]) == [n(0, 1, 54)]


def test_threshold_is_configurable() -> None:
    assert BelowThreshold(GRID, threshold=48)([n(0, 1, 50), n(1, 2, 45)]) == [n(1, 2, 45)]


def test_threshold_each_struck_note_cuts_the_previous_one() -> None:
    # Unlike LowestSounding, a higher note struck over a held low note enters the line.
    assert BelowThreshold(GRID)([n(0, 2, 36), n(1, 2, 43)]) == [n(0, 1, 36), n(1, 2, 43)]


def test_threshold_octave_doubling_keeps_lowest() -> None:
    assert BelowThreshold(GRID)([n(0, 1, 48), n(0, 1, 36)]) == [n(0, 1, 36)]


def test_threshold_keeps_rests_between_notes() -> None:
    assert BelowThreshold(GRID)([n(0, 0.5, 36), n(1, 2, 43)]) == [n(0, 0.5, 36), n(1, 2, 43)]


# --------------------------------------------------------------------------- LowestAtChordChange
def test_chord_holds_lowest_note_until_next_change() -> None:
    strategy = LowestAtChordChange(GRID, chords((0, 2, "C:maj"), (2, 4, "F:maj")))
    assert strategy([n(0, 0.5, 36), n(0, 0.5, 48), n(2, 2.5, 41)]) == [n(0, 2, 36), n(2, 4, 41)]


def test_chord_merges_repeated_labels() -> None:
    strategy = LowestAtChordChange(GRID, chords((0, 2, "C:maj"), (2, 4, "C:maj")))
    assert strategy([n(0, 1, 36), n(2, 3, 36)]) == [n(0, 4, 36)]


def test_chord_pedal_point_is_restruck_at_the_change() -> None:
    # C2 is held under C -> F/C while the right hand strikes F3 on the change.
    strategy = LowestAtChordChange(GRID, chords((0, 2, "C:maj"), (2, 4, "F:maj/5")))
    assert strategy([n(0, 4, 36), n(2, 3, 53)]) == [n(0, 2, 36), n(2, 4, 36)]


def test_chord_keeps_performed_onset_and_cuts_overlap() -> None:
    strategy = LowestAtChordChange(GRID, chords((0, 2, "C:maj"), (2, 4, "F:maj")))
    # F2 is pushed 30 ms ahead of the change
    assert strategy([n(0, 1, 36), n(1.97, 2.5, 41)]) == [n(0, 1.97, 36), n(1.97, 4, 41)]


def test_chord_prefers_the_struck_note_on_a_pitch_tie() -> None:
    strategy = LowestAtChordChange(GRID, chords((0, 2, "C:maj"), (2, 4, "C:maj/3")))
    line = strategy([n(0, 4, 36), n(2.03, 3, 36, velocity=120)])
    assert line[1] == n(2.03, 4, 36, velocity=120)


def test_chord_skips_no_chord_and_silent_changes() -> None:
    strategy = LowestAtChordChange(GRID, chords((0, 2, "N"), (2, 4, "C:maj"), (4, 6, "G:maj")))
    assert strategy([n(0, 1, 36), n(4, 5, 43)]) == [n(4, 6, 43)]


# --------------------------------------------------------------------------- extract_bass & friends
def test_extract_bass_reads_only_the_piano_track() -> None:
    score = score_with({"MELODY": [n(0, 1, 30)], "PIANO": [n(0, 1, 40), n(0, 1, 52)]})
    assert extract_bass(score, LowestSounding(GRID)) == [n(0, 1, 40)]
    assert extract_bass(score, LowestSounding(GRID), source="MELODY") == [n(0, 1, 30)]


def test_extract_bass_requires_a_piano_track() -> None:
    with pytest.raises(KeyError, match="PIANO"):
        extract_bass(score_with({"MELODY": [n(0, 1, 30)]}), LowestSounding(GRID))


def test_make_strategy_by_name() -> None:
    segs = chords((0, 2, "C:maj"))
    assert make_strategy("lowest", GRID, segs) == LowestSounding(GRID)
    assert make_strategy("threshold", GRID, segs, threshold=50) == BelowThreshold(GRID, 50)
    assert make_strategy("chord", GRID, segs) == LowestAtChordChange(GRID, segs)
    with pytest.raises(ValueError, match="unknown bass strategy"):
        make_strategy("highest", GRID, segs)


def test_bass_score_adds_bass_track_after_the_arrangement() -> None:
    score = score_with({"PIANO": [n(0, 1, 40)]})
    out = bass_score(score, [n(0, 1, 40)])
    assert [(t.name, t.program) for t in out.tracks] == [("PIANO", 0), ("BASS", bs.BASS_PROGRAM)]
    assert [t.name for t in score.tracks] == ["PIANO"]  # input left untouched


def test_bass_score_solo_drops_the_arrangement_but_keeps_tempo() -> None:
    score = score_with({"PIANO": [n(0, 1, 40)]}, qpm=90.0)
    out = bass_score(score, [n(0, 1, 40)], arrangement=False)
    assert [t.name for t in out.tracks] == ["BASS"]
    assert [round(t.qpm, 3) for t in out.tempos] == [90.0]
    assert track_notes(out, "BASS") == [n(0, 1, 40)]


# --------------------------------------------------------------------------- real POP909 songs
@pytest.fixture(scope="module")
def index() -> dict[str, SongInfo]:
    return read_index(DEFAULT_ROOT)


@needs_pop909
@pytest.mark.parametrize("song", ["001", "034", "806"])  # duple, triple, triplet feel
@pytest.mark.parametrize("name", bs.STRATEGY_NAMES)
def test_real_bass_lines_are_single_voices(song: str, name: str, index: dict[str, SongInfo]) -> None:
    folder = DEFAULT_ROOT / song
    grid = Grid.from_pop909(folder / "beat_midi.txt", index[song].beats_per_measure)
    score = Score(str(folder / f"{song}.mid"), ttype="second")
    strategy = make_strategy(name, grid, read_chord_midi(folder / "chord_midi.txt"))
    piano, line = track_notes(score, "PIANO"), extract_bass(score, strategy)

    assert len(line) > 50
    assert all(a.start < b.start and a.end <= b.start for a, b in zip(line, line[1:]))
    assert {x.pitch for x in line} <= {p.pitch for p in piano}
    assert {x.velocity for x in line} <= {p.velocity for p in piano}
    if name == "threshold":
        assert max(x.pitch for x in line) < bs.DEFAULT_THRESHOLD


@needs_pop909
@needs_fluidsynth
def test_cli_renders_a_comparison(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bs.main(["1", "--strategy", "chord", "--out", str(tmp_path)])
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "chord.mid", "chord.wav", "chord_solo.mid", "chord_solo.wav", "original.wav",
    ]
    assert [(t.name, t.program) for t in Score(str(tmp_path / "chord_solo.mid")).tracks] == [("BASS", 33)]
    assert "chord:" in capsys.readouterr().out
