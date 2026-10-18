from __future__ import annotations

from pathlib import Path

import pytest
from symusic import Note as SymusicNote
from symusic import Score, Tempo, Track

from secondchair.notes import Note, to_track, track_notes


def tick_score(tracks: dict[str, list[tuple[int, int, int, int]]], qpm: float = 120.0) -> Score:
    """Tick-based score at `qpm` (480 ticks per quarter) with (time, duration, pitch, vel) notes."""
    score = Score(480)
    score.tempos.append(Tempo(time=0, qpm=qpm))
    for name, notes in tracks.items():
        track = Track(name=name, program=0, is_drum=False)
        for time, duration, pitch, velocity in notes:
            track.notes.append(SymusicNote(time, duration, pitch, velocity))
        score.tracks.append(track)
    return score


@pytest.mark.parametrize(
    ("start", "end", "pitch", "velocity"),
    [(1.0, 1.0, 60, 80), (1.0, 0.5, 60, 80), (0.0, 1.0, 128, 80), (0.0, 1.0, 60, 0)],
)
def test_note_rejects_invalid_values(start: float, end: float, pitch: int, velocity: int) -> None:
    with pytest.raises(ValueError):
        Note(start, end, pitch, velocity)


def test_note_duration_and_ordering() -> None:
    a, b = Note(0.5, 1.25, 40, 90), Note(0.5, 1.0, 38, 90)
    assert a.duration == 0.75
    assert sorted([a, b]) == [b, a]  # ordered by start, then end


def test_track_notes_reads_named_track_in_seconds() -> None:
    score = tick_score({
        "MELODY": [(0, 480, 72, 100)],
        "PIANO": [(480, 240, 48, 70), (0, 960, 36, 90)],
    })
    assert track_notes(score, "PIANO") == [Note(0.0, 1.0, 36, 90), Note(0.5, 0.75, 48, 70)]


def test_track_notes_drops_zero_length_notes() -> None:
    score = tick_score({"PIANO": [(0, 0, 40, 80), (0, 480, 41, 80)]})
    assert [n.pitch for n in track_notes(score, "PIANO")] == [41]


def test_track_notes_rejects_missing_track() -> None:
    score = tick_score({"MELODY": []})
    with pytest.raises(KeyError, match="found 0 in \\['MELODY'\\]"):
        track_notes(score, "PIANO")


def test_track_notes_rejects_duplicate_names() -> None:
    score = tick_score({"PIANO": []})
    score.tracks.append(Track(name="PIANO", program=0, is_drum=False))
    with pytest.raises(KeyError, match="found 2"):
        track_notes(score, "PIANO")


def test_to_track_round_trips_through_midi(tmp_path: Path) -> None:
    notes = [Note(0.0, 0.5, 36, 100), Note(0.75, 1.9, 43, 64)]
    score = tick_score({}, qpm=97.0).to("second")
    score.tracks.append(to_track(notes, "BASS", program=33))
    score.dump_midi(str(tmp_path / "bass.mid"))

    back = Score(str(tmp_path / "bass.mid"))
    assert [(t.name, t.program) for t in back.tracks] == [("BASS", 33)]
    for got, want in zip(track_notes(back, "BASS"), notes, strict=True):
        assert (got.pitch, got.velocity) == (want.pitch, want.velocity)
        assert got.start == pytest.approx(want.start, abs=0.002)
        assert got.end == pytest.approx(want.end, abs=0.002)
