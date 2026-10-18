from __future__ import annotations

from pathlib import Path

import pytest

from secondchair.chords import (
    CHORD_CLASSES,
    QUALITY_TO_CLASS,
    ChordEvent,
    ChordSegment,
    chord_events,
    collapse,
    dedupe_changes,
    merge_repeats,
    read_chord_midi,
)
from secondchair.grid import Grid

POP909 = Path(__file__).resolve().parents[1] / "data" / "POP909"
needs_pop909 = pytest.mark.skipif(not POP909.is_dir(), reason="POP909 not extracted to data/")


def write_chord_file(path: Path, rows: list[tuple[float, float, str]]) -> Path:
    """Write a chord file exactly as POP909 does: tab-separated, CRLF, trailing newline."""
    path.write_bytes("".join(f"{s:.6f}\t{e:.6f}\t{label}\r\n" for s, e, label in rows).encode("ascii"))
    return path


def test_read_chord_midi_parses_pop909_format(tmp_path: Path) -> None:
    path = write_chord_file(tmp_path / "chord_midi.txt", [(0.055333, 0.721998, "N"), (0.721998, 2.0, "F#:maj7/5")])
    assert read_chord_midi(path) == [ChordSegment(0.055333, 0.721998, "N"), ChordSegment(0.721998, 2.0, "F#:maj7/5")]


def test_read_chord_midi_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "chord_midi.txt"
    path.write_text("0.0\t1.0\tC:maj\n1.0\t2.0\n")
    with pytest.raises(ValueError, match=":2: expected 3 columns"):
        read_chord_midi(path)


def test_chord_segment_must_have_positive_length() -> None:
    with pytest.raises(ValueError):
        ChordSegment(2.0, 2.0, "C:maj")


def test_merge_repeats_joins_contiguous_identical_labels() -> None:
    segs = [ChordSegment(0, 1, "N"), ChordSegment(1, 2, "N"), ChordSegment(2, 4, "C:maj"),
            ChordSegment(4, 6, "C:maj"), ChordSegment(6, 7, "C:maj7")]
    assert merge_repeats(segs) == [ChordSegment(0, 2, "N"), ChordSegment(2, 6, "C:maj"), ChordSegment(6, 7, "C:maj7")]


def test_merge_repeats_keeps_separated_identical_labels_apart() -> None:
    segs = [ChordSegment(0, 1, "C:maj"), ChordSegment(1.5, 2, "C:maj")]
    assert merge_repeats(segs) == segs


def test_merge_repeats_of_nothing_is_nothing() -> None:
    assert merge_repeats([]) == []


@needs_pop909
def test_real_chord_file() -> None:
    segs = read_chord_midi(POP909 / "001" / "chord_midi.txt")
    assert len(segs) == 155
    assert [s.label for s in segs[:5]] == ["N", "N", "N", "N", "B:maj"]
    assert all(a.end == b.start for a, b in zip(segs, segs[1:]))
    merged = merge_repeats(segs)
    assert merged[0] == ChordSegment(segs[0].start, segs[3].end, "N")
    assert all(a.label != b.label for a, b in zip(merged, merged[1:]))


# --------------------------------------------------------------------------- coarse vocabulary
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("C:maj", "C:maj"),
        ("F#:maj7/5", "F#:maj"),  # inversion dropped
        ("Bb:sus4(b7)", "Bb:maj"),  # extension dropped
        ("A:7", "A:maj"),
        ("E:aug", "E:maj"),
        ("D:sus2", "D:maj"),
        ("G:maj6", "G:maj"),
        ("C:hdim7", "C:min"),
        ("B:dim7", "B:min"),
        ("F:dim", "F:min"),
        ("Eb:min7/b7", "Eb:min"),
        ("Ab:minmaj7", "Ab:min"),
        ("C#:min6", "C#:min"),
        ("Db:min", "C#:min"),  # enharmonic roots share one spelling
        ("Cb:maj", "B:maj"),
        ("E#:min", "F:min"),
        ("N", "N"),
    ],
)
def test_collapse_map(label: str, expected: str) -> None:
    assert collapse(label) == expected


def test_quality_table_maps_onto_major_and_minor_only() -> None:
    assert set(QUALITY_TO_CLASS.values()) == {"maj", "min"}


def test_chord_classes_are_n_plus_twelve_roots_times_two() -> None:
    assert len(CHORD_CLASSES) == len(set(CHORD_CLASSES)) == 25
    assert CHORD_CLASSES[0] == "N"
    assert all(collapse(c) == c for c in CHORD_CLASSES)  # idempotent


@pytest.mark.parametrize("label", ["", "C", "H:maj", "C:sus", "C:maj(9", "X"])
def test_collapse_rejects_labels_it_cannot_classify(label: str) -> None:
    with pytest.raises(ValueError):
        collapse(label)


@needs_pop909
def test_every_pop909_label_collapses() -> None:
    labels = {seg.label for song in POP909.iterdir() if song.is_dir() for seg in read_chord_midi(song / "chord_midi.txt")}
    assert len(labels) == 320
    assert {collapse(label) for label in labels} == set(CHORD_CLASSES)


# --------------------------------------------------------------------------- chord events
def test_chord_event_requires_a_chord_class() -> None:
    with pytest.raises(ValueError, match="CHORD_CLASSES"):
        ChordEvent(0, 0, "C:maj7")
    with pytest.raises(ValueError, match="non-negative"):
        ChordEvent(0, -1, "C:maj")


def test_dedupe_changes_orders_and_drops_repeats() -> None:
    events = [ChordEvent(1, 0, "G:maj"), ChordEvent(0, 0, "C:maj"), ChordEvent(0, 8, "C:maj"), ChordEvent(1, 8, "G:maj")]
    assert dedupe_changes(events) == [ChordEvent(0, 0, "C:maj"), ChordEvent(1, 0, "G:maj")]


def test_dedupe_changes_drops_a_leading_no_chord() -> None:
    events = [ChordEvent(0, 0, "N"), ChordEvent(1, 0, "C:maj"), ChordEvent(2, 0, "N")]
    assert dedupe_changes(events) == [ChordEvent(1, 0, "C:maj"), ChordEvent(2, 0, "N")]


def test_dedupe_changes_later_event_on_the_same_grid_point_wins() -> None:
    events = [ChordEvent(0, 0, "C:maj"), ChordEvent(1, 0, "A:min"), ChordEvent(1, 0, "C:maj")]
    assert dedupe_changes(events) == [ChordEvent(0, 0, "C:maj")]  # bar 1 ends up unchanged


def test_chord_events_collapse_quantize_and_merge() -> None:
    grid = Grid.constant_tempo(bpm=60, beats_per_bar=4)  # one beat per second
    segs = [ChordSegment(0, 2, "C:maj"), ChordSegment(2, 4, "C:maj7"), ChordSegment(4, 6, "A:min7"), ChordSegment(6, 8, "N")]
    assert chord_events(segs, grid) == [ChordEvent(0, 0, "C:maj"), ChordEvent(1, 0, "A:min"), ChordEvent(1, 8, "N")]
