from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from secondchair import dataset as ds
from secondchair.dataset import SongTokens, read_dataset, read_skiplist, split_of, write_dataset
from secondchair.pop909 import DEFAULT_ROOT, read_index
from secondchair.tokenizer import RemiTokenizer

TOK = RemiTokenizer()
needs_pop909 = pytest.mark.skipif(not DEFAULT_ROOT.is_dir(), reason="POP909 not extracted to data/")


# --------------------------------------------------------------------------- skiplist & splits
def test_read_skiplist(tmp_path: Path) -> None:
    path = tmp_path / "skiplist.txt"
    path.write_text("# header\n\n7 beat file truncated\n034 chords missing  # trailing comment\n")
    assert read_skiplist(path) == {"007": "beat file truncated", "034": "chords missing"}


def test_read_skiplist_requires_a_reason(tmp_path: Path) -> None:
    path = tmp_path / "skiplist.txt"
    path.write_text("007\n")
    with pytest.raises(ValueError, match="needs a reason"):
        read_skiplist(path)


def test_missing_skiplist_skips_nothing(tmp_path: Path) -> None:
    assert read_skiplist(tmp_path / "nope.txt") == {}


def test_repo_skiplist_is_well_formed() -> None:
    read_skiplist()  # raises on a malformed line


def test_split_is_fixed_and_roughly_90_5_5() -> None:
    splits = Counter(split_of(f"{k:03d}") for k in range(1, 910))
    assert split_of("001") == split_of("001")
    assert set(splits) == {"train", "val", "test"}
    assert 0.85 < splits["train"] / 909 < 0.95


# --------------------------------------------------------------------------- file format
def test_dataset_file_round_trip(tmp_path: Path) -> None:
    songs = [SongTokens("001", "train", 1, ("BOS", "Bar", "Position_0", "Chord_N", "EOS")),
             SongTokens("002", "test", -1, ("BOS", "EOS"))]
    path = write_dataset(songs, TOK, tmp_path)
    assert read_dataset(path) == songs
    assert (tmp_path / "vocab.json").exists()


# --------------------------------------------------------------------------- real corpus
@needs_pop909
def test_build_tokenizes_every_song_with_tokens_in_the_vocabulary() -> None:
    songs = ds.build(TOK, DEFAULT_ROOT, skiplist={})
    assert len(songs) == len(read_index(DEFAULT_ROOT))
    for s in songs:
        assert s.tokens[0] == "BOS" and s.tokens[-1] == "EOS" and len(s.tokens) > 2
        TOK.to_ids(s.tokens)  # raises on anything outside the vocabulary


@needs_pop909
def test_corpus_sweep(tmp_path: Path) -> None:
    """Brief's test 3: every song not on the skiplist round-trips; skipped songs exist and have reasons."""
    skiplist = read_skiplist()
    index = read_index(DEFAULT_ROOT)
    assert set(skiplist) <= set(index), "skiplist names songs that are not in POP909"

    results, failures = ds.sweep(TOK, DEFAULT_ROOT, skiplist)
    assert failures == {}, f"add these to skiplist.txt with a reason, or fix them: {failures}"
    assert len(results) == len(index) - len(skiplist)
    assert ds.plot_onset_p95(results, tmp_path / "p95.png").stat().st_size > 0
