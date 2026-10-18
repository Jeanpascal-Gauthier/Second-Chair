from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from openpyxl import Workbook

from secondchair import pop909
from secondchair.pop909 import SongInfo, read_index, song_dir

HEADER = ["song_id", "name", "artist", "modify_times", "num_beats_per_measure", "num_quavers_per_beat"]


def test_root_defaults_to_repo_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pop909.ROOT_ENV, raising=False)
    assert pop909.pop909_root() == pop909.DEFAULT_ROOT
    assert pop909.DEFAULT_ROOT.parts[-2:] == ("data", "POP909")


def test_root_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(pop909.ROOT_ENV, str(tmp_path))
    assert pop909.pop909_root() == tmp_path
    assert song_dir("007") == tmp_path / "007"


def test_read_index_parses_meter_and_pads_ids(tmp_path: Path) -> None:
    sheet = Workbook()
    rows = sheet.active
    rows.append(HEADER)
    rows.append(["001", "Song A", "Artist A", 3, 2, 2])
    rows.append([34, "Song B", "Artist B", 1, 3, 2])  # id stored as a number
    sheet.save(tmp_path / "index.xlsx")

    index = read_index(tmp_path)
    assert index == {
        "001": SongInfo("001", "Song A", "Artist A", beats_per_measure=2, quavers_per_beat=2),
        "034": SongInfo("034", "Song B", "Artist B", beats_per_measure=3, quavers_per_beat=2),
    }


@pytest.mark.skipif(not pop909.DEFAULT_ROOT.is_dir(), reason="POP909 not extracted to data/")
def test_real_index() -> None:
    index = read_index(pop909.DEFAULT_ROOT)
    assert len(index) == 909
    assert Counter((s.beats_per_measure, s.quavers_per_beat) for s in index.values()) == {
        (2, 2): 857, (3, 2): 28, (2, 3): 23, (3, 3): 1,
    }
    assert all(song_dir(k, pop909.DEFAULT_ROOT).is_dir() for k in index)
