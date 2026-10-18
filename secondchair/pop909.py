"""Where POP909 lives on disk, and its song index (index.xlsx).

The dataset root comes from $POP909_ROOT, defaulting to data/POP909 in the repo. Each song is a
folder named by its three-digit id holding NNN.mid, beat_midi.txt and chord_midi.txt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT_ENV = "POP909_ROOT"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "POP909"


@dataclass(frozen=True)
class SongInfo:
    song_id: str
    name: str
    artist: str
    beats_per_measure: int  # 2 (duple) or 3 (triple)
    quavers_per_beat: int  # 2 (straight) or 3 (triplet feel)


def pop909_root() -> Path:
    """Dataset root: $POP909_ROOT if set, else the repo's data/POP909."""
    return Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def song_dir(song_id: str, root: Path | None = None) -> Path:
    return (root or pop909_root()) / song_id


def read_index(root: Path | None = None) -> dict[str, SongInfo]:
    """Song metadata from index.xlsx, keyed by three-digit song id."""
    frame = pd.read_excel((root or pop909_root()) / "index.xlsx", dtype=str)
    return {
        row.song_id.zfill(3): SongInfo(
            song_id=row.song_id.zfill(3),
            name=row.name,
            artist=row.artist,
            beats_per_measure=int(float(row.num_beats_per_measure)),
            quavers_per_beat=int(float(row.num_quavers_per_beat)),
        )
        for row in frame.itertuples(index=False)
    }
