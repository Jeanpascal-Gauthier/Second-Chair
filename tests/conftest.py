"""Fixtures shared by the audio tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from symusic import Note as SymusicNote
from symusic import Score, Tempo, Track

from secondchair.render import render, soundfont_path

BEAT_THIS_WEIGHTS = Path(torch.hub.get_dir()) / "checkpoints" / "beat_this-final0.ckpt"


@pytest.fixture
def beat_tracker() -> None:
    """Skip unless beat_this's weights are cached: tests never download them."""
    if not BEAT_THIS_WEIGHTS.exists():
        pytest.skip("beat_this weights not cached; run the audio pipeline once to download them")


@pytest.fixture
def groove_wav(tmp_path: Path) -> Path:
    """Eight 2-second bars of C-G-Am-F at 120 bpm: drums, a piano chord on every beat, bass roots."""
    if shutil.which("fluidsynth") is None or not soundfont_path().is_file():
        pytest.skip("fluidsynth or the soundfont is not installed")
    progression = [(36, [60, 64, 67]), (43, [59, 62, 67]), (45, [57, 60, 64]), (41, [57, 60, 65])]
    score = Score(480)
    score.tempos.append(Tempo(time=0, qpm=120.0))
    piano, bass, drums = (Track(name=n, program=p, is_drum=d) for n, p, d in
                          (("PIANO", 0, False), ("BASS", 33, False), ("DRUMS", 0, True)))
    for bar in range(8):
        root, triad = progression[bar % 4]
        bass.notes.append(SymusicNote(bar * 1920, 1800, root, 100))
        for beat in range(4):
            t = bar * 1920 + beat * 480
            piano.notes.extend(SymusicNote(t, 400, p, 70) for p in triad)
            drums.notes.append(SymusicNote(t, 120, 36 if beat % 2 == 0 else 38, 110))  # kick / snare
            drums.notes.extend(SymusicNote(t + off, 60, 42, 60) for off in (0, 240))  # hi-hat
    score.tracks.extend([piano, bass, drums])
    score.dump_midi(str(tmp_path / "groove.mid"))
    return render(tmp_path / "groove.mid", tmp_path / "groove.wav")
