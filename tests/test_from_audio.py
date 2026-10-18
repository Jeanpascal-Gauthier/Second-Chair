from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from secondchair.analysis import Analysis, grid_from_beats
from secondchair.chords import ChordSegment, chord_events
from secondchair.from_audio import MIX_RATE, accompany_file, bars_to_play, bassline, describe, mix
from secondchair.model import GPT, GPTConfig
from secondchair.tokenizer import RemiTokenizer


TOK = RemiTokenizer()


@pytest.fixture(scope="module")
def untrained() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=len(TOK.vocab), block_size=64, n_layer=1, n_head=2, n_embd=32)).eval()


def fake_analysis() -> Analysis:
    """120 bpm 4/4 from 0.5 s: C for a bar, G for a bar, then A minor to the end (17 s)."""
    beats = np.arange(1, 34) * 0.5
    chords = [ChordSegment(0.5, 2.5, "C:maj"), ChordSegment(2.5, 4.5, "G:maj"), ChordSegment(4.5, 17.0, "A:min")]
    return Analysis(17.0, beats, beats[::4], grid_from_beats(beats, beats[::4]), chords)


def test_bars_to_play_run_from_the_first_chord_to_the_last_beat() -> None:
    analysis = fake_analysis()
    assert bars_to_play(analysis, chord_events(analysis.chords, analysis.grid)) == range(0, 9)


def test_describe_lists_tempo_meter_and_chords_per_bar() -> None:
    analysis = fake_analysis()
    text = describe(analysis, chord_events(analysis.chords, analysis.grid))
    assert "tempo ~120 bpm, 4 beats per bar" in text.splitlines()[0]
    assert any(line.startswith("bar    1") and line.endswith("G:maj@1") for line in text.splitlines())


def test_bassline_follows_the_recognized_chords(untrained: GPT) -> None:
    analysis = fake_analysis()
    tokens, notes, chords = bassline(untrained, TOK, analysis, seed=0)
    assert chords == chord_events(analysis.chords, analysis.grid)
    decoded_notes, decoded_chords = TOK.decode(tokens, analysis.grid, 0)
    assert decoded_chords == chords
    assert tokens.count("Bar") == 9
    assert all(n.start >= 0 for n in notes)
    assert all(a.end <= b.start for a, b in zip(notes, notes[1:]))


def test_mix_is_stereo_44k_as_long_as_the_longer_input_and_peaks_at_minus_1_dbfs(tmp_path: Path) -> None:
    t = np.arange(22050) / 22050
    soundfile.write(tmp_path / "song.wav", 0.3 * np.sin(2 * np.pi * 440 * t), 22050)  # mono, 1 s
    t = np.arange(int(1.5 * MIX_RATE)) / MIX_RATE
    bass = 0.9 * np.sin(2 * np.pi * 55 * t)
    soundfile.write(tmp_path / "bass.wav", np.stack([bass, bass], axis=1), MIX_RATE)  # stereo, 1.5 s

    audio, rate = soundfile.read(mix(tmp_path / "song.wav", tmp_path / "bass.wav", tmp_path / "mix.wav"))
    assert rate == MIX_RATE and audio.shape == (int(1.5 * MIX_RATE), 2)
    assert np.abs(audio).max() == pytest.approx(10 ** (-1 / 20), abs=1e-3)


def test_mix_bass_level(tmp_path: Path) -> None:
    t = np.arange(MIX_RATE) / MIX_RATE
    song, bass = 0.5 * np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 55 * t)
    soundfile.write(tmp_path / "song.wav", song, MIX_RATE)
    soundfile.write(tmp_path / "bass.wav", bass, MIX_RATE)
    quiet, _ = soundfile.read(mix(tmp_path / "song.wav", tmp_path / "bass.wav", tmp_path / "q.wav", bass_db=-40))
    loud, _ = soundfile.read(mix(tmp_path / "song.wav", tmp_path / "bass.wav", tmp_path / "l.wav", bass_db=0))
    share = lambda audio: abs(np.dot(audio[:, 0], bass)) / np.linalg.norm(audio[:, 0])  # noqa: E731
    assert share(loud) > 10 * share(quiet)


def test_accompany_file_end_to_end(untrained: GPT, groove_wav: Path, beat_tracker: None, tmp_path: Path) -> None:
    mixed = accompany_file(groove_wav, untrained, TOK, tmp_path / "out",
                           device="cuda" if torch.cuda.is_available() else "cpu")
    assert mixed == tmp_path / "out" / "mix.wav"
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"mix.wav", "bass.wav", "bass.mid", "analysis.txt",
                                                             "tokens.txt"}
    assert "C:maj@1" in (tmp_path / "out" / "analysis.txt").read_text()
