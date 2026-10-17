from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
from symusic import Note, Score, Tempo, Track

from secondchair import render as rnd

TARGET_PEAK = round(32767 * 10 ** (rnd.PEAK_DBFS / 20))


def write_wav(path: Path, samples: np.ndarray, rate: int = 44100, channels: int = 2) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype("<i2").tobytes())
    return path


def read_wav(path: Path) -> tuple[wave._wave_params, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        return params, np.frombuffer(w.readframes(params.nframes), dtype="<i2").astype(np.int32)


def write_bass_midi(path: Path, seconds_per_note: float = 0.5, n_notes: int = 2) -> Path:
    """Tiny bassline: n quarter notes on C2 at 120 bpm, Electric Bass (finger)."""
    score = Score(480)
    score.tempos.append(Tempo(time=0, qpm=60 / seconds_per_note))
    track = Track(name="BASS", program=33, is_drum=False)
    for k in range(n_notes):
        track.notes.append(Note(time=480 * k, duration=480, pitch=36, velocity=100))
    score.tracks.append(track)
    score.dump_midi(str(path))
    return path


# --------------------------------------------------------------------------- configuration
def test_soundfont_defaults_to_repo_generaluser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rnd.SOUNDFONT_ENV, raising=False)
    assert rnd.soundfont_path() == rnd.DEFAULT_SOUNDFONT
    assert rnd.DEFAULT_SOUNDFONT.parts[-2:] == ("soundfonts", "GeneralUser-GS.sf2")


def test_soundfont_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(rnd.SOUNDFONT_ENV, str(tmp_path / "other.sf2"))
    assert rnd.soundfont_path() == tmp_path / "other.sf2"


def test_fluidsynth_env_var_overrides_path_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rnd.FLUIDSYNTH_ENV, "C:/tools/fluidsynth.exe")
    assert rnd.fluidsynth_executable() == "C:/tools/fluidsynth.exe"


def test_missing_fluidsynth_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rnd.FLUIDSYNTH_ENV, raising=False)
    monkeypatch.setattr(rnd.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="fluidsynth"):
        rnd.fluidsynth_executable()


def test_build_command_puts_options_before_positionals() -> None:
    cmd = rnd.build_command("fs", Path("gu.sf2"), Path("in.mid"), Path("out.wav"), 22050)
    assert cmd[0] == "fs"
    assert cmd[-2:] == ["gu.sf2", "in.mid"]
    assert all(arg.startswith("-") or cmd[i - 1].startswith("-") for i, arg in enumerate(cmd[1:-2], 1))
    for flag, value in (("-F", "out.wav"), ("-r", "22050"), ("-T", "wav"), ("-O", "s16")):
        assert cmd[cmd.index(flag) + 1] == value
    assert {"-n", "-i"} <= set(cmd)


# --------------------------------------------------------------------------- input validation
def test_render_rejects_non_wav_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.wav"):
        rnd.render(write_bass_midi(tmp_path / "in.mid"), tmp_path / "out.mp3")


def test_render_rejects_missing_midi(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="MIDI"):
        rnd.render(tmp_path / "nope.mid", tmp_path / "out.wav")


def test_render_rejects_missing_soundfont(tmp_path: Path) -> None:
    midi = write_bass_midi(tmp_path / "in.mid")
    with pytest.raises(FileNotFoundError, match="soundfont"):
        rnd.render(midi, tmp_path / "out.wav", soundfont=tmp_path / "nope.sf2")


# --------------------------------------------------------------------------- normalization
def test_normalize_wav_moves_peak_to_target(tmp_path: Path) -> None:
    samples = np.array([0, 1000, -2000, 500, 0, -1500], dtype=np.int32)
    path = write_wav(tmp_path / "x.wav", samples)
    gain = rnd.normalize_wav(path)
    params, out = read_wav(path)
    assert gain == pytest.approx(TARGET_PEAK / 2000, rel=1e-4)
    assert np.abs(out).max() == TARGET_PEAK
    assert out[2] == -TARGET_PEAK  # sign and position preserved
    assert (params.nchannels, params.framerate, params.nframes) == (2, 44100, 3)


def test_normalize_wav_leaves_silence_untouched(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "silent.wav", np.zeros(8, dtype=np.int32))
    before = path.read_bytes()
    assert rnd.normalize_wav(path) == 1.0
    assert path.read_bytes() == before


def test_normalize_wav_handles_int16_minimum(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "full.wav", np.array([-32768, 16384], dtype=np.int32))
    rnd.normalize_wav(path)
    _, out = read_wav(path)
    assert out[0] == -TARGET_PEAK
    assert out[1] == round(16384 * TARGET_PEAK / 32768)


def test_normalize_wav_rejects_non_16_bit(tmp_path: Path) -> None:
    path = tmp_path / "x24.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(44100)
        w.writeframes(b"\x00\x10\x00" * 4)
    with pytest.raises(ValueError, match="16-bit"):
        rnd.normalize_wav(path)


# --------------------------------------------------------------------------- real FluidSynth
needs_fluidsynth = pytest.mark.skipif(
    shutil.which("fluidsynth") is None or not rnd.soundfont_path().is_file(),
    reason="fluidsynth or the soundfont is not installed",
)


@needs_fluidsynth
def test_render_produces_normalized_audio(tmp_path: Path) -> None:
    midi = write_bass_midi(tmp_path / "bass.mid", seconds_per_note=0.5, n_notes=2)
    out = rnd.render(midi, tmp_path / "nested" / "bass.wav")

    assert out == tmp_path / "nested" / "bass.wav"
    params, samples = read_wav(out)
    assert (params.framerate, params.nchannels, params.sampwidth) == (44100, 2, 2)
    # two half-second notes, plus the release tail FluidSynth renders after the last event (~2.3 s)
    assert 1.0 <= params.nframes / params.framerate < 6.0
    assert np.abs(samples).max() == TARGET_PEAK


@needs_fluidsynth
def test_render_without_normalization_keeps_fluidsynth_level(tmp_path: Path) -> None:
    midi = write_bass_midi(tmp_path / "bass.mid")
    _, samples = read_wav(rnd.render(midi, tmp_path / "raw.wav", normalize=False))
    assert 0 < np.abs(samples).max() < TARGET_PEAK


@needs_fluidsynth
def test_render_surfaces_fluidsynth_errors(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mid"
    corrupt.write_bytes(b"not a midi file at all")
    with pytest.raises(RuntimeError, match="fluidsynth failed"):
        rnd.render(corrupt, tmp_path / "out.wav")
    assert not (tmp_path / "out.wav").exists()


@needs_fluidsynth
def test_render_overwrites_existing_output(tmp_path: Path) -> None:
    midi = write_bass_midi(tmp_path / "bass.mid")
    out = tmp_path / "out.wav"
    out.write_bytes(b"stale")
    rnd.render(midi, out)
    assert read_wav(out)[0].nframes > 0
