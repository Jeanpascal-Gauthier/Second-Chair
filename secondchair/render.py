"""Render MIDI files to WAV by shelling out to the FluidSynth command-line player.

Usage:
    from secondchair.render import render
    render("song.mid", "renders/song.wav")

or from a shell:
    uv run python -m secondchair.render song.mid renders/song.wav

The soundfont comes from $SECONDCHAIR_SF2, defaulting to soundfonts/GeneralUser-GS.sf2 in the
repo. The fluidsynth executable comes from $SECONDCHAIR_FLUIDSYNTH, defaulting to `fluidsynth`
on PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

SOUNDFONT_ENV = "SECONDCHAIR_SF2"
FLUIDSYNTH_ENV = "SECONDCHAIR_FLUIDSYNTH"
DEFAULT_SOUNDFONT = Path(__file__).resolve().parent.parent / "soundfonts" / "GeneralUser-GS.sf2"
SAMPLE_RATE = 44100
PEAK_DBFS = -1.0
_INT16_MAX = 32767


def soundfont_path() -> Path:
    """Soundfont to render with: $SECONDCHAIR_SF2 if set, else the repo's GeneralUser GS."""
    return Path(os.environ.get(SOUNDFONT_ENV) or DEFAULT_SOUNDFONT)


def fluidsynth_executable() -> str:
    """FluidSynth executable: $SECONDCHAIR_FLUIDSYNTH if set, else `fluidsynth` on PATH."""
    exe = os.environ.get(FLUIDSYNTH_ENV) or shutil.which("fluidsynth")
    if not exe:
        raise FileNotFoundError(
            f"fluidsynth not found on PATH; install it or set ${FLUIDSYNTH_ENV} to its path"
        )
    return exe


def build_command(
    exe: str, soundfont: Path, midi_path: Path, out_path: Path, sample_rate: int = SAMPLE_RATE
) -> list[str]:
    """FluidSynth argv for an offline render: no MIDI input, no shell, 16-bit WAV output.

    Options precede the positional soundfont/MIDI arguments so argument parsing does not depend on
    the platform's getopt permuting them.
    """
    return [
        exe, "-n", "-i", "-q",
        "-F", str(out_path), "-T", "wav", "-O", "s16",
        "-r", str(sample_rate),
        str(soundfont), str(midi_path),
    ]


def normalize_wav(path: Path, peak_dbfs: float = PEAK_DBFS) -> float:
    """Scale a 16-bit PCM WAV in place so its peak sits at `peak_dbfs`; return the gain applied.

    Silent files are left untouched (gain 1.0).
    """
    with wave.open(str(path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(params.nframes)
    if params.sampwidth != 2:
        raise ValueError(f"expected 16-bit PCM WAV, got {8 * params.sampwidth}-bit: {path}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.int32)
    peak = int(np.abs(samples).max()) if samples.size else 0
    if peak == 0:
        return 1.0
    gain = _INT16_MAX * 10 ** (peak_dbfs / 20) / peak
    scaled = np.clip(np.round(samples * gain), -_INT16_MAX - 1, _INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(scaled.tobytes())
    return gain


def render(
    midi_path: str | Path,
    out_path: str | Path,
    *,
    soundfont: str | Path | None = None,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
) -> Path:
    """Render `midi_path` to the WAV file `out_path` and return `out_path`.

    By default the result is peak-normalized to -1 dBFS so renders compare at equal loudness
    (FluidSynth's own output is quiet at its default gain, and a lone bassline even more so).
    Parent directories of `out_path` are created as needed.
    """
    midi_path, out_path = Path(midi_path), Path(out_path)
    soundfont = Path(soundfont) if soundfont is not None else soundfont_path()
    if out_path.suffix.lower() != ".wav":
        raise ValueError(f"output must be a .wav file: {out_path}")
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    if not soundfont.is_file():
        raise FileNotFoundError(f"soundfont not found: {soundfont} (set ${SOUNDFONT_ENV})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)  # so a stale file can't pass for a fresh render
    cmd = build_command(fluidsynth_executable(), soundfont, midi_path, out_path, sample_rate)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.is_file():
        raise RuntimeError(
            f"fluidsynth failed (exit {result.returncode}) rendering {midi_path}:\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if normalize:
        normalize_wav(out_path)
    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a MIDI file to WAV with FluidSynth.")
    parser.add_argument("midi", type=Path)
    parser.add_argument("out", type=Path, help="output .wav path")
    parser.add_argument("--soundfont", type=Path, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args(argv)
    print(render(args.midi, args.out, soundfont=args.soundfont, normalize=not args.no_normalize))


if __name__ == "__main__":
    main()
