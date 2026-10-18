"""Accompany an audio recording: find its beats and chords, have the model write a bassline, mix it in.

    uv run python -m secondchair.from_audio song.mp3
    uv run python -m secondchair.from_audio song.wav --temperature 0.7 --bass-db -3

Writes renders/audio/<file name>/:
    mix.wav        the recording with the generated bass
    bass.wav       the bass alone
    bass.mid       the bass, with the recognized chords as markers
    analysis.txt   tempo, meter and the chord found at each bar
    tokens.txt     the generated tokens, one bar per line
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import librosa
import numpy as np
import soundfile

from secondchair.analysis import Analysis, analyze
from secondchair.chords import NO_CHORD, ChordEvent, chord_events
from secondchair.generate import accompany, chord_skeleton
from secondchair.model import GPT
from secondchair.notes import Note, cut_overlaps, playable
from secondchair.render import render
from secondchair.tokenizer import BAR, RemiTokenizer, to_midi
from secondchair.train import load_checkpoint, pick_device

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO / "runs" / "baseline" / "best.pt"
OUT_ROOT = REPO / "renders" / "audio"
MIX_RATE = 44100
PEAK_DBFS = -1.0


def bars_to_play(analysis: Analysis, chords: Sequence[ChordEvent]) -> range:
    """From the first bar with a chord to the bar holding the last beat."""
    grid = analysis.grid
    last = grid.bar_at_step(grid.step_at(float(analysis.beats[-1])))
    first = min((c.bar for c in chords if c.chord != NO_CHORD), default=last)
    return range(first, last + 1)


def bassline(
    model: GPT, tokenizer: RemiTokenizer, analysis: Analysis, *, temperature: float = 1.0, top_k: int = 0, seed: int = 0
) -> tuple[list[str], list[Note], list[ChordEvent]]:
    """Generated tokens, the bass notes they decode to (one voice, playable) and the chord events used."""
    grid = analysis.grid
    chords = chord_events(analysis.chords, grid)
    bars = bars_to_play(analysis, chords)
    skeleton = chord_skeleton(tokenizer.encode([], chords, grid, bars).tokens)
    tokens = accompany(model, tokenizer, skeleton, [grid.positions_in_bar(b) for b in bars],
                       temperature=temperature, top_k=top_k, seed=seed)
    notes, _ = tokenizer.decode(tokens, grid, bars.start)
    return tokens, playable(cut_overlaps(notes)), chords


def mix(recording: Path, bass_wav: Path, out: Path, bass_db: float = -6.0) -> Path:
    """Write the recording with the bass added `bass_db` dB below the recording's RMS level.

    Both are resampled to 44.1 kHz stereo; the mix is peak-normalized to -1 dBFS.
    """
    tracks = []
    for path in (recording, bass_wav):
        audio, _ = librosa.load(str(path), sr=MIX_RATE, mono=False)
        audio = np.atleast_2d(audio)
        tracks.append(np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio[:2])
    length = max(t.shape[1] for t in tracks)
    song, bass = (np.pad(t, ((0, 0), (0, length - t.shape[1]))) for t in tracks)
    gain = _rms(song) * 10 ** (bass_db / 20) / max(_rms(bass), 1e-9)
    mixed = song + gain * bass
    mixed *= 10 ** (PEAK_DBFS / 20) / max(float(np.abs(mixed).max()), 1e-9)
    out.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(out), mixed.T, MIX_RATE, subtype="PCM_16")
    return out


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


def describe(analysis: Analysis, chords: Sequence[ChordEvent]) -> str:
    """Readable summary: tempo, meter, and the chords of every bar."""
    grid = analysis.grid
    bpm = 60 / float(np.median(np.diff(analysis.beats)))
    lines = [f"duration {analysis.duration:.1f}s, tempo ~{bpm:.0f} bpm, {grid.beats_per_bar} beats per bar, "
             f"{len(analysis.beats)} beats, {len(analysis.downbeats)} downbeats tracked", ""]
    by_bar: dict[int, list[str]] = {}
    for c in chords:
        by_bar.setdefault(c.bar, []).append(f"{c.chord}@{c.position // grid.beat_res + 1}")
    for bar in sorted(by_bar):
        lines.append(f"bar {bar:4d}  {grid.to_seconds(bar, 0):7.2f}s  {' '.join(by_bar[bar])}")
    return "\n".join(lines) + "\n"


def accompany_file(
    path: Path,
    model: GPT,
    tokenizer: RemiTokenizer,
    out_dir: Path,
    *,
    device: str = "cpu",
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
    bass_db: float = -6.0,
) -> Path:
    """Analyze `path`, generate its bassline and write everything into `out_dir`; return the mix."""
    analysis = analyze(path, device, tokenizer.beat_res)
    tokens, notes, chords = bassline(model, tokenizer, analysis, temperature=temperature, top_k=top_k, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.txt").write_text(describe(analysis, chords), encoding="utf-8")
    (out_dir / "tokens.txt").write_text(" ".join(tokens).replace(f" {BAR} ", f"\n{BAR} ") + "\n", encoding="utf-8")
    to_midi(notes, chords, analysis.grid).dump_midi(str(out_dir / "bass.mid"))
    bass_wav = render(out_dir / "bass.mid", out_dir / "bass.wav")
    return mix(path, bass_wav, out_dir / "mix.wav", bass_db)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add a generated bassline to an audio recording.")
    parser.add_argument("audio", type=Path, help="wav, mp3, flac, ...")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 = always the most likely token")
    parser.add_argument("--top-k", type=int, default=0, help="sample among the k likeliest (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bass-db", type=float, default=-6.0, help="bass level relative to the recording")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="default: renders/audio/<file name>")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    out_dir = args.out or OUT_ROOT / args.audio.stem
    mixed = accompany_file(args.audio, model, tokenizer, out_dir, device=str(device), temperature=args.temperature,
                           top_k=args.top_k, seed=args.seed, bass_db=args.bass_db)
    print(f"{mixed}\n{(out_dir / 'analysis.txt').read_text(encoding='utf-8').splitlines()[0]}")


if __name__ == "__main__":
    main()
