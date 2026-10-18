"""POP909 -> token sequences: grid, chords, `chord` bassline and REMI tokens for every song.

    uv run python -m secondchair.dataset build   # writes data/tokens/pop909.jsonl + vocab.json
    uv run python -m secondchair.dataset sweep   # round-trips every song, plots p95 onset error

Everything runs in memory (no MIDI files in between), so what is tokenized is exactly what the
bass extraction produced. Songs listed in skiplist.txt (repo root, "<song id> <reason>" per line)
are left out; any other song that fails stops `build` and is reported by `sweep`.

Each song gets a fixed split from a hash of its id (about 90% train, 5% val, 5% test), so the split
never changes when songs are added to or removed from the skiplist.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from symusic import Score

from secondchair import plots
from secondchair.bass import LowestAtChordChange, extract_bass
from secondchair.chords import ChordEvent, chord_events, read_chord_midi
from secondchair.grid import Grid
from secondchair.notes import Note
from secondchair.pop909 import SongInfo, pop909_root, read_index, song_dir
from secondchair.tokenizer import RemiTokenizer, TokenSequence, split_at_barlines

REPO = Path(__file__).resolve().parent.parent
SKIPLIST = REPO / "skiplist.txt"
DEFAULT_OUT = REPO / "data" / "tokens"
DEFAULT_PLOT = REPO / "reports" / "onset_error_p95.png"


@dataclass(frozen=True)
class Song:
    song_id: str
    grid: Grid
    bass: list[Note]
    chords: list[ChordEvent]


@dataclass(frozen=True)
class SongTokens:
    song_id: str
    split: str
    first_bar: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class RoundTrip:
    """MIDI-free round trip of one song: errors of the decoded notes against the tokenized ones."""

    song_id: str
    n_notes: int
    onset_ms: np.ndarray
    duration_ms: np.ndarray
    velocity: np.ndarray


# --------------------------------------------------------------------------- per song
def load_song(song_id: str, info: SongInfo, beat_res: int = 4, root: Path | None = None) -> Song:
    """Grid, collapsed chord events and `chord`-strategy bassline of one POP909 song."""
    folder = song_dir(song_id, root)
    grid = Grid.from_pop909(folder / "beat_midi.txt", info.beats_per_measure, beat_res)
    segments = read_chord_midi(folder / "chord_midi.txt")
    score = Score(str(folder / f"{song_id}.mid"), ttype="second")
    bass = extract_bass(score, LowestAtChordChange(grid, segments))
    return Song(song_id, grid, bass, chord_events(segments, grid))


def tokenize_song(song: Song, tokenizer: RemiTokenizer) -> TokenSequence:
    return tokenizer.encode(song.bass, song.chords, song.grid)


def round_trip(song: Song, tokenizer: RemiTokenizer) -> RoundTrip:
    """Encode, decode and re-encode one song; raise AssertionError if anything but timing moved."""
    seq = tokenize_song(song, tokenizer)
    notes, chords = tokenizer.decode(seq.tokens, song.grid, seq.first_bar)
    reference = split_at_barlines(song.bass, song.grid)
    assert tokenizer.encode(notes, chords, song.grid) == seq, "re-encoding changed the tokens"
    assert len(notes) == len(reference), f"{len(notes)} notes decoded, {len(reference)} encoded"
    assert [n.pitch for n in notes] == [n.pitch for n in reference], "pitch sequence changed"
    pairs = list(zip(notes, reference))
    return RoundTrip(
        song.song_id,
        len(notes),
        np.array([abs(a.start - b.start) for a, b in pairs]) * 1000,
        np.array([abs(a.duration - b.duration) for a, b in pairs]) * 1000,
        np.array([abs(a.velocity - b.velocity) for a, b in pairs]),
    )


def split_of(song_id: str) -> str:
    """Fixed train/val/test assignment from a hash of the song id."""
    bucket = zlib.crc32(song_id.encode()) % 20
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


# --------------------------------------------------------------------------- skiplist
def read_skiplist(path: Path = SKIPLIST) -> dict[str, str]:
    """Song id -> reason, from "<song id> <reason>" lines; blank lines and # comments ignored."""
    if not path.exists():
        return {}
    skipped: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        song_id, _, reason = line.partition(" ")
        if not reason.strip():
            raise ValueError(f"{path}:{lineno}: every skipped song needs a reason")
        skipped[song_id.zfill(3)] = reason.strip()
    return skipped


# --------------------------------------------------------------------------- whole corpus
def build(
    tokenizer: RemiTokenizer, root: Path | None = None, skiplist: dict[str, str] | None = None
) -> list[SongTokens]:
    """Token sequences for every song not on the skiplist, in song order."""
    skiplist = read_skiplist() if skiplist is None else skiplist
    out = []
    for song_id, info in read_index(root).items():
        if song_id in skiplist:
            continue
        try:
            seq = tokenize_song(load_song(song_id, info, tokenizer.beat_res, root), tokenizer)
        except Exception as exc:
            raise RuntimeError(f"song {song_id} failed ({exc}); fix it or add it to {SKIPLIST.name}") from exc
        out.append(SongTokens(song_id, split_of(song_id), seq.first_bar, seq.tokens))
    return out


def sweep(
    tokenizer: RemiTokenizer, root: Path | None = None, skiplist: dict[str, str] | None = None
) -> tuple[list[RoundTrip], dict[str, str]]:
    """Round-trip every song not on the skiplist; returns the results and {song id: failure}."""
    skiplist = read_skiplist() if skiplist is None else skiplist
    results, failures = [], {}
    for song_id, info in read_index(root).items():
        if song_id in skiplist:
            continue
        try:
            results.append(round_trip(load_song(song_id, info, tokenizer.beat_res, root), tokenizer))
        except Exception as exc:  # report every failure, not just the first
            failures[song_id] = f"{type(exc).__name__}: {exc}"
    return results, failures


def write_dataset(songs: Iterable[SongTokens], tokenizer: RemiTokenizer, out_dir: Path = DEFAULT_OUT) -> Path:
    """Write <out_dir>/pop909.jsonl (one song per line) and vocab.json; return the jsonl path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "beat_res": tokenizer.beat_res,
        "max_bar_beats": tokenizer.max_bar_beats,
        "n_velocities": tokenizer.n_velocities,
        "vocab": list(tokenizer.vocab),
    }
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=1), encoding="utf-8")
    path = out_dir / "pop909.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for s in songs:
            f.write(json.dumps({"song_id": s.song_id, "split": s.split, "first_bar": s.first_bar,
                                "tokens": list(s.tokens)}) + "\n")
    return path


def read_dataset(path: Path = DEFAULT_OUT / "pop909.jsonl") -> list[SongTokens]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [SongTokens(r["song_id"], r["split"], r["first_bar"], tuple(r["tokens"])) for r in rows]


def plot_onset_p95(results: Sequence[RoundTrip], path: Path = DEFAULT_PLOT) -> Path:
    """Per-song p95 onset error, sorted descending: which songs the grid represents worst."""
    ranked = sorted(((float(np.percentile(r.onset_ms, 95)), r.song_id) for r in results), reverse=True)
    values = [v for v, _ in ranked]
    fig, ax = plots.chart(f"p95 onset error per song, worst first ({len(values)} songs)",
                          "songs, ranked", "p95 onset error (ms)")
    ax.plot(range(len(values)), values, color=plots.SERIES[0], linewidth=2)
    worst = "\n".join(f"{song_id}  {value:.0f} ms" for value, song_id in ranked[:5])
    ax.text(0.98, 0.95, f"worst songs\n{worst}", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=plots.MUTED, family="monospace")
    ax.set_xlim(0, max(len(values) - 1, 1))
    ax.set_ylim(bottom=0)
    return plots.save(fig, path)


# --------------------------------------------------------------------------- command line
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tokenize POP909 or sweep it for round-trip errors.")
    parser.add_argument("command", choices=["build", "sweep"])
    parser.add_argument("--beat-res", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None, help="build: output folder; sweep: plot path")
    args = parser.parse_args(argv)
    tokenizer = RemiTokenizer(beat_res=args.beat_res)

    if args.command == "build":
        songs = build(tokenizer)
        path = write_dataset(songs, tokenizer, args.out or DEFAULT_OUT)
        lengths = [len(s.tokens) for s in songs]
        splits = {name: sum(s.split == name for s in songs) for name in ("train", "val", "test")}
        print(f"{len(songs)} songs, {sum(lengths):,} tokens (median {int(np.median(lengths))}, max {max(lengths)}), "
              f"splits {splits} -> {path}")
        return

    results, failures = sweep(tokenizer)
    onset = np.concatenate([r.onset_ms for r in results])
    p95 = np.array([np.percentile(r.onset_ms, 95) for r in results])
    print(f"{len(results)} songs round-tripped, {len(failures)} failed, {len(read_skiplist())} skipped")
    print(f"onset ms p50/p95/max {np.percentile(onset, 50):.1f}/{np.percentile(onset, 95):.1f}/{onset.max():.1f}; "
          f"songs with p95 > 50 ms: {int((p95 > 50).sum())}")
    for song_id, reason in failures.items():
        print(f"  FAILED {song_id}: {reason}")
    print(f"plot -> {plot_onset_p95(results, args.out or DEFAULT_PLOT)}")


if __name__ == "__main__":
    main()
