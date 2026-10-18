"""Generate basslines with a trained model, accompanying a song's real chords.

    uv run python -m secondchair.generate runs/baseline/best.pt                 # three test songs
    uv run python -m secondchair.generate runs/baseline/best.pt --song 328 --temperature 0.7

Accompaniment mode: the song's chords are fed to the model bar by bar, as the live chord recognizer
will supply them, and the model writes the bass around them. Bar lines and chord changes are
forced; the model chooses where notes fall and their pitch, length and velocity, within what the
token grammar allows: one note per position (a bassline is one voice), positions only moving
forward and never past the next chord change, and no note running past its bar line.

Writes renders/generated/<run>/<song>/: original.wav (the song without bass), generated.wav and
target.wav (model bass and the extracted bass it learned from, each with the arrangement),
*_solo.wav (bass alone), the MIDI files, and tokens.txt (the generated tokens, one bar per line).
"""

from __future__ import annotations

import argparse
import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from symusic import Score

from secondchair.bass import bass_score
from secondchair.chords import NO_CHORD, PITCH_CLASS_NAMES, ChordEvent
from secondchair.dataset import DEFAULT_OUT, Song, load_song, read_dataset
from secondchair.grid import Grid
from secondchair.model import GPT
from secondchair.notes import Note, cut_overlaps, playable
from secondchair.pop909 import pop909_root, read_index, song_dir
from secondchair.render import render
from secondchair.tokenizer import BAR, BOS, EOS, RemiTokenizer, split_at_barlines
from secondchair.train import load_checkpoint, pick_device

OUT_ROOT = Path(__file__).resolve().parent.parent / "renders" / "generated"


@dataclass(frozen=True)
class _Ids:
    """Token ids the sampler needs, by role."""

    bos: int
    eos: int
    bar: int
    position: list[int]  # index = position
    pitch: list[int]  # index = MIDI pitch
    velocity: list[int]
    duration: list[int]  # index = steps - 1
    chord: dict[str, int]
    position_of: dict[int, int]  # Position id -> position

    @classmethod
    def of(cls, tokenizer: RemiTokenizer) -> _Ids:
        def ids(prefix: str) -> list[int]:
            return [i for i, t in enumerate(tokenizer.vocab) if t.startswith(prefix)]

        position = ids("Position_")
        return cls(
            *tokenizer.to_ids([BOS, EOS, BAR]),
            position=position,
            pitch=ids("Pitch_"),
            velocity=ids("Velocity_"),
            duration=ids("Duration_"),
            chord={tokenizer.vocab[i][len("Chord_"):]: i for i in ids("Chord_")},
            position_of={token_id: p for p, token_id in enumerate(position)},
        )


@dataclass(frozen=True)
class Accompaniment:
    song_id: str
    tokens: tuple[str, ...]
    first_bar: int
    notes: list[Note]  # the model's bass: one voice, playable
    target: list[Note]  # the extracted bass the model learned from, split at barlines
    root_agreement: float  # share of notes on the root of the chord they start under
    target_root_agreement: float


# --------------------------------------------------------------------------- sampling
def chord_skeleton(tokens: Sequence[str]) -> list[list[tuple[int, str]]]:
    """Per bar, the (position, chord) pairs of a token sequence; position 0 always comes first."""
    bars: list[list[tuple[int, str]]] = []
    position = 0
    for token in tokens:
        kind, _, value = token.partition("_")
        if token == BAR:
            bars.append([])
        elif kind == "Position":
            position = int(value)
        elif kind == "Chord":
            bars[-1].append((position, value))
    return bars


def _context(tokens: list[int], bar_starts: list[int], block_size: int) -> list[int]:
    """The longest tail of `tokens` that fits the model and starts at a bar line, as in training."""
    if len(tokens) <= block_size:
        return tokens
    i = bisect_left(bar_starts, len(tokens) - block_size)
    return tokens[bar_starts[i]:] if i < len(bar_starts) else tokens[-block_size:]


def _choose(
    logits: torch.Tensor, allowed: Sequence[int], temperature: float, top_k: int, generator: torch.Generator
) -> int:
    """One of `allowed`, sampled from the model's logits (the most likely one if temperature <= 0)."""
    candidates = torch.as_tensor(allowed)
    scores = logits[candidates]
    if temperature <= 0:
        return int(candidates[scores.argmax()])
    scores = scores / temperature
    if 0 < top_k < len(candidates):
        scores = scores.masked_fill(scores < torch.topk(scores, top_k).values[-1], float("-inf"))
    return int(candidates[torch.multinomial(torch.softmax(scores, dim=0), 1, generator=generator)])


@torch.no_grad()
def accompany(
    model: GPT,
    tokenizer: RemiTokenizer,
    skeleton: Sequence[Sequence[tuple[int, str]]],
    bar_steps: Sequence[int],
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    seed: int = 0,
) -> list[str]:
    """Tokens for the bars of `skeleton` (chords forced where given), with the bass sampled around them.

    `bar_steps` gives each bar's length in grid positions.
    """
    ids = _Ids.of(tokenizer)
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    tokens, bar_starts = [ids.bos], []

    def pick(allowed: Sequence[int]) -> int:
        context = torch.tensor(_context(tokens, bar_starts, model.cfg.block_size), device=device)[None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(context)
        return _choose(logits[0, -1].float().cpu(), allowed, temperature, top_k, generator)

    for chords, n_steps in zip(skeleton, bar_steps, strict=True):
        if not chords or chords[0][0] != 0:
            raise ValueError("every bar must open with a chord at position 0")
        bar_starts.append(len(tokens))
        tokens += [ids.bar, ids.position[0], ids.chord[chords[0][1]]]
        pending = list(chords[1:])
        position, note_here = 0, False
        while True:
            limit = pending[0][0] if pending else n_steps - 1
            allowed = (([] if note_here else ids.pitch) + ids.position[position + 1 : limit + 1]
                       + ([] if pending else [ids.bar]))
            choice = pick(allowed)
            if choice == ids.bar:
                break
            if choice in ids.position_of:
                position = ids.position_of[choice]
                tokens.append(choice)
                if pending and position == pending[0][0]:
                    tokens.append(ids.chord[pending.pop(0)[1]])
                    note_here = False
                    continue
                choice = pick(ids.pitch)  # a position the model moved to must hold a note
            tokens.append(choice)
            tokens.append(pick(ids.velocity))
            tokens.append(pick(ids.duration[: n_steps - position]))
            note_here = True
    tokens.append(ids.eos)
    return tokenizer.to_tokens(tokens)


# --------------------------------------------------------------------------- songs
def root_agreement(notes: Sequence[Note], chords: Sequence[ChordEvent], grid: Grid) -> float:
    """Share of notes whose pitch class is the root of the chord in effect where they start.

    Notes under no chord (N) are not counted; NaN if no note is.
    """
    points = [(c.bar, c.position) for c in chords]
    hits = counted = 0
    for note in notes:
        i = bisect_right(points, grid.quantize(note.start)) - 1
        chord = chords[i].chord if i >= 0 else NO_CHORD
        if chord != NO_CHORD:
            counted += 1
            hits += note.pitch % 12 == PITCH_CLASS_NAMES.index(chord.split(":")[0])
    return hits / counted if counted else math.nan


def accompany_song(
    model: GPT, tokenizer: RemiTokenizer, song: Song, *, temperature: float = 1.0, top_k: int = 0, seed: int = 0
) -> Accompaniment:
    """The model's bass over the same bars as the song's extracted bass, under the song's chords."""
    target = tokenizer.encode(song.bass, song.chords, song.grid)
    bars = range(target.first_bar, target.first_bar + target.tokens.count(BAR))
    skeleton = chord_skeleton(tokenizer.encode([], song.chords, song.grid, bars).tokens)
    tokens = accompany(model, tokenizer, skeleton, [song.grid.positions_in_bar(b) for b in bars],
                       temperature=temperature, top_k=top_k, seed=seed)
    notes, _ = tokenizer.decode(tokens, song.grid, bars.start)
    notes = playable(cut_overlaps(notes))
    target_notes = split_at_barlines(song.bass, song.grid)
    return Accompaniment(
        song.song_id, tuple(tokens), bars.start, notes, target_notes,
        root_agreement(notes, song.chords, song.grid), root_agreement(target_notes, song.chords, song.grid),
    )


def write_accompaniment(result: Accompaniment, score: Score, out_dir: Path, original_midi: Path) -> list[Path]:
    """MIDI and WAV files to compare the generated bass with the target, plus tokens.txt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokens.txt").write_text(" ".join(result.tokens).replace(f" {BAR} ", f"\n{BAR} ") + "\n",
                                        encoding="utf-8")
    written = [render(original_midi, out_dir / "original.wav")]
    for name, notes in (("generated", result.notes), ("target", result.target)):
        for suffix, arrangement in (("", True), ("_solo", False)):
            midi = out_dir / f"{name}{suffix}.mid"
            bass_score(score, notes, arrangement=arrangement).dump_midi(str(midi))
            written.append(render(midi, midi.with_suffix(".wav")))
    return written


def held_out_songs(dataset_path: Path = DEFAULT_OUT / "pop909.jsonl") -> list[str]:
    """Ids of the songs held out of training and validation."""
    return [s.song_id for s in read_dataset(dataset_path) if s.split == "test"]


# --------------------------------------------------------------------------- command line
def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Accompany POP909 songs' chords with a trained model's bass.")
    parser.add_argument("checkpoint", type=Path, help="e.g. runs/baseline/best.pt")
    parser.add_argument("--song", action="append", help="song id (repeatable; default: the first three test songs)")
    parser.add_argument("--all-test", action="store_true", help="every test song (scores only unless --render)")
    parser.add_argument("--render", action="store_true", help="with --all-test, also write audio")
    parser.add_argument("--temperature", type=float, default=1.0, help="0 = always the most likely token")
    parser.add_argument("--top-k", type=int, default=0, help="sample among the k likeliest (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="default: renders/generated/<run>")
    args = parser.parse_args(argv)

    model, tokenizer, _ = load_checkpoint(args.checkpoint, pick_device(args.device))
    if args.all_test:
        songs = held_out_songs()
    else:
        songs = [s.zfill(3) for s in args.song] if args.song else held_out_songs()[:3]
    write_audio = args.render or not args.all_test
    out_root = args.out or OUT_ROOT / args.checkpoint.parent.name
    index, root = read_index(), pop909_root()

    scores = []
    for song_id in songs:
        song = load_song(song_id, index[song_id], tokenizer.beat_res, root)
        result = accompany_song(model, tokenizer, song, temperature=args.temperature, top_k=args.top_k,
                                seed=args.seed)
        scores.append((result.root_agreement, result.target_root_agreement))
        line = (f"{song_id}: {len(result.notes)} notes (target {len(result.target)}), on the chord root "
                f"{result.root_agreement:.0%} (target {result.target_root_agreement:.0%})")
        if write_audio:
            midi = song_dir(song_id, root) / f"{song_id}.mid"
            write_accompaniment(result, Score(str(midi), ttype="second"), out_root / song_id, midi)
            line += f" -> {out_root / song_id}"
        print(line)
    if len(scores) > 1:
        mean = [sum(column) / len(column) for column in zip(*scores)]
        print(f"mean over {len(scores)} songs: on the chord root {mean[0]:.1%} (target {mean[1]:.1%})")


if __name__ == "__main__":
    main()
