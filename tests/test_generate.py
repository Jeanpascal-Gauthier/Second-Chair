from __future__ import annotations

import math
import shutil
from collections import Counter
from pathlib import Path

import pytest
import torch
from symusic import Score

from secondchair.chords import ChordEvent
from secondchair.dataset import DEFAULT_OUT, load_song
from secondchair.generate import (
    _choose,
    _context,
    accompany,
    accompany_song,
    chord_skeleton,
    root_agreement,
    held_out_songs,
    write_accompaniment,
)
from secondchair.grid import Grid
from secondchair.model import GPT, GPTConfig
from secondchair.notes import Note
from secondchair.pop909 import DEFAULT_ROOT, read_index, song_dir
from secondchair.render import soundfont_path
from secondchair.tokenizer import RemiTokenizer, split_at_barlines

TOK = RemiTokenizer()
GRID = Grid.constant_tempo(bpm=120)  # two seconds per 4/4 bar, 16 positions
CHORDS = [ChordEvent(0, 0, "C:maj"), ChordEvent(1, 8, "G:maj"), ChordEvent(2, 4, "A:min"), ChordEvent(3, 0, "N")]
BARS = range(0, 4)

needs_pop909 = pytest.mark.skipif(not DEFAULT_ROOT.is_dir(), reason="POP909 not extracted to data/")
needs_fluidsynth = pytest.mark.skipif(
    shutil.which("fluidsynth") is None or not soundfont_path().is_file(), reason="fluidsynth or soundfont missing"
)


@pytest.fixture(scope="module")
def untrained() -> GPT:
    """Random weights: near-uniform choices exercise every branch of the sampler."""
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=len(TOK.vocab), block_size=64, n_layer=1, n_head=2, n_embd=32)).eval()


def skeleton() -> list[list[tuple[int, str]]]:
    return chord_skeleton(TOK.encode([], CHORDS, GRID, BARS).tokens)


def generate(model: GPT, **kwargs: float) -> list[str]:
    return accompany(model, TOK, skeleton(), [GRID.positions_in_bar(b) for b in BARS], **kwargs)


# --------------------------------------------------------------------------- pieces
def test_chord_skeleton_lists_each_bars_chords() -> None:
    assert skeleton() == [[(0, "C:maj")], [(0, "C:maj"), (8, "G:maj")], [(0, "G:maj"), (4, "A:min")], [(0, "N")]]


def test_context_keeps_the_longest_bar_aligned_tail() -> None:
    tokens = list(range(20))
    assert _context(tokens, [1, 6, 12], block_size=32) == tokens
    assert _context(tokens, [1, 6, 12], block_size=15) == tokens[6:]
    assert _context(tokens, [1, 6], block_size=10) == tokens[-10:]  # no bar line fits: plain tail


def test_choose_stays_within_the_allowed_tokens() -> None:
    logits = torch.tensor([5.0, 1.0, 2.0, 0.0, 3.0])
    generator = torch.Generator().manual_seed(0)
    assert _choose(logits, [1, 2, 3], 0.0, 0, generator) == 2  # greedy among allowed, not the global best
    draws = Counter(_choose(logits, [1, 3, 4], 1.0, 0, generator) for _ in range(300))
    assert set(draws) <= {1, 3, 4} and draws[4] > draws[3]
    assert {_choose(logits, [1, 3, 4], 1.0, 1, generator) for _ in range(20)} == {4}  # top-1 = greedy


def test_root_agreement() -> None:
    chords = [ChordEvent(0, 0, "C:maj"), ChordEvent(1, 0, "N")]
    assert root_agreement([Note(0, 1, 36, 80), Note(1, 2, 48, 80)], chords, GRID) == 1.0
    assert root_agreement([Note(0, 1, 40, 80), Note(1, 2, 48, 80)], chords, GRID) == 0.5  # E under C
    assert math.isnan(root_agreement([Note(2.5, 3, 40, 80)], chords, GRID))  # under N: not counted


# --------------------------------------------------------------------------- the sampler's guarantees
@pytest.mark.parametrize("seed", range(8))
def test_generated_tokens_are_well_formed_and_keep_the_chords(untrained: GPT, seed: int) -> None:
    tokens = generate(untrained, seed=seed)
    notes, chords = TOK.decode(tokens, GRID, BARS.start)  # raises on any grammar violation
    assert chords == [c for c in CHORDS if c.chord != "N"] + [ChordEvent(3, 0, "N")]
    assert tokens.count("Bar") == len(BARS)
    starts = [GRID.quantize(n.start) for n in notes]
    assert len(starts) == len(set(starts))  # one voice: at most one note per position


def test_same_seed_same_bassline(untrained: GPT) -> None:
    assert generate(untrained, seed=3) == generate(untrained, seed=3)
    assert len({tuple(generate(untrained, seed=s)) for s in range(4)}) > 1


def test_greedy_ignores_the_seed(untrained: GPT) -> None:
    assert generate(untrained, temperature=0.0, seed=1) == generate(untrained, temperature=0.0, seed=2)


def test_context_longer_than_the_model_is_cropped(untrained: GPT) -> None:
    long_bars = range(0, 24)
    skeleton_24 = chord_skeleton(TOK.encode([], CHORDS, GRID, long_bars).tokens)
    tokens = accompany(untrained, TOK, skeleton_24, [16] * 24, seed=0)  # well past block_size 64
    TOK.decode(tokens, GRID, 0)


def test_every_bar_must_open_with_a_chord(untrained: GPT) -> None:
    with pytest.raises(ValueError, match="position 0"):
        accompany(untrained, TOK, [[(4, "C:maj")]], [16])


# --------------------------------------------------------------------------- real songs
@needs_pop909
def test_accompany_song(untrained: GPT) -> None:
    song = load_song("010", read_index(DEFAULT_ROOT)["010"])
    result = accompany_song(untrained, TOK, song, seed=0)
    assert result.target == split_at_barlines(song.bass, song.grid)
    assert all(n.start >= 0 for n in result.notes)
    assert all(a.end <= b.start for a, b in zip(result.notes, result.notes[1:]))  # one voice
    assert 0.0 <= result.root_agreement <= 1.0 and result.target_root_agreement > 0.8


@needs_pop909
@needs_fluidsynth
def test_write_accompaniment(untrained: GPT, tmp_path: Path) -> None:
    song = load_song("010", read_index(DEFAULT_ROOT)["010"])
    result = accompany_song(untrained, TOK, song, seed=0)
    midi = song_dir("010", DEFAULT_ROOT) / "010.mid"
    write_accompaniment(result, Score(str(midi), ttype="second"), tmp_path, midi)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"original.wav", "generated.wav", "generated_solo.wav", "target.wav", "target_solo.wav", "tokens.txt"} <= names
    assert (tmp_path / "tokens.txt").read_text().splitlines()[1].startswith("Bar Position_0 Chord_")


@pytest.mark.skipif(not (DEFAULT_OUT / "pop909.jsonl").exists(), reason="dataset not built")
def test_test_songs_are_the_held_out_split() -> None:
    assert len(held_out_songs()) == 41
