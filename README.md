# Second-Chair

A real-time AI bass accompanist: you play, it listens, and it generates a bassline that follows
along. This repo currently holds the symbolic half — generating a bassline from a stream of
`(bar, position, chord)` events.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [FluidSynth](https://www.fluidsynth.org/) on `PATH`.

```sh
uv sync
```

Local assets (gitignored, not committed):

- `soundfonts/GeneralUser-GS.sf2` — [GeneralUser GS](https://github.com/mrbumpy409/GeneralUser-GS)
  (or point `SECONDCHAIR_SF2` at any other `.sf2`).
- `data/POP909/` — the [POP909](https://github.com/music-x-lab/POP909-Dataset) dataset.

If `fluidsynth` isn't on `PATH`, set `SECONDCHAIR_FLUIDSYNTH` to the executable.

## Listening

```sh
uv run python -m secondchair.render data/POP909/001/001.mid renders/001.wav
```

or `secondchair.render.render(midi_path, out_path)` from Python. Output is peak-normalized to
-1 dBFS so renders compare at equal loudness (`--no-normalize` to disable).

## Comparing bass extraction strategies

POP909 has no bass track, so the bassline is derived from the piano (`secondchair.bass`):

```sh
uv run python -m secondchair.bass 001
```

writes `renders/001/` with `original.wav` and, per strategy (`lowest`, `threshold`, `chord`), the
bass with the arrangement (`<strategy>.wav`) and alone (`<strategy>_solo.wav`), plus the MIDI files.

## Tokens

`secondchair.tokenizer.RemiTokenizer` turns a bassline and its chords into REMI tokens, bar by bar:

```
Bar Position_0 Chord_B:maj Pitch_47 Velocity_4 Duration_8 Position_8 Chord_C#:maj Pitch_49 Velocity_6 Duration_8
```

Positions and durations count grid steps (16ths by default) from the bar's downbeat. Chords are
reduced to 12 roots × major/minor plus `N`, every bar restates the chord in effect, and held notes
are restruck at each barline.

## Dataset and training

```sh
uv run python -m secondchair.dataset build      # POP909 -> data/tokens/pop909.jsonl
uv run python -m secondchair.dataset sweep      # round-trip check + reports/onset_error_p95.png
uv run python -m secondchair.train --name baseline
```

Training uses PyTorch's CUDA 13.0 build (configured in `pyproject.toml`) and writes
`runs/<name>/` with checkpoints (`best.pt`, `last.pt`), `log.csv` and `loss.png`. Run
`--help` for model and optimizer settings.

## Generating

```sh
uv run python -m secondchair.generate runs/baseline/best.pt --song 010
uv run python -m secondchair.generate runs/baseline/best.pt --all-test   # scores only
```

Feeds a song's real chords to the model bar by bar and lets it write the bass. Writes
`renders/generated/<run>/<song>/` with `original.wav`, `generated.wav` / `target.wav` (model bass and
the extracted bass, each with the arrangement), `*_solo.wav` and `tokens.txt`. `--temperature 0`
always plays the most likely note; higher values add variety.

## Tests

```sh
uv run pytest
```
