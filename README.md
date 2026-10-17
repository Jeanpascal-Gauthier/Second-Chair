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

## Tests

```sh
uv run pytest
```
