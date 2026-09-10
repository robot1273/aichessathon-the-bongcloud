# AI Chessathon starter

Fork this to build an agent for [AI Chessathon](https://aichessathon.com). It gives you a working
submission, baselines to beat, and a local harness that speaks the same protocol and enforces the
same clock as the platform, so you can see whether a change actually helped before you upload it.

```
git clone https://github.com/advitrocks9/aichessathon-starter
cd aichessathon-starter
make setup
make play
```

That plays your agent against a baseline over a full 120 s + 0.5 s game and prints the result.
When you like it, `make zip` and drop `submission.zip` on your dashboard.

## Writing an agent

`agent.py` is the whole submission. One function.

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The fork ships a legal random-mover, so the loop works before you write anything. Replace the body.

```
make play                                          # one game, real time control
make arena                                         # 16 fast games, prints a score
make play FEN="<fen>"                              # start from a given position
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
uv run python -m harness.arena --opponent ../my-old-version --games 200
uv run python -m harness.arena --pgn-dir games
```

- Anything your agent prints shows up under the result, so `print` debugging works. The platform
  keeps the first 4 KB and the last 4 KB, and so does the harness.
- Every rated game leaves a log on your dashboard beside the PGN with your output, your init time,
  your move times and your clock. Only your team can read it.
- Games replay. The opening and the baseline's seed both come from the game number, so a
  deterministic agent plays the same games every run and a score change is a change you made. The
  random mover it ships with is not deterministic, so `make arena` wanders until you replace it.

## The ladder

Measured with `harness/arena.py`. Beating greedy is a search. Beating minimax is a search plus an
evaluation worth searching with.

| Matchup | Games | Time control | Score |
|---|---|---|---|
| random vs greedy | 32 | 10 s + 0.1 s | 4.7% +- 5.1% (+0 =3 -29) |
| greedy vs minimax | 16 | 120 s + 0.5 s | 0.0% (+0 =0 -16) |
| numba vs minimax | 16 | 10 s + 0.5 s | 59.4% +- 16.1% (+5 =9 -2) |

Read the third row twice. 59.4% looks like an edge, but the interval runs from -47 to +195 elo, so
sixteen games have not found one. That is why `make arena` prints it.

```
uv run python -m harness.arena --agent baselines/random --opponent baselines/greedy --games 32
uv run python -m harness.arena --agent baselines/greedy --opponent baselines/minimax --games 16 \
  --base-ms 120000 --increment-ms 500
uv run python -m harness.arena --agent baselines/numba --opponent baselines/minimax --games 16 \
  --increment-ms 500
```

- `baselines/random` plays a uniformly random legal move. It is what `agent.py` starts as, minus
  the seed the baselines take from the harness.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is the
  point. Jitting a shallow search buys headroom, not depth. Read it for the warm-up call at the
  bottom, which is how you keep compilation off your clock.

## What's here

```
agent.py             your submission
baselines/           random, greedy, minimax, numba; each is a directory with an agent.py
harness/runner.py    the process the platform runs your agent in
harness/referee.py   the clock, legality, draw and cap rules
harness/rules.py     the event constants, and eight openings the rated ladder plays
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score and an interval
harness/package.py   builds submission.zip and plays the platform's two smoke games from it
docs/IDEAS.md        where the strength actually comes from
```

- `make zip` ships `agent.py`, every python file beside it, `weights/`, and any package you
  import. Add the rest with `--include`.
- It then plays the platform's two smoke games out of the zip it just built, so a file you never
  packaged fails here in a minute instead of costing one of your ten daily uploads.
- The platform decides acceptance and its validation log is the authority.
- Local games start from one of the eight openings unless you pass `--fen`. Rated games draw from
  the full set, which is not published. Treat the eight as a sample, not preparation.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
