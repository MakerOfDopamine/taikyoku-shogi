# Taikyoku Shogi engine

The rules of `../game.js` in C (`tky.c`), and `nnplay`, which plays them with a neural
network: self-play for training data, Elo matches, and label-quality measurements. This
directory contains no hand-retyped game data.

```
./build.sh                   # regenerate tables.h from game.js; build the tky test CLI
./run_tests.sh               # rules validation (see "Correctness")
./build_nn.sh                # build nnplay; needs third_party/onnxruntime, see the end
python3 export_onnx.py       # checkpoint.pt -> net_fp16.onnx, net_fp32.onnx
./nnplay selfcheck           # the runtime reproduces the exported weights
./nnplay play --positions 1000000 --out selfplay.tkn --children selfplay.kids
python3 tkn_stats.py selfplay.tkn --brief
./bundle.sh nnplay-bundle    # a tree that runs all this on another GPU
```

## Where the rules come from

`gen_tables.js` loads `../game.js` in a `vm` context and *executes* it. It writes
`tables.h` (and `tables.json`) containing:

* every `PIECES` entry's `dydx` / `djump` / `tp`, resolved for **both** colours by calling
  the reference `invert_color()` -- the rotation is never re-derived;
* `PROMOTE`, remapped to dense piece indices;
* `VALUES` / `VALUES_PROMOTED` as integers in 1/1000 pawn;
* the initial 1296-square board, read out of a real `new Board()`.

It asserts on the way through that every `tp` entry is `[int, int, int[8]]` and every
`dydx` / `djump` has length 8 (CLAUDE.md §11.8). Re-run `./build.sh` after any change to
`game.js`.

## Correctness

All in `run_tests.sh`:

| check | what it proves |
|---|---|
| `./tky perft 2` | 262 / 68614 and the 68x special-0 + 194x special-9 split, the anchors in CLAUDE.md §10 |
| `node probe.js lone` | 2408 positions: every piece id x both colours x 4 squares, full move list vs `game.js` (a lone Lion gives 88 moves) |
| `node probe.js random` | dense random boards, every hardcoded-behaviour id forced onto the board |
| `node xcheck.js` | seeded playouts: at every ply the full **sorted** move list, material, side to move and piece count must equal `game.js` |
| `./difftest.sh` | the optimised generator vs `tky_ref.c`, the straight transliteration |
| `./tky kingtest` | the King-capture detector vs a reference that walks every emptied square, the cached King square vs a full scan, and the rule's own invariant |
| `nnplay safetytest` | King safety against the slow detector (below) |
| unpack step | `unpack.py` rebuilds the engine's board from the moves at every ply, and the engine's move set for a 180-degree-rotated position is exactly the rotation of the original's |

`tky_ref.c` is kept deliberately: it is the literal transliteration, so any future
optimisation can be diffed against it without going back through Node.

Two optimisations change *how* moves are found but not *which*, and are covered by
`difftest.sh`:

* rays are walked on a 44x44 board with an off-board sentinel ring, so a step is one add
  and one compare instead of two bounds checks and a multiply;
* the hook movers' second leg walked all four hook directions from every midpoint, and the
  two *parallel* walks only ever re-emit the ray itself. That is replayed as a direct emit
  of the midpoint list -- including the quirk that a ray with a single midpoint contributes
  nothing, because that midpoint's backward walk stops immediately on the mover.

## Forced King capture

`Game` has no notion of check, so a player that does not look walks past a hanging King
constantly. `nnplay` overrides that: **if any legal move captures the enemy King, that move
is played** (`--no-regicide` turns it off). Under uniformly random play it turns 47.8%
decisive games into 99.97%, at a mean of 522 plies instead of 2460.

"Captures the King" means the King stands on any square the move empties, which is exactly
`Game.squares_emptied`: the target, the intermediate square of a special 1-8 move, and
every square strictly between origin and target of a special 9. The mover's own origin is
excluded -- for a jitto the encoded target *is* the origin and holds the mover.

Special 9 needs care. It is emitted by two different things: a trampler's ranging capture,
whose ray breaks on any piece of level >= the mover's and so can never reach a level-4
King, and the **Free Eagle's** ranging capture, which has no level check and will sweep
straight over one. Assuming CLAUDE.md's "a trample can never capture a king" covers all of
special 9 is wrong, and `tky kingtest` exists to catch exactly that.

# Neural-network self-play (`nnplay`)

The rules are not reimplemented: `nnplay.c` `#include`s `tky.c` behind `TKY_NO_MAIN`, so
the generator, `do_move` and the King-capture detector are the object code "Correctness"
covers. Everything the network sees is in the frame of the side to move: black to move is
the board as stored, white to move is rotated 180 degrees with every colour flipped
(CLAUDE.md §13). The network is one ONNX graph, value and policy over one pass of the
trunk (see "The export").

## The search

`play` and `elo` run PUCT over the policy's priors and the value head, as AlphaZero:

* each simulation walks from the root by `Q + c * P * sqrt(N) / (1 + n)` to an unexpanded
  node, the network gives that position's value and a logit for every legal move, and the
  value is backed up the path;
* **Q is min-max normalised** over the tree before it meets the exploration term (MuZero).
  Children's raw values differ by ~0.01 while the exploration term is of order 1, so raw Q
  would barely steer the visits. `--cpuct` (default 1.25);
* an unvisited child is scored at its **parent's network value**;
* priors are `softmax(logits)` over **every** legal move;
* in self-play the root's priors get **Dirichlet noise**, alpha = `--noise-alpha` / legal
  moves (default 10, AlphaZero's scaling: 0.3 at chess's ~35 moves, ~0.01 here), mixed in
  at `--noise-frac` (default 0.25). `elo` uses none;
* the subtree under the move played is **kept** for the next move, and `--sims` counts the
  new simulations each move adds (default 128, a placeholder until it is measured -- see
  "Measuring the search").

The tree itself models these rules: a position where the side to move can take the King is a
**win** for it, with no network call (this is forced regicide, and it is also how a move
that hangs the King is found: its child is that terminal), and a position seen
`--rep-limit` times, counting the game so far and the path, is a **draw**. With King
safety on it models that rule too: a move that hangs the mover's King is **pruned** the
first time a walk reaches it -- the walk is undone and counts for nothing -- and a node
whose every move is pruned is **lost**. Before this, a King threat that left the defender
a handful of escapes among hundreds of replies averaged out near -1 for it, so a plain
threat looked like a forced win: 4.4% of one self-play batch's stored values sat below
-0.9 with the mover going on to win only 42% of those games. Pruning costs one move
generation per pruned reply (about 6.5 us) and no network call. The ply cap and
resignation are deliberately left out: the value head is trained on results that already
include them, so modelling them in the tree would count them twice.

**Contempt** (`--contempt c`, default 0.1, 0 off): a draw in the tree -- repetition, or
stalemate when `stalemate_loses` is off -- is worth `-c * sigmoid(20 * (v_root - 0.1))` to
the side searching instead of 0, and the same amount the other way to its opponent.
`v_root` is the network's value of the root, so a side that thinks it is ahead steers away
from draws, and the 0.1 offset keeps near-level positions from flipping between the two
attitudes: the cost is 0.12c at `v_root = 0`, 0.5c at 0.1 and essentially c from 0.3. The
amount is set once per move, and a draw's value is worked out when it is backed up, so a
tree reused across moves never carries the previous side's contempt in a terminal. `elo`
applies it to both sides; the one-ply search has none. **Self-play makes it flat**: in
`play` a draw costs the side searching the full `c` whatever the root's value (header word
156 bit 3). Level positions -- value within +-0.01 -- had begun ending in pairs of pieces
shuffling into repetition, where the scaled cost is only ~0.012c.

**The move played** is drawn by visits^(1/T): `--temperature` and the opening schedule
(`--opening-plies`, `--opening-temperature`) keep their meaning but now apply to visit
counts, so T=1 is proportional to visits and 0 the most visited. A forced King capture is
played without searching. **King safety** applies to the pick: a pick that leaves the
mover's own King capturable is struck out and the draw repeated, which is exactly the
distribution restricted to the safe moves; if every visited move hangs, the unvisited ones
are tried in prior order.

### Batching: games in lockstep

One position's leaves are too few for a GPU call: in endgames a ply scored ~20 boards and
the GPU sat at 17% while one CPU thread launched kernels and waited. So `--parallel-games`
games (default 32) play in lockstep. Each round, every game whose mover is searching walks
its tree `--leaves` times (default 8), each pending visit counting as a loss on its path
(virtual loss) so the next walk goes elsewhere; all the games' leaves go to the network in
one call; then every finished search plays its move. Games end at different times and their
slots start new ones, so a file's games come out in the order they finish.

On the development RTX 3050, 8 games x 8 leaves gave 61 boards a call and 2,770
evaluations/s, against 4,000 boards/s for values alone: the move logits cost ~30%. The CPU
side -- tree walks, move generation, encoding -- is one thread; a status line every
`--progress-every` seconds reports the network's share of the wall clock, and well under
~80% means the CPU has become the limit, where a second process (`parallel_play_cell.py`)
uses a second core.

### What is recorded

* **`.tkn` value** of each ply: minus the played move's searched Q -- the mean of the values
  found below it -- for the side then to move. Same meaning as before, a better estimate.
* **`n_scored`**: the network evaluations the move cost (fewer than `--sims` when walks end
  on terminals, 0 for a forced capture).
* **`.kids` version 2** (`--children`): every root child the search visited, with its visits
  and its Q. Training uses the visit distribution as the policy target (below).
* the search settings, in the version 7 header.

### Measuring the search

`elo` with the bot's own graph as the baseline plays the net against itself, so
`--sims` and `--baseline-sims` compare search budgets with nothing else different:

```
./nnplay elo --baseline net_fp16.onnx --sims 128 --baseline-sims 64 \
             --opening-plies 8 --opening-temperature 1.0
```

The default `--sims` should come from that and from positions per hour at each setting on
the GPU that will run it. `--sims 0` plays the one-ply search below, so
`--baseline-sims 0` prices the search itself.

## The one-ply search (`--sims 0`)

The search before this one, still what `variance` uses: the policy ranks every legal move
from one pass over the parent, the value head scores the children of its top **5%** plus
**2%** more drawn at random from the rest (the policy's blind-spot insurance, and with
`--children` the record the next policy learns from), and the move is drawn from
`softmax(-value / T)` -- the child was encoded for the opponent, so its value is negated.
King safety applies by rejection, as above; if no scored child is safe, the unscored ones
are checked and only the safe ones are sent to the network, so pruning cannot hide the
defence. On held-out positions the top 5% held the value head's best child 78% of the time
(random: 5%). Pruning even beat scoring every child, 29-1, at T=0.05: over ~1,000 children
the softmax leaks mass to near-ties and the minimum of a thousand noisy values is mostly
noise.

## Rules of play

**King safety**, on by default, the mirror of forced regicide: a move that leaves the
mover's own King capturable is never played while some move avoids it. Without it the bot
gave games away -- in a 2M-position corpus at T=0.3 the side to move escaped a threatened
King only ~14% of the time, because a handful of defences were outweighed by hundreds of
King-hanging moves in the softmax. The check is exact: the move is played on a copy and the
opponent's replies go through the forced-regicide detector, ~8 us each. Plies where it
changed the pick carry flag 16. It applies in every mode, to both sides in `elo`, the
uniform mover included. `nnplay safetytest` checks it without a GPU: against `tky.c`'s slow
detector on every legal move of positions from random games (0 mismatches over 3.2M
moves), the five answers to the opening Great General sweep, and the sampling distribution.

**Adjudication**, also on by default, because the result was a label the value head could
not learn: the side ahead on material won only 52-57% of games at leads of 200-300 pawns,
since the game was decided by whichever side first missed a King capture.

* **Resignation.** A side ahead by `--resign-lead` pawns (default 300, ~13% of a side's
  2345-pawn start) for `--resign-plies` plies in a row (default 10) is scored the winner,
  termination 5. Over 27k self-play games such a lead was later reversed in 0.1-0.7% of
  games, against 1-3% at 200; holding it stops one Great General trample (~370 pawns in a
  ply) from ending a game. `--resign-lead 0` turns it off. Re-measure with
  `tkn_stats.py --adjudication` on a batch played with `--resign-lead 0`.
* **Ply cap** (`--max-plies`). A capped game goes to the side ahead on material, and is a
  draw under a 50-pawn lead; `--cap-draw` makes every capped game a draw. `play
  --drop-capped` writes no capped game at all.
* **Endgame starts.** `play --endgames` starts every game from a random sparse position to
  teach the network to finish a won game: one King a side, fewer than
  `--endgames-max-pieces` pieces in all (default 20, Kings included, the count uniform from
  3 up), the side ahead by at least `--endgames-imbalance` pawns (default 50) holding a
  ranging piece, neither King capturable, the side to move random. Pieces are drawn
  uniformly over the 300 non-King ids; one that only arises by promotion is placed
  promoted, one from the setup as itself or, 50/50, its promotion; an unpromoted piece that
  could promote is kept out of its own zone. The side meant to be ahead gets the ranging
  piece and at least half the rest -- an even split meets a 50-pawn lead in only ~9% of
  draws; ~18% of draws pass. 101 misses in a row stop the run as unsatisfiable.
  Adjudication is off and only royal captures are written (header word 100 bit 2). The
  start position is not recorded: records hold the board after each move.

With `--drop-capped` or `--endgames`, `--games` and `--positions` count what is written.

**Soft labels** (`play --selfplay-soft-target-scale S`, default 0.75, 0 off): a game cut off
by the ply cap or by repetition is labelled `S * tanh(material / 300 pawns)` for training,
black's view, instead of its result, so the value head learns that a material lead is worth
something even when the game never converts it. Only the label changes: the result, and so
every reported score, still follows the rules above. To train on these labels with no other
adjudication, play with `--resign-lead 0 --cap-draw`. `merge_tkn` gives games from files
without soft labels their result as the label, so a mixed merge is safe.

## Sizing a run

`--positions N` plays whole games until at least N positions are written. Games in play
when the target is reached are finished rather than cut short -- a game cut short would be
written with a result it never reached -- so a run overshoots by up to `--parallel-games`
games. `--games N` keeps exactly N games. Each finished game prints a line, and each game is
flushed as it finishes, so a killed run keeps every game it completed:

```
Game 41, Positions 10230/1000000 (index 37): 300 plies, black by ply-cap, 61.2s  ETA 2h41m
```

## Sub-batching

`nnplay` measures at startup how many boards one call should take: it doubles the batch
until the throughput stops improving or the allocator refuses, and keeps the fastest
(`--sub-batch N` skips the probe, `--probe-cap N` bounds it). A round of lockstep search is
split into calls of that size. The right size differs by an order of magnitude between
cards, which is why it is measured:

| sub-batch | RTX 3050 6 GB laptop | a datacentre card |
|---|---|---|
| 1 | 184 | 1929 |
| 8 | 222 | 3612 |
| 16 | 236 | **3855** |
| 64 | out of memory | 3874 |
| 512 | -- | 3801 |

(an earlier model, boards/s). The probe takes the median of several runs per size after two
discarded warm-ups (the first call on a shape pays for algorithm selection), stops only on
two consecutive regressions, and on the CPU provider caps itself at 8 and watches host
memory, because the host allocator overcommits instead of refusing.

## Rating (`nnplay elo`)

Plays the network against a baseline anchored at 0 and reports its Elo. Colours alternate
every game, regicide is forced for both sides, and the running estimate is printed after
each game, with how it ended: by king, adjudication, ply cap, repetition, stalemate or no
progress.

```
game 7    bot=white win  by ply cap      in   80 plies (  9.1s) |    6W    0L    2D  score 0.875 | elo   +494 [+185, +898] | first-move -101 | 0.2 min elapsed
```

**Search**: both sides search with their own net -- `--sims` for the bot, `--baseline-sims`
for the opponent (default: the same) -- with no root noise and `--temperature` 0, the most
visited move. Games run in lockstep like self-play, with each side's leaves pooled into
its own calls.

**The baseline** is `baseline_<precision>.onnx`, which `export_onnx.py` writes from
`baseline.pt`: copy the checkpoint to rate against there and re-export. `--baseline FILE`
names another graph -- the bot's own plays the net against itself -- and `--baseline
random` the uniform mover, the absolute anchor, which stops discriminating as soon as the
bot beats it every game.

**Do not run net against net with no randomness.** Two deterministic players repeat the same
game per colour, and the interval comes out far too narrow; `elo` warns.
`--opening-plies 8 --opening-temperature 1.0` makes the games differ.

### When to stop: SPRT, on by default

A match stops as soon as the games settle "is the bot at least `elo1` better than the
baseline, or at most `elo0`?" -- a sequential probability ratio test, with `--games` as the
upper limit. Defaults: `--sprt 0,50`, `--alpha 0.05 --beta 0.05`, at most 1000 games. Each
game line carries the log-likelihood ratio and its bounds; the summary states which
hypothesis was accepted, or that the limit came first, in which case the Bayesian interval
is the result. Games still in play when it decides are not counted. `--no-sprt` plays
exactly `--games`.

The bounds are in **logistic Elo**, the rating a score implies, `s = 1/(1+10^(-elo/400))`.
The draw rate is a nuisance parameter maximised out under each hypothesis -- fishtest's
generalised SPRT -- by exact constrained maximum likelihood rather than the usual normal
approximation, which would accept H1 after two or three straight wins. Measured by
simulation with 7% draws and a 30-Elo first-move advantage:

| true logistic Elo | accepts H1 | accepts H0 | undecided at 1000 | games, mean (p90) |
|---|---|---|---|---|
| -50 | 0% | 100% | 0% | 94 (152) |
| 0 | 5.4% | 93.5% | 1.1% | 256 (491) |
| +25 | 48.7% | 45.5% | 5.8% | 389 (815) |
| +50 | 94.5% | 4.8% | 0.7% | 252 (492) |
| +100 | 100% | 0% | 0% | 98 (162) |

### The rating model

BayesElo's: Rao-Kupper with an explicit draw rating, plus a first-move advantage.

```
P(win)  = 1 / (1 + 10^((-delta + draw_elo) / 400))
P(loss) = 1 / (1 + 10^(( delta + draw_elo) / 400))
```

`delta` is the bot's rating plus the advantage when it moves first and minus it when it
does not. A flat prior times that likelihood is evaluated on a grid and marginalised to the
rating; the interval is the 95% credible interval. **Alternating colours is what makes the
advantage identifiable**: a bot that went 45W-5L as black and 5W-45L as white rates `elo
+0, advantage +192`. A record with no losses cannot bound the rating from above and is
flagged `at the grid edge`; read its lower bound. `nnplay elotest Wb Lb Db Ww Lw Dw` runs
the estimator alone, and `test_nnplay.py` checks it against closed-form logistic Elo.

### Keeping the positions (`--out`)

`--out FILE` keeps every position in the `.tkn` format. Every ply's value is the **rated**
net's, whoever moved: its searched Q on its own moves, one raw evaluation of the resulting
board on the opponent's. Opponent moves carry ply flag 32. With `--out` the temperature
defaults to 0.3 rather than 0, since greedy games are the least varied training data.

## Label quality (`nnplay variance`)

A value head is trained on `(position, eventual outcome)`, and the outcome is partly set by
the position and partly by the dice rolled after it. `variance` splits the two,
`Var(z) = Var(E[z|s]) + E[Var(z|s)]`: it plays games with the one-ply search, snapshots
positions every `--branch-every N` plies from a random phase, and replays each `--branches
K` times to the end. `signal / (signal + noise)` is the **achievable R2**, the most any
value head could explain from that schedule's labels. `sweep_temperature.py` runs it per
schedule with a bootstrap interval over games, and `variance_sweep_cell.py` is the same as
one marimo cell. The estimator was checked on synthetic outcomes: true R2 0.213, estimated
0.2145, the 95% interval covering the truth in 192 of 200 trials.

## Training (`--children`, `tkykids.c`, the notebook cells)

`model_policy.py` has a value head and a policy head; `train_policy_cell.py` (one global,
`policy_training`) trains them together on every `*.tkn.tgz` and bare `.tkn` in its
directory, each with its `.kids` if there is one. The value target is TD(lambda) over the
recorded values. The policy target depends on what the `.kids` holds:

* **version 2, from a search**: AlphaZero's -- visits / total over **every** legal move,
  the unvisited ones at zero, so the policy also learns to push down what the search never
  tried. The cell generates each position's legal moves with `tkykids.c`, ~1,000 policy rows
  a position;
* **version 1, from the one-ply search**: `softmax(-value / POLICY_TEMPERATURE)` over the
  scored children;
* **no `.kids`**: 16 legal moves drawn here and scored by the model's own value head.

`CAP_SHARE` (default 0.5) draws half the batches from games that reached the ply cap and
half from the rest, since capped games run the full length and otherwise supply most of
the positions. Archives are unpacked once into `tkn_cache/` and memory-mapped; a game
present twice is counted once, matched by seed, length and moves. The report ranks each
held-out position's recorded children by the policy and asks what looking at only its top
5/10/25/50% would catch, each beside what a random pick would; a second report does the
same over every legal move of a few positions, scored by the value head.

**`tkykids.c`** is a small library over `tky.c` for Python: legal moves and child boards
from a stored board, through ctypes. It is checked against self-play: its move count equals
the recorded `n_legal`, every recorded child is one of its moves, and applying the played
move reproduces the next record's board.

`merge_tkn_cell.py` (one global function, `merge_tkn`) and `merge_tkn.py` join `.tkn` files
and archives; the cell also merges `.kids` files of one version. `parallel_play_cell.py`
runs several `nnplay play` processes and merges their output. A marimo cell cannot import
from the project's files; one cell defines a global and others use it by name.

## The export

`export_onnx.py` writes `net_fp32.onnx` and `net_fp16.onnx` from a `model_policy.py`
checkpoint (material-head `model.py` checkpoints are refused):

```
board   int32 (B, 1296)  embedding indices 0..602, the board in the side to move's frame
moves   int32 (M, 4)     rows of (board row, from square, to square, special), each in
                         its own board's frame
value   (B,)             for the side to move, in [-1, 1]
logits  (M,)             one per move row
```

The embedding index is `piece_index * 2 + colour - 1` for a piece, colour 1 for the side to
move, and 0 for an empty square; the promoted bit is not an input. Both axes are dynamic:
the search sends every leaf of a round, with all their moves. `nnplay` refuses the old
value-only graph with a separate `_policy.onnx` and says to re-export. Two things in the
export are not cosmetic:

* **The fast path is disabled before tracing.** `nn.TransformerEncoderLayer` in eval mode
  collapses to one fused op with no ONNX symbolic.
* **Attention is re-expressed in the BERT shape** (`export_rewrite.py`, reading the head
  count and width off the layer) and fused offline into `com.microsoft.Attention`; the
  pre-fusion graph is kept as `net_<p>_raw.onnx` to re-fuse with on another runtime. The
  python `onnxruntime` doing the fusion must match the C library's version; the export
  warns, and `--no-fuse` opts out.

Parity is checked in torch and through onnxruntime for both outputs: fp16 lands within
~2e-4 of fp32 torch on values and ~2e-3 on logits, fp32 within 6e-5 and 5e-4. The
manifest records the start position's value and its 262 moves' logits, which `nnplay
selfcheck` reproduces move for move. `--precision fp32 --no-tf32` closes the gap to torch
to ~5e-7, the mode for comparing against it. `export_to_web.py` writes plain fp32 graphs
without fusion for onnxruntime-web: a value graph and a policy graph.

## Is the GPU busy? (`--stats`)

`--stats` adds GPU utilisation, sampled from NVML at 10 Hz (opened with `dlopen`, so it is
optional), to the status lines. NVML's "busy" is the share of time a kernel was resident,
not how much arithmetic it used: read it as "never idle". The sub-batch curve going flat is
the saturation test.

## Files

### `.tkn`

Little-endian, fixed strides. A file header, then per game a 16-byte header and one 2608-byte
record per ply.

**Header**, 128 bytes, **256 from version 7**; offset 12 holds its size, so readers take it
from the file. Magic `TKYNNSP\0` at 0, version at 8; 32 the base seed (fresh every run
unless `--seed`); 40 games and 48 plies, patched at close; 56 the temperature (f64); 64 the
ply cap; 80 the rules (bit 0 forced regicide; bit 1 King safety; bit 2 ply cap scored on
material; bits 8..23 the resignation lead in pawns, 0 off; bits 24..31 the plies it must
hold); 92 opening plies (u16) and 94 opening temperature (u16, thousandths); 96 the
sub-batch; 100 bit 0 fp16, bit 1 capped games dropped, bit 2 endgame starts, bits 8..15 and
16..31 the one-ply search's random and top shares in thousandths (0 under search); 104..127
`<graph name>@<hash of the graph>`. Version 7 adds the search: 128 simulations a move (0 the
one-ply search), 132 leaves per tree per call, 136 c_puct (f32), 140 root-noise share (f32),
144 its alpha numerator (f32), 148 games in lockstep, 152 the `elo` opponent's simulations,
156 bits 0 tree reuse, 1 min-max normalised Q, 2 unvisited children at the parent's value,
160 the contempt (f32; 0 in files written before it existed), 164 the soft labels' scale
and 168 their pawns (f32; 0 none).

**Per game**: u32 `n_plies`; i8 result, 1 black, -1 white, 0 draw (a resignation or a scored
ply cap goes to the side ahead); u8 termination, 0 royal capture, 1 stalemate, 2
repetition, 3 no progress, 4 ply cap, 5 resignation; i16 the training label in 1/10000,
black's view (when header word 164 is nonzero; the result otherwise, and 0 in older files);
u64 seed.

**Ply record**:

| off | type | field |
|---|---|---|
| 0 | u16[1296] | the board **after** the move, absolute, `x*36 + y`, packed `index << 2 \| colour << 1 \| promoted` |
| 2592 | u32 | move: `special << 28 \| origin << 17 \| target << 6` |
| 2596 | f32 | value of this board for the side now to move: searched Q, or the network's at `--sims 0` |
| 2600 | u16 | legal moves the mover chose from |
| 2602 | i16 | material after the move, black minus white, `mat_shift` units |
| 2604 | u8 | the mover: 1 black, 0 white |
| 2605 | u8 | flags: 1 capture, 2 promotion, 4 forced King capture, 8 last ply, 16 King safety changed the pick, 32 played by the `elo` opponent |
| 2606 | u16 | network evaluations the move cost |

A value near -1 means the side now to move is losing, i.e. the mover liked the move. The
move is stored beside the board, which makes the format self-checking: `test_nnplay.py`
replays every game through `unpack.apply_move` and requires the stored boards back.

### `.kids`

Written by `--children`, paired with the `.tkn` game by (seed, plies, FNV-1a 64 of the
records' move words), so the pairing survives merging and reordering:

    file header, 32 bytes: "TKYKIDS\0", u32 version, u32 header size 32, u32 child size,
        u32 game header size 24, u64 games
    per game, 24 bytes: u32 n_plies, u32 reserved, u64 seed, u64 the move hash
    per ply: u16 n, then n children
        version 1 (--sims 0), 6 bytes: u32 move, f16 value -- every child scored
        version 2 (search),   8 bytes: u32 move, u16 visits, f16 value -- every root
                                       child visited, its value its mean

Ply k's children belong to the position before move k, i.e. record k-1's board (the start
position for k = 0, which training skips). A move uses the record's packing, absolute; a
value is for the side to move after that child, so the mover prefers low.

### Reading them

```python
from unpack_nn import unpack_nn, games
from unpack import embedding_index

for board, info in unpack_nn("selfplay.tkn"):
    x = embedding_index(board)     # exactly the array nnplay fed the network
    v = info.value                 # this board's value, side to move's view
    z = info.result_stm            # +1 if the side to move went on to win
```

`unpack_nn` canonicalises to the side to move by default (`canonical=False` for the stored
frame); `games()` walks game headers without decoding plies. `tkn_notebook.py` is the same
as one marimo cell. `tkn_stats.py FILE` reports calibration, sign accuracy, diversity,
lengths, material against outcome and how games end; `--brief` gives the headline numbers.

## Validation

`test_nnplay.py` (with no argument it plays a one-ply and a searched game first) checks,
none of it against `nnplay.c` itself:

| check | what it proves |
|---|---|
| `encode` | `encode_stm` equals `canonical_board` + `embedding_index` on positions of both colours |
| `replay` | every stored board is the previous one with the stored move applied by `unpack.apply_move`, i.e. game.js's `Board.move` |
| `symmetry` | the canonical move and the canonical board agree about the same rotation |
| `values` | at `--sims 0`, the stored boards re-evaluated reproduce the stored values |
| `torch` | `nnplay eval` against torch: ~5e-4 in fp16, ~5e-7 in fp32 without TF32 |
| `elo` | the rating posterior against closed-form logistic Elo, its sqrt(n) narrowing, and the first-move, draw and unbeaten cases |

## Deprecated

These refuse to run, with the reason: `nnplay subsample` and its `--proportions`,
`--repeats`, `--stride` and `--verify` (scoring random subsets of children, from before the
policy); `elo --vs-full` (compare `--sims` settings instead); `--policy-top` and
`--policy-explore` (the one-ply search's 5% + 2% is fixed); `--subset-fraction`; and
`--policy-model` (the policy is part of the model graph).

## Running it somewhere else

`./bundle.sh DIR` assembles a directory with one entry point. Upload it, run `setup.sh`
once, use `nnplay`. The bundle carries no runtime; `setup.sh` downloads it.

```
sh setup.sh [PYTHON]     # PYTHON only downloads things; nothing is installed into it
```

It installs `onnxruntime-gpu` with `pip --target` into the bundle's own **`ortlib/`**,
compiles, and proves the result: the CUDA provider loads, the start position's value and
policy match `net_manifest.json`, and the probe prints this GPU's throughput. It stops with
the reason if any of that fails, and is safe to re-run.

**Why a private directory.** `onnxruntime` and `onnxruntime-gpu` install into the *same*
package directory, so whichever went in last wins -- and a notebook package manager
(marimo's among them) installs the CPU wheel whenever a cell does `import onnxruntime`.
Nothing in `ortlib/` is visible to a package manager, and the binary finds it through an
rpath relative to itself.

* the runtime version is whatever `ORT_SPEC` resolves to (default `onnxruntime-gpu>=1.22`);
  pin it when the driver is older than the newest wheel's CUDA, e.g.
  `ORT_SPEC='onnxruntime-gpu==1.20.2' sh setup.sh`;
* only the **header** is shipped; the C API is versioned inside it and newer runtimes still
  serve older API versions;
* the SONAME aliases are made inside `ortlib/.../capi`, beside the provider libraries,
  because ONNX Runtime loads its providers from its own directory;
* CUDA libraries are found through `nvidia.__path__` and the usual locations and baked in
  as `DT_RPATH` -- not `DT_RUNPATH`, which the `dlopen`ed provider would not inherit.

**`nnplay` refuses to run on the CPU** unless given `--allow-cpu`: a silent fallback is
~100x slower and looks like a hang. The error names the cause -- a CPU-only wheel, a driver
older than the runtime's CUDA, or a missing library.

## Getting the runtime

For development here, `build_nn.sh` needs `third_party/onnxruntime`, the GPU C API release;
its header says how to fetch it. The CUDA provider also needs cuDNN 9 and the
cuBLAS/cuFFT/cuRAND runtimes. `build_nn.sh` finds them -- including the copies torch ships
in its own wheels -- and bakes them in as `DT_RPATH`, so `./nnplay` runs with no
`LD_LIBRARY_PATH`. Rebuild after moving the tree or changing the torch install.
