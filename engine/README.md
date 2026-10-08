# Taikyoku Shogi corpus generator

Plays random games at scale and writes a packed binary corpus for neural-network training.
The rules are the ones in `../game.js`; this directory contains no hand-retyped game data.

```
./build.sh                                          # regenerate tables.h from game.js, compile
./run_tests.sh                                      # full validation (see "Correctness")
./tky sample --games 3000 --threads 20              # statistics pass -> samples.bin
python3 fit_tanh.py                                 # fit the label squash -> tanh_fit.json
./tky gen --games 2000000 --threads 20 --out corpus --tanh-scale 246.8  # --save-percentage 5 by default
python3 tky_manifest.py corpus                      # -> corpus/manifest.json
```

`gen` writes one shard per thread while running and fuses them into a single
`corpus/corpus.tky` at the end; `--keep-shards` keeps the per-thread files instead.

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

Three independent comparisons, all in `run_tests.sh`:

| check | what it proves |
|---|---|
| `./tky perft 2` | 262 / 68614 and the 68x special-0 + 194x special-9 split, the anchors in CLAUDE.md §10 |
| `node probe.js lone` | 2408 positions: every piece id x both colours x 4 squares, full move list vs `game.js` (a lone Lion gives 88 moves) |
| `node probe.js random` | dense random boards, every hardcoded-behaviour id forced onto the board |
| `node xcheck.js` | seeded playouts: at every ply the full **sorted** move list, material, side to move and piece count must equal `game.js` |
| `./difftest.sh` | the optimised generator vs `tky_ref.c`, the straight transliteration |
| `run_tests.sh` unpack step | `unpack.py` on both formats against each other and against `tky_reader.py`, plus the start-position anchors |
| `run_tests.sh` sampling step | the 5% selection, its label balance, and `.tkp` snapshots against `.tky` replay |
| `./tky kingtest` | the King-capture detector vs a reference that walks every emptied square, the cached King square vs a full scan, and the rule's own invariant |
| `python3 tky_reader.py` | the reader rebuilds the engine's exact board from the stored moves at every ply |

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

`Game` has no notion of check, so a random mover walks past a hanging King constantly. By
default the generator now overrides the random choice: **if any legal move captures the
enemy King, that move is played**. `--no-regicide` restores pure random play, and the file
header records which was used.

"Captures the King" means the King stands on any square the move empties, which is exactly
`Game.squares_emptied`: the target, the intermediate square of a special 1-8 move, and
every square strictly between origin and target of a special 9. The mover's own origin is
excluded -- for a jitto the encoded target *is* the origin and holds the mover.

Special 9 needs care. It is emitted by two different things: a trampler's ranging capture,
whose ray breaks on any piece of level >= the mover's and so can never reach a level-4
King, and the **Free Eagle's** ranging capture, which has no level check and will sweep
straight over one. Assuming CLAUDE.md's "a trample can never capture a king" covers all of
special 9 is wrong, and `tky kingtest` exists to catch exactly that: it runs the fast
detector against a reference that walks every emptied square of every move one at a time,
checks the incrementally maintained King square against a full board scan, and asserts that
no non-final position is ever left with the enemy King en prise.

The effect on the corpus is large:

| | random | forced King capture |
|---|---|---|
| decisive | 47.8% | **99.97%** |
| mean game length | 2460 plies | **522 plies** |
| median / p90 length | 3000 / 3000 | 451 / 1057 |
| black wins | 23.7% | 50.4% |
| games/s (20 threads) | 598 | **1855** |
| positions/s | 1.48 M | 0.97 M |
| mean legal moves per position | 1542 | 1329 |

Positions per second falls 31%, but only ~2% of that is the detector. The rest is the
position mix: games that end at ply 522 instead of 3000 spend their whole life in the
piece-dense opening, and a position with 760 pieces on the board costs more to generate
for than one with 155, even though it yields fewer moves. Holding the mix fixed at a
524-ply cap, the detector costs 1.120 -> 1.101 Mpos/s.

## Sampling policy

`--policy move` (default) picks **uniformly among all legal moves**, which is the natural
reading of "random game" and matches what a generator over `game.js` would do.

`--policy piece` picks uniformly among the pieces that have a move, then uniformly among
that piece's moves. It is ~50x faster because it generates moves for a handful of pieces
instead of all ~400, but it is a genuinely different distribution: it strips the weighting
that the move-uniform policy gives to pieces with many moves, and since 194 of the 262
opening moves are trample moves belonging to six pieces, games become far less violent
(royal captures fall from 48% to 0.9%). Under this policy the `n_legal` field holds the
*sampled piece's* move count, not the position's; `header.policy` records which was used.

## Termination

Exactly `Game`'s rules: royal capture (only id 1000 is royal), 4-fold repetition of
position + side to move, stalemate (side to move loses), optional no-progress limit, plus a
`--max-plies` cap that the reference does not have. Repetition and stalemate never fire in
practice on a 36x36 board -- across 24.6M random positions, not one had zero legal moves,
and the fewest ever seen was 90. With forced King capture on, 99.97% of games end by royal
capture and the ply cap is nearly dead too.

## Position sampling (`--save-percentage`)

`--save-percentage P` (default **5.0**) marks roughly P% of positions as the training set,
choosing them so their **tanh labels come out evenly spread over [-1, 1]** rather than
following the natural distribution.

Bins are equal-width in label space, 64 of them. The per-bin keep rate comes from
water-filling: find the level L where keeping `min(seen[b], L)` from every bin exhausts the
budget `P% * total_seen`, keep rarer bins whole and redistribute their shortfall, thin the
rest to `L/seen[b]`. Two corrections make it behave on short runs as well as long ones:

* a bounded proportional term on the accumulated per-bin deficit, so a bin that was thinned
  too hard under an earlier, cruder rate catches back up;
* a running ceiling at `1.03 * L`, because a game's material barely moves from ply to ply
  and without it one lopsided game can dump hundreds of consecutive positions into a single
  bin before the next rate update.

The natural label distribution is already close to uniform -- that is exactly what
`fit_tanh.py` maximises -- so this is a modest correction, not a heavy resample. Over 1000
games (521k positions) it keeps 4.87% and flattens the 16-bin spread from 1.23x to 1.12x,
which is Poisson noise on 1585 counts; the standard deviation of the per-bin share drops
from 0.059 to 0.026. Selection stays flat across the course of each game too: the accept
rate runs 5.04% over the first tenth of a game down to 4.64% over the last.

Because the sampler is online it cannot un-keep a position, so very short runs are still
warming up: 60 games lands at 4.3% with a worst bin 80% off even, 500 games at 4.7% and
13%, 3000 games at 4.9% and 5.5%. `chi2/bin` printed at the end is the diagnostic -- around
1.0 means the residual unevenness is pure sampling noise, and it typically settles near 0.5
because the ceiling suppresses variance.

## Two output formats

`--format games` (default) writes whole games with every move, and flags the selected plies
with bit 3 of the move word. `--format positions` writes only the selected positions, as
standalone board snapshots that need no replay.

**The snapshots are 16x larger**, which is counter-intuitive but not close: a move is 8
bytes and a board is 2624. On 1000 games the same 25363 selected positions cost 4.19 MB as
a game stream and 66.55 MB as snapshots. The game stream is also lossless -- you can change
your mind about the 5% -- so prefer it unless the training pipeline genuinely cannot
replay. `run_tests.sh` checks the two agree square for square.

## Corpus format

Little-endian throughout. Each worker thread writes its own `shard_NNNN.tky` while it runs,
and when they have all finished the shards are **fused into a single `corpus.tky`** (or
`corpus.tkp`) and removed. `--keep-shards` leaves them in place instead.

Fusing is a concatenation: every shard carries the same 128-byte header except for its
first-game index and its two counts, so the fused header is shard 0's with the counts
summed and the first-game index reset to 0. Nothing downstream changes -- `unpack.py`,
`tky_reader.py` and `tky_manifest.py` all take a single file as happily as a directory --
and the game index the game format derives as `header.first_game + n` stays correct,
because the shards are concatenated in worker order and worker *i* owns a contiguous run
of game indices. `run_tests.sh` checks a fused corpus and a `--keep-shards` one built from
the same seed: identical payload bytes, identical positions, identical game indices.

### File header, 128 bytes

| off | type | field |
|---|---|---|
| 0 | char[8] | `TKYSHOGI` |
| 8 | u32 | format version (1) |
| 12 | u32 | file header bytes (128) |
| 16 | u32 | game header bytes (16) |
| 20 | u32 | ply record bytes (8) |
| 24 | u32 | `VALUE_FP` (1000) -- piece values are in 1/1000 pawn |
| 28 | u32 | `mat_shift` -- the material field's unit is `1 >> mat_shift` pawns (default 3, i.e. 1/8) |
| 32 | u64 | seed base |
| 40 | u64 | index of this shard's first game |
| 48 | u64 | games in this shard |
| 56 | u64 | plies in this shard |
| 64 | u32 | tanh scale S, in 1/1000 pawn |
| 68 | u32 | policy: 0 uniform-move, 1 uniform-piece |
| 72 | u32 | max plies |
| 76 | u32 | repetition limit |
| 80 | u32 | no-progress limit |
| 84 | u32 | stalemate loses |
| 88 | u32 | board size (36) |
| 92 | u32 | distinct piece types (302) |
| 96 | u32 | forced King capture was on |
| 100 | u32 | save percentage x100 (0 = no sampling, everything counts as selected) |
| 104 | u32 | label bins used by the sampler (64) |

### Per game, 16 bytes, then `n_plies + 1` ply records

| off | type | field |
|---|---|---|
| 0 | u32 | `n_plies` |
| 4 | u8 | result: 0 white wins, 1 black wins, 2 draw |
| 5 | u8 | termination: 0 royal capture, 1 stalemate, 2 repetition, 3 no progress, 4 ply cap |
| 6 | u16 | flags (reserved) |
| 8 | u64 | seed -- reproduces the whole game from `--seed` plus the game index |

### Ply record, 8 bytes

| bits | field |
|---|---|
| `move` u32, 31..28 | special: 0 ordinary, 1..8 two-step (leg 1 = `DIRS[special-1]`), 9 trample |
| `move` 27..17 | origin square, `x*36 + y` |
| `move` 16..6 | target square |
| `move` 5 | this move captured something |
| `move` 4 | this move promoted the mover |
| `move` 3 | selected by `--save-percentage` |
| `move` 2..0 | reserved |
| `mat` i16 | material balance, **black minus white**, in `mat_shift` units, King excluded |
| `nmoves` u16 | legal moves for the side to move, capped at 65535 |

A record describes the position *before* the move it carries. Every game ends with one
extra record for the final position; a terminal record is identified by its top nibble
being `0xF`, which no real move can produce because `special` never exceeds 9. Side to
move is `1 - (ply & 1)`; black moves first.

### Position records (`--format positions`, `shard_NNNN.tkp`)

Fixed 2624-byte stride, so a shuffled trainer can mmap the file and index straight to
record `i` with no index and no replay. The C struct is field-ordered to need no padding
and carries a `_Static_assert` on its size.

| off | type | field |
|---|---|---|
| 0 | u16[1296] | board, packed codes, `x*36 + y` |
| 2592 | u64 | seed of the game this came from |
| 2600 | u32 | game index |
| 2604 | u32 | total plies of that game |
| 2608 | u16 | ply |
| 2610 | u16 | legal moves for the side to move |
| 2612 | i16 | material balance, same units as the game format |
| 2614 | u16 | occupied squares |
| 2616 | u8 | side to move |
| 2617 | u8 | result |
| 2618 | u8 | termination |
| 2619 | u8 | label bin the sampler assigned |
| 2620 | u32 | reserved |

Header is the same 128-byte layout, magic `TKYPOSNS`; offset 16 holds the record size and
20 the board square count, offset 48 the record count and 56 how many positions were looked
at to produce them.

### Why this shape

* **Moves, not boards.** A board snapshot is 1296 squares; a move is 4 bytes. Replaying a
  move needs the three move-application rules and the promotion table -- roughly 30 lines,
  in `tky_reader.apply_move` -- and **no move generation**, so a training pipeline never
  links the engine. That is the whole reason the format is 8 bytes per position rather
  than kilobytes.
* **Fixed-width records.** No varints, no bit unpacking at read time; `np.frombuffer` over
  a `memmap` gives you the whole game as a struct array.
* **Raw material, not the squashed label.** The label is `tanh(material / S)` with S in
  the header, so a reader recovers it with one `np.tanh` (or a 65536-entry lookup table,
  since the field is an i16). Storing the raw value costs nothing extra, loses no
  precision to saturation, and lets the squash be refitted later without regenerating.
* **Extras that cost nothing to produce.** `nmoves` is a free mobility feature and target;
  the capture and promotion bits let a reader find the interesting plies without decoding;
  the per-game seed makes any game reproducible and any bug bisectable.

## The label

`label = tanh(material_pawns / S)`, from black's point of view; `Position.label_stm` flips
it to the side to move. Material excludes the King -- both sides hold exactly one in every
non-terminal position, so its 100000 would only add noise -- but *does* include the Crown
Prince, which is capturable in this variant.

S is fitted by `fit_tanh.py` from a `tky sample` pass. The criterion is that **S maximises
the entropy of the label distribution**, which is the two-sided form of "most balances land
where tanh still has a gradient": too small an S piles mass onto +/-1 where the derivative
vanishes, too large an S piles it onto 0 where the label stops separating positions.

On 2.6M sampled positions from the default settings the optimum is **S = 247 pawns**, and
the resulting labels come out almost exactly uniform on (-1, 1):

```
deciles   -0.81 -0.60 -0.39 -0.19 +0.00 +0.18 +0.38 +0.60 +0.81
5.0% of labels saturate (|label| > 0.95);  89.9% sit in |x| < 1.5
cross-check: |material| p95 / atanh(0.9) = 307 pawns, same order
```

S depends on the game distribution: pure random play at natural length gives S = 314,
because those games run to ply 3000 and drift further apart. Refit with `tky sample` +
`fit_tanh.py` after changing `--policy`, `--max-plies` or `--no-regicide`.

Material spread grows with game stage (sd 67 pawns before ply 25, 271 by ply 800), so
`tanh_fit.json` also reports a per-ply-bucket S if you would rather normalise by stage.
Because the raw material is what is stored, switching costs nothing.

## Unpacking it

`unpack.py` is the single entry point for both formats. It yields `(board, info)` where
`board` is a `(36, 36)` uint16 of packed codes indexed `board[x][y]` -- x the rank, y the
file, matching `board[x][y]` in game.js -- and `info` carries everything stored with that
position. It depends only on numpy and `tables.json`, so it can be copied into a training
repo on its own.

```python
from unpack import unpack, embedding_index, piece_id, colour, promoted

for board, info in unpack("corpus/"):        # .tky or .tkp, a file or a directory
    x = embedding_index(board)               # (36,36) int, 0..602, for nn.Embedding(603, d)
    y = info.label                           # tanh(material / S), side to move's view
```

A square is one uint16: `piece_index << 2 | colour << 1 | promoted`. `piece_index`,
`piece_id`, `colour`, `promoted`, `occupied`, `embedding_index` and `material_plane` decode
it; `PIECE_NAME[i]` names a piece index.

`info` holds `game_index`, `seed`, `ply`, `plies_total`, `side_to_move`, `material` (pawns,
side to move minus opponent by default; black minus white when `canonical=False`), `label`, `label_bin`, `n_legal`, `n_occupied`, `result`, `termination`,
`selected`, and -- game format only -- `move`, the decoded move played from this position.
`label_stm`, `result_stm` and `plies_remaining` are derived; `as_dict()` gives the lot.

Boards are copied on the way out. Pass `copy=False` to get a view of the array being
replayed, which is faster but is overwritten by the next step.

### Perspective

On disk the board is **absolute**, always in black's frame: black's King starts at (0,17)
and black moves toward increasing x, whoever is to move.

`unpack()` **canonicalises to the side to move by default**. White-to-move positions are
rotated 180 degrees with every colour bit swapped, so the side to move always plays as
black up the board and the network never has to learn two perspectives. Pass
`canonical=False` (or `--absolute`) for the board exactly as stored. That is exact rather than approximate: the setup is a 180
degree rotation and not a mirror, a white piece's move table is by construction the 180
degree rotation of the black one, and the promotion zones map onto each other (white's
`x <= 10` becomes `x >= 25`). `run_tests.sh` proves it by asking the engine for the legal
moves of a position and of its canonical form and requiring one to be the rotation of the
other -- including the `special` nibble, since a 180 degree rotation sends `DIRS[i]` to
`DIRS[i+4]`.

`material`, `label`, `result` and `move` are rewritten into the same frame and
`side_to_move` is pinned to 1, so `label == label_stm` and `result == 1` means the side to
move won. `Info.canonical` records which frame you are in. The absolute frame is gone once
canonicalised, so use `canonical=False` if anything downstream needs it -- and note
`tky_reader.py` is always absolute.

As a script it inspects a corpus:

```bash
python3 unpack.py corpus/ --show 2 --board
```

`--all` walks every ply rather than just the sampled ones, `--stride`/`--limit` subsample,
`--json` prints `info` as JSON, `--count` just summarises.

## Reading it

```python
from tky_reader import Corpus, PositionCorpus, to_index_planes

# game stream: replays each game, yields the sampled positions
c = Corpus("corpus")
for pos in c.positions(shuffle_games=True, seed=0):
    x = to_index_planes(pos.board, pos.side_to_move)   # int16[36,36], values < 604
    y = pos.label_stm                                  # tanh material, side to move

# position corpus: a struct array over an mmap, shuffle it however you like
pc = PositionCorpus("corpus")
order = np.random.permutation(len(pc))
boards = pc.records["board"][order[:1024]]             # (1024, 1296) uint16
labels = pc.labels()[order[:1024]]
```

`positions()` yields only the sampled subset when the shard was written with a
`--save-percentage` below 100; pass `selected_only=False` to walk every ply.

`positions()` replays each game once and yields every ply, so the amortised cost per
position is a handful of array writes. **Do not seek to a single ply by replaying from the
start each time** -- sample whole games into a shuffle buffer instead, the usual pattern.

`to_index_planes` returns a dense index in `0..602` -- 0 for empty, then piece index * 2 +
colour -- the natural input to a single `nn.Embedding(603, d)`; one-hotting 603 piece types
into planes would be 3 MB per position. The count is 301 real pieces (including the King)
in two colours, plus one code for empty. The raw `code >> 1` would run to 603 and leave a
hole at 0, because empty squares carry a colour bit that `game.js` pins to 1 so "empty,
colour 0" never occurs; `embedding_index` subtracts one to close it. `to_feature_planes` is a 5-plane dense alternative. The promoted bit is
dropped by the index encoding because it never changes how a piece moves -- feed it as an
extra binary plane if the value head wants it, since `VALUES_PROMOTED` differs.

## Throughput

Measured on an i9-13900H (6 P-cores + 8 E-cores, 20 threads).

| configuration | games/hour | positions/hour | GB/hour | decisive |
|---|---|---|---|---|
| **default** (forced King capture) | **6.7 M** | 3.5 B | 28 | 99.97% |
| `--no-regicide` | 2.2 M | 5.3 B | 43 | 48% |
| `--no-regicide --max-plies 300` | 16.2 M | 4.9 B | 39 | 0.8% |
| `--no-regicide --max-plies 200` | 24.3 M | 4.9 B | 39 | 0.4% |

Without the King rule, position throughput is flat in the ply cap and the cap only trades
games against decisive results, because random royal captures happen at median ply 1938.
The King rule gets both at once: 3.1x the games and 6.5x the *decisive* games per second,
paying 31% of raw position throughput for it.

`--policy piece` multiplies games/hour by ~50 at the cost of a different game distribution.
Scaling is 10.5x from 1 thread to 20 on this laptop part; a many-core server should scale
close to linearly, since the workers share nothing.

---

# Neural-network self-play (`nnplay`)

`tky` plays randomly. `nnplay` plays the network in `checkpoint.pt`: at every ply it
generates the legal moves, evaluates the position after **each** of them, and samples the
move from a softmax over the negated leaf values.

```
./build_nn.sh                                   # needs third_party/onnxruntime, see below
./bundle.sh nnplay-bundle                       # 13 MB tree to run this on another GPU
python3 export_onnx.py                          # checkpoint.pt -> net_fp32.onnx, net_fp16.onnx
./nnplay play --games 100 --max-plies 400 --out selfplay.tkn
python3 unpack_nn.py selfplay.tkn --show 5      # inspect it
python3 test_nnplay.py selfplay.tkn             # validate it
```

**It is slow, and unavoidably so.** A ply costs one network evaluation per legal move, and
the mean branching factor is ~1300, so a ply is ~1300 forward passes of a 4-layer
transformer over 1296 tokens -- about 22 GFLOP each. On the development GPU (a 6 GB
RTX 3050 laptop part) that is 352 boards/s, so **3.8 s per ply**, and the `--max-plies`
default of 3000 inherited from `tky` is hours. Set a cap you actually want. `--games N`
runs sequentially and `--first-game N` lets separate runs write shards that do not
collide, which is how to use more than one GPU.

The rules are not reimplemented. `nnplay.c` `#include`s `tky.c` behind `TKY_NO_MAIN`, so
the generator, `do_move`, the King-capture detector and the termination rules are the same
object code the corpus generator uses, and everything in "Correctness" above still covers
them.

## The ply

1. `gen_all` gives every legal move, already deduplicated by `(origin, move code)`. Two
   encodings that reach the same square are two candidates, not one -- a Lion's special-0
   `tp` jump captures only at the destination while the special-k double step also
   captures on the intermediate square (CLAUDE.md section 6), so collapsing them would be
   a rules bug. Where the intermediate is empty they do produce identical boards, and both
   copies are evaluated and both carry softmax mass; that is the move-uniform reading.
2. Each candidate is played on a scratch position and the result encoded for **the side to
   move in it**, which is the opponent. That is `unpack.canonical_board` followed by
   `unpack.embedding_index`: black to move is the board as stored, white to move is
   rotated 180 degrees with every colour bit flipped.
3. The batch goes through the network; the first output, `value`, is taken.
4. The move is drawn from `softmax(-value / T)`. The negation is because the leaf was
   scored from the opponent's side of the board. `--temperature 0` is greedy.

**Forced King capture.** As in `tky`, if any legal move captures the enemy King it is
played without consulting the network -- same detector, same
`Game.squares_emptied` definition, including the Free Eagle's ranging capture that the
trample level rule does not cover. The chosen leaf still goes through the network on its
own so every ply carries a real value, and the record is flagged. `--no-regicide` hands
the decision back to the network.

**King safety**, the mirror rule, on by default: a move that leaves the mover's own King
capturable is never played while some move avoids it. Without it the bot gave games away.
In a 2M-position corpus at T=0.3 the side to move escaped a threatened King only ~14% of
the time, and 7.1% of games opened with the Great General sweeping its file to the square
beside the enemy King -- a threat with five one-move answers, all of which take the
General, that White found 35% of the time. The cause is the softmax, not the net: the net
saw the danger, but a handful of defences at weight ~1 are outweighed by hundreds of
King-hanging moves at `exp(-0.99/0.3) ~ 0.037` each.

The check is exact. The candidate move is played on a copy and the opponent's replies go
through the forced-regicide detector; a move that removes its own King (a Free Eagle
sweeping over it) counts as hanging it. It is applied by rejection: sample from the
softmax, and if the pick hangs the King, drop it and sample again from the rest. That is
exactly the softmax restricted to safe moves, and on an ordinary ply it costs one check,
6.6 us against a ply's ~14 ms of network time. If no scored child is safe -- the policy's
picks are only a few percent of the moves -- the unscored children are checked and only the
safe ones are sent to the network, so pruning cannot hide the defence. If nothing is safe, the King is lost
anyway and the pick is made as before. Plies where the rule changed the pick carry flag 16,
and the run summary counts them.

It applies in `play`, `variance` and to both sides in `elo`, the uniform-random baseline
included. `subsample` is unchanged. `--no-king-safety` turns it off, and the file header
records which was used. `nnplay safetytest` checks it without a GPU: the check against
`tky.c`'s slow detector on every legal move of positions from random games (0 mismatches
over 3.2M moves), the five answers to the sweep, and the sampling distribution against the
restricted softmax.

**Adjudication**, also on by default, because the result was a label the value head could
not learn: in the 2M-position T=0.3 corpus the side ahead on material won only 52-57% of
games at leads of 200-300 pawns, since the game was decided by whichever side first missed
a King capture. Two rules make the label follow material instead:

* **Resignation.** A side ahead by `--resign-lead` pawns (default 300) for
  `--resign-plies` plies in a row (default 10) is scored the winner, termination 5
  `resignation`. 300 pawns is ~13% of a side's 2345-pawn start. The threshold is the lead
  that does not come back: over 27k self-play games, a 300-pawn lead held 10 plies was later
  reversed -- the other side going 300 ahead -- in 0.1-0.7% of games, against 1-3% at 200.
  Holding it is what stops a single Great General trample, which can swing ~370 pawns in one
  ply, from ending a game. `--resign-lead 0` turns it off.
* **Ply cap.** A capped game goes to the side ahead on material, and is a draw when the
  lead is under 50 pawns (fixed; format version 5 on, version 4 drew only an exact tie). It
  used to be written as a draw it never was: 91% of capped games in the T=1.0 corpus ended
  at least 50 pawns apart. `--cap-draw` makes every capped game a draw again.

  For a softer target than +/-1 on capped games, e.g. `tanh(material / scale)`, compute it
  at training time: termination 4 in the game header marks them, and the last ply record's
  `material` is the final balance.

Both apply wherever games are played: `play`, the `variance` main line and its playouts
(which inherit the main line's count of plies the lead has held), and `elo`. The header
records the settings. Re-measure the threshold once play changes -- King safety changes it
-- with `tkn_stats.py --adjudication` on a batch generated with `--resign-lead 0`; a corpus
that already resigned stops at the rule in force and cannot show what lies past it.

## Sizing a run by positions

`--positions N` plays whole games until at least N positions have been written, however
many games that takes, which is usually what you want when filling a training corpus.

```
./nnplay play --positions 200000 --temperature 0.3 --out selfplay.tkn
```

It **overshoots rather than truncating the last game**: a game cut short would be written
with a result it never reached, and every position in it would carry a value target that is
simply wrong. Game lengths vary enormously here -- 3 to 20 plies inside one short run -- so
expect to land a little over the target. Passing `--games` as well turns it into a cap on
the number of games, for bounding a run that might otherwise take longer than you want.

The progress line counts toward the target:

```
game 19: 20 plies, draw by ply-cap, 3.9s (2835 boards/s)  [193/200 positions]
game 20: 10 plies, white by royal, 1.1s (2836 boards/s)  [203/200 positions]
```

The file header records the games and plies actually written rather than the number asked
for, so a `--positions` shard is self-describing in the same way a `--games` one is.

## Sub-batching

A ply is ~1300 boards and will not fit on a GPU at once, so `nnplay` measures at startup
what does: it doubles the batch until the allocator refuses or the throughput stops
improving, and keeps the fastest size. `--sub-batch N` skips the probe, `--probe-cap N`
bounds it. Measured curves, same graph, two very different cards:

| sub-batch | RTX 3050 6 GB laptop | a datacentre card (170 GB host) |
|---|---|---|
| 1 | 184 | 1929 |
| 4 | 206 | 3236 |
| 8 | 222 | 3612 |
| 16 | 236 | **3855** |
| 64 | out of memory | 3874 |
| 512 | -- | 3801 |

Both plateau, and neither plateaus where the other does: the 3050 is saturated by batch 8
and runs out of memory at 64, the bigger card climbs to 16 and is then flat to 512, at 11x
the throughput. There is no single right answer to bake in, which is the whole reason the
probe measures.

**Measuring this correctly is harder than it looks**, and getting it wrong picked
sub-batch 4 on a card that wanted 16. Three things were needed:

* **two warm-up runs per size, discarded.** The first call on a shape pays for cuDNN
  algorithm selection and arena growth. On a card where a batch is 2 ms that one-off is
  three times the steady-state cost, and including it made sub-batch 8 look like a cliff
  (6.1 ms measured, 2.21 ms real);
* **the median of the reps, over at least 0.25 s of wall clock.** Not the mean, which one
  allocation stall can veto a size with. Not the minimum either: small batches have the
  higher variance, so best-of rewards them for it, and it inverted the 3050 curve outright
  (277/247/191 falling, where the median shows 184/206/236 rising);
* **two consecutive regressions to stop**, not one, so a genuine dip cannot hide whatever
  is above it.

**The probe must never stop only because the allocator refused.** That was the original
rule and it is safe on a device allocator -- the GPU says no and the probe stops -- but the
CPU provider's host allocator does not say no, it overcommits. Since the one unfused
attention layer holds an 8 x 1296 x 1296 score matrix per board, 26.9 MB in fp16, a probe
to the default cap of 8192 asks for 220 GB, and on a CPU fallback it will take a 64 GB
machine and its swap down with it. It did. Four rules stop it now:

* on the CPU provider the cap is forced to 8, whatever `--probe-cap` says, since CPU
  self-play is ~37x slower here and not worth tuning anyway;
* the host-memory guard below applies **only** to the CPU provider. CUDA is excluded
  because its allocator does refuse, and because the RSS estimate is wrong there anyway --
  on a 170 GB host it predicted 74 GB for sub-batch 1024 and stopped a walk that had
  plateaued six doublings earlier;
* throughput falling more than 10% below the best seen. It has to be a real regression
  rather than a small gain, because the curve rises gently -- 266 to 271 boards/s from
  batch 1 to 2 on the development GPU -- and a "stop when it stops improving" rule picks
  batch 2 there instead of 32;
* the next batch's predicted host memory, from the **marginal** RSS growth per board,
  exceeding a third of `MemAvailable`. Marginal, not average: the first batch also pays
  for loading the provider and building the CUDA context, and charging that per board
  over-estimates by orders of magnitude and stops the probe immediately;
* a single batch already taking more than 10 s.

The allocator refusing is still honoured, but as a backstop. Throughput is flat in the batch size on the development GPU -- it saturates at
batch 2 -- but that is a property of that card, not of the model, so the probe measures
rather than assumes. The measurement is printed:

```
  sub-batch    16 :    327.3 boards/s  (48.9 ms/batch)
  sub-batch    32 :    351.6 boards/s  (91.0 ms/batch)
  sub-batch    64 : Failed to allocate memory for requested buffer of size 1719926784
  chosen sub-batch 32 (351.6 boards/s)
```

Candidates are built on the CPU by copying the position, playing the move and encoding the
result -- about 25 MB of memory traffic per ply, against seconds of GPU time, so it has
been left simple rather than made incremental. On a GPU fast enough for that to matter,
patching only the squares the move empties (`Game.squares_emptied`, at most 35 for a
special 9) is the cheap version.

## Rating the bot (`nnplay elo`)

Plays the network against a uniformly random mover and reports its Elo, with the baseline
anchored at 0. Colours alternate every game, regicide is forced for both sides so games
finish, and the running estimate is printed after each game.

```
./nnplay elo --games 200 --progress-every 25
game 7    bot=white win  in    4 plies (  1.4s) |    8W    0L    0D  score 1.000 | elo  +1500 [+315, +1476] (at the grid edge) | first-move +0 | 0.2 min elapsed
```

`--temperature` defaults to **0** here, which measures playing strength; pass it explicitly
to rate the sampling policy you actually generate data with instead.

### When to stop: SPRT, on by default

A match stops as soon as the games settle the question "is the bot at least `elo1` better
than the baseline, or at most `elo0`?" -- a sequential probability ratio test, with
`--games` as the upper limit. Defaults: `--sprt 0,50`, `--alpha 0.05 --beta 0.05`, at most
1000 games. Each game line carries the log-likelihood ratio and its bounds, `LLR +1.23
[-2.94, 2.94]`, and the summary states which hypothesis was accepted, or that the limit was
reached first, in which case the Bayesian interval is the result.

The bounds are in **logistic Elo**, the rating a score implies, `s = 1/(1+10^(-elo/400))`;
the BayesElo rating printed beside it uses a different scale once there are draws. The draw
rate is a nuisance parameter maximised out under each hypothesis -- fishtest's generalised
SPRT -- but computed exactly, by constrained maximum likelihood, not with the usual normal
approximation. Against a random baseline the bot wins every game, the observed variance is
zero, and the approximation would accept H1 after two or three wins; exactly, N straight
wins is the binomial `N log(s1/s0)`, so the default needs 23.

Measured by simulation, on game results drawn from the model with 7% draws and a 30-Elo
first-move advantage alternating with colour:

| true logistic Elo | accepts H1 | accepts H0 | undecided at 1000 | games, mean (p90) |
|---|---|---|---|---|
| -50 | 0% | 100% | 0% | 94 (152) |
| 0 | 5.4% | 93.5% | 1.1% | 256 (491) |
| +25 | 48.7% | 45.5% | 5.8% | 389 (815) |
| +50 | 94.5% | 4.8% | 0.7% | 252 (492) |
| +100 | 100% | 0% | 0% | 98 (162) |
| +800 (a random baseline) | 100% | 0% | 0% | 23 (23) |

The error rates hold at their nominal 5% despite the colour effect, and the 1000-game limit
leaves ~1% of matches on the boundaries undecided, against ~15% at 400. A finer band costs
more: `--sprt 0,30` averages ~590 games at +0. `--sprt -30,0` asks the opposite question,
"is the new net no worse?". `--no-sprt` plays exactly `--games` games, as before.

### Keeping the positions (`--out`)

A rating match is hundreds of games of real play, so `--out FILE` keeps them: every
position, in the same `.tkn` format as self-play, written a game at a time so a killed run
keeps what it finished. It merges with self-play files (`merge_tkn.py`) and reads with the
same tools.

Every ply's `value` is the **rated** net's value of the resulting position, whoever moved,
so the file holds one network's opinions just as a self-play file does. The opponent's moves
cost one extra board through the rated net for that; the baseline's own values are not kept.
Moves made by the baseline or the random mover carry ply flag 32 (`opponent_move` in
`unpack_nn.py` and the notebook reader), and the header records the match's temperature.
With `--out` that defaults to **0.3** rather than `elo`'s usual 0: greedy games measure
strength but are the least varied training data there is. The rating then describes the
sampling policy, not greedy play; pass `--temperature 0` to keep greedy games anyway.
Against the random mover, `n_scored` is 1 on its moves: it scores nothing.

### The model

BayesElo's, not a normal approximation to a win rate: Rao-Kupper with an explicit draw
rating, plus a first-move advantage, since black moves first.

```
P(win)  = 1 / (1 + 10^((-delta + draw_elo) / 400))
P(loss) = 1 / (1 + 10^(( delta + draw_elo) / 400))
P(draw) = 1 - P(win) - P(loss)
```

`delta` is the bot's rating plus the advantage when it moves first and minus it when it
does not, so the two colours are two likelihood terms over one rating. A flat prior times
that likelihood is evaluated on a grid over (rating, draw rating, advantage) and
marginalised down to the rating; the reported interval is the 95% credible interval of that
marginal. **Alternating colours is what makes the advantage identifiable** -- a bot that
went 45W-5L as black and 5W-45L as white rates `elo +0, advantage +192`, where either
colour alone would have read as +/-192 of strength.

`nnplay elotest Wb Lb Db Ww Lw Dw` runs the estimator alone on given counts. Against the
closed-form logistic Elo, on symmetric colours with no draws:

| record | closed form | posterior |
|---|---|---|
| 50W 50L | 0.0 | +0 [-70, +68] |
| 150W 50L | +190.8 | +192 [+138, +250] |
| 90W 10L | +381.7 | +395 [+291, +530] |
| 600W 400L | +70.4 | +71 [+48, +92] |

The interval narrows as sqrt(n) -- +/-70 at 100 games, +/-22 at 1000. The few-Elo excess
over the closed form is the draw rating: it is constrained to be non-negative, so with no
draws observed the mass at `draw_elo > 0` slightly inflates the rating needed to explain
the wins. BayesElo has the same property.

A record with no losses cannot bound the rating from above, so the posterior runs into the
edge of the grid. That is flagged (`at the grid edge`) and the sensible reading is the
lower bound alone: 100 wins in 100 games says "at least +846", not "+1500".

`test_nnplay.py` checks all of this -- `elotest` opens no ONNX session, so the estimator is
tested without a GPU.

### Rating the cost of the policy's pruning (`--vs-full`)

`--vs-full` answers the question `subsample` cannot: what does scoring only the policy's
picks cost in *strength*, rather than in the net's own value. The net plays itself with one
session driving both sides -- the rated side scoring the policy's top `--policy-top` plus
`--policy-explore` at random, its opponent scoring every child -- so no baseline graph is
involved and nothing but the evaluation budget differs between the players.

```
./nnplay elo --vs-full --temperature 0.3
```

The rated side really does evaluate only those children, so the reported work ratio is a
genuine throughput figure (6.7% at the defaults, in a test). `--policy-top 1` is the
control: both sides score everything and it must rate 0. This replaces `--subset-fraction`,
which priced a *random* slice the same way and is gone with random subsetting itself.

**Budget the match before reading it.** The interval narrows as sqrt(games), and at these
branching factors 8 games gives about +/-250 Elo, which is worth nothing:

| games | 95% interval, roughly |
|---|---|
| 8 | +/-250 |
| 100 | +/-71 |
| 400 | +/-35 |
| 1600 | +/-18 |

The SPRT (on by default) stops as soon as the result is clear.

### Which baseline

The default opponent is `baseline_<precision>.onnx`, which `export_onnx.py` writes from
**`baseline.pt`**: copy the checkpoint you want to rate against there and re-export. If the
graph is missing, `elo` says so and names the step that is absent rather than falling back
to something else. `--baseline random` selects the uniform mover instead.

Random is the absolute anchor, and it stops being useful almost immediately. The current
network scores **199W-1L in 200 games against it, Elo +1039 [+765, +1449]**, in games
lasting 3 to 4 plies -- and every future checkpoint will also win ~100%, so the rating
saturates and stops distinguishing them. Rating against the previous checkpoint is what
keeps resolving.

**Two identical networks must rate 0, and that is the control worth running.** The same
graph on both sides gives:

```
  as black : 10W 0L 0D
  as white : 0W 10L 0D
  Elo +0, 95% credible interval [-172, +170]
  draw rating 24, first-move advantage +176
```

Every game is decided by who moves first, and the model puts all of it on the advantage
term and none on the rating. That is also the sharpest illustration of why colours
alternate: measured from one side only, this identical pair would have read as +176 or
-176 Elo of strength.

**Do not run net-versus-net at temperature 0.** Both players are then deterministic, so
every game with a given colour is the same game: the 20-game record above is two distinct
games copied ten times, and its interval is correspondingly far too narrow to believe.
`elo` warns when it sees this. Against the *random* baseline temperature 0 is fine, because
the opponent supplies the variation.

## What does evaluating only some children cost? (`nnplay subsample`)

A ply costs one network evaluation per legal move, so the obvious saving is to score only
a fraction of them. This measures what that fraction costs.

```
./nnplay subsample --games 20 --max-plies 200 --temperature 0.3 \
                   --proportions 0.8,0.6,0.4,0.2,0.1,0.05,0.02 --repeats 16 --csv sub.csv
```

Every proportion is measured on the **same positions**, from one self-play pass. The
subset's values are read out of the full evaluation rather than run as their own batch,
which is exact and not an approximation: a child's value does not depend on which other
children shared its batch. Only fp16 batch-shape rounding does, and `--verify N` re-runs
subsets as real GPU batches to measure it -- 1.2e-04 on the development GPU. Being exact
is what makes every extra proportion and repeat free.

Reported per proportion: top-1 agreement with the full evaluation, the mean/sd/percentiles
and max of the argmax loss, the policy loss, the rank of the chosen child, and the subset's
own spread statistics. `--csv` writes one row per position, proportion and repeat.

**Two losses, because there are two questions.** `loss` is `best_full - best_subset` in the
mover's frame -- what a greedy player gives up. `policy loss` compares the *expected* value
of sampling at the temperature in use from the subset against sampling from everything, and
is the relevant number when the engine samples rather than taking the argmax. The argmax
loss is its `T -> 0` case. Policy loss can come out slightly negative: dropping children at
random removes bad ones as often as good ones, and a softmax over the survivors can expect
marginally more than one over the full set.

**Read top-1 agreement, not mean loss.** On the current network the value landscape is
bimodal: at most positions one child sits near the tanh ceiling (~+0.96) while the whole
rest of the pack is around -0.05, a gap of ~0.68 to the 95th percentile. The argmax loss is
therefore all-or-nothing -- you either sample that one move or you lose ~0.95 -- so its mean
is just the miss rate times 0.95 and its median is 0. Measured over 24 positions:

| fraction | top-1 agreement | mean loss | median loss | policy loss at T=0.3 |
|---|---|---|---|---|
| 0.80 | 78.6% | 0.110 | 0.000 | -0.000 |
| 0.40 | 36.2% | 0.418 | 0.007 | +0.014 |
| 0.10 | 10.4% | 0.583 | 0.937 | +0.024 |
| 0.02 | 3.4% | 0.673 | 0.979 | +0.047 |

Agreement tracks the fraction, which is exactly the identity `P(best child in a uniform
k-subset) = k/n`. That is the sanity check on the sampler, and also the headline: **random
subsetting buys nothing clever for a greedy player** -- keeping the best move with
probability p costs p of the work. Departures from that line would be the interesting
result, and there are none here.

For a player that *samples*, the answer is the opposite. At T = 0.3 one child at +0.96
against 400 near -0.05 still only draws ~7% of the softmax mass, so the policy rarely takes
it and dropping it costs almost nothing: the policy loss stays under 0.05 even at 2% of the
children. Which column matters depends entirely on the temperature the engine will run at.

## Which temperature schedule gives better labels? (`nnplay variance`)

A value head is trained on `(position, eventual outcome)`, and the outcome is partly set by
the position and partly by the dice rolled after it. `variance` splits the two:

```
Var(z) = Var(E[z|s])  +  E[Var(z|s)]
         signal          noise
```

It plays games under the given schedule, snapshots positions along the way, and replays
each one `--branches K` times to the end. The spread within a snapshot's K outcomes is the
noise; the spread of their means across snapshots, less `noise/K` of sampling error, is the
signal. `signal / (signal + noise)` is the **achievable R2**: the most any value head could
explain from that schedule's labels. It is a property of the schedule, not of how well the
generating net happens to predict, which is why it can compare schedules where
`corr(value, outcome)` on an ordinary corpus cannot -- a net trained on one schedule's games
predicts that schedule best whatever its labels are worth.

Snapshots are taken every `--branch-every N` plies from a random phase per game, so every
ply is equally likely to be sampled and positions are weighted the way the corpus weights
them.

Use it through the sweep, which runs it per schedule and puts a bootstrap interval over
games on every number. `variance_sweep_cell.py` is the same sweep as a single marimo cell
to paste into a notebook where there is no shell; its settings are at the top. Locally:

```
python3 sweep_temperature.py --schedules 1.0:0:1.0,1.0:0:0.5,1.5:20:0.3 --games 400
python3 sweep_temperature.py --mode variance --schedules 1.0:0:1.0,1.0:0:0.5,1.5:20:0.3 \
                             --games 300 --dry-run
```

The play sweep comes first: variance mode reads each schedule's play shard to set the
stride (mean game length over `--points-per-game`, default 8) and to estimate cost. Cost
grows with game length squared -- each snapshot's playouts run to the end of the game --
so read `--dry-run` before committing. On the T=1.0 production corpus it comes to ~21k
plies per game at K=8, about 26 h for 300 games at 68 pos/s; halving K or the points per
game halves that. `--reuse` resumes without redoing finished schedules, and `--jobs J`
splits a schedule's games over J processes, which only pays if one leaves the GPU idle.

The estimator was checked on synthetic outcomes with a known answer: true R2 0.213,
estimated 0.2145 (sd 0.012 at 200 games), with the 95% interval covering the truth in
192 of 200 trials, and the noise-corrected net correlation matching its true value to three
places.

## Which children get scored: the policy

`play`, `elo` and `variance` no longer score a random slice of the children. Each ply the
**policy graph** ranks every legal move from one pass over the parent, and the value head
scores the policy's top `--policy-top` (default **5%**) plus `--policy-explore` (default
**2%**) more drawn at random from the rest; the move is then sampled from those as before,
with forced regicide and King safety unchanged. The random extras are the policy's
blind-spot insurance: on held-out games it missed the value head's best child ~10% of the
time even in its top half, and children it never ranks highly are only ever evaluated --
and, with `--children`, recorded for the next policy to learn from -- if something other
than the policy picks them.

What it bought, measured on the policy before this was built: over every legal move of
held-out positions, the top 5% held the value head's best child 78% of the time (random:
5%) and gave up 0.013 of value against the best child, where a random 25% gives up 0.027 --
half the loss at a fifth of the evaluations.

The policy graph sits beside the value graph -- `net_fp16_policy.onnx` beside
`net_fp16.onnx`, `--policy-model` to name another -- and `export_onnx.py` writes it for any
`model_policy.py` checkpoint. **A model without one is refused** by `play`, `elo` and
`variance`, and so is an `elo` baseline without one; `--baseline random` still works, and
`eval`, `selfcheck` and `subsample` need only the value graph. `--subset-fraction` is gone
and says so. `elo --vs-full` replaces its rating use: the net plays itself, one side pruning
with the policy and the other scoring every child, which prices the pruning directly.

`selfcheck` now verifies the policy move for move: every legal move of the start position
goes through the C move generator, the rows nnplay builds and the policy graph, against the
logits `export_onnx.py` computed in torch for the same moves (`policy_tolerance` in the
manifest, 0.05; fp16 lands at ~0.006). The `.tkn` header records the settings (version 6:
bits 8..15 and 16..31 of word 100, in thousandths), and the run summary prints the policy's
share of the time.

## Training the policy head (`--children`, `tkykids.c`, two notebook cells)

`model_policy.py` adds a policy head: one forward pass over a position gives a logit for
every legal move, and the aim is for those logits to rank children the way the value head
would, so the search can evaluate only the policy's top few instead of hundreds.

**`nnplay play --out X.tkn --children X.kids`** keeps what the policy learns from: at
every ply, every child the search scored and the value it got. Nothing extra is computed;
the values existed anyway. About 2 KB a position.

    file header, 32 bytes: "TKYKIDS\0", u32 version 1, u32 header size 32, u32 child size 6,
        u32 game header size 24, u64 games
    per game, 24 bytes: u32 n_plies, u32 reserved, u64 seed, u64 FNV-1a 64 of the ply
        records' move words -- (seed, n_plies, hash) finds the matching .tkn game
    per ply: u16 n, then n x { u32 move (ply-record packing, absolute), f16 value }

Ply k's children belong to the position before move k, i.e. record k-1's board. A value is
for the side to move after that child, as the search saw it; the mover prefers low. A
forced King capture is stored as its one child. Tar the `.kids` into the same `.tkn.tgz`.
`merge_tkn.py` does not merge `.kids` files and warns when it meets one.

**`tkykids.c`** is a small library over `tky.c` for archives with no `.kids`: legal moves
and child boards from a stored board, called from Python through ctypes. It is checked
against self-play: its move count equals the recorded `n_legal` at every position, every
recorded child is one of its moves, and applying the played move reproduces the next
record's board exactly. The training cell compiles it with gcc on first use.

**`distill_value_cell.py`** (one global, `distilled`) trains the new model's value head to
reproduce the old model's, on every position in the `*.tkn.tgz` archives in its directory.
It warm-starts from `checkpoint.pt`: every tensor whose name and shape match is copied, the
old `conv_stem` mapping onto `conv_norm`/`conv_down`; the feed-forward layers (2048 wide
before, 512 now) start fresh with their output layer zeroed, so the student begins as the
teacher minus those blocks rather than as noise -- 0.74 correlation before any training.
It writes `checkpoint_policy.pt`.

**`train_policy_cell.py`** (one global, `policy_training`) trains value and policy
together. The value target is `engine_marimo.py`'s TD(lambda), unchanged. The policy target
is the search's own distribution, `softmax(-child value / 0.3)`, against the policy's
softmax over the same children: recorded children with their recorded values where a
`.kids` exists, otherwise 16 legal moves drawn here and scored by the model's own value
head. Archives are unpacked once into `tkn_cache/` and memory-mapped, so memory does not
grow with the corpus: a 1M-position archive indexes in 2 s at ~4 GB resident.

It reads every `*.tkn.tgz` in its directory and every bare `.tkn` beside them (with a
`.kids` next to it, if any). A game present twice -- an archive beside its own unpacked
`.tkn`, a merged file beside its sources -- is counted once, matched by seed, length and
move sequence.

Its report, on held-out games, ranks each position's children by policy and asks what the
search would get evaluating only the policy's top 5/10/25/50%: whether the value head's
best child is in there, how much of the target softmax is, and how much value is given up
against the best child -- each beside what a random pick of the same size would get (the
random value loss computed exactly, not sampled), with the median child's gap to the best
for scale. A second report does the same over **every legal move** of 200 held-out
positions, scored by the model's own value head, at the start, every 2500 steps and at the
end; that is the one that says how far the search could be cut. Read the ideal-ordering row
first: if it is barely above random, the value head rates the children nearly alike at this
temperature and no ordering can concentrate much there.

## Is the GPU actually busy? (`--stats`)

`--stats` adds a second line to each progress line: where the interval's wall clock went,
and GPU utilisation sampled from NVML at 10 Hz by a background thread. NVML is opened with
`dlopen`, so there is no link-time dependency and the binary still runs where it is absent.

```
    ply    4  moves   523  value -0.2522  material  -2.1  pieces 773  234.0 boards/s
              wall   3.82s = nn 99.94% + movegen  0.01% + encode  0.04% + other  0.01%
              gpu busy  99.9%  memory bus  75.1%  vram 5.65/6.44 GB
```

That is the development GPU. Two separate readings, and they answer different questions:

* **`nn 99.94%`** is host-side: nnplay is not the bottleneck. Move generation and candidate
  encoding together cost 0.05% of a ply, which is why building candidates by copying the
  whole `Pos` has been left alone rather than made incremental.
* **`gpu busy 99.9%`** is NVML's `utilization.gpu`, and it means *the share of the sample
  period during which at least one kernel was resident* -- not how much of the card's
  arithmetic those kernels used. It cannot distinguish a saturated GPU from one running a
  small kernel continuously, so read it as "never idle", not "at peak FLOPs".

**The sub-batch curve is the real saturation test.** Throughput going flat from 16 to 512
means more work in flight buys nothing, which is what saturation looks like; if there were
headroom, the curve would still be climbing. `gpu busy` near 100% only rules out the
opposite failure -- a GPU sitting idle between calls waiting on the host.

`memory bus` is NVML's `utilization.memory`, the share of the period spent reading or
writing device memory. At 75% against a flat throughput curve, this workload is closer to
memory-bound than compute-bound, which fits: the one unfused attention layer writes and
re-reads a 1296x1296 score matrix per head per board.

## Why the export is shaped the way it is

`export_onnx.py` writes both an fp32 and an fp16 graph from the same weights. Two things
in it are not cosmetic:

* **The fast path is disabled before tracing.** `nn.TransformerEncoderLayer` in eval mode
  collapses to one fused `aten::_transformer_encoder_layer_fwd` with no ONNX symbolic, so
  the export fails outright. Unfusing it costs 1.8e-4 on the value, which is measured and
  printed.
* **Attention is re-expressed in the BERT shape** (`export_rewrite.py`) and then fused
  offline into `com.microsoft.Attention`. `nn.MultiheadAttention` exports its packed
  `in_proj` as chunk arithmetic that ONNX Runtime's fusion cannot match, and the
  unfused graph keeps a literal `8 x 1296 x 1296` score matrix per layer -- **27 MB per
  board per layer** in fp16. The rewrite is a re-expression of the same weights, checked
  at 4.8e-7, and it is worth **2.2x**: 158 -> 352 boards/s on the development GPU.

  Three of the four layers fuse. The fourth is layer 0, whose `LayerNorm` sits behind the
  positional embedding's broadcast `Add`, and ORT's `SkipLayerNormalization` fusion
  explicitly refuses a `(1, seq, hidden)` initializer skip. That one layer is what sets
  the memory ceiling, and therefore the sub-batch: **27 MB per board**, so batch 64 asks
  for 1.7 GB.

The python `onnxruntime` used for the fusion must be the same version as the C library in
`third_party/onnxruntime`, because a contrib op is only guaranteed loadable by the runtime
that defines it. `export_onnx.py` checks and warns; `--no-fuse` opts out.

## Precision

Measured against torch on real corpus positions, on the value head:

| mode | max deviation from torch | throughput |
|---|---|---|
| `--precision fp32 --no-tf32` | 4.6e-07 | slowest |
| `--precision fp32` | 7.8e-04 | TF32 matmuls |
| `--precision fp16` (default) | 3.8e-04 | 2.2x fp32 on the dev GPU |

fp16's error is ~4e-4 on a `tanh` output in [-1, 1] going into a softmax at T = 1, so it
does not move the policy. The fp32 gap is TF32 accumulation in the CUDA kernels, not the
export: `--no-tf32` closes it to 5e-7, which is what makes it the mode to reach for when
comparing against torch.

## The value head, and training it

The checkpoint this was built against has an **untrained value head**. `engine.py` used to
compute `_, result_mat = tkybot(data_in)` and take the loss against `result_mat` alone, so
no gradient ever reached `lin_value`: the first output of the tuple -- the one the
self-play policy is defined on -- sat at its initialisation and barely varied with the
position. Measured over the 262 children of the start position:

| head | range over the 262 children | sd | effective moves, T=1.0 | T=0.1 |
|---|---|---|---|---|
| `value` | 0.0105 | 0.0026 | **262.0 of 262** | 261.9 |
| `material` | 0.2776 | 0.0488 | 261.7 | 231.2 |

At T = 1.0 the most likely move is 1.00x uniform: **the policy is uniform random to within
0.4%**, and dropping to T = 0.01 only reaches 1.47x. Over a whole 200-ply game the chosen
values move from -0.26 to -0.15, which is drift with game stage rather than discrimination
between moves.

Nothing was wrong with the pipeline -- the value read out of the graph matches torch to
5e-7 -- there was simply nothing in that head yet.

### Training it

`engine.py` now takes both losses: `label_stm` for the material head as before, and
`result_stm` for the value head, weighted by `VALUE_WEIGHT`. No self-play corpus is needed
to start, because a `tky` corpus already carries a per-game result and `unpack` already
exposes it in the side-to-move frame.

Three things to expect, measured on `shard_0000.tkp` -- 9957 positions, every one ending in
a royal capture, balanced 49.9 / 50.1:

* **The signal is weak, and that is the point.** Correlation between the material label and
  the eventual result is **+0.174**, and their signs agree only **57%** of the time. Under
  uniformly random continuations a material lead mostly does not convert, so the winner is
  close to a coin flip and the value head cannot beat that ceiling by much. It is still
  learning something the material head cannot, which is the whole reason to have it.
* **The value loss starts around three times the material loss.** MSE of a near-zero `tanh`
  against a +/-1 target starts at 1.0, while the material loss starts near 0.3.
  `VALUE_WEIGHT` is there to rebalance that if the material head starts regressing.
* **The head responds immediately.** 29 optimiser steps on 469 samples already move its
  spread from 0.0105 to 0.17. Whether that spread is *informative* is a different question,
  which is what the held-out numbers answer.

**The sign is the thing to get right**, and it is checked rather than argued. `unpack`
canonicalises to the side to move, so a white-to-move position is rotated 180 degrees and
its result has to be flipped with it, or half the corpus trains against its own negation --
which would look like a network that simply refuses to learn. `run_tests.sh` verifies it
two ways over 61k positions, 30678 of them white to move:

* `result_stm` against a ground truth recomputed from the absolute frame -- the side to
  move won exactly when `result == side_to_move`, with 0 = white won and 1 = black won;
* the stored `result` itself against the **board**, which is independent of both: after a
  royal capture the loser's King is gone, so counting id 1000 by colour in a terminal
  position names the winner without consulting any stored field. The side to move in a
  terminal position is always the loser, so canonical `result_stm` there is uniformly -1.

`held-out value sign accuracy` is the metric to watch -- 0.5 is chance, and R2 against a
+/-1 target is hard to read. `value head spread` is printed beside it because that, not the
accuracy, is what the self-play softmax actually consumes: a head with a 0.0105 range gives
a uniform policy no matter how accurate it is.

9957 positions is far too few for a 6.5M-parameter network, and it is cheap to fix. `tky
gen` writes about 26 selected positions per game at the default 5%, at ~1855 games/s on 20
threads, so a million positions is well under a minute of CPU.

## Corpus format (`.tkn`)

Little-endian, fixed strides. Same 128-byte header style as `.tky`, magic `TKYNNSP`;
offset 32 holds the base seed -- drawn fresh every run unless `--seed` is given, because a
fixed default made separate sessions replay each other's games -- 56 the temperature as a
double, 80 the rules (bit 0 forced regicide; from
version 3 bit 1 King safety; from version 4 bit 2 ply cap scored on material, bits 8..23
the resignation lead in pawns, 0 off, and bits 24..31 the plies it must hold), 96 the
sub-batch used, 104..127 the model file name. Then, per game, a 16-byte header and one record per ply.

### Per game, 16 bytes

| off | type | field |
|---|---|---|
| 0 | u32 | `n_plies` |
| 4 | i8 | result: **1 black, -1 white, 0** draw. From version 4 a resignation goes to the side ahead on material, and so does a ply cap -- from version 5 only at a lead of 50 pawns or more |
| 5 | u8 | termination: 0 royal capture, 1 stalemate, 2 repetition, 3 no progress, 4 ply cap, 5 resignation |
| 6 | u16 | flags (reserved) |
| 8 | u64 | seed |

### Ply record, 2608 bytes

| off | type | field |
|---|---|---|
| 0 | u16[1296] | the board **after** the move, absolute, `x*36 + y` |
| 2592 | u32 | move: `special << 28 \| origin << 17 \| target << 6` |
| 2596 | f32 | the network's value for the leaf it chose |
| 2600 | u16 | legal moves the mover chose from |
| 2602 | i16 | material after the move, black minus white, `mat_shift` units |
| 2604 | u8 | the mover: 1 black, 0 white |
| 2605 | u8 | flags: 1 capture, 2 promotion, 4 forced King capture, 8 last ply of the game, 16 King safety changed the pick, 32 played by the `elo` opponent |
| 2606 | u16 | reserved |

Unlike `.tky` this stores boards, not just moves, because that is what the request asked
for -- 2608 bytes a ply against 8, so a 500-ply game is 1.3 MB. The move is kept anyway,
which makes the format self-checking: `test_nnplay.py` replays every game through
`unpack.apply_move` and requires the stored boards back.

**The value's sign.** It is the raw network output for the child board, and that board was
encoded for the side to move *in it* -- the mover's opponent. So a value near -1 means the
opponent is losing, i.e. the mover liked the move. `unpack_nn` canonicalises by default,
and in the canonical frame `info.value` is simply the value of the board you are holding;
`info.value_mover` is the negation.

## Reading it

```python
from unpack_nn import unpack_nn, games
from unpack import embedding_index

for board, info in unpack_nn("selfplay.tkn"):
    x = embedding_index(board)     # exactly the array nnplay fed the network
    v = info.value                 # the network's value of this board
    z = info.result_stm            # +1 if the side to move went on to win
```

Each game is flushed to disk as it finishes, so killing a long run keeps the games that
completed. `--progress-every N` prints a line every N plies within a game.

`unpack_nn` mirrors `unpack`: absolute on disk, canonicalised to the side to move by
default, `canonical=False` for the stored frame. `move`, `material` and `result` are
rewritten into the same frame; `result_black` and `side_black` keep the absolute answers.
`games()` walks game headers without decoding plies. As a script it summarises a shard.

## Validation

`test_nnplay.py` checks six things, none of them against `nnplay.c` itself:

| check | what it proves |
|---|---|
| `encode` | `encode_stm` equals `canonical_board` + `embedding_index` on real positions of both colours -- the C and python frames agree |
| `replay` | every stored board is the previous one with the stored move applied, by `unpack.apply_move`, i.e. by game.js's `Board.move` rather than nnplay's |
| `symmetry` | the canonical move and the canonical board agree about the same rotation |
| `values` | the stored boards re-evaluated reproduce the stored values -- a record's value belongs to that record's board |
| `torch` (fp16) | `nnplay eval` matches torch to 3.8e-04 |
| `torch` (fp32, no TF32) | ... and to 4.6e-07, so export + fusion + C plumbing are exact |

## Running it somewhere else

`./bundle.sh DIR` assembles a directory with one entry point. Upload it, run `setup.sh`
once, use `nnplay`. The bundle is ~13 MB and carries no runtime; `setup.sh` downloads that.

```
sh setup.sh [PYTHON]     # PYTHON only downloads things; nothing is installed into it
```

It installs `onnxruntime-gpu` with `pip --target` into the bundle's own **`ortlib/`**,
compiles, and then proves the result: the CUDA provider loads, the start position
evaluates to the value `export_onnx.py` recorded in `net_manifest.json`, and the probe
prints this GPU's throughput. It stops with the reason if any of that fails, and is safe
to re-run. Measured end to end from a clean bundle: 1m40s including the download.

**Why a private directory.** `onnxruntime` and `onnxruntime-gpu` install into the *same*
package directory, so whichever went in last wins -- and a notebook package manager
(marimo's sandbox among them) installs the CPU wheel whenever a cell does
`import onnxruntime`. Three separate sessions were lost to the GPU runtime being silently
replaced that way. Nothing in `ortlib/` is visible to a package manager, and the binary
finds it through `-Wl,-rpath,'$ORIGIN/ortlib/onnxruntime/capi'`, relative to itself.

Consequences worth knowing:

* the runtime version is whatever `ORT_SPEC` resolves to (default `onnxruntime-gpu>=1.22`);
  pin it when the driver is older than the newest wheel's CUDA, e.g.
  `ORT_SPEC='onnxruntime-gpu==1.20.2' sh setup.sh`;
* only the **header** is shipped. The C API is versioned inside it and newer runtimes still
  serve older API versions, so one header works with whatever gets downloaded;
* the SONAME aliases are made **inside** `ortlib/.../capi`, beside the provider libraries.
  ONNX Runtime loads its providers from the directory its own `libonnxruntime` came from,
  so aliasing it into the bundle directory instead produces
  `Failed to load library .../libonnxruntime_providers_shared.so`;
* CUDA libraries are found by asking several interpreters for `nvidia.__path__` and by
  globbing the usual site-packages and `/usr/local/cuda` locations, then baked in as
  `DT_RPATH` -- not `DT_RUNPATH`, because the provider is `dlopen`ed by `libonnxruntime.so`
  and only `DT_RPATH` is inherited that far. No `LD_LIBRARY_PATH` at run time.

**`nnplay` refuses to run on the CPU** unless given `--allow-cpu`. A silent fallback is
~100x slower, looks like a hang, and was twice mistaken for a working setup. The error
names the cause: a CPU-only wheel, a driver older than the runtime's CUDA, or a missing
library, which `setup.sh` then lists via `ldd`.

`nnplay selfcheck` is the same check on demand, in pure C -- no Python, numpy or
`unpack.py` needed on the far end. The bundle's own `README.md` carries the notebook cells.

### The exporter reads the architecture, it does not assume it

`export_rewrite.py` takes the head count and width off the layer
(`self_attn.num_heads`, `embed_dim`) and `export_onnx.py` passes those to the fusion.
Hardcoding them is not a style problem: splitting a 128-wide layer into 8 heads instead of
4 is still a valid tensor program, just not this network, and the only symptom was the
export check drifting to `max|dv| = 0.09` -- which looks like a tolerance that wants
widening. Reading the shape put it back to `1.9e-07`. The export prints what it found
(`encoder: 3 layers, 4 heads, d_model 128`); if that line does not match `model.py`,
nothing downstream is measuring the network you trained.

## Getting the runtime

For development in this directory, `build_nn.sh` needs `third_party/onnxruntime`, the GPU
C API release; its header says how to fetch it. (`bundle.sh` needs it only for the one
header and the version number, so a bundle built elsewhere does not.) The CUDA provider also needs cuDNN 9 and the cuBLAS/cuFFT/cuRAND runtimes.
`build_nn.sh` finds them -- including the copies torch ships in its own wheels, which is
where they live on a machine with no system CUDA -- and bakes them in as `DT_RPATH`, not
`DT_RUNPATH`, because the provider library is `dlopen`ed by `libonnxruntime.so` and only
`DT_RPATH` is inherited that far. So `./nnplay` runs with no `LD_LIBRARY_PATH`. Rebuild
after moving the tree or changing the torch install.
