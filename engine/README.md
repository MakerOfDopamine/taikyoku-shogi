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
offset 56 holds the temperature as a double, 96 the sub-batch used, 104..127 the model
file name. Then, per game, a 16-byte header and one record per ply.

### Per game, 16 bytes

| off | type | field |
|---|---|---|
| 0 | u32 | `n_plies` |
| 4 | i8 | result: **1 black, -1 white, 0** any draw or ply cap |
| 5 | u8 | termination: 0 royal capture, 1 stalemate, 2 repetition, 3 no progress, 4 ply cap |
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
| 2605 | u8 | flags: 1 capture, 2 promotion, 4 forced King capture, 8 last ply of the game |
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

`./bundle.sh DIR` assembles the smallest tree that plays self-play on another machine --
a notebook, a rented GPU box. **13 MB, ten files, no node, no torch and no
`checkpoint.pt` on the far end**, because the rules are already baked into `tables.h` and
the weights into `net_fp16.onnx`:

| file | why |
|---|---|
| `nnplay.c`, `tky.c`, `tables.h` | the whole rules and self-play stack; `nnplay.c` includes the other two |
| `onnxruntime_c_api.h` | the only header needed, and it pulls in nothing but libc |
| `net_fp16.onnx` | the weights, already fused |
| `unpack.py`, `unpack_nn.py`, `tables.json` | reading the `.tkn` back |
| `setup.sh`, `README.md` | generated: the pip line and the exact gcc line |

The 258 MB C API tarball is **not** needed there. `pip install onnxruntime-gpu` ships
`libonnxruntime.so` and the CUDA provider in the wheel, so the only thing missing from pip
is the header, which the bundle carries. `setup.sh` does the pip install, links the two
SONAME aliases the wheel does not provide, and compiles.

The pip version must match the one the graphs were fused with -- `bundle.sh` reads it from
`third_party/onnxruntime/VERSION_NUMBER` and writes it into the generated `setup.sh` --
because `com.microsoft.Attention` is a contrib op and only the runtime that defines it is
guaranteed to load it. Add `--with-export` (+75 MB, adds `checkpoint.pt` and the export
scripts) to re-export on the far end instead, and `--with-tests` for `test_nnplay.py` and
the corpus shard its encode and torch checks read. `--with-fp32` adds the 25 MB fp32 graph,
which only earns its place for the `--no-tf32` parity check.

Without a working CUDA provider `nnplay` warns and runs on the CPU rather than dying --
verified by building the bundle against the CPU-only wheel -- but it is ~100x slower there
and not worth starting.

## Getting the runtime

For development in this directory, `build_nn.sh` needs `third_party/onnxruntime`, the GPU
C API release; its header says how to fetch it. (`bundle.sh` needs it only for the one
header and the version number, so a bundle built elsewhere does not.) The CUDA provider also needs cuDNN 9 and the cuBLAS/cuFFT/cuRAND runtimes.
`build_nn.sh` finds them -- including the copies torch ships in its own wheels, which is
where they live on a machine with no system CUDA -- and bakes them in as `DT_RPATH`, not
`DT_RUNPATH`, because the provider library is `dlopen`ed by `libonnxruntime.so` and only
`DT_RPATH` is inherited that far. So `./nnplay` runs with no `LD_LIBRARY_PATH`. Rebuild
after moving the tree or changing the torch install.
