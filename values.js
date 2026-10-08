// Material values for Taikyoku Shogi.
//
// The first version of this file scored pieces from their move tables with hand-picked
// constants for range, jumping, trampling and so on. Measurement killed it: on a board
// that stays 40-60% occupied, modelled range and real range are not the same quantity.
// A Free King, which slides forever in eight directions, actually reaches 35 squares; a
// Hook Mover, which turns one corner, actually reaches 265. No hand-set constant was
// going to bridge that.
//
// So a piece is scored by what it does in real positions instead. Greedy games are
// played out, boards are snapshotted along the way, and every piece type is dropped onto
// uniformly random squares of those boards to measure four things:
//
//   dest  distinct squares it can move to        - flexibility, control, escape
//   atk   enemy pieces it attacks                - pressure
//   net   most pieces one move can remove        - material per turn (trample, lion)
//   igui  enemies it can take without moving     - risk-free capture
//
// That single measurement replaces every special case: tramplers, lions, hook movers,
// jumpers and igui all show up in the numbers without being named. Three things it
// cannot see, because it only looks one move ahead, stay analytic: long-run reach
// (colourbinding, forward-only pieces that can never come back), promotion, and royalty.
//
// Values are rescaled at the end so the Pawn is exactly 1.
// Load after game.js. Run under node to regenerate the VALUES table in game.js.

const V = {
    // Squares you can reach. Heavy diminishing returns: the 200th square a Hook Mover
    // controls is mostly empty board far from any fight.
    CONTROL: 0.35,
    CONTROL_EXP: 0.55,

    // Enemy pieces you attack. Mild diminishing returns: the opponent has to answer all
    // of them, but you still only get to take one per turn.
    THREAT: 1.30,
    THREAT_EXP: 0.80,

    // Pieces removed by a single move beyond the first. This one is linear and it is the
    // reason the generals dominate: a Great General deletes ~11 pieces per move, which is
    // eleven times the material rate of an ordinary capture.
    HAUL: 2.50,

    // Capturing without leaving your square cannot be answered by recapture.
    SAFE: 1.50,

    // Simply being a body: blocks a line, costs the opponent a move to remove. Small,
    // because one body out of four hundred is not worth much.
    PRESENCE: 0.10,

    // Long-run reach, which a one-move measurement cannot see: how much of the board the
    // piece can ever get to and how many moves it takes. Colourbound and forward-only
    // pieces score low here.
    SCOPE_STEP: 0.75,
    SCOPE_FLOOR: 0.55,

    // Level 1-3 pieces cannot be trampled by their own level or below.
    IMMUNITY: 0.04,

    // Promotion is forced and automatic, but the zone is 25 ranks away for a Pawn and
    // sits inside the enemy camp. Fast forward pieces get there sooner. A promotion that
    // loses value is avoidable by staying out of the zone, so it is discounted harder,
    // and no piece is worth more than twice itself for a promotion it has not made.
    PROMO_BASE: 0.12,
    PROMO_SPEED: 0.10,
    PROMO_LOSS: 1 / 3,
    PROMO_CAP: 1.0,

    KING: 100000,

    GAMES: 14,
    PLIES: 500,
    SNAPSHOT_EVERY: 20,
    SNAPSHOT_UNTIL: 400,
    PROBES_PER_SNAPSHOT: 30,
    CANDIDATES: 24,
    NOISE: 0.5,
}

const V_DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

// Seeded so the table in game.js can be regenerated exactly.
let v_seed = 0x9e3779b9
function v_rand() {
    v_seed |= 0
    v_seed = (v_seed + 0x6d2b79f5) | 0
    let t = Math.imul(v_seed ^ (v_seed >>> 15), 1 | v_seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

function v_emptied(x, y, move) {
    const special = Math.floor(move / 1296)
    const tx = Math.floor(move / 36) % 36
    const ty = move % 36
    const out = [[tx, ty]]
    if (special >= 1 && special <= 8) {
        const o = V_DIRS[special - 1]
        if (!out_of_bounds(x + o[0]) && !out_of_bounds(y + o[1])) out.push([x + o[0], y + o[1]])
    } else if (special === 9) {
        const ux = Math.sign(tx - x)
        const uy = Math.sign(ty - y)
        const n = Math.max(Math.abs(tx - x), Math.abs(ty - y))
        for (let i = 1; i < n; i++) out.push([x + i * ux, y + i * uy])
    }
    return out
}

// ---- playouts ------------------------------------------------------------------
// The policy counts pieces rather than weighing them, so that generating the sample
// does not depend on the values being generated. It does have to keep its king alive:
// a player that does not will hang it inside three plies, because a trample opens a
// lane through its own army and there is no check rule to stop the loss.

function v_pick(game) {
    const all = game.all_legal_moves()
    if (all.length === 0) return null
    const us = game.turn
    const scored = all.map((mv) => {
        const origin = game.board.getPiece(mv[0], mv[1])
        let gain = 0
        for (const [sx, sy] of v_emptied(mv[0], mv[1], mv[2])) {
            if (sx === mv[0] && sy === mv[1]) continue
            const t = game.board.getPiece(sx, sy)
            if (t === null || t.is_empty()) continue
            if (t.id === 1000 && t.color !== origin.color) gain += 1000
            else gain += t.color === origin.color ? -1 : 1
        }
        return [gain + v_rand() * V.NOISE, mv]
    })
    scored.sort((a, b) => b[0] - a[0])
    for (const [, mv] of scored.slice(0, V.CANDIDATES)) {
        const probe = game.clone()
        probe.move(mv[0], mv[1], mv[2])
        if (probe.is_over && probe.winner === us) return mv
        if (!probe.is_royal_capturable(us)) return mv
    }
    return scored[0][1]
}

function v_snapshot(game) {
    const rows = []
    for (let i = 0; i < 36; i++) {
        const row = []
        for (let j = 0; j < 36; j++) {
            const p = game.board.board[i][j]
            row.push(p.is_empty() ? 0 : [p.id, p.color, p.promoted ? 1 : 0])
        }
        rows.push(row)
    }
    return rows
}

function v_playouts() {
    const snaps = []
    for (let g = 0; g < V.GAMES; g++) {
        const game = new Game(null, { repetition_limit: 0, no_progress_limit: 0 })
        for (let ply = 0; ply <= V.PLIES; ply++) {
            if (ply % V.SNAPSHOT_EVERY === 0 && ply <= V.SNAPSHOT_UNTIL) snaps.push(v_snapshot(game))
            if (game.is_game_over()) break
            const mv = v_pick(game)
            if (mv === null) break
            game.move(mv[0], mv[1], mv[2])
            game.history = game.history.slice(-2)
        }
    }
    return snaps
}

// ---- measurement ---------------------------------------------------------------

function v_measure(snaps) {
    const boards = snaps.map((rows) => new Board(rows.map((row, i) => row.map((cell, j) => {
        if (cell === 0) return PIECES[0].copy(i, j)
        const p = PIECES[cell[0]].copy(i, j)
        if (cell[1] === 0) p.invert_color()
        p.promoted = !!cell[2]
        return p
    }))))

    const stats = {}
    for (const key of Object.keys(PIECES)) {
        const id = Number(key)
        if (id === 0) continue
        let n = 0, dest = 0, atk = 0, net = 0, igui = 0

        for (const board of boards) {
            for (let k = 0; k < V.PROBES_PER_SNAPSHOT; k++) {
                const x = (v_rand() * 36) | 0
                const y = (v_rand() * 36) | 0
                const saved = board.board[x][y]
                const piece = PIECES[id].copy(x, y)
                board.board[x][y] = piece

                const moves = piece.get_legal_moves(board)
                const squares = new Set(moves.map((v) => v % 1296))
                let attacked = 0
                for (const d of squares) {
                    const t = board.board[(d / 36) | 0][d % 36]
                    if (!t.is_empty() && t.color === 0) attacked++
                }
                let best = 0
                const safe = new Set()
                for (const mv of moves) {
                    let enemy = 0
                    let friend = 0
                    for (const [sx, sy] of v_emptied(x, y, mv)) {
                        if (sx === x && sy === y) continue
                        const t = board.board[sx][sy]
                        if (t.is_empty()) continue
                        if (t.color === 0) { enemy++; if (mv % 1296 === x * 36 + y) safe.add(sx * 36 + sy) }
                        else friend++
                    }
                    if (enemy - friend > best) best = enemy - friend
                }

                board.board[x][y] = saved
                n++
                dest += squares.size
                atk += attacked
                net += best
                igui += safe.size
            }
        }
        stats[id] = { dest: dest / n, atk: atk / n, net: net / n, igui: igui / n }
    }
    return stats
}

// ---- long-run reach ------------------------------------------------------------
// Breadth-first over an empty board: how much of it the piece can ever get to, with
// squares discounted by how many moves they take. Unreachable squares count zero, so a
// colourbound piece scores half and a Pawn scores almost nothing.

let v_scope_board = null
const v_scope_cache = {}
function v_scope(id) {
    if (v_scope_cache[id] !== undefined) return v_scope_cache[id]
    if (v_scope_board === null) {
        const empty = []
        for (let i = 0; i < 36; i++) empty.push(new Array(36).fill(0))
        v_scope_board = new Board(empty)
    }
    const board = v_scope_board
    const dist = new Int16Array(1296).fill(-1)
    const start = 17 * 36 + 17
    dist[start] = 0
    const queue = [start]

    for (let head = 0; head < queue.length; head++) {
        const square = queue[head]
        const d = dist[square]
        if (d >= 20) continue
        const x = Math.floor(square / 36)
        const y = square % 36
        board.board[x][y] = PIECES[id].copy(x, y)
        const reachable = board.board[x][y].get_legal_moves(board)
        board.board[x][y] = PIECES[0].copy(x, y)
        for (const mv of reachable) {
            const t = mv % 1296
            if (dist[t] < 0) { dist[t] = d + 1; queue.push(t) }
        }
    }

    let total = 0
    for (let i = 0; i < 1296; i++) if (dist[i] >= 0) total += Math.pow(V.SCOPE_STEP, dist[i])
    v_scope_cache[id] = total / 1296
    return v_scope_cache[id]
}

let v_scope_max = null
function v_scope_factor(id) {
    if (v_scope_max === null) {
        v_scope_max = 0
        for (const key of Object.keys(PIECES)) {
            const n = Number(key)
            if (n !== 0 && n !== 1000) v_scope_max = Math.max(v_scope_max, v_scope(n))
        }
    }
    return V.SCOPE_FLOOR + (1 - V.SCOPE_FLOOR) * (v_scope(id) / v_scope_max)
}

// ---- the value function --------------------------------------------------------

let v_stats = null
function v_measurements() {
    if (v_stats === null) v_stats = v_measure(v_playouts())
    return v_stats
}

// How quickly the piece can drive into the promotion zone.
function v_promo_weight(id) {
    const p = PIECES[id]
    let speed = Math.max(p.dydx[0], p.dydx[1], p.dydx[7])
    for (const entry of p.tp) speed = Math.max(speed, entry[0])
    return V.PROMO_BASE + V.PROMO_SPEED * Math.min(1, Math.max(0, speed) / 6)
}

// Value with no promotion bonus: what the piece is worth once it has already promoted.
const v_base_cache = {}
function v_base(id) {
    if (v_base_cache[id] !== undefined) return v_base_cache[id]
    if (id === 0) return 0
    if (id === 1000) return V.KING

    const s = v_measurements()[id]
    const raw = V.PRESENCE
        + V.CONTROL * Math.pow(s.dest, V.CONTROL_EXP)
        + V.THREAT * Math.pow(s.atk, V.THREAT_EXP)
        + V.HAUL * Math.max(0, s.net - 1)
        + V.SAFE * s.igui

    v_base_cache[id] = raw * v_scope_factor(id) * (1 + V.IMMUNITY * Math.min(PIECES[id].level, 3))
    return v_base_cache[id]
}

// Value of a piece that has not promoted yet, which carries part of what it will become.
function v_natural(id) {
    const value = v_base(id)
    if (id === 1000 || PROMOTE[id] === undefined) return value
    const delta = v_base(PROMOTE[id]) - value
    const weight = v_promo_weight(id)
    if (delta < 0) return value + delta * weight * V.PROMO_LOSS
    return value + Math.min(delta * weight, value * V.PROMO_CAP)
}

let v_pawn = null
function piece_value(piece) {
    if (piece.id === 0) return 0
    if (piece.id === 1000) return V.KING
    if (v_pawn === null) v_pawn = v_natural(1)
    return (piece.promoted ? v_base(piece.id) : v_natural(piece.id)) / v_pawn
}

if (typeof module !== "undefined") {
    module.exports = { piece_value, v_base, v_natural, v_scope, v_measurements, V }
}
