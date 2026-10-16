async function load_bot() {
    const session = await ort.InferenceSession.create('./net_web.onnx', {
        executionProviders: ['webgpu', 'wasm']
    });
    return session
}
async function load_bot_policy() {
    const session = await ort.InferenceSession.create('./net_web_policy.onnx', {
        executionProviders: ['webgpu', 'wasm']
    });
    return session
}

let session;
load_bot().then(s => { session = s; });
let policySession;
load_bot_policy().then(s => { policySession = s; });

function board_to_tensor(boards, turn) {
    if (!Array.isArray(boards)) {
        boards = [boards]
    }
    turn = Array.isArray(turn) ? turn : boards.map(() => turn)
    let data = new Int32Array(boards.length * 1296)
    for (let b = 0; b < boards.length; b++) {
        const white = turn[b] === 0
        for (let i = 0; i < 1296; i++) {
            const s = white ? 1295 - i : i              // white to move: rotate 180 degrees
            const piece = boards[b].board[Math.floor(s / 36)][s % 36]
            if (piece.id === 0) { data[b * 1296 + i] = 0; continue }   // empties are colour 1 in game.js
            const index = piece.id === 1000 ? 301 : piece.id
            const mine = white ? 1 - piece.color : piece.color         // 1 = the side to move
            data[b * 1296 + i] = index * 2 + mine - 1
        }
    }
    return data
}

async function get_values(tensor, size) {
    const board = new ort.Tensor('int32', tensor, [size, 1296]);
    const out = await session.run({ board });
    return out.value.data
}

async function get_policy(boards, moves) {   // moveList: [[from, to, type], ...]
    const M = moves.length;
    const mv = new Int32Array(M * 4);
    for (let i = 0; i < M; i++) {
        mv[i * 4 + 0] = 0;                       // batch index
        mv[i * 4 + 1] = moves[i][0];      // from square
        mv[i * 4 + 2] = moves[i][1];      // to square
        mv[i * 4 + 3] = moves[i][2];      // move type
    }
    const board = new ort.Tensor('int32', boards, [1, 1296]);
    const moves_new = new ort.Tensor('int32', mv, [M, 4]);
    const out = await policySession.run({ board, moves: moves_new });
    return out.logits.data;
}

// temp: softmax temperature over the negated child values, as nnplay samples; 0 plays the best
async function get_engine_move(temp = 0.1) {
    const me = board.turn
    const white = (me === 0)
    const legal = []                                    
    for (let x = 0; x < 36; x++) {
        for (let y = 0; y < 36; y++) {
            const p = board.getPiece(x, y)
            if (p.id === 0 || p.color !== me) continue
            for (const m of p.get_legal_moves(board)) legal.push([x, y, m])
        }
    }

    const canon = (s) => white ? 1295 - s : s           // the policy sees the mover as black
    const mvs = legal.map(([x, y, m]) => {
        let sp = Math.floor(m / 1296)
        if (white && sp >= 1 && sp <= 8) sp = (sp - 1 + 4) % 8 + 1
        return [canon(x * 36 + y), canon(m % 1296), sp]
    })

    const logits = await get_policy(board_to_tensor(board, me), mvs)
    const children_checked = Math.ceil(legal.length / 20)
    const top = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, children_checked)
    const kids = top.map(i => { const b = new Board(board); b.move(...legal[i]); return b })

    const values = await get_values(board_to_tensor(kids, 1 - me), children_checked)   // for the opponent
    let pick = values.indexOf(Math.min(...values))
    if (temp > 0) {
        const vmin = values[pick]
        const w = Array.from(values, v => Math.exp((vmin - v) / temp))
        let r = Math.random() * w.reduce((a, b) => a + b, 0)
        for (pick = 0; pick < w.length - 1 && (r -= w[pick]) > 0; pick++);
    }
    board.move(...legal[top[pick]])
    refresh()
}

class Node {
    constructor(move, prior, parent = null) {
        this.parent = parent     // parent node
        this.move = move         // move that led here
        this.prior = prior       // P(policy) for this move from parent

        this.N = 0               // # of visits
        this.W = 0               // values from this node
        this.vLoss = 0           // leaf batching

        this.children = null     // children nodes
        this.terminal = null     // 1/0/-1 if this node is terminal (draw, royal capture)
    }

    get Q() { // so today I learnt about this thing called get
        return (this.N == 0 ? 0 : (this.W / this.N))
    }
}

/*
class GameTree {
    constructor(board) {
        this.
    }
}*/