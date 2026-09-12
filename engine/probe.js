// Exhaustive per-piece equivalence check: builds positions, asks game.js and the C engine
// for the full legal-move list of the side to move, and diffs them.
const fs = require('fs'), vm = require('vm'), path = require('path'), cp = require('child_process')
const ROOT = path.join(__dirname, '..')
const sb = { console, alert: (m) => { throw new Error('alert: ' + m) } }
vm.createContext(sb)
const { Board, Game, PIECES } = vm.runInContext(
    fs.readFileSync(path.join(ROOT, 'game.js'), 'utf8') + '\n;({ Board, Game, PIECES })', sb)
const T = JSON.parse(fs.readFileSync(path.join(__dirname, 'tables.json'), 'utf8'))
const IDS = T.ids

function run(cells, turn) {
    // C side
    const stdin = `turn ${turn}\n` + cells.map(([sq, idx, c, pr]) => `${sq} ${idx} ${c} ${pr}`).join('\n') + '\n'
    const out = cp.execFileSync(path.join(__dirname, 'tky'), ['probe'], { input: stdin, maxBuffer: 1 << 28 })
        .toString().trim().split('\n')
    const cmoves = out.slice(1).filter(Boolean).map((l) => { const [o, m] = l.split(':'); return (+o << 14) | +m })
    // JS side
    const grid = Array.from({ length: 36 }, () => Array(36).fill(0))
    for (const [sq, idx, c, pr] of cells) {
        const p = PIECES[IDS[idx]].copy(Math.floor(sq / 36), sq % 36)
        if (c === 0) p.invert_color()
        p.promoted = !!pr
        grid[Math.floor(sq / 36)][sq % 36] = p
    }
    const g = new Game(new Board(grid))
    g.turn = turn
    const jmoves = g.all_legal_moves().map(([x, y, m]) => ((x * 36 + y) << 14) | m)
    const a = new Set(cmoves), b = new Set(jmoves)
    const onlyC = [...a].filter((v) => !b.has(v)), onlyJs = [...b].filter((v) => !a.has(v))
    return { cn: cmoves.length, jn: new Set(jmoves).size, onlyC, onlyJs }
}

let fails = 0, cases = 0
function check(label, cells, turn) {
    cases++
    const r = run(cells, turn)
    if (r.onlyC.length || r.onlyJs.length || r.cn !== r.jn) {
        fails++
        console.log(`FAIL ${label}: c=${r.cn} js=${r.jn} c-only=${r.onlyC.slice(0, 6).map(v => (v >> 14) + ':' + (v & 16383))} js-only=${r.onlyJs.slice(0, 6).map(v => (v >> 14) + ':' + (v & 16383))}`)
    }
    return r
}

const mode = process.argv[2] || 'lone'

if (mode === 'lone') {
    // every piece type, both colours, alone in the middle and in a corner
    for (let idx = 0; idx < IDS.length; idx++) {
        const id = IDS[idx]
        if (id === 0) continue
        for (const color of [0, 1]) for (const sq of [18 * 36 + 18, 0, 35 * 36 + 35, 2 * 36 + 33]) {
            const r = check(`id=${id} ${PIECES[id].name} c=${color} sq=${sq}`, [[sq, idx, color, 0]], color)
            if (id === 297 && sq === 18 * 36 + 18 && color === 1) {
                const dests = new Set(r.onlyC.length ? [] : [])
                console.log(`  lone Lion mid-board: ${r.cn} moves`)
            }
        }
    }
} else {
    // random dense boards: a target piece surrounded by random junk
    const seed = +(process.argv[3] || 1)
    let s = BigInt(seed) * 6364136223846793005n + 1442695040888963407n
    const rnd = (n) => { s = (s * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n); return Number((s >> 33n) % BigInt(n)) }
    const N = +(process.argv[4] || 60)
    for (let t = 0; t < N; t++) {
        const cells = [], used = new Set()
        const density = 20 + rnd(300)
        for (let k = 0; k < density; k++) {
            const sq = rnd(1296)
            if (used.has(sq)) continue
            used.add(sq)
            let idx = 1 + rnd(IDS.length - 1)
            cells.push([sq, idx, rnd(2), rnd(2)])
        }
        // guarantee some exotic pieces are present and on move
        const turn = rnd(2)
        for (const id of [276, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 297, 298, 299, 300, 1000]) {
            let sq; do { sq = rnd(1296) } while (used.has(sq))
            used.add(sq)
            cells.push([sq, IDS.indexOf(id), turn, 0])
        }
        check(`random board #${t} (density ${cells.length}, turn ${turn})`, cells, turn)
    }
}
console.log(`${cases} positions compared, ${fails} mismatched`)
process.exit(fails ? 1 : 0)
