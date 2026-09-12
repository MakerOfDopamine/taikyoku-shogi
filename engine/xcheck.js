// Replays a C `trace` transcript through game.js and compares, at every ply, the full
// sorted legal-move list (as origin:move pairs) and the material balance.
const fs = require('fs'), vm = require('vm'), path = require('path')
const ROOT = path.join(__dirname, '..')
const sandbox = { console, alert: (m) => { throw new Error('alert: ' + m) } }
vm.createContext(sandbox)
const src = fs.readFileSync(path.join(ROOT, 'game.js'), 'utf8')
const { Game, VALUES, VALUES_PROMOTED } = vm.runInContext(
    src + '\n;({ Game, VALUES, VALUES_PROMOTED })', sandbox, { filename: 'game.js' })

function mat(g) {
    let t = 0
    for (let i = 0; i < 36; i++) for (let j = 0; j < 36; j++) {
        const p = g.board.board[i][j]
        if (p.id === 0 || p.id === 1000) continue
        const v = (p.promoted && VALUES_PROMOTED[p.id] !== undefined) ? VALUES_PROMOTED[p.id] : VALUES[p.id]
        t += (p.color === 1 ? 1 : -1) * Math.round(v * 1000)
    }
    return t
}
function digest(list) {          // FNV-1a over sorted uint32 (origin<<14 | move)
    let h = 0xcbf29ce484222325n
    for (const v of list) { h = (h ^ BigInt(v)) * 0x100000001b3n & 0xffffffffffffffffn }
    return h.toString()
}

const lines = fs.readFileSync(process.argv[2], 'utf8').split('\n')
const game = new Game()
let ply = 0, checked = 0, moveset = null, expect = null, bad = 0
for (const line of lines) {
    const t = line.split(' ')
    if (t[0] === 'PLY') {
        expect = { ply: +t[1] }
        for (const kv of t.slice(2)) { const i = kv.indexOf("="); const k = kv.slice(0, i), v = kv.slice(i + 1); expect[k] = (k === "digest") ? v : +v }
        moveset = []
    } else if (t[0] === 'M') {
        moveset.push((+t[1] << 14) | +t[2])
    } else if (t[0] === 'PICK') {
        const all = game.all_legal_moves().map(([x, y, m]) => ((x * 36 + y) << 14) | m)
        all.sort((a, b) => a - b)
        const problems = []
        if (all.length !== expect.n) problems.push(`nmoves js=${all.length} c=${expect.n}`)
        if (digest(all) !== expect.digest) problems.push(`digest mismatch`)
        if (mat(game) !== expect.mat) problems.push(`material js=${mat(game)} c=${expect.mat}`)
        if (game.turn !== expect.turn) problems.push(`turn js=${game.turn} c=${expect.turn}`)
        if (moveset.length) {
            const a = new Set(all), b = new Set(moveset)
            const onlyJs = all.filter((v) => !b.has(v)), onlyC = moveset.filter((v) => !a.has(v))
            if (onlyJs.length || onlyC.length)
                problems.push(`set diff: js-only ${onlyJs.length} [${onlyJs.slice(0, 5).map(v => (v >> 14) + ':' + (v & 16383))}] c-only ${onlyC.length} [${onlyC.slice(0, 5).map(v => (v >> 14) + ':' + (v & 16383))}]`)
        }
        if (problems.length) { console.log(`ply ${expect.ply}: ` + problems.join('; ')); if (++bad > 8) process.exit(1) }
        checked++
        const o = +t[1], m = +t[2]
        if (!game.move(Math.floor(o / 36), o % 36, m)) { console.log(`ply ${expect.ply}: game.js REJECTED move ${o}:${m}`); process.exit(1) }
        ply++
    } else if (t[0] === 'END') {
        console.log(`transcript end: ${t[1]} at ply ${t[2]}; js says over=${game.is_over} result=${game.result_string()}`)
    }
}
console.log(`checked ${checked} plies, ${bad} mismatched`)
process.exit(bad ? 1 : 0)
