const grid = document.getElementById('grid');
const display_grid = document.getElementById('piece-move-grid')
for (let i = 0; i < 1296; i++) {
    const cell = document.createElement('div');
    cell.className = 'cell';
    row = 35 - Math.floor(i / 36);
    col = i % 36;
    cell.dataset.row = row
    cell.dataset.col = col
    cell.id = `cell-${row}-${col}`
    cell.title = `cell-${row}-${col}`
    grid.appendChild(cell);
}

let global_game = new Game()
let board = global_game.board // for ONCE the js laziness of only assigning references works in my favor

const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

const COLOR_EMPTY = "#d4b855"    // board background
const COLOR_SELECTED = "#0eac62" // selected square
const COLOR_MOVE = "#888888"     // ordinary move (special 0)
const COLOR_INF_MOVE = "#ff3300" // display board only
const COLOR_TRAMPLE = "#905c4c"  // trample move (special 9, SHIFT)
const COLOR_FIRST = "#6b3fa0"    // medium move, first leg (special 1-8, X)
const COLOR_STANDING = "#3f7fd0" // square the piece is provisionally standing on
const COLOR_SECOND = "#16cc5c"   // medium move, second leg
const COLOR_SEARCH = "#26cad6"   // highlighted by search

// selected: origin square of the piece being moved, or [null, null].
// intermediate: chosen first-leg direction as a special index 1-8, or null.
let selected = [null, null]
let intermediate = null
let SHIFT = false
let X_HELD = false

function cell(x, y) {
    return document.getElementById(`cell-${x}-${y}`)
}

function piece_color(piece) {
    return piece.promoted
        ? (piece.color == 0 ? "#00aaff" : "#ff0000")
        : (piece.color == 0 ? "#ffffff" : "#000000")
}

function clear() {
    for (let i = 0; i < 36; i++) {
        for (let j = 0; j < 36; j++) {
            cell(i, j).style.background = COLOR_EMPTY
        }
    }
}

function draw_pieces() {
    for (let i = 0; i < 36; i++) {
        for (let j = 0; j < 36; j++) {
            cell(i, j).innerHTML = board.board[i][j].short_name
            cell(i, j).style.color = piece_color(board.board[i][j])
        }
    }
}

function selected_moves() {
    if (selected[0] == null) {
        return []
    }
    return board.board[selected[0]][selected[1]].get_legal_moves(board)
}

// The set of first-leg directions (special 1-8) the selected piece can currently use.
function first_leg_directions(moves) {
    let dirs = new Set()
    for (let move of moves) {
        let special = Math.floor(move / 1296)
        if (special >= 1 && special <= 8) {
            dirs.add(special)
        }
    }
    return dirs
}

// Ordinary moves.
function render() {
    clear()
    draw_pieces()
    for (let square of selected_moves()) {
        if (square >= 1296) continue
        cell(Math.floor(square / 36), square % 36).style.background = COLOR_MOVE
    }
}

// SHIFT: trample moves.
function render_special() {
    clear()
    draw_pieces()
    for (let square of selected_moves()) {
        if (square < 9 * 1296) continue
        cell(Math.floor(square / 36) % 36, square % 36).style.background = COLOR_TRAMPLE
    }
}

// X, stage one: the intermediate squares a medium move may step through.
function render_medium() {
    clear()
    draw_pieces()
    if (selected[0] == null) return
    for (let special of first_leg_directions(selected_moves())) {
        let dir = DIRS[special - 1]
        let mid_x = selected[0] + dir[0]
        let mid_y = selected[1] + dir[1]
        if (out_of_bounds(mid_x) || out_of_bounds(mid_y)) continue
        cell(mid_x, mid_y).style.background = COLOR_FIRST
    }
}

// X, stage two: provisional board state with the piece standing on the chosen
// intermediate square, showing the second-leg destinations from there.
// This only draws, the real board is untouched until the move is committed.
function render_intermediate() {
    clear()
    draw_pieces()
    if (selected[0] == null || intermediate == null) return

    let piece = board.board[selected[0]][selected[1]]
    let dir = DIRS[intermediate - 1]
    let mid_x = selected[0] + dir[0]
    let mid_y = selected[1] + dir[1]

    // Provisional state: origin vacated, piece drawn on the intermediate square
    // (covering whatever it would capture there).
    cell(selected[0], selected[1]).innerHTML = ""
    cell(mid_x, mid_y).innerHTML = piece.short_name
    cell(mid_x, mid_y).style.color = piece_color(piece)
    cell(mid_x, mid_y).style.background = COLOR_STANDING

    for (let move of selected_moves()) {
        if (Math.floor(move / 1296) !== intermediate) continue
        // Returning to the origin square is a legal jitto, so it is highlighted too.
        cell(Math.floor(move / 36) % 36, move % 36).style.background = COLOR_SECOND
    }
}

function _get_grid_sizing_max(piece) {
    // Helper function for side display.
    // Returns max squares, including jumps, apart from "infinite" directions.
    let max_squares = 2
    for (let i in piece.dydx) {
        if (piece.dydx[i] != 99 && piece.dydx[i] > max_squares) {
            max_squares = piece.dydx[i] + 1
        }
    }

    for (let i of piece.tp) {
        if (Math.max(Math.abs(i[0]), Math.abs(i[1])) >= max_squares) {
            max_squares = Math.max(Math.abs(i[0]), Math.abs(i[1])) + 1
        } 
    }
    return max_squares
}

function update_ui() {
    display_grid.innerHTML = ""
    let piece = board.getPiece(selected[0], selected[1])
    let display = 'inline-grid'
    if (piece === null || piece.is_empty()) {
        display_grid.style.display = 'none'
        return
    }
    if (selected[0] != null) {
        let max_squares = _get_grid_sizing_max(piece)
        let size = (max_squares * 2 + 1)
        display_grid.style = `display: ${display}; grid-template-columns: repeat(${size}, 26px); gap: 1px; background: #333; padding: 1px;`
        for (let i = 0; i < (max_squares * 2 + 1) ** 2; i++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            row = (size - 1 - Math.floor((size - 1) / 2)) - Math.floor(i / size);
            col = i % size - Math.floor((size - 1) / 2);
            cell.dataset.drow = row
            cell.dataset.dcol = col
            cell.id = `display-cell-[${row},${col}]`
            cell.title = `display-cell-[${row},${col}]`
            cell.style.alignContent = "center"
            display_grid.appendChild(cell);
        }

        document.getElementById("display-cell-[0,0]").innerHTML = piece.short_name
        document.getElementById("display-cell-[0,0]").style.color = piece_color(piece)
        
        let basex = 0
        let basey = 0
        let dirs = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
        let chars = ["|", "/", "—", "\\", "|", "/", "—", "\\"]
        let move = "○"
        let star = "☆"
        for (let i in piece.dydx) {
            for (let step = 1; step < piece.dydx[i] + 1; step++) {
                let mcell = document.getElementById(`display-cell-[${step * dirs[i][0]},${step * dirs[i][1]}]`)
                try {
                    mcell.style.background = piece.dydx[i] < 99 ? COLOR_MOVE : COLOR_INF_MOVE
                    mcell.style.fontSize = "18px"
                    if (piece.dydx[i] == 99) {
                        mcell.innerHTML = chars[i]
                    } else {
                        mcell.innerHTML = move
                    }

                    if (["0", "2", "4", "6"].includes(i) && [286, 289, 291].includes(piece.id)) {
                        mcell.style.background = COLOR_TRAMPLE
                    }

                    if (["1", "3", "5", "7"].includes(i) && [286, 289, 291].includes(piece.id)) {
                        mcell.style.background = COLOR_TRAMPLE
                    }
                }
                catch {break}
            }
        }

        for (let dest of piece.tp) {
            console.log(dest[2])
            document.getElementById(`display-cell-[${dest[0]},${dest[1]}]`).style.background = "#e9d946"
            document.getElementById(`display-cell-[${dest[0]},${dest[1]}]`).innerHTML = star
            for (let j in dest[2]) {
                for (let step = 1; step < dest[2][j] + 1; step++) {
                    let mcell = document.getElementById(`display-cell-[${dest[0] + step * dirs[j][0]},${dest[1] + step * dirs[j][1]}]`)
                    try {
                        mcell.style.background = "#e9d946"
                        mcell.style.fontSize = "18px"
                        if (dest[2][j] == 99) {
                            mcell.innerHTML = chars[j]
                        } else {
                            mcell.innerHTML = move
                        }
                    }
                    catch {break}
                }
            }
        }

        for (let d in dirs) {
            if (piece.djump[d] != 0) {
                let mcell = document.getElementById(`display-cell-[${dirs[d][0]},${dirs[d][1]}]`)
                mcell.style.background = "#e9d946"
                mcell.style.fontSize = "14px"
                mcell.innerHTML = piece.djump[d]
            }
        }

        let info = document.getElementById("piece-info")
        info.innerHTML = 
        `
        Piece ID: ${piece.id}<br>
        Piece name: ${piece.name}<br>
        Piece short name: ${piece.short_name}
        `
    }
}

// Draw whatever the current modifier state is.
function refresh() {
    if (X_HELD && intermediate != null) {
        render_intermediate()
    } else if (X_HELD) {
        render_medium()
    } else if (SHIFT) {
        render_special()
    } else {
        render()
    }

    update_ui()
}

function click(x, y) {
    if (selected[0] == null) {
        selected = [x, y]
        intermediate = null
        refresh()
        document.getElementById(`cell-${x}-${y}`).style.background = COLOR_SELECTED
        return
    }

    let moves_able = selected_moves()

    if (X_HELD) {
        if (intermediate == null) {
            // Did the player pick a usable first leg?
            for (let special of first_leg_directions(moves_able)) {
                let dir = DIRS[special - 1]
                if (selected[0] + dir[0] === x && selected[1] + dir[1] === y) {
                    intermediate = special
                    render_intermediate()
                    return
                }
            }
        } else {
            // Make the move if this is a real destination.
            let move = intermediate * 1296 + 36 * x + y
            if (moves_able.includes(move)) {
                board.move(selected[0], selected[1], move)
                selected = [null, null]
                intermediate = null
                refresh()
                return
            }
        }
        // Abandon the medium move and re-select.
        intermediate = null
        selected = [null, null]
        click(x, y)
        return
    }

    let move = 1296 * (SHIFT ? 9 : 0) + 36 * x + y
    if (moves_able.includes(move)) {
        board.move(selected[0], selected[1], move)
        selected = [null, null]
        refresh()
        return
    } else {
        selected = [null, null]
        click(x, y)
        return
    }

    document.getElementById(`cell-${x}-${y}`).style.background = COLOR_SELECTED
}

function lookup_piece(x) {
    clear()
    let id = Number(x)
    if (isNaN(id)) {
        // String Lookup
        for (let i = 0; i < 36; i++) {
            for (let j = 0; j < 36; j++) {
                if (board.board[i][j].name.toLowerCase().includes(x.toLowerCase()) || board.board[i][j].short_name.includes(x)) {
                    document.getElementById(`cell-${i}-${j}`).style.background = COLOR_SEARCH
                } 
            }
        }
    } else {
        for (let i = 0; i < 36; i++) {
            for (let j = 0; j < 36; j++) {
                if (board.board[i][j].id == x) {
                    document.getElementById(`cell-${i}-${j}`).style.background = COLOR_SEARCH
                } 
            }
        }
    }
}

for (let i = 0; i < 36; i++) {
    for (let j = 0; j < 36; j++) {
        cell(i, j).addEventListener('click', () => {click(i, j)})
    }
}

window.addEventListener('keydown', (event) => {
    if (event.key === 'Shift') {
        if (SHIFT) return
        SHIFT = true
        refresh()
    } else if (event.code === 'KeyX' || event.key === 'x' || event.key === 'X') {
        if (X_HELD) return
        X_HELD = true
        intermediate = null
        refresh()
    }
});

window.addEventListener('keyup', (event) => {
    if (event.key === 'Shift') {
        SHIFT = false
        refresh()
    } else if (event.code === 'KeyX' || event.key === 'x' || event.key === 'X') {
        X_HELD = false
        // Abandon a half-finished medium move without touching board state.
        intermediate = null
        refresh()
    }
});

window.addEventListener('mousedown', function(e) {
    if (e.shiftKey) {
        e.preventDefault();
    }
});

document.getElementById("lookup-form").addEventListener("submit", (e) => {
    e.preventDefault();
    lookup_piece(document.getElementById('piece-id-lookup').value)
});

render()

