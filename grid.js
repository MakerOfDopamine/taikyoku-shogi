const grid = document.getElementById('grid');
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

let board = new Board()

// Must match the direction order used by Piece.get_legal_moves / Board.move.
const DIRS = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

const COLOR_EMPTY = "#d4b855"   // board background
const COLOR_MOVE = "#888888"    // ordinary move (special 0)
const COLOR_TRAMPLE = "#905c4c" // trample move (special 9, SHIFT)
const COLOR_FIRST = "#6b3fa0"   // medium move, first leg (special 1-8, X)
const COLOR_STANDING = "#3f7fd0"// square the piece is provisionally standing on
const COLOR_SECOND = "#16cc5c"  // medium move, second leg

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

// Draw whatever the current modifier state calls for.
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
}

function click(x, y) {
    if (selected[0] == null) {
        selected = [x, y]
        intermediate = null
        refresh()
        return
    }

    let moves_able = selected_moves()

    if (X_HELD) {
        if (intermediate == null) {
            // Stage one: did the player pick a usable first leg?
            for (let special of first_leg_directions(moves_able)) {
                let dir = DIRS[special - 1]
                if (selected[0] + dir[0] === x && selected[1] + dir[1] === y) {
                    intermediate = special
                    render_intermediate()
                    return
                }
            }
        } else {
            // Stage two: commit the full medium move if this is a real destination.
            let move = intermediate * 1296 + 36 * x + y
            if (moves_able.includes(move)) {
                board.move(selected[0], selected[1], move)
                selected = [null, null]
                intermediate = null
                refresh()
                return
            }
        }
        // Anything else abandons the medium move and re-selects.
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
    } else {
        selected = [null, null]
        click(x, y)
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

render()

if (false) {//confirm('Do you want to random move')) {
    // Comment this out
    let turn = 1
    let count = 0
    let a = setInterval(() => {
        let all_moves = []
        let choose_x = Math.floor(Math.random() * 36)
        let choose_y = Math.floor(Math.random() * 36)
        while (board.board[choose_x][choose_y].id == 0 || board.board[choose_x][choose_y].color != turn) {
            choose_x = Math.floor(Math.random() * 36)
            choose_y = Math.floor(Math.random() * 36)
        }
        all_moves = board.board[choose_x][choose_y].get_legal_moves(board)
        if (all_moves.length == 0) {
            ;
        } else {
            let chosen_move = all_moves[Math.floor(Math.random() * all_moves.length)]
            chosen_move = [Math.floor(chosen_move / 36) % 36, chosen_move % 36]
            if (board.board[chosen_move[0]][chosen_move[1]].id == 1000 || board.board[chosen_move[0]][chosen_move[1]].id == 11000) {
                alert(turn == 1 ? "Black won!" : "White won!")
            }
            try {
                board.board[chosen_move[0]][chosen_move[1]] = board.board[choose_x][choose_y].copy(chosen_move[0], chosen_move[1])
                board.board[choose_x][choose_y] = PIECES[0].copy(choose_x, choose_y)
                //render()
            } catch {
                ;
            }
            turn = 1 - turn
            count += 1
            //console.log(count)
            if (count % 100 == 0) {
                render()
            }
        }
    }, 1)
}