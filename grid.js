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

function render() {
    for (let i = 0; i < 36; i++) {
        for (let j = 0; j < 36; j++) {
            document.getElementById(`cell-${i}-${j}`).innerHTML = board.board[i][j].short_name
            document.getElementById(`cell-${i}-${j}`).style.color = board.board[i][j].color == 0 ? "#ffffff" : "#000000" 
        }
    }
    if (selected[0] != null) {
        let moves_able = board.board[selected[0]][selected[1]].get_legal_moves(board)
        for (let square of moves_able) {
            if (square > 1296) continue
            document.getElementById(`cell-${Math.floor(square / 36) % 36}-${square % 36}`).style.background = "#888888"
        }
    }
}

function render_special() {
    clear()
    let moves_able = board.board[selected[0]][selected[1]].get_legal_moves(board)
    for (let i of moves_able) {
        if (i >= 10 * 1296) {
            let j = [Math.floor((i - 10 * 1296) / 36) % 36, (i - 10 * 1296) % 36]
            document.getElementById(`cell-${j[0]}-${j[1]}`).style.background = "#2a6a88"
        } else if (i >= 9 * 1296) {
            let j = [Math.floor((i - 9 * 1296) / 36) % 36, (i - 9 * 1296) % 36]
            document.getElementById(`cell-${j[0]}-${j[1]}`).style.background = "#905c4c"
        } else if (i >= 1296) {
            let j = [Math.floor(i / 1296), Math.floor(i / 36) % 36, i % 36]
            let anti_dir = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
            anti_dir = anti_dir[j[0]-1]
            if (!out_of_bounds(j[1] + anti_dir[0]) && !out_of_bounds(j[2] + anti_dir[1])) {
                document.getElementById(`cell-${j[1] + anti_dir[0]}-${j[2] + anti_dir[1]}`).style.background = "#16cc5c"
            }
        }
    }
}

function clear() {
    for (let i = 0; i < 36; i++) {
        for (let j = 0; j < 36; j++) {
            document.getElementById(`cell-${i}-${j}`).style.background = "#d4b855"
        }
    }
}

selected = [null, null]
function click(x, y) {
    clear()
    if (selected[0] == null) {
        selected = [x, y]
        render()
    } else {
        let moves_able = board.board[selected[0]][selected[1]].get_legal_moves(board)
        if (moves_able.includes(1296 * (SHIFT ? 9 : 0) + 36 * x + y)) {
            board.board[x][y] = board.board[selected[0]][selected[1]].copy(x, y)
            board.board[selected[0]][selected[1]] = PIECES[0].copy(selected[0], selected[1])
            selected = [null, null]
        } else {
            for (let square of moves_able) {
                document.getElementById(`cell-${Math.floor(square / 36) % 36}-${square % 36}`).style.background = "#888888"
            }
            selected = [null, null]
            click(x, y)
        }
    }
    render()
}

for (let i = 0; i < 36; i++) {
    for (let j = 0; j < 36; j++) {
        document.getElementById(`cell-${i}-${j}`).addEventListener('click', () => {click(i, j)})
    }
}

let SHIFT = false;

window.addEventListener('keydown', (event) => {
    if (event.key === 'Shift') {
        SHIFT = true;
        if (selected[0] != null) {
            render_special()
        }
    }
});

window.addEventListener('keyup', (event) => {
    if (event.key === 'Shift') {
        SHIFT = false
        if (selected[0] != null) {
            clear()
            render()
        }
    };
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