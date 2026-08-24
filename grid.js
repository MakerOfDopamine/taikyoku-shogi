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
            document.getElementById(`cell-${Math.floor(square / 36) % 36}-${square % 36}`).style.background = "#888888"
        }
    }
}

function render_special() {
    clear()
    let moves_able = board.board[selected[0]][selected[1]].get_legal_moves(board)
    for (let i of moves_able) {
        if (i >= 9 * 1296) {
            console.log("hi")
            let j = [Math.floor((i - 9 * 1296) / 36) % 36, (i - 9 * 1296) % 36]
            document.getElementById(`cell-${j[0]}-${j[1]}`).style.background = "#905c4c"
        } else {
            console.log(i)
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
        let chosen_move = all_moves[Math.floor(Math.random() * all_moves.length)]
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
    }, 1)
}