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
}

function clear() {
    for (let i = 0; i < 36; i++) {
        for (let j = 0; j < 36; j++) {
            document.getElementById(`cell-${i}-${j}`).style.background = "#d4b855"
        }
    }
}


function click(x, y) {
    clear()
    let moves_able = board.board[x][y].get_legal_moves(board)
    for (let square of moves_able) {
        document.getElementById(`cell-${square[0]}-${square[1]}`).style.background = "#888888"
    }
    render()
}

for (let i = 0; i < 36; i++) {
    for (let j = 0; j < 36; j++) {
        document.getElementById(`cell-${i}-${j}`).addEventListener('click', () => {click(i, j)})
    }
}

render()

if (confirm('Do you want to random move')) {
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