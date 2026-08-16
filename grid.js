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
