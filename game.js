const out_of_bounds = (n) => !Number.isInteger(n) || (0 > n) || (35 < n)
const reverse = (n) => (35 - n)
const RESULT_ROYAL_CAPTURE = "royal_capture"
const RESULT_STALEMATE = "stalemate"
const RESULT_REPETITION = "repetition"
const RESULT_NO_PROGRESS = "no_progress"
const NONE = [0,0,0,0,0,0,0,0]
const Q = [99,99,99,99,99,99,99,99]

class Board {
    constructor(base_board = null, turn = 1, viewing = false) {
        this.board = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        this.turn = turn
        this.turn_no = 1
        this.viewing = viewing
        if (base_board != null) {
            // Deep copy: never alias the caller's array. Coordinates are re-derived
            // from the actual position so a hand-built board can't disagree with itself.
            this.board = base_board.board.map((row, i) => row.map(
                (piece, j) => (piece === 0 ? PIECES[0] : piece).copy(i, j)
            ))
            this.turn = base_board.turn
        } else {
            this.initiate()
        }
    }

    initiate() {
        let pieces_to_add = [
            [1, 10, 0],
            [1, 10, 1],
            [1, 10, 2],
            [1, 10, 3],
            [1, 10, 4],
            [1, 10, 5],
            [1, 10, 6],
            [1, 10, 7],
            [1, 10, 8],
            [1, 10, 9],
            [1, 10, 10],
            [1, 10, 11],
            [1, 10, 12],
            [1, 10, 13],
            [1, 10, 14],
            [1, 10, 15],
            [1, 10, 16],
            [1, 10, 17],
            [1, 10, 18],
            [1, 10, 19],
            [1, 10, 20],
            [1, 10, 21],
            [1, 10, 22],
            [1, 10, 23],
            [1, 10, 24],
            [1, 10, 25],
            [1, 10, 26],
            [1, 10, 27],
            [1, 10, 28],
            [1, 10, 29],
            [1, 10, 30],
            [1, 10, 31],
            [1, 10, 32],
            [1, 10, 33],
            [1, 10, 34],
            [1, 10, 35],
            [2, 8, 10],
            [2, 8, 25],
            [3, 11, 10],
            [3, 11, 25],
            [4, 8, 12],
            [4, 8, 23],
            [5, 8, 16],
            [5, 8, 19],
            [6, 11, 5],
            [6, 11, 14],
            [6, 11, 21],
            [6, 11, 30],
            [7, 7, 4],
            [7, 7, 31],
            [8, 7, 13],
            [8, 7, 22],
            [9, 5, 11],
            [9, 5, 24],
            [10, 8, 14],
            [10, 8, 21],
            [11, 9, 10],
            [11, 9, 25],
            [12, 2, 16],
            [12, 2, 19],
            [13, 5, 10],
            [13, 5, 25],
            [14, 7, 5],
            [14, 7, 30],
            [15, 7, 16],
            [15, 7, 19],
            [16, 7, 15],
            [16, 7, 20],
            [17, 6, 9],
            [17, 6, 26],
            [18, 7, 6],
            [18, 7, 29],
            [19, 6, 7],
            [19, 6, 28],
            [20, 1, 16],
            [20, 1, 19],
            [21, 6, 14],
            [21, 6, 21],
            [22, 5, 12],
            [22, 5, 23],
            [23, 6, 11],
            [23, 6, 24],
            [24, 6, 10],
            [24, 6, 25],
            [25, 0, 16],
            [25, 0, 19],
            [26, 6, 15],
            [26, 6, 20],
            [27, 5, 6],
            [27, 5, 29],
            [28, 7, 8],
            [28, 7, 27],
            [29, 7, 12],
            [29, 7, 23],
            [30, 6, 6],
            [30, 6, 29],
            [31, 1, 18],
            [32, 1, 17],
            [34, 4, 17],
            [35, 4, 18],
            [36, 7, 9],
            [36, 7, 26],
            [37, 0, 15],
            [38, 0, 20],
            [39, 0, 18],
            [42, 8, 8],
            [42, 8, 27],
            [43, 6, 3],
            [43, 6, 32],
            [44, 7, 2],
            [44, 7, 33],
            [45, 6, 4],
            [45, 6, 31],
            [46, 0, 7],
            [46, 0, 28],
            [47, 1, 34],
            [48, 1, 1],
            [49, 2, 11],
            [49, 2, 24],
            [50, 6, 5],
            [50, 6, 30],
            [51, 6, 13],
            [52, 6, 12],
            [53, 6, 23],
            [54, 6, 22],
            [55, 5, 4],
            [55, 5, 31],
            [56, 5, 9],
            [56, 5, 26],
            [57, 2, 12],
            [57, 2, 23],
            [58, 5, 7],
            [58, 5, 28],
            [59, 1, 20],
            [60, 1, 15],
            [61, 3, 13],
            [61, 3, 22],
            [62, 3, 6],
            [62, 3, 29],
            [63, 3, 7],
            [63, 3, 28],
            [64, 8, 3],
            [64, 8, 32],
            [65, 8, 4],
            [65, 8, 31],
            [66, 5, 14],
            [66, 5, 21],
            [67, 5, 13],
            [67, 5, 22],
            [68, 8, 5],
            [68, 8, 30],
            [69, 1, 14],
            [70, 1, 21],
            [72, 1, 4],
            [72, 1, 31],
            [73, 2, 8],
            [73, 2, 27],
            [75, 3, 4],
            [75, 3, 31],
            [76, 3, 5],
            [76, 3, 30],
            [77, 3, 3],
            [77, 3, 32],
            [78, 1, 7],
            [78, 1, 28],
            [80, 1, 2],
            [80, 1, 33],
            [81, 9, 16],
            [81, 9, 19],
            [82, 9, 7],
            [82, 9, 28],
            [83, 0, 0],
            [83, 0, 35],
            [84, 7, 10],
            [84, 7, 25],
            [85, 9, 15],
            [85, 9, 20],
            [86, 1, 0],
            [86, 1, 35],
            [87, 2, 1],
            [87, 2, 34],
            [89, 5, 1],
            [89, 5, 34],
            [92, 4, 3],
            [92, 4, 32],
            [93, 4, 8],
            [93, 4, 27],
            [94, 4, 12],
            [94, 4, 23],
            [95, 7, 14],
            [95, 7, 21],
            [99, 0, 3],
            [99, 0, 32],
            [100, 4, 2],
            [100, 4, 33],
            [104, 9, 3],
            [104, 9, 32],
            [108, 6, 16],
            [108, 6, 19],
            [113, 0, 13],
            [113, 0, 22],
            [116, 5, 2],
            [117, 5, 33],
            [118, 9, 6],
            [118, 9, 29],
            [119, 9, 13],
            [119, 9, 22],
            [120, 9, 5],
            [120, 9, 30],
            [124, 9, 14],
            [124, 9, 21],
            [127, 4, 13],
            [127, 4, 22],
            [128, 1, 3],
            [128, 1, 32],
            [129, 5, 0],
            [129, 5, 35],
            [130, 6, 0],
            [130, 6, 35],
            [132, 1, 8],
            [132, 1, 27],
            [133, 1, 9],
            [133, 1, 26],
            [134, 7, 0],
            [134, 7, 35],
            [135, 9, 2],
            [135, 9, 33],
            [136, 4, 14],
            [136, 4, 21],
            [137, 4, 4],
            [137, 4, 31],
            [138, 6, 2],
            [138, 6, 33],
            [139, 5, 3],
            [139, 5, 32],
            [140, 7, 7],
            [140, 7, 28],
            [141, 9, 1],
            [141, 9, 34],
            [144, 7, 11],
            [144, 7, 24],
            [147, 7, 34],
            [148, 0, 1],
            [149, 3, 9],
            [149, 3, 26],
            [150, 9, 0],
            [151, 9, 35],
            [153, 2, 29],
            [154, 2, 6],
            [156, 0, 2],
            [156, 0, 33],
            [160, 2, 3],
            [160, 2, 32],
            [163, 9, 9],
            [163, 9, 26],
            [164, 9, 8],
            [164, 9, 27],
            [166, 2, 9],
            [166, 2, 26],
            [167, 4, 10],
            [167, 4, 25],
            [168, 1, 5],
            [168, 1, 30],
            [172, 4, 7],
            [172, 4, 28],
            [175, 1, 12],
            [175, 1, 23],
            [176, 1, 13],
            [176, 1, 22],
            [177, 3, 11],
            [177, 3, 24],
            [178, 3, 10],
            [178, 3, 25],
            [181, 2, 28],
            [182, 2, 7],
            [183, 7, 1],
            [184, 0, 34],
            [187, 2, 2],
            [187, 2, 33],
            [188, 0, 14],
            [188, 0, 21],
            [189, 0, 11],
            [191, 7, 3],
            [191, 7, 32],
            [193, 3, 16],
            [193, 3, 19],
            [197, 8, 0],
            [197, 8, 35],
            [198, 0, 4],
            [198, 0, 31],
            [199, 5, 5],
            [199, 5, 30],
            [201, 6, 8],
            [201, 6, 27],
            [202, 3, 1],
            [202, 3, 34],
            [203, 8, 2],
            [203, 8, 33],
            [205, 0, 10],
            [205, 0, 25],
            [206, 3, 0],
            [206, 3, 35],
            [207, 4, 0],
            [207, 4, 35],
            [208, 8, 1],
            [208, 8, 34],
            [209, 2, 0],
            [209, 2, 35],
            [210, 8, 11],
            [210, 8, 24],
            [211, 8, 13],
            [211, 8, 22],
            [212, 8, 15],
            [212, 8, 20],
            [213, 2, 10],
            [213, 2, 25],
            [217, 4, 9],
            [217, 4, 26],
            [218, 4, 11],
            [218, 4, 24],
            [219, 1, 6],
            [219, 1, 29],
            [220, 4, 15],
            [220, 4, 20],
            [221, 3, 8],
            [221, 3, 27],
            [222, 8, 6],
            [222, 8, 29],
            [223, 1, 11],
            [223, 1, 24],
            [224, 8, 18],
            [225, 6, 1],
            [225, 6, 34],
            [226, 1, 10],
            [226, 1, 25],
            [227, 4, 1],
            [227, 4, 34],
            [229, 8, 7],
            [229, 8, 28],
            [230, 8, 9],
            [230, 8, 26],
            [236, 9, 4],
            [236, 9, 31],
            [238, 2, 15],
            [238, 2, 20],
            [239, 2, 14],
            [239, 2, 21],
            [243, 0, 9],
            [243, 0, 26],
            [244, 0, 12],
            [244, 0, 23],
            [250, 3, 2],
            [250, 3, 33],
            [251, 5, 8],
            [251, 5, 27],
            [252, 5, 15],
            [253, 5, 20],
            [254, 4, 5],
            [254, 4, 30],
            [255, 0, 8],
            [255, 0, 27],
            [256, 4, 6],
            [256, 4, 29],
            [257, 5, 18],
            [258, 3, 14],
            [258, 3, 21],
            [259, 0, 5],
            [260, 0, 30],
            [261, 2, 17],
            [262, 5, 17],
            [263, 2, 18],
            [264, 8, 17],
            [265, 9, 11],
            [265, 9, 24],
            [266, 9, 12],
            [266, 9, 23],
            [267, 9, 18],
            [268, 9, 17],
            [273, 0, 24],
            [274, 6, 18],
            [275, 6, 17],
            [276, 7, 18],
            [283, 4, 16],
            [283, 4, 19],
            [286, 2, 5],
            [286, 2, 30],
            [287, 2, 4],
            [287, 2, 31],
            [288, 3, 15],
            [288, 3, 20],
            [290, 3, 18],
            [291, 3, 17],
            [292, 5, 19],
            [293, 0, 6],
            [293, 0, 29],
            [294, 5, 16],
            [295, 2, 13],
            [295, 2, 22],
            [297, 3, 12],
            [297, 3, 23],
            [300, 7, 17]
        ]

        for (let i of pieces_to_add) {
            if (this.board[i[1]][i[2]] != 0 || this.board[reverse(i[1])][reverse(i[2])] != 0) {
                alert(`Duplicate piece placement! [${i[0]} @ ${i[1]},${i[2]}]`)
            }
            this.board[i[1]][i[2]] = PIECES[i[0]].copy(i[1], i[2])
            this.board[reverse(i[1])][reverse(i[2])] = PIECES[i[0]].copy(reverse(i[1]), reverse(i[2]))
            this.board[reverse(i[1])][reverse(i[2])].invert_color()
        }

        if (this.board[0][17] != 0 || this.board[35][18] != 0) {
            alert("Duplicate piece placement! [king]")
        }
        this.board[0][17] = PIECES[1000].copy(0, 17)
        this.board[35][18] = PIECES[1000].copy(35, 18)
        this.board[35][18].invert_color()
        this.board[35][18].short_name = "<b>玉将</b>"

        for (let i = 0; i < 36; i++) {
            for (let j = 0; j < 36; j++) {
                if (this.board[i][j] === 0) {
                    this.board[i][j] = PIECES[0].copy(i, j)
                }
            }
        }
    }

    getPiece(x, y) {
        if (out_of_bounds(x) || out_of_bounds(y)) {
            return null
        }
        return this.board[x][y]
    }

    move(start_x, start_y, move, no_promote = false) {
        // DOES NOT VALIDATE THE MOVE OR WHETHER A MOVEMENT IS "IN BOUNDS"!
        // Expects moves fed directly from the generator.

        if (this.viewing) {
            return; // Viewing mode does not allow moving
        }

        let special = Math.floor(move / 1296)
        let target_x = Math.floor(move / 36) % 36
        let target_y = move % 36

        if (this.turn != this.board[start_x][start_y].color) {
            return
        }
        if (move < 1296) {
            // Regular moves.
            this.board[target_x][target_y] = this.board[start_x][start_y].copy(target_x, target_y)
            this.board[start_x][start_y] = PIECES[0].copy(start_x, start_y)
        } else if (move < 9 * 1296) {
            // "Intermediate" moves: Move to an intermediate square first, then move to target square.
            // Also handles igui-type moves.
            let offset = ([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])[special - 1]
            let mid_x = start_x + offset[0]
            let mid_y = start_y + offset[1]
            if (out_of_bounds(mid_x) || out_of_bounds(mid_y)) {
                // x * 36 + y silently wraps into the neighbouring rank for an off-board
                // square, and can wrap past 1296 and re-enter this branch forever.
                alert(`Invalid move: intermediate square off board [${move}] ${mid_x},${mid_y}`)
                return
            }
            this.move(start_x, start_y, mid_x * 36 + mid_y, true)
            this.turn = 1 - this.turn
            this.turn_no -= 1
            this.move(mid_x, mid_y, target_x * 36 + target_y, true)
            this.turn = 1 - this.turn
            this.turn_no -= 1
        } else if (move < 10 * 1296) {
            // "Trample" moves: remove all intermediate pieces. Note that this can include friendly pieces!!
            let dx = target_x - start_x
            let dy = target_y - start_y

            if ((dx === 0 || dy === 0 || Math.abs(dx) === Math.abs(dy)) && (dx !== 0 || dy !== 0)) {
                let unit = [Math.sign(dx), Math.sign(dy)]
                let size = Math.max(Math.abs(dx), Math.abs(dy))
                this.board[target_x][target_y] = this.board[start_x][start_y].copy(target_x, target_y)
                for (let i = 0; i < size; i++) {
                    this.board[start_x + i * unit[0]][start_y + i * unit[1]] = PIECES[0].copy(start_x + i * unit[0], start_y + i * unit[1])
                }
            } else {
                throw "Invalid move"
                //alert(`Invalid move... somehow ${move} ${start_x} ${start_y} ${target_x} ${target_y}`)
                //return
            }
        } else {
            throw "Invalid move (invalid special type)"
            //alert(`Invalid move: invalid special type [${move}]`)
            //return
        }

        if (!no_promote && ((this.board[target_x][target_y].color == 1 && target_x >= 25) || (this.board[target_x][target_y].color == 0 && target_x <= 10))) {
            this.promote(target_x, target_y)
        }

        this.turn = 1 - this.turn
        this.turn_no += 1
    }

    promote(x, y) {
        if (this.board[x][y].promoted) {
            return
        }

        if (PROMOTE[this.board[x][y].id] !== undefined) {
            let color = this.board[x][y].color
            this.board[x][y] = PIECES[PROMOTE[this.board[x][y].id]].copy(x, y)
            if (color == 0) {
                this.board[x][y].invert_color()
            }
            this.board[x][y].promoted = true
        }
    }
}

class Piece {
    constructor(id, name, short_name = "", dydx = [0, 0, 0, 0, 0, 0, 0, 0], tp = [], djump = [0, 0, 0, 0, 0, 0, 0, 0], color = 1, promoted = false, x = 0, y = 0) {
        this.id = id
        this.name = name
        this.short_name = short_name
        this.dydx = dydx
        this.tp = tp
        this.djump = djump
        this.color = color // 1 = BLACK, 0 = WHITE
        this.promoted = promoted
        if (this.id === 1000 || this.id === 39) {
            this.level = 4
        } else if (this.id === 291) {
            this.level = 3
        } else if (this.id === 290) {
            this.level = 2
        } else if (285 < this.id && 290 > this.id) {
            this.level = 1
        } else {this.level = 0}
        // 4 = King / Crown Prince
        // 3 = Great General
        // 2 = Vice General
        // 1 = All Range-capturing
        // 0 = All Pieces

        this.x = x
        this.y = y
    }

    copy(x, y) {
        // dydx / tp / djump are cloned so an instance can never mutate the PIECES template
        // (and thus every other piece of the same type) in place.
        return new Piece(
            this.id, this.name, this.short_name,
            this.dydx.slice(),
            this.tp.map((entry) => { let e = entry.slice(); e[2] = entry[2].slice(); return e }),
            this.djump.slice(),
            this.color, this.promoted, x, y
        )
    }

    invert_color() {
        // MUTATIVE!
        this.color = 1 - this.color
        this.dydx = this.dydx.slice(4, 8).concat(this.dydx.slice(0, 4))
        this.djump = this.djump.slice(4, 8).concat(this.djump.slice(0, 4))
        this.tp = this.tp.map((value) => [-value[0], -value[1], value[2].slice(4, 8).concat(value[2].slice(0, 4))])
    }

    is_empty() {
        return this.id === 0
    }

    is_friendly(piece) {
        return (this.color === piece.color) && (!(this.id === 0 || piece.id === 0))
    }

    get_legal_moves(board) {
        let moves = []
        let base_x = this.x
        let base_y = this.y
        let offset_x = 0
        let offset_y = 0
        let working_x = base_x
        let working_y = base_y
        let offset = 0

        // dydx
        let count = 0
        for (let dir of [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]) {
            working_x = base_x
            working_y = base_y
            offset = 1
            let pieces_jumped = 0

            while (offset <= this.dydx[count]) {
                working_x += dir[0]
                working_y += dir[1]
                if (out_of_bounds(working_x) || out_of_bounds(working_y)) {
                    break
                }
                if ((board.getPiece(working_x, working_y).color === (1 - this.color)) && !board.getPiece(working_x, working_y).is_empty()) {
                    moves.push(working_x * 36 + working_y)
                    pieces_jumped += 1
                }
                if (board.getPiece(working_x, working_y).is_friendly(this)) {
                    pieces_jumped += 1
                } else {
                    moves.push(working_x * 36 + working_y)
                }
                if (pieces_jumped > this.djump[count]) {
                    break
                }
                offset += 1
            }
            count += 1
        }

        // tp
        if (this.tp.length > 0) {
            for (let item of this.tp) {
                let dest_x = base_x + item[0]
                let dest_y = base_y + item[1]

                if (out_of_bounds(dest_x) || out_of_bounds(dest_y)) {
                    continue
                }

                if (this.is_friendly(board.getPiece(dest_x, dest_y))) {
                    continue
                }

                if (!board.getPiece(dest_x, dest_y).is_empty()) {
                    moves.push(dest_x * 36 + dest_y)
                    continue
                }

                moves.push(dest_x * 36 + dest_y)
                count = 0
                for (let dir of [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]) {
                    working_x = dest_x
                    working_y = dest_y
                    offset = 1

                    while (offset <= item[2][count]) {
                        working_x += dir[0]
                        working_y += dir[1]
                        if (out_of_bounds(working_x) || out_of_bounds(working_y)) {
                            break
                        }
                        if ((board.getPiece(working_x, working_y).color === (1 - this.color)) && !board.getPiece(working_x, working_y).is_empty()) {
                            moves.push(working_x * 36 + working_y)
                            break
                        }
                        if ((!board.getPiece(working_x, working_y).is_empty()) || (this.is_friendly(board.getPiece(working_x, working_y)))) {
                            break
                        }
                        moves.push(working_x * 36 + working_y)
                        offset += 1
                    }
                    count += 1
                }
            }
        }

        // free eagle moment
        if (this.id === 276) {
            let igui_count = 0
            for (let dir of [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]) {
                igui_count += 1
                let max = 0
                if (this.color === 1) {max = (dir[0] === 1 && Math.abs(dir[1]) === 1) ? 4 : 3}
                else {max = (dir[0] === -1 && Math.abs(dir[1]) === 1) ? 4 : 3}
                
                // IGUI handling
                let adjacent_piece = board.getPiece(base_x + dir[0], base_y + dir[1])
                if (adjacent_piece === null) {
                    continue
                } else {
                    if (!this.is_friendly(adjacent_piece)) {
                        moves.push(igui_count * 1296 + base_x * 36 + base_y)
                    }
                }

                // Can we move that far?
                if (board.getPiece(base_x + max * dir[0], base_y + max * dir[1]) === null) {
                    continue
                }

                // Is there a piece in the area that we can take?
                let has_friendly = false
                let has_enemy = false
                for (let i = 1; i < max + 1; i++) {
                    let piece = board.getPiece(base_x + i * dir[0], base_y + i * dir[1])
                    if (this.is_friendly(piece)) {
                        has_friendly = true
                        break
                    }
                    if ((piece.color === 1 - this.color) && !piece.is_empty()) {
                        has_enemy = true
                    }
                }

                if (has_enemy && !has_friendly) {
                    let jump_x = base_x + max * dir[0]
                    let jump_y = base_y + max * dir[1]
                    moves.push(9 * 1296 + jump_x * 36 + jump_y)
                    jump_x += dir[0]
                    jump_y += dir[1]
                    while (true) {
                        if (board.getPiece(jump_x, jump_y) === null || this.is_friendly(board.getPiece(jump_x, jump_y))) break
                        moves.push(9 * 1296 + jump_x * 36 + jump_y)
                        if (!board.getPiece(jump_x, jump_y).is_empty()) break
                        jump_x += dir[0]
                        jump_y += dir[1]
                    }
                }
            }
        }

        // trampling pieces
        if ([286, 287, 288, 289, 290, 291].includes(this.id)) {
            let trample_dir = []
            switch (this.id) {
                case 286:
                case 289:
                    trample_dir = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                    break
                case 287:
                case 288:
                case 290:
                    trample_dir = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
                    break
                case 291:
                    trample_dir = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
                    break
            }

            for (let dir of trample_dir) {
                let jump_x = base_x + dir[0]
                let jump_y = base_y + dir[1]
                while (!out_of_bounds(jump_x) && !out_of_bounds(jump_y)) {
                    if (board.getPiece(jump_x, jump_y).level >= this.level) {
                        break
                    }
                    // Friendly pieces in the path are still removed by Board.move, but the
                    // trample may not *end* on one.
                    if (!this.is_friendly(board.getPiece(jump_x, jump_y))) {
                        moves.push(9 * 1296 + jump_x * 36 + jump_y)
                    }
                    jump_x += dir[0]
                    jump_y += dir[1]
                }
            }
        }

        // hook pieces
        if ([292, 293, 294].includes(this.id)) {
            let hook_dir = []
            switch (this.id) {
                case 292:
                    hook_dir = [[1, 0], [0, 1], [-1, 0], [0, -1]]
                    break
                case 293:
                case 294:
                    hook_dir = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
                    break
            }
            let middle = []
            for (let dir of hook_dir) {
                working_x = base_x + dir[0]
                working_y = base_y + dir[1]
                while (!out_of_bounds(working_x) && !out_of_bounds(working_y)) {
                    if (this.is_friendly(board.getPiece(working_x, working_y))) break
                    middle.push(working_x * 36 + working_y)
                    if ((!board.getPiece(working_x, working_y).is_empty())) {
                        middle.pop()
                        moves.push(working_x * 36 + working_y)
                        break
                    }
                    working_x += dir[0]
                    working_y += dir[1]
                }
            }

            for (let midpoint of middle) {
                for (let dir of hook_dir) {
                    working_x = Math.floor(midpoint / 36) + dir[0]
                    working_y = (midpoint % 36) + dir[1]
                    while (!out_of_bounds(working_x) && !out_of_bounds(working_y)) {
                        if (this.is_friendly(board.getPiece(working_x, working_y))) break
                        moves.push(working_x * 36 + working_y)
                        if ((!board.getPiece(working_x, working_y).is_empty())) {
                            break
                        }
                        working_x += dir[0]
                        working_y += dir[1]
                    }
                }
            }
        }

        // peacock (separate section because its logic is slightly different)
        if (this.id === 295) {
            let left = []
            let right = []
            let count = 0
            for (let dir of (this.color === 0 ? [[-1, -1], [-1, 1]] : [[1, -1], [1, 1]])) {
                working_x = base_x + dir[0]
                working_y = base_y + dir[1]
                while (!out_of_bounds(working_x) && !out_of_bounds(working_y)) {
                    if (this.is_friendly(board.getPiece(working_x, working_y))) break
                    ((count === 0) ? left : right).push(working_x * 36 + working_y)
                    if ((!board.getPiece(working_x, working_y).is_empty())) {
                        ((count === 0) ? left : right).pop()
                        moves.push(working_x * 36 + working_y)
                        break
                    }
                    working_x += dir[0]
                    working_y += dir[1]
                }
                count += 1
            }
            
            for (let midpoint of (this.color === 0 ? right : left)) {
                for (let dir of [[-1, -1], [1, 1]]) {
                    working_x = Math.floor(midpoint / 36) + dir[0]
                    working_y = (midpoint % 36) + dir[1]
                    while (!out_of_bounds(working_x) && !out_of_bounds(working_y)) {
                        if (this.is_friendly(board.getPiece(working_x, working_y))) break
                        moves.push(working_x * 36 + working_y)
                        if ((!board.getPiece(working_x, working_y).is_empty())) {
                            break
                        }
                        working_x += dir[0]
                        working_y += dir[1]
                    }
                }
            }

            for (let midpoint of (this.color === 0 ? left : right)) {
                for (let dir of [[1, -1], [-1, 1]]) {
                    working_x = Math.floor(midpoint / 36) + dir[0]
                    working_y = (midpoint % 36) + dir[1]
                    while (!out_of_bounds(working_x) && !out_of_bounds(working_y)) {
                        if (this.is_friendly(board.getPiece(working_x, working_y))) break
                        moves.push(working_x * 36 + working_y)
                        if ((!board.getPiece(working_x, working_y).is_empty())) {
                            break
                        }
                        working_x += dir[0]
                        working_y += dir[1]
                    }
                }
            }
        }

        if (297 <= this.id && this.id <= 300) {
            // Lion.
            let first = []
            let count = 0
            for (let dir of [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]) {
                let piece = board.getPiece(base_x + dir[0], base_y + dir[1])
                if (piece != null && !this.is_friendly(piece)) {
                    first.push([count + 1, base_x + dir[0], base_y + dir[1]])
                }
                count += 1
            }

            for (let inter of first) {
                for (let dir of [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]) {
                    let piece = board.getPiece(inter[1] + dir[0], inter[2] + dir[1])
                    if (piece != null && !(this.is_friendly(piece) && (inter[1] + dir[0] != base_x || inter[2] + dir[1] != base_y))) {
                        moves.push(inter[0] * 1296 + (inter[1] + dir[0]) * 36 + (inter[2] + dir[1]))
                    }
                }
            }
        }

        // return

        moves = [...new Set(moves)]//.map((value) => [Math.floor(value / 1296), Math.floor(value / 36) % 36, value % 36])
        return moves
    }
}

class Game {
    constructor(positions = [], moves = [], metadata = {}) {
        // positions = packed board after each ply (as in .tkn), moves = [x, y, move] for each ply
        this.positions = positions
        this.moves = moves
        this.metadata = metadata
        this.move = 0
        this.tomove = 1
        this.viewing = false
    }

    position(n) {
        // Boards are built when shown. position n = ply n (0 = start)
        if (n == 0) return new Board().board
        return decode_board(this.positions[n - 1])
    }

    push(x, y, move) {
        // Records the move just played on board, dropping any moves after the current one
        this.positions.length = this.move
        this.moves.length = this.move
        this.positions.push(encode_board(board))
        this.moves.push([x, y, move])
        this.move += 1
        this.tomove = 1 - this.tomove
    }

    load() {
        // jesus I should have named this something else
        this.move = 0
        this.tomove = 1
        this.viewing = true
        board = new Board(null, this.move, true)
        board.board = this.position(0)
        refresh()
        build_move_list(this)
    }

    goto(n) {
        this.move = n
        this.tomove = n % 2 == 0 ? 1 : 0
        board.board = this.position(n)
        board.turn = this.tomove
        board.turn_no = n + 1
        refresh()
        if (this.viewing) update_move_list(n)
    }

    next_move() {
        if (this.move == this.positions.length) return;
        this.goto(this.move + 1)
    }

    prev_move() {
        if (this.move == 0) return;
        this.goto(this.move - 1)
    }
}

const PIECES = {
    1000: new Piece(1000, "King", "<b>王将</b>", [2,2,2,2,2,2,2,2]),
    0: new Piece(0, "Empty", ""),
    1: new Piece(1, "Pawn", "歩兵", [1,0,0,0,0,0,0,0]),
    2: new Piece(2, "Earth General", "土将", [1,0,0,0,1,0,0,0]),
    3: new Piece(3, "Go-Between", "仲人", [1,0,0,0,1,0,0,0]),
    4: new Piece(4, "Stone General", "石将", [0,1,0,0,0,0,0,1]),
    5: new Piece(5, "Iron General", "鉄将", [1,1,0,0,0,0,0,1]),
    6: new Piece(6, "Dog", "犬", [1,1,0,0,0,0,0,1]),
    7: new Piece(7, "Swooping Owl", "鴟行", [1,0,0,1,0,1,0,0]),
    8: new Piece(8, "Old Rat", "老鼠", [1,0,0,1,0,1,0,0]),
    9: new Piece(9, "Strutting Crow", "烏行", [1,0,0,1,0,1,0,0]),
    10: new Piece(10, "Tile General", "瓦将", [0,1,0,0,1,0,0,1]),
    11: new Piece(11, "Sword Soldier", "刀兵", [0,1,0,0,1,0,0,1]),
    12: new Piece(12, "Copper General", "銅将", [1,1,0,0,1,0,0,1]),
    13: new Piece(13, "Flying Goose", "鳫飛", [1,1,0,0,1,0,0,1]),
    14: new Piece(14, "Climbing Monkey", "登猿", [1,1,0,0,1,0,0,1]),
    15: new Piece(15, "Reclining Dragon", "臥龍", [1,0,1,0,1,0,1,0]),
    16: new Piece(16, "Coiled Serpent", "蟠蛇", [1,0,0,1,1,1,0,0]),
    17: new Piece(17, "Flying Chicken", "鶏飛", [0,1,1,0,0,0,1,1]),
    18: new Piece(18, "Cat Sword", "猫刀", [0,1,0,1,0,1,0,1]),
    19: new Piece(19, "Evil Wolf", "悪狼", [1,1,1,0,0,0,1,1]),
    20: new Piece(20, "Silver General", "銀将", [1,1,0,1,0,1,0,1]),
    21: new Piece(21, "Fierce Stag", "猛鹿", [1,1,0,1,0,1,0,1]),
    22: new Piece(22, "Blind Dog", "盲犬", [0,1,1,0,1,0,1,1]),
    23: new Piece(23, "Huai Chicken", "淮鶏", [0,1,1,0,1,0,1,1]),
    24: new Piece(24, "Old Monkey", "古猿", [0,1,0,1,1,1,0,1]),
    25: new Piece(25, "Gold General", "金将", [1,1,1,0,1,0,1,1]),
    26: new Piece(26, "Fierce Wolf", "猛狼", [1,1,1,0,1,0,1,1]),
    27: new Piece(27, "Fierce Leopard", "猛豹", [1,1,0,1,1,1,0,1]),
    28: new Piece(28, "Blind Monkey", "盲猿", [0,1,1,1,0,1,1,1]),
    29: new Piece(29, "Blind Bear", "盲熊", [0,1,1,1,0,1,1,1]),
    30: new Piece(30, "Angry Boar", "嗔猪", [0,1,1,1,0,1,1,1]),
    31: new Piece(31, "Drunken Elephant", "酔象", [1,1,1,1,0,1,1,1]),
    32: new Piece(32, "Neighboring King", "近王", [1,1,1,1,0,1,1,1]),
    33: new Piece(33, "Rushing Boar", "行猪", [1,1,1,1,0,1,1,1]), // PROMOTE
    34: new Piece(34, "Deva", "提婆", [1,0,1,1,1,1,1,1]),
    35: new Piece(35, "Dark Spirit", "無明", [1,1,1,1,1,1,1,0]),
    36: new Piece(36, "Blind Tiger", "盲虎", [0,1,1,1,1,1,1,1]),
    37: new Piece(37, "Left General", "左将", [1,1,1,1,1,0,0,0]),
    38: new Piece(38, "Right General", "右将", [1,0,0,0,1,1,1,1]),
    39: new Piece(39, "Crown Prince", "太子", [1,1,1,1,1,1,1,1]),
    40: new Piece(40, "Bear's Eyes", "熊眼", [1,1,1,1,1,1,1,1]),
    41: new Piece(41, "Poisonous Wolf", "毒狼", [1,1,1,1,1,1,1,1]),
    42: new Piece(42, "Wood General", "木将", [0,2,0,0,0,0,0,2]),
    43: new Piece(43, "Donkey", "驢馬", [2,0,2,0,2,0,2,0]),
    44: new Piece(44, "Enchanted Badger", "変狸", [2,0,2,0,2,0,2,0]),
    45: new Piece(45, "Flying Horse", "馬麟", [0,2,0,2,0,2,0,2]),
    46: new Piece(46, "Beast Cadet", "獣曹", [2,2,2,2,0,2,2,2]),
    47: new Piece(47, "Fragrant Elephant", "香象", [2,2,2,2,2,2,2,2]),
    48: new Piece(48, "White Elephant", "白象", [2,2,2,2,2,2,2,2]),
    49: new Piece(49, "Rushing Bird", "行鳥", [2,1,1,1,0,1,1,1]),
    50: new Piece(50, "Fierce Bear", "猛熊", [1,2,1,0,0,0,1,2]),
    51: new Piece(51, "Eastern Barbarian", "東夷", [2,1,1,0,2,0,1,1]),
    52: new Piece(52, "Western Barbarian", "西戎", [2,1,1,0,2,0,1,1]),
    53: new Piece(53, "Northern Barbarian", "北狄", [1,1,2,0,1,0,2,1]),
    54: new Piece(54, "Southern Barbarian", "南蛮", [1,1,2,0,1,0,2,1]),
    55: new Piece(55, "Prancing Stag", "踊鹿", [1,1,2,0,1,0,2,1]),
    56: new Piece(56, "Poisonous Serpent", "毒蛇", [2,1,2,0,1,0,2,1]),
    57: new Piece(57, "Old Kite", "古鵄", [0,2,1,2,0,2,1,2]),
    58: new Piece(58, "Fierce Eagle", "猛鷲", [1,2,1,2,0,2,1,2]),
    59: new Piece(59, "Guardian of the Gods", "金剛", [3,0,3,0,3,0,3,0]),
    60: new Piece(60, "Sumo Wrestler", "力士", [0,3,0,3,0,3,0,3]),
    61: new Piece(61, "Fowl Cadet", "禽曹", [3,3,3,3,0,3,3,3]),
    62: new Piece(62, "Horse General", "馬将", [3,1,0,0,1,0,0,1]),
    63: new Piece(63, "Ox General", "牛将", [3,1,0,0,1,0,0,1]),
    64: new Piece(64, "Wind General", "風将", [3,1,0,0,1,0,0,1]),
    65: new Piece(65, "River General", "川将", [3,1,0,0,1,0,0,1]),
    66: new Piece(66, "Fire General", "火将", [3,1,0,0,3,0,0,1]),
    67: new Piece(67, "Water General", "水将", [1,3,0,0,1,0,0,3]),
    68: new Piece(68, "Mountain General", "山将", [1,3,0,0,1,0,0,3]),
    69: new Piece(69, "Buddhist Devil", "羅刹", [0,3,1,0,1,0,1,3]),
    70: new Piece(70, "Nature Spirit", "夜叉", [0,1,3,0,1,0,3,1]),
    71: new Piece(71, "Sword General", "刀将", [3,3,0,0,3,0,0,3]), // PROMOTE
    72: new Piece(72, "Fowl Officer", "禽吏", [2,3,2,3,0,3,2,3]),
    73: new Piece(73, "Beast Officer", "獣吏", [3,3,2,3,0,3,2,3]),
    74: new Piece(74, "Heavenly Tetrarch", "四天", [4,4,4,4,4,4,4,4]), // PROMOTE
    75: new Piece(75, "Chicken General", "鶏将", [4,0,0,1,0,1,0,0]),
    76: new Piece(76, "Pup General", "狗将", [4,0,0,1,0,1,0,0]),
    77: new Piece(77, "Pig General", "豚将", [0,4,0,0,2,0,0,4]),
    78: new Piece(78, "Mountain Stag", "山鹿", [1,3,2,0,4,0,2,3]),
    79: new Piece(79, "Leopard King", "豹王", [5,5,5,5,5,5,5,5]), // PROMOTE
    80: new Piece(80, "Turtle Dove", "山鳩", [0,5,1,0,1,0,1,5]),
    81: new Piece(81, "Crossbow Soldier", "弓兵", [5,3,3,0,1,0,3,3]),
    82: new Piece(82, "Cannon Soldier", "炮兵", [7,5,3,0,1,0,3,5]),
    // 99 in a direction: rook/bishop movement
    83: new Piece(83, "Incense Chariot", "香車", [99,0,0,0,0,0,0,0]), 
    84: new Piece(84, "Ox Chariot", "牛車", [99,0,0,0,0,0,0,0]),
    85: new Piece(85, "Fierce Tiger", "猛虎", [99,0,0,0,0,0,0,0]),
    86: new Piece(86, "Reverse Chariot", "反車", [99,0,0,0,99,0,0,0]),
    87: new Piece(87, "Side Dragon", "横龍", [99,0,99,0,0,0,99,0]),
    88: new Piece(88, "Mountain Witch", "山母", [0,99,0,99,99,99,0,99]), // PROMOTE
    89: new Piece(89, "White Foal", "白駒", [99,99,0,0,99,0,0,99]),
    90: new Piece(90, "Mockingbird", "寺鳥", [99,99,0,0,99,0,0,99]), // PROMOTE
    91: new Piece(91, "Multi General", "雜将", [99,99,0,0,99,0,0,99]), // PROMOTE
    92: new Piece(92, "Flying Chariot", "飛車", [99,0,99,0,99,0,99,0]),
    93: new Piece(93, "Soldier", "兵士", [99,0,99,0,99,0,99,0]),
    94: new Piece(94, "Running Chariot", "走車", [99,0,99,0,99,0,99,0]),
    95: new Piece(95, "Square Mover", "方行", [99,0,99,0,99,0,99,0]),
    96: new Piece(96, "Gliding Swallow", "燕行", [99,0,99,0,99,0,99,0]), // PROMOTE
    97: new Piece(97, "Free Serpent", "奔蛇", [99,0,0,99,99,99,0,0]), // PROMOTE
    98: new Piece(98, "Coiled Dragon", "蟠龍", [99,0,0,99,99,99,0,0]), // PROMOTE
    99: new Piece(99, "Whale", "鯨鯢", [99,0,0,99,99,99,0,0]),
    100: new Piece(100, "Angle Mover", "角行", [0,99,0,99,0,99,0,99]),
    101: new Piece(101, "Free Wolf", "奔狼", [99,99,99,0,0,0,99,99]), // PROMOTE
    102: new Piece(102, "Running Leopard", "走豹", [99,99,99,0,0,0,99,99]), // PROMOTE
    103: new Piece(103, "Wizard Stork", "仙鷦", [0,99,99,0,99,0,99,99]), // PROMOTE
    104: new Piece(104, "Flying Ox", "飛牛", [99,99,0,99,99,99,0,99]),
    105: new Piece(105, "Free Bear", "奔熊", [99,99,0,99,99,99,0,99]), // PROMOTE
    106: new Piece(106, "Free Leopard", "奔豹", [99,99,0,99,99,99,0,99]), // PROMOTE
    107: new Piece(107, "Great Whale", "大鯨", [99,99,0,99,99,99,0,99]), // PROMOTE
    108: new Piece(108, "Treacherous Fox", "隠狐", [99,99,0,99,99,99,0,99], [
        [2, 0, [99,0,0,0,0,0,0,0]], [3, 0, [99,0,0,0,0,0,0,0]],
        [2, 2, [0,99,0,0,0,0,0,0]], [3, 3, [0,99,0,0,0,0,0,0]],
        [-2, 2, [0,0,0,99,0,0,0,0]], [-3, 3, [0,0,0,99,0,0,0,0]],
        [-2, 0, [0,0,0,0,99,0,0,0]], [-3, 0, [0,0,0,0,99,0,0,0]],
        [-2, -2, [0,0,0,0,0,99,0,0]], [-3, -3, [0,0,0,0,0,99,0,0]],
        [2, -2, [0,0,0,0,0,0,0,99]], [3, -3, [0,0,0,0,0,0,0,99]]
    ]),
    109: new Piece(109, "Cavalier", "騎士", [99,99,99,0,99,0,99,99]), // PROMOTE
    110: new Piece(110, "Strong Chariot", "強車", [99,99,99,0,99,0,99,99]), // PROMOTE
    111: new Piece(111, "Free Dragon", "奔龍", [0,99,99,99,99,99,99,99]), // PROMOTE
    112: new Piece(112, "Free Tiger", "奔虎", [0,99,99,99,99,99,99,99]), // PROMOTE
    113: new Piece(113, "Free King", "奔王", Q),
    114: new Piece(114, "Free Stag", "奔鹿", Q), // PROMOTE
    115: new Piece(115, "Strong Eagle", "勁鷲", Q), // PROMOTE
    116: new Piece(116, "Howling Dog (left)", "𠵇犬", [99,0,0,0,1,0,0,0]),
    117: new Piece(117, "Howling Dog (right)", "𠵇犬", [99,0,0,0,1,0,0,0]),
    118: new Piece(118, "Vertical Horse", "竪馬", [99,1,0,0,1,0,0,1]),
    119: new Piece(119, "Spear Soldier", "鎗兵", [99,0,1,0,1,0,1,0]),
    120: new Piece(120, "Vertical Pup", "竪狗", [99,0,0,1,1,1,0,0]),
    121: new Piece(121, "Raiding Hawk", "延鷹", [99,1,1,0,0,0,1,1]), // PROMOTE
    122: new Piece(122, "Right Iron Chariot", "右鉄", [0,0,1,99,0,99,0,99]), // PROMOTE
    123: new Piece(123, "Left Iron Chariot", "左鉄", [0,99,0,99,0,99,1,0]), // PROMOTE
    124: new Piece(124, "Vertical Leopard", "竪豹", [99,1,1,0,1,0,1,1]),
    125: new Piece(125, "Right Dog", "右犬", [99,0,0,0,1,99,0,0]), // PROMOTE
    126: new Piece(126, "Left Dog", "左犬", [99,0,0,99,1,0,0,0]), // PROMOTE
    127: new Piece(127, "Ram's-head Soldier", "羊兵", [0,99,0,0,1,0,0,99]),
    128: new Piece(128, "Flying Swallow", "飛燕", [0,99,0,0,1,0,0,99]),
    129: new Piece(129, "Wood Chariot", "木車", [99,0,0,1,99,0,0,1]),
    130: new Piece(130, "Tile Chariot", "瓦車", [99,1,0,0,99,1,0,0]),
    131: new Piece(131, "Running Boar", "走猪", [99,0,1,0,99,0,1,0]), // PROMOTE
    132: new Piece(132, "Running Pup", "走狗", [99,0,1,0,99,0,1,0]),
    133: new Piece(133, "Running Serpent", "走蛇", [99,0,1,0,99,0,1,0]),
    134: new Piece(134, "Earth Chariot", "土車", [99,0,1,0,99,0,1,0]),
    135: new Piece(135, "Vertical Mover", "竪行", [99,0,1,0,99,0,1,0]),
    136: new Piece(136, "Fierce Ox", "猛牛", [1,99,0,0,1,0,0,99]),
    137: new Piece(137, "Side Wolf", "横狼", [0,0,99,1,0,0,99,1]),
    138: new Piece(138, "Side Ox", "横牛", [0,1,99,0,0,1,99,0]),
    139: new Piece(139, "Side Mover", "横行", [1,0,99,0,1,0,99,0]),
    140: new Piece(140, "Swallow's Wings", "燕羽", [1,0,99,0,1,0,99,0]),
    141: new Piece(141, "Side Monkey", "横猿", [0,1,99,0,1,0,99,1]),
    142: new Piece(142, "Divine Sparrow", "神雀", [1,1,1,99,1,99,1,99]), // PROMOTE
    143: new Piece(143, "Plodding Ox", "歬牛", [99,1,0,1,99,1,0,1]), // PROMOTE
    144: new Piece(144, "Side Flyer", "横飛", [0,1,99,1,0,1,99,1]),
    145: new Piece(145, "Flying Stag", "飛鹿", [99,1,1,1,99,1,1,1]), // PROMOTE
    146: new Piece(146, "Copper Elephant", "銅象", [99,1,1,1,99,1,1,1]), // PROMOTE
    147: new Piece(147, "Vermillion Sparrow", "朱雀", [1,1,1,99,1,1,1,99]),
    148: new Piece(148, "Turtle Snake", "玄武", [1,99,1,1,1,99,1,1]), 
    149: new Piece(149, "Side Boar", "横猪", [1,1,99,1,1,1,99,1]),
    150: new Piece(150, "Left Chariot", "左車", [99,99,0,0,0,99,1,0]),
    151: new Piece(151, "Right Chariot", "右車", [99,0,1,99,0,0,0,99]),
    152: new Piece(152, "Great Tiger", "大虎", [1,0,99,0,99,0,99,0]), // PROMOTE
    153: new Piece(153, "Right Tiger", "右虎", [0,1,0,1,0,99,99,99]), 
    154: new Piece(154, "Left Tiger", "左虎", [0,99,99,99,0,1,0,1]),
    155: new Piece(155, "Great Bear", "大熊", [99,99,1,0,1,0,1,99]), // PROMOTE
    156: new Piece(156, "Running Rabbit", "走兎", [99,99,0,1,1,1,0,99]),
    157: new Piece(157, "Left Army", "左軍", [1,1,1,1,1,99,99,99]), // PROMOTE
    158: new Piece(158, "Right Army", "右軍", [1,99,99,99,1,1,1,1]), // PROMOTE
    159: new Piece(159, "Divine Turtle", "神亀", [1,99,1,99,1,99,1,1]), // PROMOTE
    160: new Piece(160, "Running Wolf", "走狼", [1,99,99,0,0,0,99,99]),
    161: new Piece(161, "Flying Hawk", "飛鷹", [1,99,0,99,0,99,0,99]), // PROMOTE
    162: new Piece(162, "Cannon Chariot", "炮車", [99,99,1,0,99,0,1,99]), // PROMOTE
    163: new Piece(163, "Dragon King", "龍王", [99,1,99,1,99,1,99,1]), 
    164: new Piece(164, "Dragon Horse", "龍馬", [1,99,1,99,1,99,1,99]), 
    165: new Piece(165, "Free Boar", "奔猪", [99,99,99,0,1,0,99,99]), // PROMOTE
    166: new Piece(166, "Wind Dragon", "風龍", [0,99,99,99,0,1,99,99]),
    167: new Piece(167, "Cloud Dragon", "雲龍", [1,99,1,99,99,99,1,99]),
    168: new Piece(168, "Rain Dragon", "雨龍", [1,1,99,99,99,99,99,1]),
    169: new Piece(169, "Fire Ox", "火牛", [99,99,1,99,99,99,1,99]), // PROMOTE
    170: new Piece(170, "Fierce Wind", "暴風", [99,99,1,99,99,99,1,99]), // PROMOTE
    171: new Piece(171, "Huai River", "淮川", [1,99,99,99,1,99,99,99]), // PROMOTE
    172: new Piece(172, "Vertical Tiger", "竪虎", [99,0,0,0,2,0,0,0]),
    173: new Piece(173, "Wind Snapping Turtle", "風鼈", [99,2,0,0,99,0,0,2]), // PROMOTE
    174: new Piece(174, "Running Tile", "走瓦", [99,0,2,0,99,0,2,0]), // PROMOTE
    175: new Piece(175, "Running Tiger", "走虎", [99,0,2,0,99,0,2,0]),
    176: new Piece(176, "Running Bear", "走熊", [99,0,2,0,99,0,2,0]),
    177: new Piece(177, "Gold Stag", "金鹿", [0,99,0,2,0,2,0,99]),
    178: new Piece(178, "Silver Rabbit", "銀兎", [0,2,0,99,0,99,0,2]),
    179: new Piece(179, "Walking Heron", "歩䳲", [99,2,2,0,99,0,2,2]), // PROMOTE
    180: new Piece(180, "Reed Bird", "芦鳥", [99,0,2,2,99,2,2,0]), // PROMOTE
    181: new Piece(181, "Right Dragon", "右龍", [0,0,2,0,0,99,99,99]),
    182: new Piece(182, "Left Dragon", "左龍", [0,99,99,99,0,0,2,0]),
    183: new Piece(183, "Blue Dragon", "青龍", [99,99,2,0,99,0,2,0]),
    184: new Piece(184, "White Tiger", "白虎", [2,0,99,0,2,0,99,99]),
    185: new Piece(185, "Divine Tiger", "神虎", [99,0,99,0,2,0,99,99]), // PROMOTE
    186: new Piece(186, "Divine Dragon", "神龍", [99,99,99,0,99,0,2,0]), // PROMOTE
    187: new Piece(187, "Running Stag", "走鹿", [0,99,99,0,2,0,99,99]),
    188: new Piece(188, "Rear Standard", "後旗", [99,2,99,2,99,2,99,2]),
    189: new Piece(189, "Ceramic Dove", "鳩盤", [2,99,2,99,2,99,2,99]),
    190: new Piece(190, "Elephant King", "象王", [2,99,2,99,2,99,2,99]), // PROMOTE
    191: new Piece(191, "Horseman", "騎兵", [99,99,2,0,99,0,2,99]),
    192: new Piece(192, "Great Foal", "大駒", [99,99,2,0,99,0,2,99]), // PROMOTE
    193: new Piece(193, "Woodland Demon", "林鬼", [99,99,2,2,99,2,2,99]), 
    194: new Piece(194, "Free Chicken", "奔鶏", [99,99,2,2,99,2,2,99]), // PROMOTE
    195: new Piece(195, "Free Dog", "奔犬", [99,99,2,2,99,2,2,99]), // PROMOTE
    196: new Piece(196, "Running Ox", "走牛", [99,99,99,0,2,0,99,99]), // PROMOTE
    197: new Piece(197, "Chariot Soldier", "車兵", [99,99,2,99,99,99,2,99]),
    198: new Piece(198, "Fire Demon", "火鬼", [2,99,99,99,2,99,99,99]),
    199: new Piece(199, "Water Ox", "水牛", [2,99,99,99,2,99,99,99]),
    200: new Piece(200, "Strong Bear", "強熊", [99,99,99,99,2,99,99,99]), // PROMOTE
    201: new Piece(201, "Wind Horse", "風馬", [99,1,0,0,2,0,0,1]),
    202: new Piece(202, "Vertical Bear", "竪熊", [99,0,2,0,1,0,2,0]),
    203: new Piece(203, "Vertical Soldier", "竪兵", [99,0,2,0,1,0,2,0]),
    204: new Piece(204, "Tiger Soldier", "虎兵", [2,99,0,0,1,0,0,99]), // PROMOTE
    205: new Piece(205, "Earth Dragon", "地龍", [2,1,0,99,1,99,0,1]),
    206: new Piece(206, "Silver Chariot", "銀車", [99,2,0,1,99,1,0,2]),
    207: new Piece(207, "Stone Chariot", "石車", [99,1,2,0,99,0,2,1]),
    208: new Piece(208, "Side Soldier", "横兵", [2,0,99,0,1,0,99,0]),
    209: new Piece(209, "Gold Chariot", "金車", [99,1,2,1,99,1,2,1]),
    210: new Piece(210, "Boar Soldier", "猪兵", [99,99,2,0,1,0,2,99]),
    211: new Piece(211, "Leopard Soldier", "豹兵", [99,99,2,0,1,0,2,99]),
    212: new Piece(212, "Bear Soldier", "熊兵", [99,99,2,0,1,0,2,99]),
    213: new Piece(213, "Free Pup", "奔狗", [99,99,2,1,99,1,2,99]),
    214: new Piece(214, "Free Ox", "奔牛", [99,99,2,1,99,1,2,99]), // PROMOTE
    215: new Piece(215, "Free Horse", "奔馬", [99,99,2,1,99,1,2,99]), // PROMOTE
    216: new Piece(216, "Free Pig", "奔豚", [99,99,2,1,99,1,2,99]), // PROMOTE
    217: new Piece(217, "Little Standard", "小旗", [99,2,99,1,99,1,99,2]),
    218: new Piece(218, "Copper Chariot", "銅車", [99,3,0,0,99,0,0,3]),
    219: new Piece(219, "Forest Demon", "森鬼", [3,99,3,0,99,0,3,99]),
    220: new Piece(220, "Great Dragon", "大龍", [3,99,0,99,3,99,0,99]),
    221: new Piece(221, "Center Standard", "中旗", [99,3,99,3,99,3,99,3]),
    222: new Piece(222, "Front Standard", "前旗", [99,3,99,3,99,3,99,3]),
    223: new Piece(223, "Great Dove", "大鳩", [3,99,3,99,3,99,3,99]),
    224: new Piece(224, "Great Standard", "大旗", [99,99,99,3,99,3,99,99]),
    225: new Piece(225, "Vertical Wolf", "竪狼", [99,0,1,0,3,0,1,0]),
    226: new Piece(226, "Side Serpent", "横蛇", [3,0,99,0,1,0,99,0]),
    227: new Piece(227, "Cloud Eagle", "雲鷲", [99,3,1,0,99,0,1,3]),
    228: new Piece(228, "Goose Wing", "鴻翼", [99,1,3,1,99,1,3,1]), // PROMOTE
    229: new Piece(229, "Horse Soldier", "馬兵", [99,99,3,0,1,0,3,99]),
    230: new Piece(230, "Ox Soldier", "牛兵", [99,99,3,0,1,0,3,99]),
    231: new Piece(231, "Spear General", "鎗将", [99,0,3,0,2,0,3,0]), // PROMOTE
    232: new Piece(232, "Cannon General", "炮将", [99,99,3,0,2,0,3,99]), // PROMOTE
    233: new Piece(233, "Beast Bird", "獣鳥", [99,99,3,99,2,99,3,99]), // PROMOTE
    234: new Piece(234, "Fowl", "禽鳥", [99,99,3,99,2,99,3,99]), // PROMOTE
    235: new Piece(235, "Great Leopard", "大豹", [99,3,2,0,1,0,2,3]), // PROMOTE
    236: new Piece(236, "Longbow Soldier", "弩兵", [99,5,2,0,1,0,2,5]),
    237: new Piece(237, "Thunder Runner", "雷走", [99,99,4,0,4,0,4,99]), // PROMOTE
    238: new Piece(238, "Fire Dragon", "火龍", [99,4,99,2,99,2,99,4]),
    239: new Piece(239, "Water Dragon", "水龍", [99,2,99,4,99,4,99,2]),
    240: new Piece(240, "Longbow General", "弩将", [99,99,5,0,99,0,5,99]), // PROMOTE
    241: new Piece(241, "Stone Peng", "石鵬", [0,99,5,99,0,99,5,99]), // PROMOTE
    242: new Piece(242, "Mount Tai", "泰山", [5,99,5,99,0,99,5,99]), // PROMOTE
    243: new Piece(243, "Free Demon", "奔鬼", [5,99,99,99,5,99,99,99]),
    244: new Piece(244, "Free Dream-Eater", "奔獏", [99,99,5,99,99,99,5,99]),
    245: new Piece(245, "Free Fire", "奔火", [5,99,99,99,5,99,99,99]), // PROMOTE
    246: new Piece(246, "Running Dragon", "走龍", [99,99,99,99,5,99,99,99]), // PROMOTE
    247: new Piece(247, "Great Shark", "大鱗", [99,5,99,2,99,2,99,5]), // PROMOTE
    248: new Piece(248, "Crossbow General", "弓将", [99,5,3,0,2,0,3,5]), // PROMOTE
    249: new Piece(249, "Playful Parrot", "遊䳇", [99,3,5,2,99,2,5,3]), // PROMOTE
    // Jumping Pieces
    250: new Piece(250, "Cassia Horse", "桂馬", NONE, [
        [2, 1, NONE], [2, -1, NONE]
    ]),
    251: new Piece(251, "Flying Dragon", "飛龍", NONE, [
        [2, 2, NONE], [2, -2, NONE],
        [-2, -2, NONE], [-2, 2, NONE]
    ]),
    252: new Piece(252, "Kirin", "麒麟", [1,1,0,1,1,1,0,1], [
        [0, 2, NONE], [0, -2, NONE]
    ]),
    253: new Piece(253, "Phoenix", "鳳凰", [1,0,1,0,1,0,1,0], [
        [2, 2, NONE], [2, -2, NONE],
        [-2, -2, NONE], [-2, 2, NONE]
    ]),
    254: new Piece(254, "Flying Cat", "飛猫", [0,0,0,1,1,1,0,0], [
        [0, 3, NONE], [3, 3, NONE], [3, 0, NONE],
        [0, -3, NONE], [3, -3, NONE]
    ]),
    255: new Piece(255, "Running Horse", "走馬", [99,99,0,0,1,0,0,99], [
        [-2, 2, NONE], [-2, -2, NONE]
    ]),
    256: new Piece(256, "Mountain Hawk", "山鷹", [99,99,99,2,99,2,99,99], [
        [2, 0, NONE]
    ]),
    257: new Piece(257, "Little Turtle", "小亀", [99,99,2,99,99,99,2,99], [
        [2, 0, NONE], [-2, 0, NONE]
    ]),
    258: new Piece(258, "Great Stag", "大鹿", [99,0,99,2,99,2,99,0], [
        [2, 2, NONE], [2, -2, NONE]
    ]),
    259: new Piece(259, "Left Mountain Eagle", "左鷲", [99,99,2,99,99,99,99,99], [
        [-2, -2, NONE], [2, -2, NONE]
    ]),
    260: new Piece(260, "Right Mountain Eagle", "右鷲", [99,99,99,99,99,2,99,99], [
        [-2, 2, NONE], [2, 2, NONE]
    ]),
    261: new Piece(261, "Kirin Master", "麟師", [99,99,3,99,99,99,3,99], [
        [3, 0, NONE], [-3, 0, NONE]
    ]),
    262: new Piece(262, "Great Turtle", "大亀", [99,99,3,99,99,99,3,99], [
        [3, 0, NONE], [-3, 0, NONE]
    ]),
    263: new Piece(263, "Phoenix Master", "鳳師", [99,99,3,99,99,99,3,99], [
        [3, 3, NONE], [3, -3, NONE]
    ]),
    264: new Piece(264, "Great Master", "大師", [99,99,5,5,99,5,5,99], [
        [3, 0, NONE], [3, 3, NONE], [3, -3, NONE]
    ]),
    265: new Piece(265, "Horned Hawk", "角鷹", Q, [
        [2, 0, NONE]
    ]),
    266: new Piece(266, "Flying Eagle", "飛鷲", Q, [
        [2, 2, NONE], [2, -2, NONE]
    ]),
    267: new Piece(267, "Roaring Dog", "吼犬", [99,99,99,3,99,3,99,99], [
        [3, 0, NONE], [3, 3, NONE],
        [0, 3, NONE],
        [-3, 0, NONE],
        [0, -3, NONE], [3, -3, NONE]
    ]),
    268: new Piece(268, "Lion Dog", "狛犬", Q, [
        [3, 0, NONE], [3, 3, NONE],
        [0, 3, NONE], [-3, 3, NONE],
        [-3, 0, NONE], [-3, -3, NONE],
        [0, -3, NONE], [3, -3, NONE]
    ]),
    269: new Piece(269, "Great Dream-Eater", "大獏", Q, [ // PROMOTE
        [0, 3, NONE], [0, -3, NONE]
    ]), 
    270: new Piece(270, "Heavenly Horse", "天馬", [99,0,0,0,0,0,0,0], [ // PROMOTE
        [2, 1, NONE], [-2, 1, NONE],
        [-2, -1, NONE], [2, -1, NONE]
    ]), 
    271: new Piece(271, "Spirit Turtle", "霊亀", Q, [ // PROMOTE
        [3, 0, NONE], [0, 3, NONE],
        [-3, 0, NONE], [0, -3, NONE]
    ]),
    272: new Piece(272, "Treasure Turtle", "宝亀", Q, [ // PROMOTE
        [2, 0, NONE], [0, 2, NONE],
        [-2, 0, NONE], [0, -2, NONE]
    ]),
    273: new Piece(273, "Wooden Dove", "鳩槃", [2,99,2,99,2,99,2,99], [
        [3, 3, [0,2,0,0,0,0,0,0]], [-3, 3, [0,0,0,2,0,0,0,0]],
        [-3, -3, [0,0,0,0,0,2,0,0]], [3, -3, [0,0,0,0,0,0,0,2]]
    ]),
    274: new Piece(274, "Center Master", "中師", [99,99,3,3,99,3,3,99], [
        [2, 0, NONE], [2, 2, NONE],
        [-2, 0, NONE],
        [2, -2, NONE]
    ]),
    275: new Piece(275, "Peng Master", "鵬師", [99,99,5,5,99,5,5,99], [
        [3, 3, NONE], [3, -3, NONE]
    ]),
    276: new Piece(276, "Free Eagle", "奔鷲", Q, [
        [2, 0, [99,0,0,0,0,0,0,0]], [3, 0, [99,0,0,0,0,0,0,0]],
        [2, 2, [0,99,0,0,0,0,0,0]], [3, 3, [0,99,0,0,0,0,0,0]], [4, 4, [0,99,0,0,0,0,0,0]],
        [0, 2, [0,0,99,0,0,0,0,0]], [0, 3, [0,0,99,0,0,0,0,0]],
        [-2, 2, [0,0,0,99,0,0,0,0]], [-3, 3, [0,0,0,99,0,0,0,0]],
        [-2, 0, [0,0,0,0,99,0,0,0]], [-3, 0, [0,0,0,0,99,0,0,0]],
        [-2, -2, [0,0,0,0,0,99,0,0]], [-3, -3, [0,0,0,0,0,99,0,0]],
        [0, -2, [0,0,0,0,0,0,99,0]], [0, -3, [0,0,0,0,0,0,99,0]],
        [2, -2, [0,0,0,0,0,0,0,99]], [3, -3, [0,0,0,0,0,0,0,99]], [4, -4, [0,0,0,0,0,0,0,99]]
    ]),
    277: new Piece(277, "Free Bird", "奔翅", [99,99,99,3,99,3,99,99], [], [0,3,0,0,0,0,0,3]), // PROMOTE
    278: new Piece(278, "Great Hawk", "大鷹", Q, [ // PROMOTE
        [2, 0, [99,0,0,0,0,0,0,0]]
    ]),
    279: new Piece(279, "King of Teachings", "教王", Q, [], [3,3,3,3,3,3,3,3]), // PROMOTE
    280: new Piece(280, "Mountain Crane", "山鶻", Q, [ // PROMOTE
        [2, 0, [99,0,0,0,0,0,0,0]], [3, 0, [99,0,0,0,0,0,0,0]],
        [2, 2, [0,99,0,0,0,0,0,0]], [3, 3, [0,99,0,0,0,0,0,0]],
        [0, 2, [0,0,99,0,0,0,0,0]], [0, 3, [0,0,99,0,0,0,0,0]],
        [-2, 2, [0,0,0,99,0,0,0,0]], [-3, 3, [0,0,0,99,0,0,0,0]],
        [-2, 0, [0,0,0,0,99,0,0,0]], [-3, 0, [0,0,0,0,99,0,0,0]],
        [-2, -2, [0,0,0,0,0,99,0,0]], [-3, -3, [0,0,0,0,0,99,0,0]],
        [0, -2, [0,0,0,0,0,0,99,0]], [0, -3, [0,0,0,0,0,0,99,0]],
        [2, -2, [0,0,0,0,0,0,0,99]], [3, -3, [0,0,0,0,0,0,0,99]],
    ]), 
    281: new Piece(281, "Great Eagle", "大鷲", Q, [ // PROMOTE
        [2, 2, [0,99,0,0,0,0,0,0]], [2, -2, [0,0,0,0,0,0,0,99]]
    ]),
    282: new Piece(282, "Great Elephant", "大象", [99,3,99,99,99,99,99,3], [], [3,0,3,3,3,3,3,0]), // PROMOTE
    283: new Piece(283, "Gold Bird", "金翅", [99,99,3,3,99,3,3,99], [], [0,3,0,0,0,0,0,3]),
    284: new Piece(284, "Ancient Dragon", "元龍", [99,99,0,99,99,99,0,99], [], [3,0,0,0,3,0,0,0]), // PROMOTE
    285: new Piece(285, "Rain Demon", "霖鬼", [3,99,2,2,99,2,2,99], [ // PROMOTE
        [2, 2, [0,99,0,0,0,0,0,0]], [2, -2, [0,0,0,0,0,0,0,99]]
    ]),
    286: new Piece(286, "Flying General", "飛将", [99,0,99,0,99,0,99,0]),
    287: new Piece(287, "Angle General", "角将", [0,99,0,99,0,99,0,99]),
    288: new Piece(288, "Fierce Dragon", "猛龍", [2,99,2,99,2,99,2,99]),
    289: new Piece(289, "Flying Crocodile", "飛鰐", [99,3,99,2,99,2,99,3]), // PROMOTE
    290: new Piece(290, "Vice General", "副将", [0,99,0,99,0,99,0,99], [
        [2, 0, NONE], [0, 2, NONE],
        [-2, 0, NONE], [0, -2, NONE]
    ]),
    291: new Piece(291, "Great General", "大将", Q),
    292: new Piece(292, "Hook Mover", "鉤行", [99,0,99,0,99,0,99,0]),
    293: new Piece(293, "Tengu", "天狗", [0,99,0,99,0,99,0,99]),
    294: new Piece(294, "Capricorn", "摩羯", [1,99,1,99,1,99,1,99]),
    295: new Piece(295, "Peacock", "孔雀", [0,99,0,2,0,2,0,99]),
    296: new Piece(296, "Heavenly Tetrarch King", "天王", Q, [ // PROMOTE
        [2, 0, [99,0,0,0,0,0,0,0]], [2, 2, [0,99,0,0,0,0,0,0]], 
        [0, 2, [0,0,99,0,0,0,0,0]], [-2, 2, [0,0,0,99,0,0,0,0]], 
        [-2, 0, [0,0,0,0,99,0,0,0]], [-2, -2, [0,0,0,0,0,99,0,0]], 
        [0, -2, [0,0,0,0,0,0,99,0]], [2, -2, [0,0,0,0,0,0,0,99]]
    ]),
    297: new Piece(297, "Lion", "獅子", [1,1,1,1,1,1,1,1], [
        [2, 0, NONE], [2, 1, NONE], [2, 2, NONE],
        [1, 2, NONE], [0, 2, NONE], [-1, 2, NONE], [-2, 2, NONE],
        [-2, 1, NONE], [-2, 0, NONE], [-2, -1, NONE], [-2, -2, NONE],
        [-1, -2, NONE], [0, -2, NONE], [1, -2, NONE], [2, -2, NONE],
        [2, -1, NONE]
    ]),
    298: new Piece(298, "Furious Fiend", "奮迅", [3,3,3,3,3,3,3,3], [
        [2, 0, NONE], [2, 1, NONE], [2, 2, NONE],
        [1, 2, NONE], [0, 2, NONE], [-1, 2, NONE], [-2, 2, NONE],
        [-2, 1, NONE], [-2, 0, NONE], [-2, -1, NONE], [-2, -2, NONE],
        [-1, -2, NONE], [0, -2, NONE], [1, -2, NONE], [2, -2, NONE],
        [2, -1, NONE]
    ]),
    299: new Piece(299, "Buddhist Spirit", "法性", Q, [
        [2, 0, NONE], [2, 1, NONE], [2, 2, NONE],
        [1, 2, NONE], [0, 2, NONE], [-1, 2, NONE], [-2, 2, NONE],
        [-2, 1, NONE], [-2, 0, NONE], [-2, -1, NONE], [-2, -2, NONE],
        [-1, -2, NONE], [0, -2, NONE], [1, -2, NONE], [2, -2, NONE],
        [2, -1, NONE]
    ]),
    300: new Piece(300, "Lion Hawk", "獅鷹", [0,99,0,99,0,99,0,99], [
        [2, 0, NONE], [2, 1, NONE], [2, 2, NONE],
        [1, 2, NONE], [0, 2, NONE], [-1, 2, NONE], [-2, 2, NONE],
        [-2, 1, NONE], [-2, 0, NONE], [-2, -1, NONE], [-2, -2, NONE],
        [-1, -2, NONE], [0, -2, NONE], [1, -2, NONE], [2, -2, NONE],
        [2, -1, NONE]
    ])
}

const PROMOTE = {
    1: 25,
    2: 48,
    3: 31,
    4: 48,
    5: 48,
    6: 91,
    7: 227,
    8: 90,
    9: 161,
    10: 48,
    11: 71,
    12: 139,
    13: 140,
    14: 21,
    15: 220,
    16: 98,
    17: 121,
    18: 164,
    19: 41,
    20: 135,
    21: 33,
    22: 21,
    23: 103,
    24: 88,
    25: 92,
    26: 40,
    27: 100,
    28: 145,
    29: 145,
    30: 165,
    31: 39,
    32: 222,
    34: 279,
    35: 299,
    36: 145,
    37: 157,
    38: 158,
    42: 48,
    43: 189,
    44: 189,
    45: 113,
    46: 73,
    47: 190,
    48: 190,
    49: 243,
    50: 155,
    51: 297,
    52: 268,
    53: 273,
    54: 283,
    55: 95,
    56: 292,
    57: 293,
    58: 266,
    59: 74,
    60: 74,
    61: 72,
    62: 215,
    63: 214,
    64: 170,
    65: 171,
    66: 291,
    67: 290,
    68: 242,
    69: 74,
    70: 74,
    72: 234,
    73: 233,
    75: 194,
    76: 213,
    77: 216,
    78: 258,
    80: 223,
    81: 248,
    82: 232,
    83: 89,
    84: 143,
    85: 152,
    86: 99,
    87: 246,
    89: 192,
    92: 163,
    93: 109,
    94: 162,
    95: 110,
    99: 107,
    100: 164,
    104: 169,
    108: 280,
    113: 291,
    116: 126,
    117: 125,
    118: 164,
    119: 231,
    120: 79,
    124: 235,
    127: 204,
    128: 92,
    129: 173,
    130: 174,
    132: 106,
    133: 97,
    134: 180,
    135: 104,
    136: 104,
    137: 101,
    138: 104,
    139: 165,
    140: 96,
    141: 208,
    144: 87,
    147: 142,
    148: 159,
    149: 165,
    150: 123,
    151: 122,
    153: 184,
    154: 148,
    156: 108,
    160: 101,
    163: 266,
    164: 265,
    166: 111,
    167: 220,
    168: 220,
    172: 112,
    175: 112,
    176: 105,
    177: 89,
    178: 99,
    181: 183,
    182: 147,
    183: 186,
    184: 185,
    187: 114,
    188: 221,
    191: 109,
    193: 241,
    197: 296,
    198: 245,
    199: 269,
    201: 270,
    202: 105,
    203: 197,
    205: 168,
    206: 228,
    207: 179,
    208: 199,
    209: 249,
    210: 131,
    211: 102,
    212: 200,
    213: 195,
    217: 188,
    218: 146,
    219: 237,
    220: 284,
    221: 222,
    222: 224,
    223: 273,
    225: 160,
    226: 247,
    227: 115,
    229: 255,
    230: 196,
    236: 240,
    238: 261,
    239: 263,
    243: 113,
    244: 113,
    250: 208,
    251: 163,
    252: 283,
    253: 283,
    254: 92,
    255: 243,
    256: 265,
    257: 272,
    258: 114,
    259: 266,
    260: 266,
    262: 271,
    265: 278,
    266: 281,
    267: 268,
    268: 282,
    283: 277,
    286: 289,
    287: 285,
    288: 220,
    290: 291,
    294: 292,
    295: 293,
    297: 298
}

// BELOW IS IRRELEVANT FOR ONLY THE BOARD VIEWER!

// Material value of each piece, Pawn = 1. AUTOGENERATED!
const VALUES = {
    0: 0,         // Empty
    1: 1,         // Pawn
    2: 1.59,      // Earth General
    3: 1.39,      // Go-Between
    4: 1.6,       // Stone General
    5: 1.92,      // Iron General
    6: 2.28,      // Dog
    7: 2.04,      // Swooping Owl
    8: 2.25,      // Old Rat
    9: 2.12,      // Strutting Crow
    10: 1.94,     // Tile General
    11: 1.74,     // Sword Soldier
    12: 2.1,      // Copper General
    13: 2.13,     // Flying Goose
    14: 1.93,     // Climbing Monkey
    15: 2.56,     // Reclining Dragon
    16: 2.39,     // Coiled Serpent
    17: 2.07,     // Flying Chicken
    18: 2.57,     // Cat Sword
    19: 2.34,     // Evil Wolf
    20: 2.43,     // Silver General
    21: 2.33,     // Fierce Stag
    22: 2.22,     // Blind Dog
    23: 2.94,     // Huai Chicken
    24: 2.81,     // Old Monkey
    25: 3.06,     // Gold General
    26: 2.62,     // Fierce Wolf
    27: 2.79,     // Fierce Leopard
    28: 2.81,     // Blind Monkey
    29: 2.82,     // Blind Bear
    30: 3.17,     // Angry Boar
    31: 2.91,     // Drunken Elephant
    32: 3.61,     // Neighboring King
    33: 2.82,     // Rushing Boar
    34: 5.18,     // Deva
    35: 4.88,     // Dark Spirit
    36: 3.07,     // Blind Tiger
    37: 2.61,     // Left General
    38: 2.61,     // Right General
    39: 3.52,     // Crown Prince
    40: 3.05,     // Bear's Eyes
    41: 3.08,     // Poisonous Wolf
    42: 1.93,     // Wood General
    43: 3.3,      // Donkey
    44: 3.27,     // Enchanted Badger
    45: 3.58,     // Flying Horse
    46: 4.06,     // Beast Cadet
    47: 4.74,     // Fragrant Elephant
    48: 4.81,     // White Elephant
    49: 3.91,     // Rushing Bird
    50: 2.96,     // Fierce Bear
    51: 4,        // Eastern Barbarian
    52: 4.14,     // Western Barbarian
    53: 3.57,     // Northern Barbarian
    54: 4.17,     // Southern Barbarian
    55: 3.32,     // Prancing Stag
    56: 6,        // Poisonous Serpent
    57: 5.88,     // Old Kite
    58: 4.65,     // Fierce Eagle
    59: 3.5,      // Guardian of the Gods
    60: 3.37,     // Sumo Wrestler
    61: 4.76,     // Fowl Cadet
    62: 3.21,     // Horse General
    63: 3.22,     // Ox General
    64: 3.28,     // Wind General
    65: 3.26,     // River General
    66: 4.74,     // Fire General
    67: 4.98,     // Water General
    68: 3.39,     // Mountain General
    69: 3.37,     // Buddhist Devil
    70: 3.15,     // Nature Spirit
    71: 3.05,     // Sword General
    72: 5.27,     // Fowl Officer
    73: 5.3,      // Beast Officer
    74: 5.89,     // Heavenly Tetrarch
    75: 3.15,     // Chicken General
    76: 3.11,     // Pup General
    77: 3.65,     // Pig General
    78: 4.51,     // Mountain Stag
    79: 6.62,     // Leopard King
    80: 4.16,     // Turtle Dove
    81: 4.32,     // Crossbow Soldier
    82: 4.95,     // Cannon Soldier
    83: 3.07,     // Incense Chariot
    84: 2.47,     // Ox Chariot
    85: 2.55,     // Fierce Tiger
    86: 3.38,     // Reverse Chariot
    87: 5.77,     // Side Dragon
    88: 6.62,     // Mountain Witch
    89: 7.01,     // White Foal
    90: 6.81,     // Mockingbird
    91: 6.88,     // Multi General
    92: 6.84,     // Flying Chariot
    93: 7.02,     // Soldier
    94: 6.75,     // Running Chariot
    95: 7.04,     // Square Mover
    96: 6.51,     // Gliding Swallow
    97: 5.79,     // Free Serpent
    98: 5.8,      // Coiled Dragon
    99: 6.37,     // Whale
    100: 5.25,    // Angle Mover
    101: 6.64,    // Free Wolf
    102: 6.58,    // Running Leopard
    103: 7.35,    // Wizard Stork
    104: 8.27,    // Flying Ox
    105: 8.07,    // Free Bear
    106: 8.16,    // Free Leopard
    107: 8.21,    // Great Whale
    108: 11.71,   // Treacherous Fox
    109: 8.75,    // Cavalier
    110: 8.78,    // Strong Chariot
    111: 8.67,    // Free Dragon
    112: 8.54,    // Free Tiger
    113: 20.01,   // Free King
    114: 9.87,    // Free Stag
    115: 9.89,    // Strong Eagle
    116: 2.57,    // Howling Dog (left)
    117: 2.56,    // Howling Dog (right)
    118: 3.91,    // Vertical Horse
    119: 3.24,    // Spear Soldier
    120: 3.82,    // Vertical Pup
    121: 3.27,    // Raiding Hawk
    122: 4.09,    // Right Iron Chariot
    123: 4.14,    // Left Iron Chariot
    124: 3.79,    // Vertical Leopard
    125: 3.54,    // Right Dog
    126: 3.57,    // Left Dog
    127: 3.66,    // Ram's-head Soldier
    128: 4.18,    // Flying Swallow
    129: 3.76,    // Wood Chariot
    130: 3.74,    // Tile Chariot
    131: 3.66,    // Running Boar
    132: 4.64,    // Running Pup
    133: 4.08,    // Running Serpent
    134: 3.91,    // Earth Chariot
    135: 4.61,    // Vertical Mover
    136: 4.78,    // Fierce Ox
    137: 3.91,    // Side Wolf
    138: 4.11,    // Side Ox
    139: 3.98,    // Side Mover
    140: 3.85,    // Swallow's Wings
    141: 3.74,    // Side Monkey
    142: 5.53,    // Divine Sparrow
    143: 4.17,    // Plodding Ox
    144: 4.1,     // Side Flyer
    145: 4.63,    // Flying Stag
    146: 4.66,    // Copper Elephant
    147: 4.86,    // Vermillion Sparrow
    148: 4.79,    // Turtle Snake
    149: 4.87,    // Side Boar
    150: 4.74,    // Left Chariot
    151: 4.76,    // Right Chariot
    152: 4.49,    // Great Tiger
    153: 4.75,    // Right Tiger
    154: 4.55,    // Left Tiger
    155: 5.41,    // Great Bear
    156: 6.86,    // Running Rabbit
    157: 5.4,     // Left Army
    158: 5.5,     // Right Army
    159: 5.59,    // Divine Turtle
    160: 5.97,    // Running Wolf
    161: 5.93,    // Flying Hawk
    162: 7.53,    // Cannon Chariot
    163: 8.36,    // Dragon King
    164: 7.74,    // Dragon Horse
    165: 7.21,    // Free Boar
    166: 7.84,    // Wind Dragon
    167: 7.58,    // Cloud Dragon
    168: 6.36,    // Rain Dragon
    169: 8.81,    // Fire Ox
    170: 8.8,     // Fierce Wind
    171: 8.63,    // Huai River
    172: 3.73,    // Vertical Tiger
    173: 4.21,    // Wind Snapping Turtle
    174: 4.19,    // Running Tile
    175: 5.14,    // Running Tiger
    176: 5.08,    // Running Bear
    177: 4.58,    // Gold Stag
    178: 3.61,    // Silver Rabbit
    179: 5.02,    // Walking Heron
    180: 4.97,    // Reed Bird
    181: 4.72,    // Right Dragon
    182: 4.51,    // Left Dragon
    183: 6.01,    // Blue Dragon
    184: 5.86,    // White Tiger
    185: 6.55,    // Divine Tiger
    186: 6.43,    // Divine Dragon
    187: 7.07,    // Running Stag
    188: 8.31,    // Rear Standard
    189: 7.42,    // Ceramic Dove
    190: 7.49,    // Elephant King
    191: 7.96,    // Horseman
    192: 7.74,    // Great Foal
    193: 8.53,    // Woodland Demon
    194: 8.47,    // Free Chicken
    195: 8.5,     // Free Dog
    196: 7.59,    // Running Ox
    197: 9.73,    // Chariot Soldier
    198: 8.85,    // Fire Demon
    199: 9.17,    // Water Ox
    200: 9.71,    // Strong Bear
    201: 3.18,    // Wind Horse
    202: 4.47,    // Vertical Bear
    203: 4.7,     // Vertical Soldier
    204: 3.94,    // Tiger Soldier
    205: 4.31,    // Earth Dragon
    206: 4.99,    // Silver Chariot
    207: 4.87,    // Stone Chariot
    208: 4.54,    // Side Soldier
    209: 5.71,    // Gold Chariot
    210: 5.68,    // Boar Soldier
    211: 5.97,    // Leopard Soldier
    212: 6.72,    // Bear Soldier
    213: 8.33,    // Free Pup
    214: 8.39,    // Free Ox
    215: 8.35,    // Free Horse
    216: 8.39,    // Free Pig
    217: 8.12,    // Little Standard
    218: 4.69,    // Copper Chariot
    219: 6.87,    // Forest Demon
    220: 7.87,    // Great Dragon
    221: 8.49,    // Center Standard
    222: 8.81,    // Front Standard
    223: 7.99,    // Great Dove
    224: 9.52,    // Great Standard
    225: 3.8,     // Vertical Wolf
    226: 4.76,    // Side Serpent
    227: 6.32,    // Cloud Eagle
    228: 5.7,     // Goose Wing
    229: 6,       // Horse Soldier
    230: 6.44,    // Ox Soldier
    231: 3.92,    // Spear General
    232: 6.31,    // Cannon General
    233: 8.79,    // Beast Bird
    234: 8.81,    // Fowl
    235: 4.52,    // Great Leopard
    236: 5.61,    // Longbow Soldier
    237: 6.74,    // Thunder Runner
    238: 8.78,    // Fire Dragon
    239: 8.64,    // Water Dragon
    240: 8.07,    // Longbow General
    241: 7.08,    // Stone Peng
    242: 7.76,    // Mount Tai
    243: 9.31,    // Free Demon
    244: 9.4,     // Free Dream-Eater
    245: 9.17,    // Free Fire
    246: 9.73,    // Running Dragon
    247: 8.69,    // Great Shark
    248: 5.31,    // Crossbow General
    249: 6.97,    // Playful Parrot
    250: 1.57,    // Cassia Horse
    251: 2.69,    // Flying Dragon
    252: 4.52,    // Kirin
    253: 4.77,    // Phoenix
    254: 3.86,    // Flying Cat
    255: 6.57,    // Running Horse
    256: 9.83,    // Mountain Hawk
    257: 9.66,    // Little Turtle
    258: 8.38,    // Great Stag
    259: 9.98,    // Left Mountain Eagle
    260: 10.18,   // Right Mountain Eagle
    261: 9.54,    // Kirin Master
    262: 9.85,    // Great Turtle
    263: 9.58,    // Phoenix Master
    264: 9.83,    // Great Master
    265: 10.21,   // Horned Hawk
    266: 10.43,   // Flying Eagle
    267: 11,      // Roaring Dog
    268: 12.59,   // Lion Dog
    269: 10.36,   // Great Dream-Eater
    270: 3.32,    // Heavenly Horse
    271: 10.77,   // Spirit Turtle
    272: 10.51,   // Treasure Turtle
    273: 8.54,    // Wooden Dove
    274: 9.49,    // Center Master
    275: 9.45,    // Peng Master
    276: 23.36,   // Free Eagle
    277: 13.66,   // Free Bird
    278: 10.32,   // Great Hawk
    279: 20.26,   // King of Teachings
    280: 13.79,   // Mountain Crane
    281: 10.56,   // Great Eagle
    282: 16.35,   // Great Elephant
    283: 12.97,   // Gold Bird
    284: 11.77,   // Ancient Dragon
    285: 8.18,    // Rain Demon
    286: 54.31,   // Flying General
    287: 26.18,   // Angle General
    288: 32.36,   // Fierce Dragon
    289: 55.82,   // Flying Crocodile
    290: 40.03,   // Vice General
    291: 75.72,   // Great General
    292: 37.43,   // Hook Mover
    293: 20.21,   // Tengu
    294: 28.23,   // Capricorn
    295: 15.61,   // Peacock
    296: 12.06,   // Heavenly Tetrarch King
    297: 10.59,   // Lion
    298: 11.45,   // Furious Fiend
    299: 17.81,   // Buddhist Spirit
    300: 15.34,   // Lion Hawk
    1000: 100000, // King
}

// Material value, not including promotion possibilities
// Not exist in this object = same as in VALUES (or the piece cannot promote at all)
const VALUES_PROMOTED = {
    1: 0.76,    // Pawn
    2: 1.16,    // Earth General
    3: 1.16,    // Go-Between
    4: 1.16,    // Stone General
    5: 1.54,    // Iron General
    6: 1.55,    // Dog
    7: 1.52,    // Swooping Owl
    8: 1.53,    // Old Rat
    9: 1.52,    // Strutting Crow
    10: 1.57,   // Tile General
    11: 1.53,   // Sword Soldier
    12: 1.89,   // Copper General
    13: 1.92,   // Flying Goose
    14: 1.88,   // Climbing Monkey
    15: 1.9,    // Reclining Dragon
    16: 1.85,   // Coiled Serpent
    17: 1.87,   // Flying Chicken
    18: 1.86,   // Cat Sword
    19: 2.22,   // Evil Wolf
    20: 2.25,   // Silver General
    21: 2.25,   // Fierce Stag
    22: 2.21,   // Blind Dog
    23: 2.25,   // Huai Chicken
    24: 2.2,    // Old Monkey
    25: 2.51,   // Gold General
    26: 2.55,   // Fierce Wolf
    27: 2.48,   // Fierce Leopard
    28: 2.53,   // Blind Monkey
    29: 2.53,   // Blind Bear
    30: 2.53,   // Angry Boar
    31: 2.82,   // Drunken Elephant
    32: 2.82,   // Neighboring King
    34: 2.79,   // Deva
    35: 2.83,   // Dark Spirit
    36: 2.82,   // Blind Tiger
    37: 2.16,   // Left General
    38: 2.15,   // Right General
    42: 1.49,   // Wood General
    43: 2.56,   // Donkey
    44: 2.52,   // Enchanted Badger
    45: 2.42,   // Flying Horse
    46: 3.97,   // Beast Cadet
    47: 4.24,   // Fragrant Elephant
    48: 4.32,   // White Elephant
    49: 2.96,   // Rushing Bird
    50: 2.52,   // Fierce Bear
    51: 2.83,   // Eastern Barbarian
    52: 2.8,    // Western Barbarian
    53: 2.78,   // Northern Barbarian
    54: 2.81,   // Southern Barbarian
    55: 2.81,   // Prancing Stag
    56: 3,      // Poisonous Serpent
    57: 3.28,   // Old Kite
    58: 3.61,   // Fierce Eagle
    59: 3.01,   // Guardian of the Gods
    60: 2.86,   // Sumo Wrestler
    61: 4.77,   // Fowl Cadet
    62: 2.16,   // Horse General
    63: 2.16,   // Ox General
    64: 2.16,   // Wind General
    65: 2.16,   // River General
    66: 2.37,   // Fire General
    67: 2.49,   // Water General
    68: 2.49,   // Mountain General
    69: 2.86,   // Buddhist Devil
    70: 2.71,   // Nature Spirit
    72: 4.55,   // Fowl Officer
    73: 4.59,   // Beast Officer
    75: 1.93,   // Chicken General
    76: 1.93,   // Pup General
    77: 2.56,   // Pig General
    78: 3.8,    // Mountain Stag
    80: 3.22,   // Turtle Dove
    81: 4.07,   // Crossbow Soldier
    82: 4.57,   // Cannon Soldier
    83: 2.01,   // Incense Chariot
    84: 1.99,   // Ox Chariot
    85: 2,      // Fierce Tiger
    86: 2.68,   // Reverse Chariot
    87: 4.66,   // Side Dragon
    89: 6.8,    // White Foal
    92: 6.57,   // Flying Chariot
    93: 6.54,   // Soldier
    94: 6.54,   // Running Chariot
    95: 6.55,   // Square Mover
    99: 5.85,   // Whale
    100: 4.74,  // Angle Mover
    104: 8.12,  // Flying Ox
    108: 11.12, // Treacherous Fox
    113: 10.01, // Free King
    116: 2.29,  // Howling Dog (left)
    117: 2.28,  // Howling Dog (right)
    118: 3.02,  // Vertical Horse
    119: 3.05,  // Spear Soldier
    120: 3.04,  // Vertical Pup
    124: 3.59,  // Vertical Leopard
    127: 3.58,  // Ram's-head Soldier
    128: 3.51,  // Flying Swallow
    129: 3.63,  // Wood Chariot
    130: 3.61,  // Tile Chariot
    132: 3.65,  // Running Pup
    133: 3.6,   // Running Serpent
    134: 3.61,  // Earth Chariot
    135: 3.62,  // Vertical Mover
    136: 3.85,  // Fierce Ox
    137: 3.48,  // Side Wolf
    138: 3.48,  // Side Ox
    139: 3.47,  // Side Mover
    140: 3.43,  // Swallow's Wings
    144: 4.01,  // Side Flyer
    147: 4.67,  // Vermillion Sparrow
    148: 4.56,  // Turtle Snake
    149: 4.5,   // Side Boar
    150: 4.79,  // Left Chariot
    151: 4.81,  // Right Chariot
    153: 4.5,   // Right Tiger
    156: 5.66,  // Running Rabbit
    160: 5.78,  // Running Wolf
    163: 7.79,  // Dragon King
    164: 7.05,  // Dragon Horse
    166: 7.61,  // Wind Dragon
    167: 7.64,  // Cloud Dragon
    168: 6.3,   // Rain Dragon
    172: 2.37,  // Vertical Tiger
    175: 4.18,  // Running Tiger
    176: 4.23,  // Running Bear
    177: 3.95,  // Gold Stag
    178: 3.21,  // Silver Rabbit
    181: 4.39,  // Right Dragon
    182: 4.46,  // Left Dragon
    183: 5.89,  // Blue Dragon
    184: 5.67,  // White Tiger
    187: 6.28,  // Running Stag
    188: 8.26,  // Rear Standard
    191: 7.74,  // Horseman
    193: 8.64,  // Woodland Demon
    197: 9.08,  // Chariot Soldier
    198: 8.76,  // Fire Demon
    199: 8.84,  // Water Ox
    201: 3.14,  // Wind Horse
    202: 3.45,  // Vertical Bear
    203: 3.46,  // Vertical Soldier
    205: 3.96,  // Earth Dragon
    206: 4.78,  // Silver Chariot
    207: 4.82,  // Stone Chariot
    208: 3.76,  // Side Soldier
    209: 5.35,  // Gold Chariot
    210: 5.85,  // Boar Soldier
    211: 5.8,   // Leopard Soldier
    212: 5.88,  // Bear Soldier
    213: 8.28,  // Free Pup
    217: 8.08,  // Little Standard
    220: 6.77,  // Great Dragon
    221: 8.45,  // Center Standard
    222: 8.61,  // Front Standard
    223: 7.84,  // Great Dove
    225: 3.24,  // Vertical Wolf
    226: 3.96,  // Side Serpent
    227: 5.31,  // Cloud Eagle
    229: 6.01,  // Horse Soldier
    230: 6.11,  // Ox Soldier
    236: 4.91,  // Longbow Soldier
    238: 8.57,  // Fire Dragon
    239: 8.38,  // Water Dragon
    243: 9.11,  // Free Demon
    244: 9.23,  // Free Dream-Eater
    250: 1.17,  // Cassia Horse
    251: 1.77,  // Flying Dragon
    252: 3.21,  // Kirin
    253: 3.33,  // Phoenix
    254: 3.31,  // Flying Cat
    255: 5.85,  // Running Horse
    256: 9.73,  // Mountain Hawk
    257: 9.42,  // Little Turtle
    258: 7.96,  // Great Stag
    259: 9.86,  // Left Mountain Eagle
    260: 10.12, // Right Mountain Eagle
    262: 9.59,  // Great Turtle
    265: 10.18, // Horned Hawk
    266: 10.39, // Flying Eagle
    267: 10.85, // Roaring Dog
    268: 11.52, // Lion Dog
    283: 12.77, // Gold Bird
    286: 53.88, // Flying General
    287: 27.6,  // Angle General
    288: 34.39, // Fierce Dragon
    290: 29.96, // Vice General
    294: 25.64, // Capricorn
    295: 14.31, // Peacock
    297: 10.44, // Lion
}

const material_value = (piece) => (piece.promoted && VALUES_PROMOTED[piece.id] !== undefined)
    ? VALUES_PROMOTED[piece.id]
    : VALUES[piece.id]

// THE STUPID GAMEFILE 
const magic = [0x54, 0x4B, 0x59, 0x4E, 0x4E, 0x53, 0x50, 0x00]
let LOADED_SAVE = {}

function u16(bytes) {
    // Helper function to convert two-byte slices into a single number.
    return bytes[0] + bytes[1] * 256
}

function u32(bytes) {
    // Helper function to convert four-byte slices into a single number.
    return bytes[0] + bytes[1] * 256 + bytes[2] * 256 * 256 + bytes[3] * 256 * 256 * 256
}

function u64(bytes) {
    // Helper function to convert eight-byte slices into a single number.
    // NOTE THAT THIS IS UNSAFE ABOVE MAX SAFE INTEGER!
    const val = (
        bytes[0] + 
        bytes[1] * 256 + 
        bytes[2] * 256 ** 2 + 
        bytes[3] * 256 ** 3 +
        bytes[4] * 256 ** 4 +
        bytes[5] * 256 ** 5 +
        bytes[6] * 256 ** 6 +
        bytes[7] * 256 ** 7
    )
    // This is equivalent to val > MAX_SAFE_INTEGER = 2^53 - 1. Above this roundoff errors make some integer values not representable.
    if (bytes[7] !== 0 || bytes[6] > 31) console.warn("U64 conversion has produced a number above maximum safe integer.")
    return val
}

let loaded_games; 
document.getElementById("gamefile-upload").addEventListener("change", async (e) => {
    // Me when I need to const
    const file = e.target.files[0]
    if (!file) return
    if (file.size > (2 ** 30)) {
        alert("File is too big!")
        return
    }
    if (file.name.slice(-4) != ".tkn") {
        alert("Invalid file extension!")
        return
    }

    const buffer = await file.arrayBuffer()
    const bytes = new Uint8Array(buffer)

    const valid_file = magic.every((v, i) => bytes[i] === v);
    if (!valid_file) {
        alert("Invalid file! (Magic number mismatch. The file may be corrupted)")
        return
    }

    // ew magic offsets
    const HEADER_SIZE = u32(bytes.slice(12, 16))
    const GAME_HEADER_SIZE = u32(bytes.slice(16, 20))
    const RECORD_SIZE = u32(bytes.slice(20, 24))
    const num_games = u32(bytes.slice(40, 48))
    
    loaded_games = {}

    let offset_so_far = HEADER_SIZE
    for (let game = 0; game < num_games; game++) {
        let N_PLIES = u32(bytes.slice(offset_so_far, offset_so_far + 4))
        let result = (bytes[offset_so_far + 4] << 24) >> 24 // 1 = black win, 0 = draw, -1 = white win
        let termination_outcome = bytes[offset_so_far + 5]
        offset_so_far += GAME_HEADER_SIZE

        let positions = []
        let moves = []
        for (let j = 0; j < N_PLIES; j++) {
            let m = u32(bytes.subarray(offset_so_far + 2592, offset_so_far + 2596))
            let origin = (m >>> 17) & 0x7FF
            positions.push(new Uint16Array(buffer, offset_so_far, 1296))
            moves.push([Math.floor(origin / 36), origin % 36, (m >>> 28) * 1296 + ((m >>> 6) & 0x7FF)])
            offset_so_far += RECORD_SIZE
        }

        let game_obj = new Game(positions, moves, {
            "result": result,
            "termination": termination_outcome,
            "plycount": N_PLIES
        })
        loaded_games[game] = game_obj
    }

    document.getElementById("viewer-control").style.display = "flex";
    for (let el of document.getElementsByClassName("game-control-button")) el.disabled = true;
    game_index_selected = 0
    loaded_games[0].load()
    // Handle picking the same file again
    e.target.value = ""
});

function decode_board(packed) {
    // Builds a board from 1296 packed squares
    let decoded = Array.from({ length: 36 }, () => Array(36).fill(0));
    for (let k = 0; k < 1296; k++) {
        let code = packed[k]
        let x = Math.floor(k / 36)
        let y = k % 36
        if (code >> 2 == 301) {
            decoded[x][y] = PIECES[1000].copy(x, y)
        } else {
            decoded[x][y] = PIECES[code >> 2].copy(x, y)
        }

        if ((code & 2) == 0) {
            decoded[x][y].invert_color()
        }

        if ((code & 1) == 1) {
            decoded[x][y].promoted = true
        }
    }
    return decoded
}

function encode_board(b) {
    // Packs b.board into 1296 squares: index << 2 | color << 1 | promoted
    let packed = new Uint16Array(1296)
    for (let x = 0; x < 36; x++) {
        for (let y = 0; y < 36; y++) {
            let piece = b.board[x][y]
            packed[x * 36 + y] = ((piece.id == 1000 ? 301 : piece.id) << 2) | (piece.color << 1) | (piece.promoted ? 1 : 0)
        }
    }
    return packed
}

function export_game() {
    // only works for PLAYED games!
    // Writes game_hist as a one-game .tkn (version 7), no search or values
    const HEADER_SIZE = 256
    const GAME_HEADER_SIZE = 16
    const RECORD_SIZE = 2608
    const plies = game_hist.positions.length
    if (plies == 0) {
        alert("No moves to export!")
        return
    }

    const bytes = new Uint8Array(HEADER_SIZE + GAME_HEADER_SIZE + plies * RECORD_SIZE)
    const view = new DataView(bytes.buffer)

    // File header
    bytes.set(magic, 0)
    view.setUint32(8, 7, true)              // version
    view.setUint32(12, HEADER_SIZE, true)
    view.setUint32(16, GAME_HEADER_SIZE, true)
    view.setUint32(20, RECORD_SIZE, true)
    view.setUint32(24, 1000, true)          // piece values are in 1/1000 pawn
    view.setUint32(28, 3, true)             // mat_shift: material is stored x 8
    view.setBigUint64(40, 1n, true)         // games
    view.setBigUint64(48, BigInt(plies), true)
    view.setUint32(84, 36, true)            // board size
    view.setUint32(88, 302, true)           // piece types
    bytes.set(Array.from("human", c => c.charCodeAt(0)), 104)

    // Game header
    const last = game_hist.positions[plies - 1]
    const kings = [0, 0]                    // [white, black]
    for (let code of last) {
        if (code >> 2 == 301) kings[(code >> 1) & 1] += 1
    }
    const result = kings[0] == 0 ? 1 : (kings[1] == 0 ? -1 : 0) // 1 = black win, 0 = draw, -1 = white win
    view.setUint32(HEADER_SIZE, plies, true)
    view.setInt8(HEADER_SIZE + 4, result)
    view.setUint8(HEADER_SIZE + 5, result == 0 ? 4 : 0) // 0 = royal capture, 4 = ply cap (unfinished)
    view.setInt16(HEADER_SIZE + 6, result * 10000, true)

    // Records (value, legal moves and evaluations stay 0)
    let prev = encode_board(new Board())
    for (let n = 0; n < plies; n++) {
        const at = HEADER_SIZE + GAME_HEADER_SIZE + n * RECORD_SIZE
        const packed = game_hist.positions[n]
        const [x, y, move] = game_hist.moves[n]
        const origin = x * 36 + y
        const target = move % 1296

        let material = 0
        let pieces_before = 0
        let pieces_after = 0
        for (let k = 0; k < 1296; k++) {
            const code = packed[k]
            const id = code >> 2
            const value = ((code & 1) && VALUES_PROMOTED[id] !== undefined) ? VALUES_PROMOTED[id] : (VALUES[id] ?? 0)
            material += ((code >> 1) & 1) ? value : -value
            if (id != 0) pieces_after += 1
            if ((prev[k] >> 2) != 0) pieces_before += 1
        }
        const mover = (packed[target] >> 1) & 1
        const captured = pieces_after < pieces_before
        const promoted = (packed[target] & 1) == 1 && (prev[origin] & 1) == 0
        const flags = (captured ? 1 : 0) | (promoted ? 2 : 0) | (n == plies - 1 ? 8 : 0)

        bytes.set(new Uint8Array(packed.buffer, packed.byteOffset, 2592), at)
        view.setUint32(at + 2592, (Math.floor(move / 1296) << 28 | origin << 17 | target << 6) >>> 0, true)
        view.setInt16(at + 2602, Math.round(material * 8), true)
        view.setUint8(at + 2604, mover)
        view.setUint8(at + 2605, flags)
        prev = packed
    }

    const link = document.createElement("a")
    link.href = URL.createObjectURL(new Blob([bytes]))
    link.download = "game.tkn"
    link.click()
    setTimeout(() => URL.revokeObjectURL(link.href), 1000)
}

function load_next_game() {
    const next_game = loaded_games[game_index_selected + 1]
    if (next_game === undefined) {
        return
    }

    game_index_selected += 1
    document.getElementById("viewer-game-no").textContent = `Game ${game_index_selected + 1}`
    next_game.load()
}

function load_prev_game() {
    const prev_game = loaded_games[game_index_selected - 1]
    if (prev_game === undefined) {
        return
    }

    game_index_selected -= 1
    document.getElementById("viewer-game-no").textContent = `Game ${game_index_selected + 1}`
    prev_game.load()
}