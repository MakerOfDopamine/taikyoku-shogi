const out_of_bounds = (n) => (0 > n) || (35 < n)
const reverse = (n) => (35 - n)

class Board {
    constructor(base_board = null) {
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
        if (base_board != null) {
            this.board = base_board
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
        ]

        for (let i of pieces_to_add) {
            this.board[i[1]][i[2]] = PIECES[i[0]].copy(i[1], i[2])
            this.board[reverse(i[1])][reverse(i[2])] = PIECES[i[0]].copy(reverse(i[1]), reverse(i[2]))
            this.board[reverse(i[1])][reverse(i[2])].invert_color()
        }

        for (let i = 0; i < 36; i++) {
            for (let j = 0; j < 36; j++) {
                if (this.board[i][j] == 0) {
                    this.board[i][j] = PIECES[0].copy(i[1], i[2])
                }
            }
        }

        this.board[0][17] = PIECES[1000].copy(0, 17)
        this.board[35][18] = PIECES[11000].copy(35, 18)
    }
}

class Piece {
    constructor(id, name, short_name = "", dydx = [0, 0, 0, 0, 0, 0, 0, 0], tp = [], djump = [0, 0, 0, 0, 0, 0, 0, 0], color = 1, x = 0, y = 0) {
        this.id = id
        this.name = name
        this.short_name = short_name
        this.dydx = dydx
        this.tp = tp
        this.djump = djump
        this.color = color // 1 = BLACK, 0 = WHITE
        this.cache = [-1, -1, []]

        this.x = x
        this.y = y
    }

    copy(x, y) {
        return new Piece(this.id, this.name, this.short_name, this.dydx, this.tp, this.djump, this.color, x, y)
    }

    invert_color() {
        // MUTATIVE!
        this.color = 1 - this.color
        this.dydx = this.dydx.slice(4, 8).concat(this.dydx.slice(0, 4))
        this.djump = this.djump.slice(4, 8).concat(this.djump.slice(0, 4))
        this.tp = this.tp.map((value) => [-value[0], -value[1], value[2].slice(4, 8).concat(value[2].slice(0, 4))])
    }

    is_empty() {
        return this.id == 0
    }

    is_friendly(piece) {
        return (this.color == piece.color) && (!(this.id == 0 || piece.id == 0))
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

            while (offset <= this.dydx[count]) {
                working_x += dir[0]
                working_y += dir[1]
                if (out_of_bounds(working_x) || out_of_bounds(working_y)) {
                    break
                }
                if ((board.board[working_x][working_y].color == (1 - this.color)) && !board.board[working_x][working_y].is_empty()) {
                    moves.push(working_x * 36 + working_y)
                    break
                }
                if ((!board.board[working_x][working_y].is_empty()) || (board.board[working_x][working_y].is_friendly(this))) {
                    break
                }
                moves.push(working_x * 36 + working_y)
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

                if (board.board[dest_x][dest_y].is_friendly(this)) {
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
                        if ((board.board[working_x][working_y].color == (1 - this.color)) && !board.board[working_x][working_y].is_empty()) {
                            moves.push(working_x * 36 + working_y)
                            break
                        }
                        if ((!board.board[working_x][working_y].is_empty()) || (board.board[working_x][working_y].is_friendly(this))) {
                            break
                        }
                        moves.push(working_x * 36 + working_y)
                        offset += 1
                    }
                    count += 1
                }
            }
        }

        // special

        // return

        console.log(moves.includes(10 * 36 + 17))
        moves = [...new Set(moves)].map((value) => [Math.floor(value / 36), value % 36])
        return moves
    }
}

const NONE = [0,0,0,0,0,0,0,0]
const Q = [99,99,99,99,99,99,99,99]
const PIECES = {
    1000: new Piece(1000, "King", "王将", [2,2,2,2,2,2,2,2]),
    11000: new Piece(11000, "King", "玉将", [2,2,2,2,2,2,2,2], color=1),
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
    56: new Piece(56, "Poisonous Snake", "毒蛇", [2,1,2,0,1,0,2,1]),
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
}

