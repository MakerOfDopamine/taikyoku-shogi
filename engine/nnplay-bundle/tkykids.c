/* Legal moves and child boards for Python, through ctypes: the policy trainer uses it to
 * score children of positions whose self-play file has no .kids side file.
 *
 *   gcc -O2 -shared -fPIC -o libtkykids.so tkykids.c -lm      (needs tky.c and tables.h)
 *
 * Boards are the .tkn's: 1296 u16 codes, absolute frame, square x*36 + y. Moves use the
 * ply-record packing, special << 28 | origin << 17 | target << 6. Not thread-safe. */
#define TKY_NO_MAIN
#include "tky.c"

static Pos P, C;
static Gen G;

int tk_init(void)
{
    tables_init();
    memset(&G, 0, sizeof(G));
    G.cap = 8192;
    G.buf = malloc((size_t)G.cap * 4);
    G.stamp = calloc(MAXMOVE, 2);
    G.kline = calloc(NSQ, 2);
    G.ksq = -1;
    return G.buf && G.stamp && G.kline ? 0 : -1;
}

static void load(const uint16_t *board, int turn)
{
    pos_init(&P);
    for (int i = 0; i < NSQ; i++) sq_set(&P, SQ2P[i], board[i]);
    P.turn = turn;
}

static uint32_t to_file(uint32_t v)
{
    int o = UNPACK_O(v), mv = UNPACK_M(v);
    return ((uint32_t)(mv / 1296) << 28) | ((uint32_t)o << 17) | ((uint32_t)(mv % 1296) << 6);
}

/* Every legal move for `turn` (1 black, 0 white), up to cap of them into out; returns how
 * many there are. *forced is the King capture forced regicide would play, or 0. */
int tk_moves(const uint16_t *board, int turn, uint32_t *out, int cap, uint32_t *forced)
{
    load(board, turn);
    gen_all(&P, &G);
    king_ctx_set(&G, &P);
    uint32_t f = find_king_capture(&G);
    *forced = f ? to_file(f) : 0;
    for (int i = 0; i < G.n && i < cap; i++) out[i] = to_file(G.buf[i]);
    return G.n;
}

/* The board after each of n moves, n * 1296 codes into out. */
void tk_children(const uint16_t *board, int turn, const uint32_t *moves, int n, uint16_t *out)
{
    load(board, turn);
    for (int k = 0; k < n; k++) {
        uint32_t m = moves[k];
        int origin = (int)((m >> 17) & 0x7FF), mv = (int)((m >> 28) & 0xF) * 1296 + (int)((m >> 6) & 0x7FF);
        memcpy(&C, &P, sizeof(Pos));
        do_move(&C, SQ2P[origin], mv);
        for (int i = 0; i < NSQ; i++) out[(size_t)k * NSQ + i] = C.code[SQ2P[i]];
    }
}
