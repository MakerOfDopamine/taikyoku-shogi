/* Taikyoku Shogi random-playout corpus generator.
 *
 * Rules are a straight transliteration of game.js (Board / Piece / Game). Every table
 * comes from tables.h, which gen_tables.js produces by executing game.js.
 *
 * Build: see build.sh
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#include "tables.h"

#define NSQ 1296
#define MAXMOVE 12960

/* ---- packed square codes ------------------------------------------------- */
/* code = idx<<2 | color<<1 | promoted */
#define CODE(idx, color, pr) (((idx) << 2) | ((color) << 1) | (pr))
#define C_IDX(c)   ((c) >> 2)
#define C_COLOR(c) (((c) >> 1) & 1)
#define C_PROM(c)  ((c) & 1)
static const uint16_t EMPTY_CODE = CODE(EMPTY_IDX, 1, 0); /* game.js: empties are colour 1 */

/* side[] is the hot array: 0 = white, 1 = black, 2 = empty */
#define SIDE_EMPTY 2

static const int8_t DIRX[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
static const int8_t DIRY[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };

#define OOB(v) ((unsigned)(v) > 35u)

#ifdef STATS
enum { CT_DYDX, CT_TP, CT_EAGLE, CT_TRAMPLE, CT_HOOK1, CT_HOOK2, CT_PEACOCK, CT_LION, CT_N };
static const char *CT_NAME[CT_N] = {"dydx","tp","eagle","trample","hook-leg1","hook-leg2","peacock","lion"};
static __thread uint64_t CT_EMIT[CT_N], CT_STEP[CT_N], CT_DUP[CT_N];
static __thread int CT_CUR;
static uint64_t CT_EMIT_G[CT_N], CT_STEP_G[CT_N], CT_DUP_G[CT_N], CT_PIECES[CT_N];
static __thread uint64_t CT_PC[CT_N];
static pthread_mutex_t CT_LOCK = PTHREAD_MUTEX_INITIALIZER;
#define CT(x) CT_CUR = (x)
#define CTSTEP() (CT_STEP[CT_CUR]++)
#else
#define CT(x) ((void)0)
#define CTSTEP() ((void)0)
#endif

static uint8_t LVL[NPIECE];      /* level by piece index */
static uint8_t IS_SIMPLE[NDEF];  /* no dedup needed */

/* ---- results ------------------------------------------------------------- */
enum { RES_WHITE = 0, RES_BLACK = 1, RES_DRAW = 2 };
enum { TERM_ROYAL = 0, TERM_STALEMATE = 1, TERM_REPETITION = 2, TERM_NOPROGRESS = 3, TERM_PLYCAP = 4 };

/* ---- position ------------------------------------------------------------ */
typedef struct {
    uint16_t code[NSQ];
    uint8_t  side[NSQ];
    uint16_t plist[2][512];   /* squares occupied by each colour */
    int16_t  pat[NSQ];        /* square -> slot in plist, -1 if empty */
    int      pn[2];
    int      turn;
    int      npieces;
    int      royals[2];
    int32_t  material;        /* black - white, VALUE_FP units */
    uint64_t hash;
} Pos;

static const uint64_t TURN_KEY = 0xC0FFEE1234567891ull;

static inline uint64_t zkey(int sq, int code)
{
    uint64_t z = (uint64_t)sq * 8192u + (uint64_t)code + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline void sq_set(Pos *p, int sq, uint16_t nc)
{
    uint16_t oc = p->code[sq];
    if (oc == nc) return;
    p->hash ^= zkey(sq, oc) ^ zkey(sq, nc);

    int oi = C_IDX(oc), ni = C_IDX(nc);
    if (oi != EMPTY_IDX) {
        int c = C_COLOR(oc);
        p->material -= c ? MATVAL[oc] : -MATVAL[oc];
        p->npieces--;
        if (oi == KING_IDX) p->royals[c]--;
        int slot = p->pat[sq], last = --p->pn[c];
        uint16_t moved = p->plist[c][last];
        p->plist[c][slot] = moved;
        p->pat[moved] = (int16_t)slot;
        p->pat[sq] = -1;
    }
    if (ni != EMPTY_IDX) {
        int c = C_COLOR(nc);
        p->material += c ? MATVAL[nc] : -MATVAL[nc];
        p->npieces++;
        if (ni == KING_IDX) p->royals[c]++;
        p->pat[sq] = (int16_t)p->pn[c];
        p->plist[c][p->pn[c]++] = (uint16_t)sq;
        p->side[sq] = (uint8_t)c;
    } else {
        p->side[sq] = SIDE_EMPTY;
    }
    p->code[sq] = nc;
}

static void pos_init(Pos *p)
{
    memset(p, 0, sizeof(*p));
    for (int i = 0; i < NSQ; i++) { p->code[i] = EMPTY_CODE; p->side[i] = SIDE_EMPTY; p->pat[i] = -1; }
    p->hash = 0;
    for (int i = 0; i < NSQ; i++) p->hash ^= zkey(i, EMPTY_CODE);
    for (int i = 0; i < NSQ; i++) if (C_IDX(INIT_BOARD[i]) != EMPTY_IDX) sq_set(p, i, INIT_BOARD[i]);
    p->turn = 1;                        /* black moves first */
    p->hash ^= TURN_KEY;                /* turn == 1 */
}

/* ---- move generation ----------------------------------------------------- */
typedef struct {
    uint32_t *buf;
    int       n;
    int       cap;
    uint16_t *stamp;   /* MAXMOVE entries */
    uint16_t  epoch;
    int       overflow;
} Gen;

/* packed: origin<<14 | move   (origin < 1296 fits 11 bits, move < 12960 fits 14) */
#define PACK(o, m) (((uint32_t)(o) << 14) | (uint32_t)(m))
#define UNPACK_O(v) ((int)((v) >> 14))
#define UNPACK_M(v) ((int)((v) & 0x3FFF))

static void gen_grow(Gen *g)
{
    g->cap *= 2;
    g->buf = (uint32_t *)realloc(g->buf, (size_t)g->cap * 4);
    if (!g->buf) { fprintf(stderr, "oom in move buffer\n"); exit(1); }
}

#ifdef STATS
#define PUSH(mv) do { \
        if (g->n == g->cap) gen_grow(g); \
        g->buf[g->n++] = PACK(base, (mv)); CT_EMIT[CT_CUR]++; \
    } while (0)
#define PUSHU(mv) do { \
        int _m = (mv); \
        if (g->stamp[_m] != g->epoch) { g->stamp[_m] = g->epoch; PUSH(_m); } else CT_DUP[CT_CUR]++; \
    } while (0)
#else
#define PUSH(mv) do { \
        if (g->n == g->cap) gen_grow(g); \
        g->buf[g->n++] = PACK(base, (mv)); \
    } while (0)
#define PUSHU(mv) do { \
        int _m = (mv); \
        if (g->stamp[_m] != g->epoch) { g->stamp[_m] = g->epoch; PUSH(_m); } \
    } while (0)
#endif

static void gen_piece(const Pos *p, Gen *g, int base, int simple)
{
    const uint8_t *side = p->side;
    uint16_t code = p->code[base];
    const PieceDef *def = &DEFS[code >> 1];
    int me = (int)C_COLOR(code), enemy = 1 - me;
    int bx = base / 36, by = base % 36;

    CT(CT_DYDX);
    /* --- dydx: per-direction slide with jump budget ------------------------ */
    for (int d = 0; d < 8; d++) {
        int rng = def->dydx[d];
        if (!rng) continue;
        int dj = def->djump[d];
        int x = bx, y = by, jumped = 0;
        for (int off = 1; off <= rng; off++) {
            x += DIRX[d]; y += DIRY[d];
            if (OOB(x) || OOB(y)) break;
            int q = x * 36 + y, s = side[q];
            CTSTEP();
            if (s == enemy)      { if (simple) PUSH(q); else PUSHU(q); jumped++; }
            else if (s == me)    { jumped++; }
            else                 { if (simple) PUSH(q); else PUSHU(q); }
            if (jumped > dj) break;
        }
    }
    if (simple) return;

    CT(CT_TP);
    /* --- tp: fixed-offset jump, then a non-jumping slide ------------------- */
    for (int k = 0; k < def->tp_n; k++) {
        const TpEntry *e = &TP_POOL[def->tp_off + k];
        int dx = bx + e->dx, dy = by + e->dy;
        if (OOB(dx) || OOB(dy)) continue;
        int dsq = dx * 36 + dy, ds = side[dsq];
        if (ds == me) continue;
        PUSHU(dsq);
        if (ds != SIDE_EMPTY) continue;    /* landed on an enemy: no secondary slide */
        for (int d = 0; d < 8; d++) {
            int rng = e->slide[d];
            if (!rng) continue;
            int x = dx, y = dy;
            for (int off = 1; off <= rng; off++) {
                x += DIRX[d]; y += DIRY[d];
                if (OOB(x) || OOB(y)) break;
                int q = x * 36 + y, s = side[q];
                if (s == enemy) { PUSHU(q); break; }
                if (s != SIDE_EMPTY) break;
                PUSHU(q);
            }
        }
    }

    switch (def->special) {
    case SP_NONE: break;

    case SP_FREE_EAGLE: { CT(CT_EAGLE);
        for (int d = 0; d < 8; d++) {
            int max = 3;
            if (me == 1) { if (DIRX[d] == 1 && DIRY[d] != 0) max = 4; }
            else         { if (DIRX[d] == -1 && DIRY[d] != 0) max = 4; }
            int ax = bx + DIRX[d], ay = by + DIRY[d];
            if (OOB(ax) || OOB(ay)) continue;
            if (side[ax * 36 + ay] != me) PUSHU((d + 1) * 1296 + base);   /* igui */
            int fx = bx + max * DIRX[d], fy = by + max * DIRY[d];
            if (OOB(fx) || OOB(fy)) continue;
            int has_f = 0, has_e = 0;
            for (int i = 1; i <= max; i++) {
                int s = side[(bx + i * DIRX[d]) * 36 + (by + i * DIRY[d])];
                if (s == me) { has_f = 1; break; }
                if (s == enemy) has_e = 1;
            }
            if (!has_e || has_f) continue;
            int jx = fx, jy = fy;
            PUSHU(9 * 1296 + jx * 36 + jy);
            jx += DIRX[d]; jy += DIRY[d];
            for (;;) {
                if (OOB(jx) || OOB(jy)) break;
                int q = jx * 36 + jy, s = side[q];
                if (s == me) break;
                PUSHU(9 * 1296 + q);
                if (s != SIDE_EMPTY) break;
                jx += DIRX[d]; jy += DIRY[d];
            }
        }
        break;
    }

    case SP_TRAMPLE_ORTHO:
    case SP_TRAMPLE_DIAG:
    case SP_TRAMPLE_ALL: {
        CT(CT_TRAMPLE);
        int d0 = (def->special == SP_TRAMPLE_DIAG) ? 1 : 0;
        int step = (def->special == SP_TRAMPLE_ALL) ? 1 : 2;
        int lvl = def->level;
        for (int d = d0; d < 8; d += step) {
            int x = bx + DIRX[d], y = by + DIRY[d];
            while (!OOB(x) && !OOB(y)) {
                int q = x * 36 + y;
                CTSTEP();
                if (LVL[C_IDX(p->code[q])] >= lvl) break;
                if (side[q] != me) PUSHU(9 * 1296 + q);
                x += DIRX[d]; y += DIRY[d];
            }
        }
        break;
    }

    case SP_HOOK_ORTHO:
    case SP_HOOK_DIAG: {
        CT(CT_HOOK1);
        int d0 = (def->special == SP_HOOK_DIAG) ? 1 : 0;
        int mid[160], nm = 0;
        for (int d = d0; d < 8; d += 2) {
            int x = bx + DIRX[d], y = by + DIRY[d];
            while (!OOB(x) && !OOB(y)) {
                int q = x * 36 + y, s = side[q];
                if (s == me) break;
                if (s != SIDE_EMPTY) { PUSHU(q); break; }
                mid[nm++] = q;
                x += DIRX[d]; y += DIRY[d];
            }
        }
        CT(CT_HOOK2);
        for (int i = 0; i < nm; i++) {
            int mx = mid[i] / 36, my = mid[i] % 36;
            for (int d = d0; d < 8; d += 2) {
                int x = mx + DIRX[d], y = my + DIRY[d];
                while (!OOB(x) && !OOB(y)) {
                    int q = x * 36 + y, s = side[q];
                    CTSTEP();
                    if (s == me) break;
                    PUSHU(q);
                    if (s != SIDE_EMPTY) break;
                    x += DIRX[d]; y += DIRY[d];
                }
            }
        }
        break;
    }

    case SP_PEACOCK: { CT(CT_PEACOCK);
        /* leg 1 runs on the two forward diagonals; leg 2 is perpendicular to the leg
         * that produced the midpoint, and which is which depends on colour. */
        int leg1[2] = { me == 0 ? 5 : 7, me == 0 ? 3 : 1 };  /* {-1,-1},{-1,1} / {1,-1},{1,1} */
        int list[2][80], nl[2] = { 0, 0 };
        for (int c = 0; c < 2; c++) {
            int d = leg1[c];
            int x = bx + DIRX[d], y = by + DIRY[d];
            while (!OOB(x) && !OOB(y)) {
                int q = x * 36 + y, s = side[q];
                if (s == me) break;
                if (s != SIDE_EMPTY) { PUSHU(q); break; }
                list[c][nl[c]++] = q;
                x += DIRX[d]; y += DIRY[d];
            }
        }
        /* midpoints of `right` (index 1) for colour 0, of `left` (0) otherwise */
        static const int leg2a[2] = { 5, 1 };   /* {-1,-1}, {1,1} */
        static const int leg2b[2] = { 7, 3 };   /* {1,-1}, {-1,1} */
        for (int pass = 0; pass < 2; pass++) {
            int which = (me == 0) ? (pass == 0 ? 1 : 0) : (pass == 0 ? 0 : 1);
            const int *dirs = (pass == 0) ? leg2a : leg2b;
            for (int i = 0; i < nl[which]; i++) {
                int mx = list[which][i] / 36, my = list[which][i] % 36;
                for (int k = 0; k < 2; k++) {
                    int d = dirs[k];
                    int x = mx + DIRX[d], y = my + DIRY[d];
                    while (!OOB(x) && !OOB(y)) {
                        int q = x * 36 + y, s = side[q];
                        if (s == me) break;
                        PUSHU(q);
                        if (s != SIDE_EMPTY) break;
                        x += DIRX[d]; y += DIRY[d];
                    }
                }
            }
        }
        break;
    }

    case SP_LION: { CT(CT_LION);
        for (int d = 0; d < 8; d++) {
            int mx = bx + DIRX[d], my = by + DIRY[d];
            if (OOB(mx) || OOB(my)) continue;
            if (side[mx * 36 + my] == me) continue;
            for (int e = 0; e < 8; e++) {
                int tx = mx + DIRX[e], ty = my + DIRY[e];
                if (OOB(tx) || OOB(ty)) continue;
                int q = tx * 36 + ty;
                if (side[q] == me && q != base) continue;   /* returning home is legal */
                PUSHU((d + 1) * 1296 + q);
            }
        }
        break;
    }
    }
}

static void gen_all(const Pos *p, Gen *g)
{
    g->n = 0;
    int c = p->turn, n = p->pn[c];
    const uint16_t *pl = p->plist[c];
    for (int i = 0; i < n; i++) {
        int sq = pl[i];
        int simple = IS_SIMPLE[p->code[sq] >> 1];
        if (!simple) {
            if (++g->epoch == 0) { memset(g->stamp, 0, MAXMOVE * 2); g->epoch = 1; }
        }
        gen_piece(p, g, sq, simple);
    }
}

/* ---- move execution (Board.move) ----------------------------------------- */
static inline void mov(Pos *p, int from, int to)
{
    uint16_t c = p->code[from];
    sq_set(p, to, c);
    sq_set(p, from, EMPTY_CODE);
}

/* returns 1 if the piece on `sq` promoted */
static inline int try_promote(Pos *p, int sq)
{
    uint16_t c = p->code[sq];
    if (C_PROM(c)) return 0;
    int np = PROMOTE_IDX[C_IDX(c)];
    if (np < 0) return 0;
    sq_set(p, sq, (uint16_t)CODE(np, C_COLOR(c), 1));
    return 1;
}

/* applies exactly what Board.move does; returns 1 if the mover promoted */
static int do_move(Pos *p, int origin, int mv)
{
    int special = mv / 1296;
    int t = mv % 1296;
    int was_promoted = C_PROM(p->code[origin]);

    if (special == 0) {
        mov(p, origin, t);
    } else if (special <= 8) {
        int d = special - 1;
        int mx = origin / 36 + DIRX[d], my = origin % 36 + DIRY[d];
        if (OOB(mx) || OOB(my)) { fprintf(stderr, "off-board intermediate\n"); exit(1); }
        int m = mx * 36 + my;
        mov(p, origin, m);
        mov(p, m, t);
    } else {
        int ox = origin / 36, oy = origin % 36, tx = t / 36, ty = t % 36;
        int dx = tx - ox, dy = ty - oy;
        int ux = (dx > 0) - (dx < 0), uy = (dy > 0) - (dy < 0);
        int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
        int size = ax > ay ? ax : ay;
        sq_set(p, t, p->code[origin]);                       /* target written first */
        for (int i = 0; i < size; i++) sq_set(p, (ox + i * ux) * 36 + (oy + i * uy), EMPTY_CODE);
    }

    int c = C_COLOR(p->code[t]);
    int tx = t / 36;
    int promoted_now = 0;
    if ((c == 1 && tx >= 25) || (c == 0 && tx <= 10)) promoted_now = try_promote(p, t);
    p->turn ^= 1;
    p->hash ^= TURN_KEY;
    return promoted_now && !was_promoted;
}

/* ---- repetition table (per game, epoch stamped) -------------------------- */
#define REP_BITS 14
#define REP_SIZE (1 << REP_BITS)
#define REP_MASK (REP_SIZE - 1)
typedef struct {
    uint64_t key[REP_SIZE];
    uint32_t ep[REP_SIZE];
    uint16_t cnt[REP_SIZE];
    uint32_t epoch;
} RepTab;

static inline int rep_bump(RepTab *r, uint64_t h)
{
    size_t i = (size_t)(h >> (64 - REP_BITS));
    for (;;) {
        if (r->ep[i] != r->epoch) { r->ep[i] = r->epoch; r->key[i] = h; r->cnt[i] = 1; return 1; }
        if (r->key[i] == h) return ++r->cnt[i];
        i = (i + 1) & REP_MASK;
    }
}

/* ---- rng ----------------------------------------------------------------- */
typedef struct { uint64_t s[4]; } Rng;
static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rng_next(Rng *r)
{
    uint64_t *s = r->s, res = rotl64(s[0] + s[3], 23) + s[0], t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl64(s[3], 45);
    return res;
}
static void rng_seed(Rng *r, uint64_t seed)
{
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ull;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        r->s[i] = z ^ (z >> 31);
    }
}
/* unbiased-enough (Lemire) index below n */
static inline uint32_t rng_below(Rng *r, uint32_t n)
{
    uint64_t m = (uint64_t)(uint32_t)rng_next(r) * n;
    uint32_t l = (uint32_t)m;
    if (l < n) {
        uint32_t th = (uint32_t)(-(int32_t)n) % n;
        while (l < th) { m = (uint64_t)(uint32_t)rng_next(r) * n; l = (uint32_t)m; }
    }
    return (uint32_t)(m >> 32);
}

/* ---- output format ------------------------------------------------------- */
#define TKY_MAGIC "TKYSHOGI"
#define TKY_VERSION 1u
#define PLY_REC 8
#define GAME_HDR 16
#define FILE_HDR 64

typedef struct {
    int fd;
    unsigned char *buf;
    size_t len, cap;
} Sink;

static void sink_flush(Sink *s)
{
    size_t off = 0;
    while (off < s->len) {
        ssize_t w = write(s->fd, s->buf + off, s->len - off);
        if (w <= 0) { perror("write"); exit(1); }
        off += (size_t)w;
    }
    s->len = 0;
}
static inline void sink_put(Sink *s, const void *d, size_t n)
{
    if (s->len + n > s->cap) sink_flush(s);
    memcpy(s->buf + s->len, d, n);
    s->len += n;
}

/* ---- worker -------------------------------------------------------------- */
typedef struct {
    int      id;
    uint64_t seed_base;
    uint64_t first_game;
    uint64_t n_games;
    int      max_plies;
    int      rep_limit;
    int      no_progress_limit;
    int      stalemate_loses;
    int      mat_shift;          /* material stored in VALUE_FP/2^k units */
    char     path[512];
    int      write_out;
    /* stats out */
    uint64_t plies, games_done;
    uint64_t res_count[3];
    uint64_t term_count[5];
    uint64_t maxlen;
    /* optional material sampling */
    int32_t *msamples;
    uint64_t msample_n, msample_cap;
    uint64_t nmoves_sum, nmoves_max;
} Worker;

typedef struct {
    uint32_t move;
    int16_t  mat;
    uint16_t nmoves;
} PlyRec;

static void *worker_main(void *arg)
{
    Worker *w = (Worker *)arg;
    Pos base, p;
    pos_init(&base);

    Gen g;
    g.cap = 4096;
    g.buf = (uint32_t *)malloc((size_t)g.cap * 4);
    g.stamp = (uint16_t *)calloc(MAXMOVE, 2);
    g.epoch = 0;
    g.n = 0;

    RepTab *rep = (RepTab *)calloc(1, sizeof(RepTab));
    rep->epoch = 0;

    Sink sink = { -1, NULL, 0, 1u << 22 };
    PlyRec *plybuf = (PlyRec *)malloc(sizeof(PlyRec) * (size_t)(w->max_plies + 2));
    if (w->write_out) {
        sink.fd = open(w->path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (sink.fd < 0) { perror(w->path); exit(1); }
        sink.buf = (unsigned char *)malloc(sink.cap);
        unsigned char hdr[FILE_HDR];
        memset(hdr, 0, sizeof(hdr));
        memcpy(hdr, TKY_MAGIC, 8);
        uint32_t v;
        v = TKY_VERSION;       memcpy(hdr + 8, &v, 4);
        v = FILE_HDR;          memcpy(hdr + 12, &v, 4);
        v = GAME_HDR;          memcpy(hdr + 16, &v, 4);
        v = PLY_REC;           memcpy(hdr + 20, &v, 4);
        v = VALUE_FP;          memcpy(hdr + 24, &v, 4);
        v = (uint32_t)w->mat_shift; memcpy(hdr + 28, &v, 4);
        memcpy(hdr + 32, &w->seed_base, 8);
        memcpy(hdr + 40, &w->first_game, 8);
        memcpy(hdr + 48, &w->n_games, 8);
        /* bytes 56..63 are patched at close with the real game / ply totals */
        sink_put(&sink, hdr, FILE_HDR);
    }

    Rng rng;
    int mat_div = VALUE_FP >> w->mat_shift;

    for (uint64_t gi = 0; gi < w->n_games; gi++) {
        uint64_t seed = w->seed_base ^ ((w->first_game + gi) * 0x9E3779B97F4A7C15ull);
        rng_seed(&rng, seed);
        memcpy(&p, &base, sizeof(Pos));
        rep->epoch++;
        rep_bump(rep, p.hash);

        int nply = 0, no_prog = 0;
        int result = RES_DRAW, term = TERM_PLYCAP;

        for (;;) {
            gen_all(&p, &g);
            int32_t m = p.material / mat_div;
            if (m > 32767) m = 32767; else if (m < -32767) m = -32767;
            plybuf[nply].mat = (int16_t)m;
            plybuf[nply].nmoves = (uint16_t)(g.n > 65535 ? 65535 : g.n);
            if (w->nmoves_max < (uint64_t)g.n) w->nmoves_max = (uint64_t)g.n;
            w->nmoves_sum += (uint64_t)g.n;
            if (w->msamples && w->msample_n < w->msample_cap && (nply & 7) == 0)
                w->msamples[w->msample_n++] = p.material;

            if (g.n == 0) {                       /* Game.has_legal_move == false */
                result = w->stalemate_loses ? (1 - p.turn) : RES_DRAW;
                term = TERM_STALEMATE;
                plybuf[nply].move = 0xFFFFFFFFu;
                break;
            }
            if (nply >= w->max_plies) {
                result = RES_DRAW; term = TERM_PLYCAP;
                plybuf[nply].move = 0xFFFFFFFFu;
                break;
            }

            uint32_t pick = g.buf[rng_below(&rng, (uint32_t)g.n)];
            int origin = UNPACK_O(pick), mv = UNPACK_M(pick);
            int mover = p.turn;
            int royals_before[2] = { p.royals[0], p.royals[1] };
            int pieces_before = p.npieces;

            int promoted_now = do_move(&p, origin, mv);
            int captured = pieces_before - p.npieces;

            plybuf[nply].move = ((uint32_t)(mv / 1296) << 28) | ((uint32_t)origin << 17)
                              | ((uint32_t)(mv % 1296) << 6)
                              | ((uint32_t)(captured > 0) << 5) | ((uint32_t)(promoted_now != 0) << 4);
            nply++;
            no_prog = (captured > 0 || promoted_now) ? 0 : no_prog + 1;

            if (p.royals[1 - mover] < royals_before[1 - mover]) { result = mover; term = TERM_ROYAL; }
            else if (p.royals[mover] < royals_before[mover])    { result = 1 - mover; term = TERM_ROYAL; }
            else {
                int seen = rep_bump(rep, p.hash);
                if (w->rep_limit > 0 && seen >= w->rep_limit) { result = RES_DRAW; term = TERM_REPETITION; }
                else if (w->no_progress_limit > 0 && no_prog >= w->no_progress_limit) { result = RES_DRAW; term = TERM_NOPROGRESS; }
                else continue;
            }
            /* terminal: record the final position with no move attached */
            int32_t fm = p.material / mat_div;
            if (fm > 32767) fm = 32767; else if (fm < -32767) fm = -32767;
            plybuf[nply].mat = (int16_t)fm;
            plybuf[nply].nmoves = 0;
            plybuf[nply].move = 0xFFFFFFFFu;
            break;
        }

        w->plies += (uint64_t)nply;
        w->games_done++;
        w->res_count[result]++;
        w->term_count[term]++;
        if ((uint64_t)nply > w->maxlen) w->maxlen = (uint64_t)nply;

        if (w->write_out) {
            unsigned char gh[GAME_HDR];
            uint32_t np = (uint32_t)nply;
            memcpy(gh, &np, 4);
            gh[4] = (unsigned char)result;
            gh[5] = (unsigned char)term;
            gh[6] = 0; gh[7] = 0;
            memcpy(gh + 8, &seed, 8);
            sink_put(&sink, gh, GAME_HDR);
            sink_put(&sink, plybuf, (size_t)(nply + 1) * PLY_REC);
        }
    }

    if (w->write_out) {
        sink_flush(&sink);
        unsigned char tail[8];
        uint32_t ng = (uint32_t)w->games_done;
        memcpy(tail, &ng, 4);
        uint32_t npl = (uint32_t)(w->plies > 0xFFFFFFFFu ? 0xFFFFFFFFu : w->plies);
        memcpy(tail + 4, &npl, 4);
        if (pwrite(sink.fd, tail, 8, 56) != 8) perror("pwrite header");
        if (pwrite(sink.fd, &w->games_done, 8, 48) != 8) perror("pwrite ngames");
        close(sink.fd);
        free(sink.buf);
    }
#ifdef STATS
    pthread_mutex_lock(&CT_LOCK);
    for (int i = 0; i < CT_N; i++) { CT_EMIT_G[i] += CT_EMIT[i]; CT_STEP_G[i] += CT_STEP[i]; CT_DUP_G[i] += CT_DUP[i]; CT_PIECES[i] += CT_PC[i]; }
    pthread_mutex_unlock(&CT_LOCK);
#endif
    free(plybuf); free(g.buf); free(g.stamp); free(rep);
    return NULL;
}

/* ---- diagnostics --------------------------------------------------------- */
static int cmp_i32(const void *a, const void *b);

static uint64_t perft(Pos *p, Gen *g, int depth)
{
    gen_all(p, g);
    if (depth == 1) return (uint64_t)g->n;
    int n = g->n;
    uint32_t *snap = (uint32_t *)malloc((size_t)n * 4);
    memcpy(snap, g->buf, (size_t)n * 4);
    uint64_t total = 0;
    Pos save = *p;
    for (int i = 0; i < n; i++) {
        do_move(p, UNPACK_O(snap[i]), UNPACK_M(snap[i]));
        total += perft(p, g, depth - 1);
        *p = save;
    }
    free(snap);
    return total;
}

static void tables_init(void)
{
    for (int i = 0; i < NPIECE; i++) LVL[i] = DEFS[i * 2 + 1].level;
    for (int i = 0; i < NDEF; i++) IS_SIMPLE[i] = DEFS[i].simple;
}

static double now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_i32(const void *a, const void *b)
{
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    tables_init();
    const char *mode = argc > 1 ? argv[1] : "help";

    if (!strcmp(mode, "perft")) {
        int depth = argc > 2 ? atoi(argv[2]) : 1;
        Pos p; pos_init(&p);
        Gen g = { malloc(4096 * 4), 0, 4096, calloc(MAXMOVE, 2), 0, 0 };
        for (int d = 1; d <= depth; d++) {
            double t0 = now_s();
            uint64_t n = perft(&p, &g, d);
            printf("perft(%d) = %llu   (%.2fs)\n", d, (unsigned long long)n, now_s() - t0);
        }
        /* also report the special-code breakdown at depth 1 */
        gen_all(&p, &g);
        uint64_t by[11] = { 0 };
        for (int i = 0; i < g.n; i++) by[UNPACK_M(g.buf[i]) / 1296]++;
        printf("special breakdown:");
        for (int i = 0; i < 11; i++) if (by[i]) printf(" %d:%llu", i, (unsigned long long)by[i]);
        printf("\npieces=%d royals=%d/%d material=%d\n", p.npieces, p.royals[0], p.royals[1], p.material);
        return 0;
    }

    if (!strcmp(mode, "dumpmoves")) {
        /* all legal moves for the side to move in the initial position, as origin:move */
        Pos p; pos_init(&p);
        Gen g = { malloc(4096 * 4), 0, 4096, calloc(MAXMOVE, 2), 0, 0 };
        if (argc > 2 && !strcmp(argv[2], "white")) { p.turn = 0; }
        gen_all(&p, &g);
        for (int i = 0; i < g.n; i++) printf("%d:%d\n", UNPACK_O(g.buf[i]), UNPACK_M(g.buf[i]));
        return 0;
    }

    if (!strcmp(mode, "probe")) {
        /* stdin: "turn T" then "S IDX COLOR PROM" lines; stdout: sorted origin:move list.
         * Used by probe.js to compare against game.js on arbitrary positions. */
        Pos p; pos_init(&p);
        for (int i = 0; i < NSQ; i++) sq_set(&p, i, EMPTY_CODE);
        char line[128];
        while (fgets(line, sizeof(line), stdin)) {
            int a, b, c, d;
            if (sscanf(line, "turn %d", &a) == 1) { if (a != p.turn) { p.turn = a; p.hash ^= TURN_KEY; } continue; }
            if (sscanf(line, "%d %d %d %d", &a, &b, &c, &d) == 4) sq_set(&p, a, (uint16_t)CODE(b, c, d));
        }
        Gen g = { malloc(4096 * 4), 0, 4096, calloc(MAXMOVE, 2), 0, 0 };
        gen_all(&p, &g);
        int n = g.n;
        qsort(g.buf, (size_t)n, 4, cmp_i32);
        printf("N %d\n", n);
        for (int i = 0; i < n; i++) printf("%d:%d\n", UNPACK_O(g.buf[i]), UNPACK_M(g.buf[i]));
        return 0;
    }

    if (!strcmp(mode, "trace")) {
        /* Play one seeded game, printing per ply the chosen move and a full sorted move
         * list digest, for cross-checking against game.js. */
        uint64_t seed = argc > 2 ? strtoull(argv[2], NULL, 10) : 1;
        int maxply = argc > 3 ? atoi(argv[3]) : 200;
        int verbose = argc > 4 ? atoi(argv[4]) : 0;
        int det = argc > 5 ? atoi(argv[5]) : 0;
        Pos p; pos_init(&p);
        Gen g = { malloc(4096 * 4), 0, 4096, calloc(MAXMOVE, 2), 0, 0 };
        RepTab *rep = calloc(1, sizeof(RepTab)); rep->epoch = 1; rep_bump(rep, p.hash);
        Rng rng; rng_seed(&rng, seed);
        for (int ply = 0; ply < maxply; ply++) {
            gen_all(&p, &g);
            if (g.n == 0) { printf("END stalemate ply=%d\n", ply); break; }
            uint32_t *snap = malloc((size_t)g.n * 4);
            memcpy(snap, g.buf, (size_t)g.n * 4);
            int n = g.n;
            qsort(snap, (size_t)n, 4, cmp_i32);
            uint64_t dig = 0xcbf29ce484222325ull;
            for (int i = 0; i < n; i++) { dig ^= snap[i]; dig *= 0x100000001b3ull; }
            printf("PLY %d turn=%d n=%d digest=%llu mat=%d pieces=%d\n",
                   ply, p.turn, n, (unsigned long long)dig, p.material, p.npieces);
            if (verbose) for (int i = 0; i < n; i++) printf("M %d %d\n", UNPACK_O(snap[i]), UNPACK_M(snap[i]));
            uint32_t *snap2 = snap;
            uint32_t pick = !det ? g.buf[rng_below(&rng, (uint32_t)n)]
                                 : snap2[((uint64_t)ply * 2654435761u + seed * 2246822519u + 12345u) % (uint64_t)n];
            int mover = p.turn, rb[2] = { p.royals[0], p.royals[1] };
            printf("PICK %d %d\n", UNPACK_O(pick), UNPACK_M(pick));
            do_move(&p, UNPACK_O(pick), UNPACK_M(pick));
            free(snap2);
            if (p.royals[1 - mover] < rb[1 - mover] || p.royals[mover] < rb[mover]) { printf("END royal ply=%d\n", ply + 1); break; }
            int seen = rep_bump(rep, p.hash);
            if (seen >= 4) { printf("END repetition ply=%d\n", ply + 1); break; }
        }
        return 0;
    }

    /* ---- gen / sample ---------------------------------------------------- */
    uint64_t games = 1000;
    int threads = 1, max_plies = 3000, rep_limit = 4, no_prog = 0, stalemate_loses = 1;
    int mat_shift = 3;
    uint64_t seed_base = 0x5EED5EED5EED5EEDull;
    const char *outdir = NULL;
    int sample = !strcmp(mode, "sample");

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--games") && i + 1 < argc) games = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--max-plies") && i + 1 < argc) max_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rep-limit") && i + 1 < argc) rep_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-progress") && i + 1 < argc) no_prog = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed_base = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mat-shift") && i + 1 < argc) mat_shift = atoi(argv[++i]);
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (!strcmp(mode, "help")) {
        fprintf(stderr,
            "usage: tky <gen|sample|perft|dumpmoves|trace> [options]\n"
            "  gen    --games N --threads T --out DIR [--max-plies N] [--rep-limit N]\n"
            "         [--no-progress N] [--seed S] [--mat-shift K]\n"
            "  sample --games N --threads T     (statistics only, no output files)\n");
        return 1;
    }

    Worker *ws = calloc((size_t)threads, sizeof(Worker));
    pthread_t *th = calloc((size_t)threads, sizeof(pthread_t));
    uint64_t per = games / (uint64_t)threads, extra = games % (uint64_t)threads, at = 0;
    for (int i = 0; i < threads; i++) {
        ws[i].id = i;
        ws[i].seed_base = seed_base;
        ws[i].first_game = at;
        ws[i].n_games = per + (i < (int)extra ? 1 : 0);
        at += ws[i].n_games;
        ws[i].max_plies = max_plies;
        ws[i].rep_limit = rep_limit;
        ws[i].no_progress_limit = no_prog;
        ws[i].stalemate_loses = stalemate_loses;
        ws[i].mat_shift = mat_shift;
        ws[i].write_out = outdir != NULL && !sample;
        if (ws[i].write_out) snprintf(ws[i].path, sizeof(ws[i].path), "%s/shard_%04d.tky", outdir, i);
        if (sample) { ws[i].msample_cap = 400000; ws[i].msamples = malloc(sizeof(int32_t) * ws[i].msample_cap); }
    }

    double t0 = now_s();
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker_main, &ws[i]);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    double dt = now_s() - t0;

    uint64_t tp = 0, tg = 0, res[3] = { 0 }, term[5] = { 0 }, maxlen = 0, nmsum = 0, nmmax = 0;
    for (int i = 0; i < threads; i++) {
        tp += ws[i].plies; tg += ws[i].games_done;
        for (int k = 0; k < 3; k++) res[k] += ws[i].res_count[k];
        for (int k = 0; k < 5; k++) term[k] += ws[i].term_count[k];
        if (ws[i].maxlen > maxlen) maxlen = ws[i].maxlen;
        nmsum += ws[i].nmoves_sum;
        if (ws[i].nmoves_max > nmmax) nmmax = ws[i].nmoves_max;
    }
    fprintf(stderr,
        "games=%llu plies=%llu (%.1f/game, max %llu) time=%.2fs\n"
        "  %.0f games/s, %.2f Mplies/s\n"
        "  result  black=%.4f white=%.4f draw=%.4f\n"
        "  term    royal=%.4f stalemate=%.4f repetition=%.4f noprogress=%.4f plycap=%.4f\n"
        "  moves/position mean=%.1f max=%llu\n",
        (unsigned long long)tg, (unsigned long long)tp, (double)tp / (double)tg, (unsigned long long)maxlen, dt,
        tg / dt, tp / dt / 1e6,
        (double)res[1] / tg, (double)res[0] / tg, (double)res[2] / tg,
        (double)term[0] / tg, (double)term[1] / tg, (double)term[2] / tg, (double)term[3] / tg, (double)term[4] / tg,
        (double)nmsum / (double)(tp + tg), (unsigned long long)nmmax);

#ifdef STATS
    { uint64_t te = 0, ts = 0, td = 0;
      for (int i = 0; i < CT_N; i++) { te += CT_EMIT_G[i]; ts += CT_STEP_G[i]; td += CT_DUP_G[i]; }
      fprintf(stderr, "  section          emits      dups     raysteps   pieces-touching\n");
      for (int i = 0; i < CT_N; i++)
          fprintf(stderr, "  %-10s %10.2fM %8.2fM %10.2fM  %6.2f%% of emits\n", CT_NAME[i],
                  CT_EMIT_G[i]/1e6, CT_DUP_G[i]/1e6, CT_STEP_G[i]/1e6, 100.0*CT_EMIT_G[i]/te);
      fprintf(stderr, "  TOTAL      %10.2fM %8.2fM %10.2fM\n", te/1e6, td/1e6, ts/1e6);
    }
#endif
    if (sample) {
        uint64_t n = 0;
        for (int i = 0; i < threads; i++) n += ws[i].msample_n;
        int32_t *all = malloc(sizeof(int32_t) * n);
        uint64_t at2 = 0;
        for (int i = 0; i < threads; i++) { memcpy(all + at2, ws[i].msamples, sizeof(int32_t) * ws[i].msample_n); at2 += ws[i].msample_n; }
        /* absolute values, for picking a tanh scale */
        int32_t *abs_ = malloc(sizeof(int32_t) * n);
        double sum = 0, sumsq = 0;
        for (uint64_t i = 0; i < n; i++) {
            double v = all[i] / (double)VALUE_FP;
            sum += v; sumsq += v * v;
            abs_[i] = all[i] < 0 ? -all[i] : all[i];
        }
        qsort(all, n, 4, cmp_i32);
        qsort(abs_, n, 4, cmp_i32);
        double mean = sum / n, sd = sqrt(sumsq / n - mean * mean);
        printf("MATSTAT n=%llu mean=%.3f sd=%.3f\n", (unsigned long long)n, mean, sd);
        const double qs[] = { 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99 };
        printf("MATQ");
        for (unsigned i = 0; i < sizeof(qs) / sizeof(qs[0]); i++)
            printf(" %.2f=%.3f", qs[i], all[(uint64_t)(qs[i] * (n - 1))] / (double)VALUE_FP);
        printf("\nABSQ");
        for (unsigned i = 0; i < sizeof(qs) / sizeof(qs[0]); i++)
            printf(" %.2f=%.3f", qs[i], abs_[(uint64_t)(qs[i] * (n - 1))] / (double)VALUE_FP);
        printf("\n");
        /* raw dump for offline fitting */
        FILE *f = fopen("material_samples.i32", "wb");
        if (f) { fwrite(all, 4, n, f); fclose(f); printf("wrote material_samples.i32 (%llu int32, VALUE_FP=%d)\n", (unsigned long long)n, VALUE_FP); }
    }
    return 0;
}
