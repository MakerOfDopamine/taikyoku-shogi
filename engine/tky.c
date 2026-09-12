/* Taikyoku Shogi random-playout corpus generator.
 *
 * Rules are a transliteration of game.js (Board / Piece / Game). Every table comes from
 * tables.h, which gen_tables.js produces by executing game.js -- nothing is retyped.
 *
 * The generator is optimised but move-set equivalent: `tky_ref` (the straight
 * transliteration) and this file are diffed position-by-position by difftest.sh, and both
 * are diffed against game.js by probe.js / xcheck.js.
 *
 * Two representations are in play:
 *   - "sq36"  = x*36 + y, 0..1295. This is what move codes use, and what we write out.
 *   - "pad"   = (x+PAD)*W + (y+PAD) on a 44x44 board with an off-board sentinel ring.
 *     All ray walking happens here so a step is one add and one compare.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#include "tables.h"

#define NSQ     1296
#define MAXMOVE 12960
#define PAD     4
#define W       44
#define NPAD    (W * W)

#define CODE(idx, color, pr) (((idx) << 2) | ((color) << 1) | (pr))
#define C_IDX(c)   ((c) >> 2)
#define C_COLOR(c) (((c) >> 1) & 1)
#define C_PROM(c)  ((c) & 1)
static const uint16_t EMPTY_CODE = CODE(EMPTY_IDX, 1, 0); /* game.js: empties are colour 1 */

enum { SIDE_WHITE = 0, SIDE_BLACK = 1, SIDE_EMPTY = 2, SIDE_OFF = 3 };

static const int8_t DIRX[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
static const int8_t DIRY[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
static int16_t DSTEP[8];              /* direction as a pad-index delta */

static uint16_t P2SQ[NPAD];           /* pad -> sq36 (garbage off board; never read there) */
static uint16_t P2SQ9[NPAD];          /* pad -> 9*1296 + sq36, for trample/ranging moves */
static uint8_t  P2RANK[NPAD];         /* pad -> x */
static uint16_t SQ2P[NSQ];            /* sq36 -> pad */
static uint8_t  LVL[NPIECE];

/* Runtime per-(id,colour) movement plan: 1-square dirs are split out from real slides so
 * the common stepper costs one load and one branch per direction. */
typedef struct {
    int16_t  step[8];      /* pad deltas for dydx[d] == 1 */
    int16_t  sl_step[8];   /* pad deltas for dydx[d] > 1  */
    uint8_t  sl_rng[8];
    uint8_t  sl_dj[8];
    uint8_t  nstep, nslide;
    uint8_t  simple;       /* no tp, no special: dedup can be skipped */
    uint8_t  special;
    uint8_t  level;
    uint8_t  color;
    uint16_t tp_off;
    uint8_t  tp_n;
} Plan;
static Plan PLAN[NDEF];

/* tp entries pre-resolved to pad deltas */
typedef struct { int16_t jump; uint8_t slide[8]; } TpFast;
static TpFast TPF[(sizeof(TP_POOL) / sizeof(TP_POOL[0]))];

enum { POLICY_MOVE = 0, POLICY_PIECE = 1 };
enum { RES_WHITE = 0, RES_BLACK = 1, RES_DRAW = 2 };
enum { TERM_ROYAL = 0, TERM_STALEMATE = 1, TERM_REPETITION = 2, TERM_NOPROGRESS = 3, TERM_PLYCAP = 4 };

/* ---- position ------------------------------------------------------------ */
typedef struct {
    uint8_t  side[NPAD];
    uint16_t code[NPAD];
    uint8_t  lvl[NPAD];
    uint16_t plist[2][512];
    int16_t  pat[NPAD];
    int      pn[2];
    int      turn;
    int      npieces;
    int      royals[2];
    int      royal_sq[2];      /* pad index of each side's King, -1 if gone */
    int32_t  material;
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

static inline void sq_set(Pos *p, int q, uint16_t nc)
{
    uint16_t oc = p->code[q];
    if (oc == nc) return;
    p->hash ^= zkey(q, oc) ^ zkey(q, nc);

    int oi = C_IDX(oc), ni = C_IDX(nc);
    if (oi != EMPTY_IDX) {
        int c = C_COLOR(oc);
        p->material -= c ? MATVAL[oc] : -MATVAL[oc];
        p->npieces--;
        if (oi == KING_IDX) { p->royals[c]--; if (p->royal_sq[c] == q) p->royal_sq[c] = -1; }
        int slot = p->pat[q], last = --p->pn[c];
        uint16_t moved = p->plist[c][last];
        p->plist[c][slot] = moved;
        p->pat[moved] = (int16_t)slot;
        p->pat[q] = -1;
    }
    if (ni != EMPTY_IDX) {
        int c = C_COLOR(nc);
        p->material += c ? MATVAL[nc] : -MATVAL[nc];
        p->npieces++;
        if (ni == KING_IDX) { p->royals[c]++; p->royal_sq[c] = q; }
        p->pat[q] = (int16_t)p->pn[c];
        p->plist[c][p->pn[c]++] = (uint16_t)q;
        p->side[q] = (uint8_t)c;
    } else {
        p->side[q] = SIDE_EMPTY;
    }
    p->code[q] = nc;
    p->lvl[q] = LVL[ni];
}

static void pos_init(Pos *p)
{
    memset(p, 0, sizeof(*p));
    p->royal_sq[0] = p->royal_sq[1] = -1;
    memset(p->side, SIDE_OFF, sizeof(p->side));
    for (int i = 0; i < NPAD; i++) p->pat[i] = -1;
    for (int i = 0; i < NSQ; i++) {
        int q = SQ2P[i];
        p->side[q] = SIDE_EMPTY;
        p->code[q] = EMPTY_CODE;
        p->lvl[q] = 0;
        p->hash ^= zkey(q, EMPTY_CODE);
    }
    for (int i = 0; i < NSQ; i++)
        if (C_IDX(INIT_BOARD[i]) != EMPTY_IDX) sq_set(p, SQ2P[i], INIT_BOARD[i]);
    p->turn = 1;                        /* black moves first */
    p->hash ^= TURN_KEY;
}

/* ---- move generation ----------------------------------------------------- */
typedef struct {
    uint32_t *buf;
    int       n, cap;
    uint16_t *stamp;
    uint16_t  epoch;
    /* king-capture scan state, refreshed once per ply */
    uint16_t *kline;      /* NSQ entries: stamped if the square lies on a King ray */
    uint16_t  kepoch;
    int       ksq, kx, ky;
} Gen;

#define PACK(o, m) (((uint32_t)(o) << 14) | (uint32_t)(m))
#define UNPACK_O(v) ((int)((v) >> 14))
#define UNPACK_M(v) ((int)((v) & 0x3FFF))

static void gen_grow(Gen *g)
{
    g->cap *= 2;
    g->buf = (uint32_t *)realloc(g->buf, (size_t)g->cap * 4);
    if (!g->buf) { fprintf(stderr, "oom in move buffer\n"); exit(1); }
}

#define RESERVE(k) do { if (g->n + (k) > g->cap) gen_grow(g); } while (0)
#define PUSH(mv)   do { RESERVE(1); g->buf[g->n++] = bp | (uint32_t)(mv); } while (0)
#define PUSHU(mv)  do { int _m = (mv); \
        if (g->stamp[_m] != g->epoch) { g->stamp[_m] = g->epoch; PUSH(_m); } } while (0)

static void gen_piece(const Pos *restrict p, Gen *restrict g, int basep)
{
    const uint8_t *restrict side = p->side;
    uint16_t code = p->code[basep];
    const Plan *pl = &PLAN[code >> 1];
    int me = pl->color, enemy = 1 - me;
    uint32_t bp = (uint32_t)P2SQ[basep] << 14;

    if (pl->simple) {
        /* pure stepper / slider: rays from one square are disjoint, so no dedup needed */
        RESERVE(8);
        for (int k = 0; k < pl->nstep; k++) {
            int q = basep + pl->step[k];
            int s = side[q];
            if (s == me || s == SIDE_OFF) continue;
            g->buf[g->n++] = bp | P2SQ[q];
        }
        for (int k = 0; k < pl->nslide; k++) {
            int st = pl->sl_step[k], rng = pl->sl_rng[k], dj = pl->sl_dj[k];
            int q = basep, jumped = 0;
            RESERVE(36);
            for (int off = 1; off <= rng; off++) {
                q += st;
                int s = side[q];
                if (s == SIDE_OFF) break;
                if (s == me) { if (++jumped > dj) break; continue; }
                g->buf[g->n++] = bp | P2SQ[q];
                if (s == enemy && ++jumped > dj) break;
            }
        }
        return;
    }

    for (int k = 0; k < pl->nstep; k++) {
        int q = basep + pl->step[k];
        int s = side[q];
        if (s == me || s == SIDE_OFF) continue;
        PUSHU(P2SQ[q]);
    }
    for (int k = 0; k < pl->nslide; k++) {
        int st = pl->sl_step[k], rng = pl->sl_rng[k], dj = pl->sl_dj[k];
        int q = basep, jumped = 0;
        for (int off = 1; off <= rng; off++) {
            q += st;
            int s = side[q];
            if (s == SIDE_OFF) break;
            if (s == me) { if (++jumped > dj) break; continue; }
            PUSHU(P2SQ[q]);
            if (s == enemy && ++jumped > dj) break;
        }
    }

    /* --- tp: fixed-offset jump, then a non-jumping slide ------------------- */
    for (int k = 0; k < pl->tp_n; k++) {
        const TpFast *e = &TPF[pl->tp_off + k];
        int dq = basep + e->jump;
        int ds = side[dq];
        if (ds == me || ds == SIDE_OFF) continue;
        PUSHU(P2SQ[dq]);
        if (ds != SIDE_EMPTY) continue;    /* landed on an enemy: no secondary slide */
        for (int d = 0; d < 8; d++) {
            int rng = e->slide[d];
            if (!rng) continue;
            int st = DSTEP[d], q = dq;
            for (int off = 1; off <= rng; off++) {
                q += st;
                int s = side[q];
                if (s == enemy) { PUSHU(P2SQ[q]); break; }
                if (s != SIDE_EMPTY) break;   /* friendly or off-board */
                PUSHU(P2SQ[q]);
            }
        }
    }

    switch (pl->special) {
    case SP_NONE: break;

    case SP_FREE_EAGLE: {
        for (int d = 0; d < 8; d++) {
            int st = DSTEP[d];
            int max = 3;
            if (me == SIDE_BLACK) { if (DIRX[d] == 1 && DIRY[d] != 0) max = 4; }
            else                  { if (DIRX[d] == -1 && DIRY[d] != 0) max = 4; }
            int aq = basep + st;
            if (side[aq] == SIDE_OFF) continue;
            if (side[aq] != me) PUSHU((d + 1) * 1296 + P2SQ[basep]);   /* igui */
            int fq = basep + max * st;
            if (side[fq] == SIDE_OFF) continue;
            int has_f = 0, has_e = 0, q = basep;
            for (int i = 1; i <= max; i++) {
                q += st;
                int s = side[q];
                if (s == me) { has_f = 1; break; }
                if (s == enemy) has_e = 1;
            }
            if (!has_e || has_f) continue;
            q = fq;
            PUSHU(P2SQ9[q]);
            for (;;) {
                q += st;
                int s = side[q];
                if (s == me || s == SIDE_OFF) break;
                PUSHU(P2SQ9[q]);
                if (s != SIDE_EMPTY) break;
            }
        }
        break;
    }

    case SP_TRAMPLE_ORTHO:
    case SP_TRAMPLE_DIAG:
    case SP_TRAMPLE_ALL: {
        int d0 = (pl->special == SP_TRAMPLE_DIAG) ? 1 : 0;
        int dstep = (pl->special == SP_TRAMPLE_ALL) ? 1 : 2;
        int lvl = pl->level;
        const uint8_t *restrict lv = p->lvl;
        for (int d = d0; d < 8; d += dstep) {
            int st = DSTEP[d], q = basep;
            for (;;) {
                q += st;
                int s = side[q];
                if (s == SIDE_OFF || lv[q] >= lvl) break;
                if (s != me) PUSHU(P2SQ9[q]);
            }
        }
        break;
    }

    case SP_HOOK_ORTHO:
    case SP_HOOK_DIAG: {
        /* Leg 1 collects the empty squares on each ray. game.js then walks all four hook
         * directions from every midpoint; the two *parallel* walks only ever re-emit the
         * ray itself, which is the midpoint list -- but only when a ray has 2+ midpoints,
         * since a lone midpoint's backward walk stops on the mover. Replaying that as a
         * direct emit removes the quadratic rewalk and nearly all the duplicates. */
        int d0 = (pl->special == SP_HOOK_DIAG) ? 1 : 0;
        int mid[160], mdir[160], nm = 0;
        for (int d = d0; d < 8; d += 2) {
            int st = DSTEP[d], q = basep, k0 = nm;
            for (;;) {
                q += st;
                int s = side[q];
                if (s == me || s == SIDE_OFF) break;
                if (s != SIDE_EMPTY) { PUSHU(P2SQ[q]); break; }
                mid[nm] = q; mdir[nm] = d; nm++;
            }
            if (nm - k0 >= 2) for (int i = k0; i < nm; i++) PUSHU(P2SQ[mid[i]]);
        }
        /* leg 2: only the pair perpendicular to the ray this midpoint came from */
        for (int i = 0; i < nm; i++) {
            int m = mid[i], d = mdir[i];
            for (int c = 0; c < 2; c++) {
                int st = DSTEP[c ? ((d + 6) & 7) : ((d + 2) & 7)], q = m;
                for (;;) {
                    q += st;
                    int s = side[q];
                    if (s == me || s == SIDE_OFF) break;
                    PUSHU(P2SQ[q]);
                    if (s != SIDE_EMPTY) break;
                }
            }
        }
        break;
    }

    case SP_PEACOCK: {
        /* leg 1 on the two forward diagonals; leg 2 perpendicular to the leg that
         * produced the midpoint. Which list pairs with which dirs depends on colour. */
        int leg1[2] = { me == SIDE_WHITE ? 5 : 7, me == SIDE_WHITE ? 3 : 1 };
        int list[2][40], nl[2] = { 0, 0 };
        for (int c = 0; c < 2; c++) {
            int st = DSTEP[leg1[c]], q = basep;
            for (;;) {
                q += st;
                int s = side[q];
                if (s == me || s == SIDE_OFF) break;
                if (s != SIDE_EMPTY) { PUSHU(P2SQ[q]); break; }
                list[c][nl[c]++] = q;
            }
        }
        static const int leg2a[2] = { 5, 1 }, leg2b[2] = { 7, 3 };
        for (int pass = 0; pass < 2; pass++) {
            int which = (me == SIDE_WHITE) ? (pass == 0 ? 1 : 0) : (pass == 0 ? 0 : 1);
            const int *dirs = (pass == 0) ? leg2a : leg2b;
            for (int i = 0; i < nl[which]; i++) {
                for (int k = 0; k < 2; k++) {
                    int st = DSTEP[dirs[k]], q = list[which][i];
                    for (;;) {
                        q += st;
                        int s = side[q];
                        if (s == me || s == SIDE_OFF) break;
                        PUSHU(P2SQ[q]);
                        if (s != SIDE_EMPTY) break;
                    }
                }
            }
        }
        break;
    }

    case SP_LION: {
        for (int d = 0; d < 8; d++) {
            int mq = basep + DSTEP[d];
            int ms = side[mq];
            if (ms == me || ms == SIDE_OFF) continue;
            int enc = (d + 1) * 1296;
            for (int e = 0; e < 8; e++) {
                int q = mq + DSTEP[e];
                int s = side[q];
                if (s == SIDE_OFF) continue;
                if (s == me && q != basep) continue;   /* returning home is legal */
                PUSHU(enc + P2SQ[q]);
            }
        }
        break;
    }
    }
}

/* Generates for one piece only, bumping the dedup epoch. */
static void gen_one(const Pos *p, Gen *g, int q)
{
    g->n = 0;
    if (!PLAN[p->code[q] >> 1].simple) {
        if (++g->epoch == 0) { memset(g->stamp, 0, MAXMOVE * 2); g->epoch = 1; }
    }
    gen_piece(p, g, q);
}

static void gen_all(const Pos *p, Gen *g)
{
    g->n = 0;
    int c = p->turn, n = p->pn[c];
    const uint16_t *pl = p->plist[c];
    uint16_t ep = g->epoch;
    for (int i = 0; i < n; i++) {
        int q = pl[i];
        if (!PLAN[p->code[q] >> 1].simple) {
            if (++ep == 0) { memset(g->stamp, 0, MAXMOVE * 2); ep = 1; }
            g->epoch = ep;
        }
        gen_piece(p, g, q);
    }
    g->epoch = ep;
}

/* ---- king capture --------------------------------------------------------- */
/* Which squares a move empties, per Game.squares_emptied: the target, the intermediate
 * for special 1-8, and everything strictly between origin and target for special 9. A
 * move captures the enemy King if the King stands on any of them. The mover's own origin
 * never counts -- for a jitto the encoded target *is* the origin and holds the mover.
 *
 * Special 9 is two different things: a trampler's ranging capture, whose ray breaks on any
 * piece of level >= the mover's and so can never reach a level-4 King, and the Free
 * Eagle's ranging capture, which has no level check and can sweep straight over one. So
 * special 9 has to be checked, but only for origins that lie on one of the King's eight
 * rays -- everything else is one stamped-array lookup away from being skipped.
 */
static void king_ctx_set(Gen *g, const Pos *p)
{
    int kp = p->royal_sq[1 - p->turn];
    if (kp < 0) { g->ksq = -1; return; }
    g->ksq = P2SQ[kp];
    g->kx = g->ksq / 36;
    g->ky = g->ksq % 36;
    if (++g->kepoch == 0) { memset(g->kline, 0, NSQ * 2); g->kepoch = 1; }
    for (int d = 0; d < 8; d++) {
        int st = DSTEP[d], q = kp;
        for (;;) {
            q += st;
            if (p->side[q] == SIDE_OFF) break;
            g->kline[P2SQ[q]] = g->kepoch;     /* pieces do not block: a sweep passes over */
        }
    }
}

static inline uint32_t find_king_capture(const Gen *g)
{
    int ksq = g->ksq;
    if (ksq < 0) return 0;
    int n = g->n, kx = g->kx, ky = g->ky;
    const uint32_t *b = g->buf;
    for (int i = 0; i < n; i++) {
        uint32_t v = b[i];
        int mv = (int)(v & 0x3FFF);
        if (mv < 1296) { if (mv == ksq) return v; continue; }
        int o = (int)(v >> 14);
        if (mv >= 9 * 1296) {
            if (g->kline[o] != g->kepoch) continue;
            int t = mv - 9 * 1296;
            int ox = o / 36, oy = o % 36;
            int dx = t / 36 - ox, dy = t % 36 - oy;
            int ux = (dx > 0) - (dx < 0), uy = (dy > 0) - (dy < 0);
            int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
            int size = ax > ay ? ax : ay;
            int kdx = kx - ox, kdy = ky - oy;
            int step = ux ? kdx * ux : kdy * uy;
            if (step >= 1 && step <= size && kdx == step * ux && kdy == step * uy) return v;
            continue;
        }
        int special = mv / 1296;
        if (mv - special * 1296 == ksq) return v;
        int d = special - 1;
        if ((o / 36 + DIRX[d]) * 36 + (o % 36 + DIRY[d]) == ksq) return v;
    }
    return 0;
}

/* Reference detector: walks every emptied square of every move, one at a time. */
static uint32_t find_king_capture_slow(const Gen *g, int ksq36)
{
    if (ksq36 < 0) return 0;
    for (int i = 0; i < g->n; i++) {
        uint32_t v = g->buf[i];
        int mv = (int)(v & 0x3FFF), o = (int)(v >> 14);
        int special = mv / 1296, t = mv % 1296;
        if (t == ksq36 && t != o) return v;
        if (special >= 1 && special <= 8) {
            int mx = o / 36 + DIRX[special - 1], my = o % 36 + DIRY[special - 1];
            if ((unsigned)mx <= 35u && (unsigned)my <= 35u && mx * 36 + my == ksq36) return v;
        } else if (special == 9) {
            int ox = o / 36, oy = o % 36, tx = t / 36, ty = t % 36;
            int dx = tx - ox, dy = ty - oy;
            int ux = (dx > 0) - (dx < 0), uy = (dy > 0) - (dy < 0);
            int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
            int size = ax > ay ? ax : ay;
            for (int k = 1; k < size; k++)
                if ((ox + k * ux) * 36 + (oy + k * uy) == ksq36) return v;
        }
    }
    return 0;
}

/* ---- move execution (Board.move) ----------------------------------------- */
static inline void mov(Pos *p, int from, int to)
{
    uint16_t c = p->code[from];
    sq_set(p, to, c);
    sq_set(p, from, EMPTY_CODE);
}

static inline int try_promote(Pos *p, int q)
{
    uint16_t c = p->code[q];
    if (C_PROM(c)) return 0;
    int np = PROMOTE_IDX[C_IDX(c)];
    if (np < 0) return 0;
    sq_set(p, q, (uint16_t)CODE(np, C_COLOR(c), 1));
    return 1;
}

/* origin is a pad index, mv a 36-space move code; returns 1 if the mover promoted */
static int do_move(Pos *p, int origin, int mv)
{
    int special = mv / 1296;
    int tq = SQ2P[mv % 1296];
    int was_promoted = C_PROM(p->code[origin]);

    if (special == 0) {
        mov(p, origin, tq);
    } else if (special <= 8) {
        int m = origin + DSTEP[special - 1];
        mov(p, origin, m);
        mov(p, m, tq);
    } else {
        int ox = P2RANK[origin], oy = P2SQ[origin] % 36;
        int tx = P2RANK[tq], ty = P2SQ[tq] % 36;
        int dx = tx - ox, dy = ty - oy;
        int ux = (dx > 0) - (dx < 0), uy = (dy > 0) - (dy < 0);
        int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
        int size = ax > ay ? ax : ay;
        int st = ux * W + uy;
        sq_set(p, tq, p->code[origin]);                  /* target written first */
        int q = origin;
        for (int i = 0; i < size; i++, q += st) sq_set(p, q, EMPTY_CODE);
    }

    int c = C_COLOR(p->code[tq]);
    int tx = P2RANK[tq];
    int promoted_now = 0;
    if ((c == SIDE_BLACK && tx >= 25) || (c == SIDE_WHITE && tx <= 10)) promoted_now = try_promote(p, tq);
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

/* ---- position selection --------------------------------------------------- */
/* Keeps roughly `save_percentage` of positions while flattening their distribution over
 * the tanh label. Bins are equal-width in label space, so "even across bins" is even
 * across the squashed material balance, which is what the training regime wants.
 *
 * The rate per bin comes from water-filling: find the level L such that keeping
 * min(seen[b], L) from every bin exhausts the budget frac*total_seen. Bins rarer than L
 * are kept whole and their shortfall is redistributed; every other bin is thinned to
 * L/seen[b]. A bounded proportional term corrects the deficit each bin accumulated while
 * earlier, cruder rates were in force.
 *
 * Note the natural label distribution is already close to uniform -- that is exactly what
 * fit_tanh.py maximises -- so this is a modest correction, not a heavy resample.
 */
#define SEL_BINS 64

static uint8_t BIN_OF[65536];          /* stored material (i16, biased) -> label bin */

static void sel_build_bins(double S_pawns, int mat_shift)
{
    double unit = 1.0 / (double)(1 << mat_shift);
    for (int i = 0; i < 65536; i++) {
        double lab = tanh(((double)i - 32768.0) * unit / S_pawns);
        int b = (int)((lab + 1.0) * 0.5 * SEL_BINS);
        BIN_OF[i] = (uint8_t)(b < 0 ? 0 : (b >= SEL_BINS ? SEL_BINS - 1 : b));
    }
}

typedef struct {
    uint64_t seen[SEL_BINS], kept[SEL_BINS];
    uint64_t total_seen, total_kept, next_recalc;
    uint32_t thresh[SEL_BINS];
    double   frac;
    double   level_rate;   /* even level per position seen; the running per-bin ceiling */
    Rng      rng;                      /* separate stream: game trajectories stay reproducible */
} Selector;

static void sel_recalc(Selector *s)
{
    double budget = s->frac * (double)s->total_seen;
    double L = budget / SEL_BINS;
    for (int it = 0; it < 40; it++) {
        double fixed = 0;
        int nfree = 0;
        for (int b = 0; b < SEL_BINS; b++) {
            if (!s->seen[b]) continue;
            if ((double)s->seen[b] <= L) fixed += (double)s->seen[b];
            else nfree++;
        }
        if (!nfree) break;
        double nl = (budget - fixed) / nfree;
        if (nl < 0) nl = 0;
        if (fabs(nl - L) <= 1e-9 * (1.0 + L)) { L = nl; break; }
        L = nl;
    }
    s->level_rate = s->total_seen ? L / (double)s->total_seen : s->frac / SEL_BINS;
    for (int b = 0; b < SEL_BINS; b++) {
        double r;
        if (!s->seen[b]) {
            r = s->frac;
        } else {
            r = L / (double)s->seen[b];
            /* Deficit feedback. A bin that over-filled early -- typically an extreme label
             * that was rare enough to be kept whole before the corpus caught up -- must be
             * able to shut off entirely until the others reach it, so this is not floored. */
            double corr = L / (s->kept[b] ? (double)s->kept[b] : 0.5);
            if (corr > 8.0) corr = 8.0;
            r *= corr;
        }
        if (r > 1.0) r = 1.0; else if (r < 0.0) r = 0.0;
        s->thresh[b] = (r >= 1.0) ? 0xFFFFFFFFu : (uint32_t)(r * 4294967296.0);
    }
}

static void sel_init(Selector *s, double frac, uint64_t seed)
{
    memset(s, 0, sizeof(*s));
    s->frac = frac;
    rng_seed(&s->rng, seed);
    uint32_t t = (frac >= 1.0) ? 0xFFFFFFFFu : (uint32_t)(frac * 4294967296.0);
    for (int b = 0; b < SEL_BINS; b++) s->thresh[b] = t;
    s->level_rate = frac / SEL_BINS;
    s->next_recalc = 128;
}

static inline int sel_take(Selector *s, int16_t mat)
{
    int b = BIN_OF[(int)mat + 32768];
    if (s->frac >= 1.0) { s->seen[b]++; s->kept[b]++; s->total_seen++; s->total_kept++; return 1; }
    s->seen[b]++;
    s->total_seen++;
    /* Thinning picks *which* position in a run of similar ones; the ceiling stops any bin
     * from running away before the rest of the corpus catches up. A game's material barely
     * moves from ply to ply, so without the ceiling one lopsided game can dump a few
     * hundred consecutive positions into a single bin. */
    int take = (uint32_t)rng_next(&s->rng) < s->thresh[b]
               && (double)s->kept[b] < 1.03 * s->level_rate * (double)s->total_seen;
    if (take) { s->kept[b]++; s->total_kept++; }
    if (s->total_seen >= s->next_recalc) {
        sel_recalc(s);
        uint64_t step = s->total_seen / 8;
        if (step < 128) step = 128; else if (step > 65536) step = 65536;
        s->next_recalc = s->total_seen + step;
    }
    return take;
}

/* ---- output -------------------------------------------------------------- */
#define TKY_MAGIC "TKYSHOGI"
#define TKY_VERSION 1u
#define PLY_REC  8
#define GAME_HDR 16
#define FILE_HDR 128

/* Standalone position corpus: fixed-stride records, so a shuffled trainer can mmap the
 * file and seek straight to record i without an index or any replay. */
#define TKP_MAGIC   "TKYPOSNS"
#define TKP_HDR     128
#define TKP_BOARD   (NSQ * 2)
#define TKP_REC     (TKP_BOARD + 32)
enum { FMT_GAMES = 0, FMT_POSITIONS = 1 };

/* A terminal ply record has no move; the top nibble is never a real special (max 9). */
#define TERMINAL_MOVE 0xF0000000u
#define MOVE_SELECTED 8u

typedef struct { int fd; unsigned char *buf; size_t len, cap; } Sink;

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
    if (s->len + n > s->cap) {
        sink_flush(s);
        if (n > s->cap) {                       /* oversized record: write it straight out */
            size_t off = 0;
            while (off < n) {
                ssize_t w = write(s->fd, (const unsigned char *)d + off, n - off);
                if (w <= 0) { perror("write"); exit(1); }
                off += (size_t)w;
            }
            return;
        }
    }
    memcpy(s->buf + s->len, d, n);
    s->len += n;
}

/* ---- per-ply move-count statistics --------------------------------------- */
#define MS_BINW    8            /* histogram bin width, in moves */
#define MS_BINS    1024         /* covers 0..8191 moves */
#define MS_BUCKETW 25           /* plies per histogram bucket */

typedef struct {
    int       nply;             /* max_plies + 1 */
    int       nbucket;
    uint64_t *cnt, *sum, *sumsq, *psum;
    uint32_t *mn, *mx;
    uint64_t *hist;             /* nbucket * MS_BINS */
} MoveStats;

static MoveStats *ms_new(int max_plies)
{
    MoveStats *m = calloc(1, sizeof(MoveStats));
    m->nply = max_plies + 1;
    m->nbucket = (m->nply + MS_BUCKETW - 1) / MS_BUCKETW;
    m->cnt = calloc((size_t)m->nply, 8);
    m->sum = calloc((size_t)m->nply, 8);
    m->sumsq = calloc((size_t)m->nply, 8);
    m->psum = calloc((size_t)m->nply, 8);
    m->mn = malloc((size_t)m->nply * 4);
    m->mx = calloc((size_t)m->nply, 4);
    for (int i = 0; i < m->nply; i++) m->mn[i] = 0xFFFFFFFFu;
    m->hist = calloc((size_t)m->nbucket * MS_BINS, 8);
    return m;
}

static inline void ms_add(MoveStats *m, int ply, uint32_t n, uint32_t pieces)
{
    if (ply >= m->nply) return;
    m->psum[ply] += pieces;
    m->cnt[ply]++;
    m->sum[ply] += n;
    m->sumsq[ply] += (uint64_t)n * n;
    if (n < m->mn[ply]) m->mn[ply] = n;
    if (n > m->mx[ply]) m->mx[ply] = n;
    uint32_t bin = n / MS_BINW;
    if (bin >= MS_BINS) bin = MS_BINS - 1;
    m->hist[(size_t)(ply / MS_BUCKETW) * MS_BINS + bin]++;
}

static void ms_merge(MoveStats *d, const MoveStats *s)
{
    for (int i = 0; i < d->nply; i++) {
        d->cnt[i] += s->cnt[i]; d->sum[i] += s->sum[i]; d->sumsq[i] += s->sumsq[i]; d->psum[i] += s->psum[i];
        if (s->mn[i] < d->mn[i]) d->mn[i] = s->mn[i];
        if (s->mx[i] > d->mx[i]) d->mx[i] = s->mx[i];
    }
    for (size_t i = 0; i < (size_t)d->nbucket * MS_BINS; i++) d->hist[i] += s->hist[i];
}

/* ---- worker -------------------------------------------------------------- */
/* one row per sampled position, for offline fitting of the squash */
typedef struct { uint16_t ply; uint16_t npieces; uint16_t nmoves; uint16_t pad; int32_t mat; } Sample;

typedef struct {
    int      id;
    uint64_t seed_base, first_game, n_games;
    int      max_plies, rep_limit, no_progress_limit, stalemate_loses, mat_shift;
    char     path[512];
    int      write_out;
    uint64_t plies, games_done, res_count[3], term_count[5], maxlen;
    Sample  *msamples;
    uint64_t msample_n, msample_cap;
    uint64_t nmoves_sum, nmoves_max, nmoves_positions, bytes_out;
    int      policy;
    uint32_t tanh_scale_fp;
    MoveStats *ms;
    int      regicide;
    int      format;
    double   save_frac;
    char     ppath[512];
    Selector sel;
    uint64_t pos_written;
} Worker;

typedef struct { uint32_t move; int16_t mat; uint16_t nmoves; } PlyRec;

/* One selected position, buffered until the game ends so the record can carry the
 * result. Layout matches TKP_REC exactly and is written out verbatim. */
typedef struct {
    uint16_t board[NSQ];      /* 2592 B; then fields ordered so nothing needs padding */
    uint64_t seed;
    uint32_t game_index;
    uint32_t plies_total;
    uint16_t ply;
    uint16_t nmoves;
    int16_t  mat;
    uint16_t n_occupied;
    uint8_t  side_to_move;
    uint8_t  result;
    uint8_t  termination;
    uint8_t  bin;
    uint32_t reserved;
} PosSnap;
_Static_assert(sizeof(PosSnap) == TKP_REC, "PosSnap must match the on-disk record exactly");

static void snap_take(PosSnap *d, const Pos *p, uint32_t game_index, int ply, int nmoves, int16_t mat)
{
    for (int i = 0; i < NSQ; i++) d->board[i] = p->code[SQ2P[i]];
    d->game_index = game_index;
    d->ply = (uint16_t)ply;
    d->nmoves = (uint16_t)(nmoves > 65535 ? 65535 : nmoves);
    d->mat = mat;
    d->side_to_move = (uint8_t)p->turn;
    d->bin = BIN_OF[(int)mat + 32768];
    d->n_occupied = (uint16_t)p->npieces;
    d->result = d->termination = 0;
    d->plies_total = 0;
    d->reserved = 0;
}


/* material -> stored units, rounded to nearest so a reader can check it to half a unit */
static inline int32_t quantise(int32_t v, int div)
{
    int32_t h = div / 2;
    int32_t q = v >= 0 ? (v + h) / div : -((-v + h) / div);
    return q > 32767 ? 32767 : (q < -32767 ? -32767 : q);
}

static void *worker_main(void *arg)
{
    Worker *w = (Worker *)arg;
    Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos));
    pos_init(base);

    Gen g;
    g.cap = 8192;
    g.buf = malloc((size_t)g.cap * 4);
    g.stamp = calloc(MAXMOVE, 2);
    g.epoch = 0; g.n = 0;
    g.kline = calloc(NSQ, 2); g.kepoch = 0; g.ksq = -1;

    RepTab *rep = calloc(1, sizeof(RepTab));
    Sink sink = { -1, NULL, 0, 1u << 22 };
    PlyRec *plybuf = malloc(sizeof(PlyRec) * (size_t)(w->max_plies + 2));

    /* snapshots are buffered per game so each record can carry the final result */
    PosSnap *snaps = NULL;
    int nsnap = 0, snapcap = 0;
    if (w->format == FMT_POSITIONS) {
        snapcap = 64 + (int)((w->max_plies + 1) * (w->save_frac * 4.0 > 1.0 ? 1.0 : w->save_frac * 4.0));
        snaps = malloc(sizeof(PosSnap) * (size_t)snapcap);
    }

    if (w->write_out) {
        const char *path = (w->format == FMT_POSITIONS) ? w->ppath : w->path;
        sink.fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (sink.fd < 0) { perror(path); exit(1); }
        sink.buf = malloc(sink.cap);
        unsigned char hdr[FILE_HDR];
        memset(hdr, 0, sizeof(hdr));
        memcpy(hdr, (w->format == FMT_POSITIONS) ? TKP_MAGIC : TKY_MAGIC, 8);
        uint32_t v;
        v = TKY_VERSION;            memcpy(hdr + 8,  &v, 4);
        v = FILE_HDR;               memcpy(hdr + 12, &v, 4);
        if (w->format == FMT_POSITIONS) {
            v = TKP_REC;            memcpy(hdr + 16, &v, 4);
            v = NSQ;                memcpy(hdr + 20, &v, 4);
        } else {
            v = GAME_HDR;           memcpy(hdr + 16, &v, 4);
            v = PLY_REC;            memcpy(hdr + 20, &v, 4);
        }
        v = VALUE_FP;               memcpy(hdr + 24, &v, 4);
        v = (uint32_t)w->mat_shift; memcpy(hdr + 28, &v, 4);
        memcpy(hdr + 32, &w->seed_base, 8);
        memcpy(hdr + 40, &w->first_game, 8);
        /* 48: game count, 56: ply count -- patched at close */
        v = (uint32_t)w->tanh_scale_fp;        memcpy(hdr + 64, &v, 4);
        v = (uint32_t)w->policy;               memcpy(hdr + 68, &v, 4);
        v = (uint32_t)w->max_plies;            memcpy(hdr + 72, &v, 4);
        v = (uint32_t)w->rep_limit;            memcpy(hdr + 76, &v, 4);
        v = (uint32_t)w->no_progress_limit;    memcpy(hdr + 80, &v, 4);
        v = (uint32_t)w->stalemate_loses;      memcpy(hdr + 84, &v, 4);
        v = (uint32_t)w->regicide;             memcpy(hdr + 96, &v, 4);
        v = (uint32_t)(w->save_frac * 10000.0 + 0.5); memcpy(hdr + 100, &v, 4);
        v = SEL_BINS;                          memcpy(hdr + 104, &v, 4);
        v = 36;                                memcpy(hdr + 88, &v, 4);
        v = NPIECE;                            memcpy(hdr + 92, &v, 4);
        sink_put(&sink, hdr, FILE_HDR);
    }

    Rng rng;
    int mat_div = VALUE_FP >> w->mat_shift;
    if (mat_div < 1) mat_div = 1;

    for (uint64_t gi = 0; gi < w->n_games; gi++) {
        uint64_t seed = w->seed_base ^ ((w->first_game + gi) * 0x9E3779B97F4A7C15ull);
        rng_seed(&rng, seed);
        memcpy(p, base, sizeof(Pos));
        rep->epoch++;
        rep_bump(rep, p->hash);

        int nply = 0, no_prog = 0, result = RES_DRAW, term = TERM_PLYCAP;
        nsnap = 0;

        for (;;) {
            int full = 1;
            if (w->policy == POLICY_PIECE) {
                /* Uniform over pieces that have a move, then uniform over that piece's
                 * moves. Falls back to a full generation if the sampler keeps missing,
                 * which is also what makes stalemate detection exact. */
                full = 0;
                int np = p->pn[p->turn];
                int tries = np < 64 ? np : 64;
                g.n = 0;
                for (int t = 0; t < tries; t++) {
                    gen_one(p, &g, p->plist[p->turn][rng_below(&rng, (uint32_t)np)]);
                    if (g.n) break;
                }
                if (!g.n) { gen_all(p, &g); full = 1; }
            } else {
                gen_all(p, &g);
            }
            int32_t m = quantise(p->material, mat_div);
            plybuf[nply].mat = (int16_t)m;
            plybuf[nply].nmoves = (uint16_t)(g.n > 65535 ? 65535 : g.n);
            int selected = sel_take(&w->sel, plybuf[nply].mat);
            if (selected && snaps) {
                if (nsnap == snapcap) { snapcap *= 2; snaps = realloc(snaps, sizeof(PosSnap) * (size_t)snapcap); }
                snap_take(&snaps[nsnap++], p, (uint32_t)(w->first_game + gi), nply, g.n, plybuf[nply].mat);
            }
            if (full) {
                if (w->nmoves_max < (uint64_t)g.n) w->nmoves_max = (uint64_t)g.n;
                w->nmoves_sum += (uint64_t)g.n;
                w->nmoves_positions++;
                if (w->ms) ms_add(w->ms, nply, (uint32_t)g.n, (uint32_t)p->npieces);
            }
            if (w->msamples && w->msample_n < w->msample_cap && (nply & 3) == 0) {
                Sample *sm = &w->msamples[w->msample_n++];
                sm->ply = (uint16_t)(nply > 65535 ? 65535 : nply);
                sm->npieces = (uint16_t)p->npieces;
                sm->nmoves = (uint16_t)(g.n > 65535 ? 65535 : g.n);
                sm->pad = (uint16_t)full;
                sm->mat = p->material;
            }

            if (g.n == 0) {                             /* Game.has_legal_move == false */
                result = w->stalemate_loses ? (1 - p->turn) : RES_DRAW;
                term = TERM_STALEMATE;
                plybuf[nply].move = TERMINAL_MOVE | (selected ? MOVE_SELECTED : 0);
                break;
            }
            if (nply >= w->max_plies) {
                result = RES_DRAW; term = TERM_PLYCAP;
                plybuf[nply].move = TERMINAL_MOVE | (selected ? MOVE_SELECTED : 0);
                break;
            }

            uint32_t pick = 0;
            if (w->regicide) {
                /* Always take the enemy King when it is available. Under --policy piece
                 * only the sampled piece's moves are in the buffer, so this sees a King
                 * capture only when that piece happens to have one. */
                king_ctx_set(&g, p);
                pick = find_king_capture(&g);
            }
            if (!pick) pick = g.buf[rng_below(&rng, (uint32_t)g.n)];
            int origin36 = UNPACK_O(pick), mv = UNPACK_M(pick);
            int mover = p->turn;
            int royals_before[2] = { p->royals[0], p->royals[1] };
            int pieces_before = p->npieces;

            int promoted_now = do_move(p, SQ2P[origin36], mv);
            int captured = pieces_before - p->npieces;

            plybuf[nply].move = ((uint32_t)(mv / 1296) << 28) | ((uint32_t)origin36 << 17)
                              | ((uint32_t)(mv % 1296) << 6)
                              | ((uint32_t)(captured > 0) << 5) | ((uint32_t)(promoted_now != 0) << 4)
                              | (selected ? MOVE_SELECTED : 0);
            nply++;
            no_prog = (captured > 0 || promoted_now) ? 0 : no_prog + 1;

            if (p->royals[1 - mover] < royals_before[1 - mover]) { result = mover; term = TERM_ROYAL; }
            else if (p->royals[mover] < royals_before[mover])    { result = 1 - mover; term = TERM_ROYAL; }
            else {
                int seen = rep_bump(rep, p->hash);
                if (w->rep_limit > 0 && seen >= w->rep_limit) { result = RES_DRAW; term = TERM_REPETITION; }
                else if (w->no_progress_limit > 0 && no_prog >= w->no_progress_limit) { result = RES_DRAW; term = TERM_NOPROGRESS; }
                else continue;
            }
            int32_t fm = quantise(p->material, mat_div);
            plybuf[nply].mat = (int16_t)fm;
            plybuf[nply].nmoves = 0;
            int fsel = sel_take(&w->sel, plybuf[nply].mat);
            if (fsel && snaps) {
                if (nsnap == snapcap) { snapcap *= 2; snaps = realloc(snaps, sizeof(PosSnap) * (size_t)snapcap); }
                snap_take(&snaps[nsnap++], p, (uint32_t)(w->first_game + gi), nply, 0, plybuf[nply].mat);
            }
            plybuf[nply].move = TERMINAL_MOVE | (fsel ? MOVE_SELECTED : 0);
            break;
        }

        w->plies += (uint64_t)nply;
        w->games_done++;
        w->res_count[result]++;
        w->term_count[term]++;
        if ((uint64_t)nply > w->maxlen) w->maxlen = (uint64_t)nply;

        if (w->write_out && w->format == FMT_POSITIONS) {
            for (int i = 0; i < nsnap; i++) {
                snaps[i].result = (uint8_t)result;
                snaps[i].termination = (uint8_t)term;
                snaps[i].plies_total = (uint32_t)nply;
                snaps[i].seed = seed;
            }
            sink_put(&sink, snaps, (size_t)nsnap * TKP_REC);
            w->bytes_out += (size_t)nsnap * TKP_REC;
            w->pos_written += (uint64_t)nsnap;
        } else if (w->write_out) {
            unsigned char gh[GAME_HDR];
            uint32_t np = (uint32_t)nply;
            memcpy(gh, &np, 4);
            gh[4] = (unsigned char)result;
            gh[5] = (unsigned char)term;
            gh[6] = 0; gh[7] = 0;
            memcpy(gh + 8, &seed, 8);
            sink_put(&sink, gh, GAME_HDR);
            sink_put(&sink, plybuf, (size_t)(nply + 1) * PLY_REC);
            w->bytes_out += GAME_HDR + (size_t)(nply + 1) * PLY_REC;
        }
    }

    if (w->write_out) {
        sink_flush(&sink);
        uint64_t nrec = (w->format == FMT_POSITIONS) ? w->pos_written : w->games_done;
        if (pwrite(sink.fd, &nrec, 8, 48) != 8) perror("pwrite");
        uint64_t nsecond = (w->format == FMT_POSITIONS) ? w->sel.total_seen : w->plies;
        if (pwrite(sink.fd, &nsecond, 8, 56) != 8) perror("pwrite");
        close(sink.fd);
        free(sink.buf);
    }
    free(plybuf); free(snaps); free(g.buf); free(g.stamp); free(g.kline); free(rep); free(base); free(p);
    return NULL;
}

/* ---- setup --------------------------------------------------------------- */
static void tables_init(void)
{
    for (int d = 0; d < 8; d++) DSTEP[d] = (int16_t)(DIRX[d] * W + DIRY[d]);
    for (int i = 0; i < NPAD; i++) { P2SQ[i] = 0; P2SQ9[i] = 0; P2RANK[i] = 0; }
    for (int x = 0; x < 36; x++) for (int y = 0; y < 36; y++) {
        int q = (x + PAD) * W + (y + PAD), s = x * 36 + y;
        P2SQ[q] = (uint16_t)s; P2SQ9[q] = (uint16_t)(9 * 1296 + s); P2RANK[q] = (uint8_t)x;
        SQ2P[s] = (uint16_t)q;
    }
    for (int i = 0; i < NPIECE; i++) LVL[i] = DEFS[i * 2 + 1].level;
    for (unsigned i = 0; i < sizeof(TP_POOL) / sizeof(TP_POOL[0]); i++) {
        TPF[i].jump = (int16_t)(TP_POOL[i].dx * W + TP_POOL[i].dy);
        memcpy(TPF[i].slide, TP_POOL[i].slide, 8);
    }
    for (int i = 0; i < NDEF; i++) {
        const PieceDef *d = &DEFS[i];
        Plan *pl = &PLAN[i];
        memset(pl, 0, sizeof(*pl));
        pl->simple = d->simple; pl->special = d->special; pl->level = d->level;
        pl->color = d->color; pl->tp_off = d->tp_off; pl->tp_n = d->tp_n;
        for (int k = 0; k < 8; k++) {
            /* a 1-square dir cannot use its jump budget, so it needs no slide state */
            if (d->dydx[k] == 1) pl->step[pl->nstep++] = DSTEP[k];
            else if (d->dydx[k] > 1) {
                pl->sl_step[pl->nslide] = DSTEP[k];
                pl->sl_rng[pl->nslide] = d->dydx[k];
                pl->sl_dj[pl->nslide] = d->djump[k];
                pl->nslide++;
            }
        }
    }
}

/* Concatenates the per-thread shards into one file and removes them.
 *
 * Every shard carries the same 128-byte header apart from its first-game index and its two
 * counts, so the fused header is shard 0's with the counts summed and the first-game index
 * reset to 0. That is what makes the result readable with no change to unpack.py or
 * tky_reader.py: both take a single file as happily as a directory, and the game index of
 * the game format is `header.first_game + n`, which stays correct because the shards are
 * concatenated in worker order and worker i owns a contiguous run of game indices.
 */
static int fuse_shards(const char *outdir, const Worker *ws, int threads, int format,
                       uint64_t count, uint64_t second, char *out_path, size_t out_sz)
{
    snprintf(out_path, out_sz, "%s/corpus.%s", outdir, format == FMT_POSITIONS ? "tkp" : "tky");

    const char *first = (format == FMT_POSITIONS) ? ws[0].ppath : ws[0].path;
    unsigned char hdr[FILE_HDR];
    int in = open(first, O_RDONLY);
    if (in < 0) { perror(first); return 0; }
    if (read(in, hdr, FILE_HDR) != (ssize_t)FILE_HDR) { perror(first); close(in); return 0; }
    close(in);

    uint64_t zero = 0;
    memcpy(hdr + 40, &zero, 8);
    memcpy(hdr + 48, &count, 8);
    memcpy(hdr + 56, &second, 8);

    int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror(out_path); return 0; }
    if (write(fd, hdr, FILE_HDR) != (ssize_t)FILE_HDR) { perror(out_path); close(fd); return 0; }

    size_t bufsz = 1u << 20;
    unsigned char *buf = malloc(bufsz);
    if (!buf) { fprintf(stderr, "oom fusing shards\n"); close(fd); return 0; }
    for (int i = 0; i < threads; i++) {
        const char *p = (format == FMT_POSITIONS) ? ws[i].ppath : ws[i].path;
        int sfd = open(p, O_RDONLY);
        if (sfd < 0) { perror(p); free(buf); close(fd); return 0; }
        if (lseek(sfd, FILE_HDR, SEEK_SET) < 0) { perror(p); close(sfd); free(buf); close(fd); return 0; }
        ssize_t n;
        while ((n = read(sfd, buf, bufsz)) > 0) {
            ssize_t at = 0;
            while (at < n) {
                ssize_t w = write(fd, buf + at, (size_t)(n - at));
                if (w <= 0) { perror(out_path); close(sfd); free(buf); close(fd); return 0; }
                at += w;
            }
        }
        if (n < 0) { perror(p); close(sfd); free(buf); close(fd); return 0; }
        close(sfd);
        unlink(p);
    }
    free(buf);
    close(fd);
    return 1;
}

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmp_i32(const void *a, const void *b)
{ int32_t x = *(const int32_t *)a, y = *(const int32_t *)b; return (x > y) - (x < y); }

static uint64_t perft(Pos *p, Gen *g, int depth)
{
    gen_all(p, g);
    if (depth == 1) return (uint64_t)g->n;
    int n = g->n;
    uint32_t *snap = malloc((size_t)n * 4);
    memcpy(snap, g->buf, (size_t)n * 4);
    uint64_t total = 0;
    Pos *save = malloc(sizeof(Pos));
    *save = *p;
    for (int i = 0; i < n; i++) {
        do_move(p, SQ2P[UNPACK_O(snap[i])], UNPACK_M(snap[i]));
        total += perft(p, g, depth - 1);
        *p = *save;
    }
    free(snap); free(save);
    return total;
}

/* nnplay.c includes this file for the rules and the generator; TKY_NO_MAIN drops the
 * CLI so there is one copy of Board/Piece/Game semantics, not two. */
#ifndef TKY_NO_MAIN
int main(int argc, char **argv)
{
    tables_init();
    const char *mode = argc > 1 ? argv[1] : "help";

    if (!strcmp(mode, "perft")) {
        int depth = argc > 2 ? atoi(argv[2]) : 1;
        Pos *p = malloc(sizeof(Pos)); pos_init(p);
        Gen g = { malloc(8192 * 4), 0, 8192, calloc(MAXMOVE, 2), 0, calloc(NSQ, 2), 0, -1, 0, 0 };
        for (int d = 1; d <= depth; d++) {
            double t0 = now_s();
            uint64_t n = perft(p, &g, d);
            printf("perft(%d) = %llu   (%.2fs)\n", d, (unsigned long long)n, now_s() - t0);
        }
        gen_all(p, &g);
        uint64_t by[11] = { 0 };
        for (int i = 0; i < g.n; i++) by[UNPACK_M(g.buf[i]) / 1296]++;
        printf("special breakdown:");
        for (int i = 0; i < 11; i++) if (by[i]) printf(" %d:%llu", i, (unsigned long long)by[i]);
        printf("\npieces=%d royals=%d/%d material=%d\n", p->npieces, p->royals[0], p->royals[1], p->material);
        return 0;
    }

    if (!strcmp(mode, "kingtest")) {
        /* Plays games with the rule on and checks, every ply:
         *   - the fast detector agrees with the reference detector (tramples included);
         *   - the royal_sq cache agrees with a full board scan;
         *   - whenever a capture was available it was played, and the game ended by
         *     royal capture on that ply;
         *   - so no non-final position ever has the enemy King en prise. */
        uint64_t games = argc > 2 ? strtoull(argv[2], NULL, 10) : 200;
        int maxply = argc > 3 ? atoi(argv[3]) : 3000;
        Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos));
        pos_init(base);
        Gen g = { malloc(8192 * 4), 0, 8192, calloc(MAXMOVE, 2), 0, calloc(NSQ, 2), 0, -1, 0, 0 };
        uint64_t bad_detect = 0, bad_cache = 0, bad_enforce = 0, plies = 0, caps = 0, avail_nonfinal = 0;
        for (uint64_t gi = 0; gi < games; gi++) {
            Rng rng; rng_seed(&rng, 0xABCDEFull ^ (gi * 0x9E3779B97F4A7C15ull));
            memcpy(p, base, sizeof(Pos));
            for (int ply = 0; ply < maxply; ply++) {
                gen_all(p, &g);
                if (!g.n) break;
                plies++;
                /* royal_sq cache vs a full scan */
                for (int c = 0; c < 2; c++) {
                    int scan = -1;
                    for (int i = 0; i < NSQ; i++) {
                        uint16_t cd = p->code[SQ2P[i]];
                        if (C_IDX(cd) == KING_IDX && (int)C_COLOR(cd) == c) scan = SQ2P[i];
                    }
                    if (scan != p->royal_sq[c]) bad_cache++;
                }
                int ksq = p->royal_sq[1 - p->turn] < 0 ? -1 : (int)P2SQ[p->royal_sq[1 - p->turn]];
                king_ctx_set(&g, p);
                uint32_t fast = find_king_capture(&g);
                uint32_t slow = find_king_capture_slow(&g, ksq);
                if ((fast != 0) != (slow != 0)) {
                    bad_detect++;
                    if (bad_detect < 5) fprintf(stderr, "  detector split at game %llu ply %d: fast=%u slow=%u ksq=%d\n",
                                                (unsigned long long)gi, ply, fast, slow, ksq);
                }
                uint32_t pick = fast ? fast : g.buf[rng_below(&rng, (uint32_t)g.n)];
                int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
                do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
                int ended = p->royals[1 - mover] < rb[1 - mover] || p->royals[mover] < rb[mover];
                if (fast) {
                    caps++;
                    if (!ended) bad_enforce++;          /* claimed a capture but no royal died */
                } else if (slow) {
                    avail_nonfinal++;                   /* missed a capture that was there */
                }
                if (ended) break;
            }
        }
        printf("games=%llu plies=%llu king-captures=%llu\n",
               (unsigned long long)games, (unsigned long long)plies, (unsigned long long)caps);
        printf("  fast vs reference detector mismatches : %llu\n", (unsigned long long)bad_detect);
        printf("  royal_sq cache mismatches             : %llu\n", (unsigned long long)bad_cache);
        printf("  claimed captures that killed no King  : %llu\n", (unsigned long long)bad_enforce);
        printf("  available captures not played         : %llu\n", (unsigned long long)avail_nonfinal);
        return (bad_detect || bad_cache || bad_enforce || avail_nonfinal) ? 1 : 0;
    }

    if (!strcmp(mode, "probe")) {
        Pos *p = malloc(sizeof(Pos)); pos_init(p);
        for (int i = 0; i < NSQ; i++) sq_set(p, SQ2P[i], EMPTY_CODE);
        char line[128];
        while (fgets(line, sizeof(line), stdin)) {
            int a, b, c, d;
            if (sscanf(line, "turn %d", &a) == 1) { if (a != p->turn) { p->turn = a; p->hash ^= TURN_KEY; } continue; }
            if (sscanf(line, "%d %d %d %d", &a, &b, &c, &d) == 4) sq_set(p, SQ2P[a], (uint16_t)CODE(b, c, d));
        }
        Gen g = { malloc(8192 * 4), 0, 8192, calloc(MAXMOVE, 2), 0, calloc(NSQ, 2), 0, -1, 0, 0 };
        gen_all(p, &g);
        qsort(g.buf, (size_t)g.n, 4, cmp_i32);
        printf("N %d\n", g.n);
        for (int i = 0; i < g.n; i++) printf("%d:%d\n", UNPACK_O(g.buf[i]), UNPACK_M(g.buf[i]));
        return 0;
    }

    if (!strcmp(mode, "trace")) {
        uint64_t seed = argc > 2 ? strtoull(argv[2], NULL, 10) : 1;
        int maxply = argc > 3 ? atoi(argv[3]) : 200, verbose = argc > 4 ? atoi(argv[4]) : 0;
        int det = argc > 5 ? atoi(argv[5]) : 0;   /* order-independent picking, for difftests */
        Pos *p = malloc(sizeof(Pos)); pos_init(p);
        Gen g = { malloc(8192 * 4), 0, 8192, calloc(MAXMOVE, 2), 0, calloc(NSQ, 2), 0, -1, 0, 0 };
        RepTab *rep = calloc(1, sizeof(RepTab)); rep->epoch = 1; rep_bump(rep, p->hash);
        Rng rng; rng_seed(&rng, seed);
        for (int ply = 0; ply < maxply; ply++) {
            gen_all(p, &g);
            if (g.n == 0) { printf("END stalemate ply=%d\n", ply); break; }
            int n = g.n;
            uint32_t *snap = malloc((size_t)n * 4);
            memcpy(snap, g.buf, (size_t)n * 4);
            qsort(snap, (size_t)n, 4, cmp_i32);
            uint64_t dig = 0xcbf29ce484222325ull;
            for (int i = 0; i < n; i++) { dig ^= snap[i]; dig *= 0x100000001b3ull; }
            printf("PLY %d turn=%d n=%d digest=%llu mat=%d pieces=%d\n",
                   ply, p->turn, n, (unsigned long long)dig, p->material, p->npieces);
            if (verbose) for (int i = 0; i < n; i++) printf("M %d %d\n", UNPACK_O(snap[i]), UNPACK_M(snap[i]));
            uint32_t *snap2 = snap;
            /* seed 0: pick from the *sorted* list, so the choice depends only on the
             * move set and not on generation order -- lets two implementations be
             * compared over a whole game rather than only up to the first pick. */
            uint32_t pick = !det ? g.buf[rng_below(&rng, (uint32_t)n)]
                                 : snap2[((uint64_t)ply * 2654435761u + seed * 2246822519u + 12345u) % (uint64_t)n];
            int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
            printf("PICK %d %d\n", UNPACK_O(pick), UNPACK_M(pick));
            do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
            free(snap2);
            if (p->royals[1 - mover] < rb[1 - mover] || p->royals[mover] < rb[mover]) { printf("END royal ply=%d\n", ply + 1); break; }
            if (rep_bump(rep, p->hash) >= 4) { printf("END repetition ply=%d\n", ply + 1); break; }
        }
        return 0;
    }

    uint64_t games = 1000, seed_base = 0x5EED5EED5EED5EEDull;
    int threads = 1, max_plies = 3000, rep_limit = 4, no_prog = 0, stalemate_loses = 1, mat_shift = 3;
    const char *outdir = NULL;
    int policy = POLICY_MOVE;
    int regicide = 1;   /* always capture the enemy King when possible */
    double save_pct = 5.0;
    int format = FMT_GAMES;
    double tanh_scale = 246.8;   /* pawns; see tanh_fit.json / fit_tanh.py */
    int fuse = 1;                /* concatenate the per-thread shards when they are done */
    int sample = !strcmp(mode, "sample");
    int movestats = !strcmp(mode, "movestats");

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--games")       && i + 1 < argc) games = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads")     && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")         && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--max-plies")   && i + 1 < argc) max_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rep-limit")   && i + 1 < argc) rep_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-progress") && i + 1 < argc) no_prog = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")        && i + 1 < argc) seed_base = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mat-shift")   && i + 1 < argc) mat_shift = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keep-shards")) fuse = 0;
        else if (!strcmp(argv[i], "--no-regicide")) regicide = 0;
        else if (!strcmp(argv[i], "--save-percentage") && i + 1 < argc) save_pct = atof(argv[++i]);
        else if (!strcmp(argv[i], "--format") && i + 1 < argc) { const char *v = argv[++i];
            if (!strcmp(v, "games")) format = FMT_GAMES; else if (!strcmp(v, "positions")) format = FMT_POSITIONS;
            else { fprintf(stderr, "--format must be games|positions\n"); return 1; } }
        else if (!strcmp(argv[i], "--tanh-scale") && i + 1 < argc) tanh_scale = atof(argv[++i]);
        else if (!strcmp(argv[i], "--policy")      && i + 1 < argc) { const char *v = argv[++i];
            if (!strcmp(v, "move")) policy = POLICY_MOVE; else if (!strcmp(v, "piece")) policy = POLICY_PIECE;
            else { fprintf(stderr, "--policy must be move|piece\n"); return 1; } }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (save_pct < 0.0 || save_pct > 100.0) { fprintf(stderr, "--save-percentage must be in [0, 100]\n"); return 1; }
    sel_build_bins(tanh_scale, mat_shift);
    if (strcmp(mode, "gen") && !sample && !movestats) {
        fprintf(stderr,
            "usage: tky <gen|sample|perft|probe|trace> [options]\n"
            "  gen    --games N --threads T --out DIR [--max-plies N] [--rep-limit N]\n"
            "         [--no-progress N] [--seed S] [--mat-shift K]\n"
            "  sample --games N --threads T      statistics + samples.bin, no corpus\n"
            "  movestats --games N --threads T   exact legal-move counts per ply index\n"
            "  perft D                           move counts from the initial position\n"
            "  trace SEED PLIES [V] [DET]        one playout, for cross-checking\n"
            "  probe                             move list for a position read from stdin\n"
            "\n  --keep-shards   leave one shard per thread instead of fusing them into\n"
            "                  <out>/corpus.tky (or .tkp), which is the default\n"
            "\n  --policy move   uniform over legal moves (matches game.js semantics)\n"
            "  --policy piece  uniform over pieces that can move, then over that piece's moves\n"
            "  --no-regicide   do not force a King capture when one is available\n"
            "\n  --save-percentage P  keep ~P%% of positions, flattened over the tanh label\n"
            "                       (default 5.0; 100 keeps everything)\n"
            "  --format games       whole games, every move, selected plies flagged (default)\n"
            "  --format positions   only the selected positions, as standalone board snapshots\n");
        return 1;
    }

    Worker *ws = calloc((size_t)threads, sizeof(Worker));
    pthread_t *th = calloc((size_t)threads, sizeof(pthread_t));
    uint64_t per = games / (uint64_t)threads, extra = games % (uint64_t)threads, at = 0;
    for (int i = 0; i < threads; i++) {
        ws[i].id = i; ws[i].seed_base = seed_base; ws[i].first_game = at;
        ws[i].n_games = per + (i < (int)extra ? 1 : 0);
        at += ws[i].n_games;
        ws[i].max_plies = max_plies; ws[i].rep_limit = rep_limit;
        ws[i].no_progress_limit = no_prog; ws[i].stalemate_loses = stalemate_loses;
        ws[i].mat_shift = mat_shift; ws[i].policy = policy;
        ws[i].tanh_scale_fp = (uint32_t)(tanh_scale * VALUE_FP + 0.5);
        ws[i].regicide = regicide;
        ws[i].format = format;
        ws[i].save_frac = save_pct / 100.0;
        sel_init(&ws[i].sel, save_pct / 100.0, seed_base ^ 0x51EDULL ^ ((uint64_t)i * 0x2545F4914F6CDD1Dull));
        ws[i].write_out = outdir != NULL && !sample && !movestats;
        if (movestats) ws[i].ms = ms_new(max_plies);
        if (ws[i].write_out) {
            snprintf(ws[i].path,  sizeof(ws[i].path),  "%s/shard_%04d.tky", outdir, i);
            snprintf(ws[i].ppath, sizeof(ws[i].ppath), "%s/shard_%04d.tkp", outdir, i);
        }
        if (sample) { ws[i].msample_cap = 2000000; ws[i].msamples = malloc(sizeof(Sample) * ws[i].msample_cap); }
    }

    double t0 = now_s();
    for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker_main, &ws[i]);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    double dt = now_s() - t0;

    uint64_t tp = 0, tg = 0, res[3] = { 0 }, term[5] = { 0 }, maxlen = 0, nmsum = 0, nmmax = 0, bytes = 0, nmpos = 0;
    for (int i = 0; i < threads; i++) {
        tp += ws[i].plies; tg += ws[i].games_done; bytes += ws[i].bytes_out;
        for (int k = 0; k < 3; k++) res[k] += ws[i].res_count[k];
        for (int k = 0; k < 5; k++) term[k] += ws[i].term_count[k];
        if (ws[i].maxlen > maxlen) maxlen = ws[i].maxlen;
        nmsum += ws[i].nmoves_sum; nmpos += ws[i].nmoves_positions;
        if (ws[i].nmoves_max > nmmax) nmmax = ws[i].nmoves_max;
    }
    fprintf(stderr,
        "games=%llu plies=%llu (%.1f/game, max %llu) time=%.2fs threads=%d\n"
        "  %.0f games/s   %.3f Mpositions/s   %.2f GB/h at this rate\n"
        "  result  black=%.4f white=%.4f draw=%.4f\n"
        "  term    royal=%.4f stalemate=%.4f repetition=%.4f noprogress=%.4f plycap=%.4f\n"
        "  moves/position mean=%.1f max=%llu\n",
        (unsigned long long)tg, (unsigned long long)tp, (double)tp / (double)tg, (unsigned long long)maxlen, dt, threads,
        tg / dt, tp / dt / 1e6, (double)(tp + tg) * PLY_REC * 3600.0 / dt / 1e9,
        (double)res[1] / tg, (double)res[0] / tg, (double)res[2] / tg,
        (double)term[0] / tg, (double)term[1] / tg, (double)term[2] / tg, (double)term[3] / tg, (double)term[4] / tg,
        nmpos ? (double)nmsum / (double)nmpos : 0.0, (unsigned long long)nmmax);
    if (bytes) fprintf(stderr, "  wrote %.2f MB\n", bytes / 1e6);

    if (outdir && !sample && !movestats) {
        char fused[512];
        uint64_t count = 0, second = 0;
        if (format == FMT_POSITIONS) {
            for (int i = 0; i < threads; i++) { count += ws[i].pos_written; second += ws[i].sel.total_seen; }
        } else {
            count = tg; second = tp;
        }
        if (fuse) {
            if (fuse_shards(outdir, ws, threads, format, count, second, fused, sizeof(fused)))
                fprintf(stderr, "  fused %d shard%s into %s\n", threads, threads == 1 ? "" : "s", fused);
            else
                fprintf(stderr, "  fusing failed; the per-thread shards are left in %s\n", outdir);
        } else {
            fprintf(stderr, "  left %d shard%s in %s\n", threads, threads == 1 ? "" : "s", outdir);
        }
    }
    {
        Selector agg;
        memset(&agg, 0, sizeof(agg));
        for (int i = 0; i < threads; i++) {
            agg.total_seen += ws[i].sel.total_seen;
            agg.total_kept += ws[i].sel.total_kept;
            for (int b = 0; b < SEL_BINS; b++) { agg.seen[b] += ws[i].sel.seen[b]; agg.kept[b] += ws[i].sel.kept[b]; }
        }
        if (agg.total_seen) {
            double ideal = (double)agg.total_kept / SEL_BINS;
            double worst = 0, chi = 0;
            int nonempty = 0;
            for (int b = 0; b < SEL_BINS; b++) {
                if (!agg.seen[b]) continue;
                nonempty++;
                double d = fabs((double)agg.kept[b] - ideal) / (ideal > 0 ? ideal : 1);
                if (d > worst) worst = d;
                chi += ((double)agg.kept[b] - ideal) * ((double)agg.kept[b] - ideal) / (ideal > 0 ? ideal : 1);
            }
            fprintf(stderr,
                "  selection: kept %llu of %llu positions (%.3f%%, asked %.3f%%), %d/%d label bins used\n"
                "             per-bin worst deviation from even %.1f%%, chi2/bin %.2f\n",
                (unsigned long long)agg.total_kept, (unsigned long long)agg.total_seen,
                100.0 * agg.total_kept / agg.total_seen, save_pct, nonempty, SEL_BINS,
                100.0 * worst, chi / (nonempty ? nonempty : 1));
            FILE *hf = fopen("selection_bins.csv", "w");
            if (hf) {
                fprintf(hf, "bin,label_lo,label_hi,seen,kept\n");
                for (int b = 0; b < SEL_BINS; b++)
                    fprintf(hf, "%d,%.5f,%.5f,%llu,%llu\n", b,
                            -1.0 + 2.0 * b / SEL_BINS, -1.0 + 2.0 * (b + 1) / SEL_BINS,
                            (unsigned long long)agg.seen[b], (unsigned long long)agg.kept[b]);
                fclose(hf);
                fprintf(stderr, "             wrote selection_bins.csv\n");
            }
        }
    }

    if (movestats) {
        MoveStats *m = ms_new(max_plies);
        for (int i = 0; i < threads; i++) ms_merge(m, ws[i].ms);
        FILE *f = fopen("movestats.bin", "wb");
        if (f) {
            uint32_t hdr[4] = { (uint32_t)m->nply, (uint32_t)m->nbucket, MS_BINS, MS_BINW };
            fwrite(hdr, 4, 4, f);
            fwrite(m->cnt, 8, (size_t)m->nply, f);
            fwrite(m->sum, 8, (size_t)m->nply, f);
            fwrite(m->sumsq, 8, (size_t)m->nply, f);
            fwrite(m->psum, 8, (size_t)m->nply, f);
            fwrite(m->mn, 4, (size_t)m->nply, f);
            fwrite(m->mx, 4, (size_t)m->nply, f);
            fwrite(m->hist, 8, (size_t)m->nbucket * MS_BINS, f);
            fclose(f);
            fprintf(stderr, "  wrote movestats.bin (per-ply-index counts + %dx%d histogram)\n",
                    m->nbucket, MS_BINS);
        }
        printf("%6s %10s %9s %9s %7s %7s %8s\n", "ply", "positions", "mean", "sd", "min", "max", "pieces");
        int marks[] = { 0,1,2,3,4,5,10,20,30,50,75,100,150,200,300,400,600,800,1200,1600,2000,2500,2999,3000 };
        for (unsigned k = 0; k < sizeof(marks)/sizeof(marks[0]); k++) {
            int i = marks[k];
            if (i >= m->nply || !m->cnt[i]) continue;
            double mu = (double)m->sum[i] / m->cnt[i];
            double sd = sqrt((double)m->sumsq[i] / m->cnt[i] - mu * mu);
            printf("%6d %10llu %9.1f %9.1f %7u %7u %8.0f\n", i,
                   (unsigned long long)m->cnt[i], mu, sd, m->mn[i], m->mx[i],
                   (double)m->psum[i] / m->cnt[i]);
        }
        return 0;
    }
    if (sample) {
        uint64_t n = 0;
        for (int i = 0; i < threads; i++) n += ws[i].msample_n;
        Sample *all = malloc(sizeof(Sample) * n);
        uint64_t at2 = 0;
        for (int i = 0; i < threads; i++) { memcpy(all + at2, ws[i].msamples, sizeof(Sample) * ws[i].msample_n); at2 += ws[i].msample_n; }
        const char *sf = "samples.bin";
        FILE *f = fopen(sf, "wb");
        if (f) { fwrite(all, sizeof(Sample), n, f); fclose(f);
                 fprintf(stderr, "  wrote %s: %llu rows of {u16 ply, u16 npieces, u16 nmoves, u16 full, i32 material}, VALUE_FP=%d\n",
                         sf, (unsigned long long)n, VALUE_FP); }
        int32_t *ab = malloc(4 * n);
        double sum = 0, sumsq = 0;
        for (uint64_t i = 0; i < n; i++) { double v = all[i].mat / (double)VALUE_FP; sum += v; sumsq += v * v; ab[i] = all[i].mat < 0 ? -all[i].mat : all[i].mat; }
        qsort(ab, n, 4, cmp_i32);
        double mean = sum / n;
        printf("MATSTAT n=%llu mean=%.3f sd=%.3f\n", (unsigned long long)n, mean, sqrt(sumsq / n - mean * mean));
        const double qs[] = { 0.50, 0.75, 0.90, 0.95, 0.99 };
        printf("ABS|mat| quantiles:"); for (unsigned i = 0; i < 5; i++) printf(" p%.0f=%.1f", qs[i]*100, ab[(uint64_t)(qs[i]*(n-1))] / (double)VALUE_FP);
        printf("\n");
    }
    return 0;
}
#endif /* TKY_NO_MAIN */
