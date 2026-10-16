/* Taikyoku Shogi rules: move generation, King-capture detection, move execution.
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

/* ---- buffered output ---------------------------------------------------- */
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

/* material -> stored units, rounded to nearest so a reader can check it to half a unit */
static inline int32_t quantise(int32_t v, int div)
{
    int32_t h = div / 2;
    int32_t q = v >= 0 ? (v + h) / div : -((-v + h) / div);
    return q > 32767 ? 32767 : (q < -32767 ? -32767 : q);
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

/* nnplay.c and tkykids.c include this file for the rules; TKY_NO_MAIN drops the test
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

    fprintf(stderr,
        "usage: tky <perft|kingtest|probe|trace> [args]\n"
        "  perft D                  move counts from the initial position\n"
        "  kingtest [GAMES] [PLIES] the King-capture detectors against each other\n"
        "  trace SEED PLIES [V] [DET]  one random playout, for cross-checking\n"
        "  probe                    move list for a position read from stdin\n");
    return 1;
}
#endif /* TKY_NO_MAIN */
