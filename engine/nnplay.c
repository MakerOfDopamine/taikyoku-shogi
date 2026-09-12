/* Neural-network self-play for Taikyoku Shogi.
 *
 * The rules, the move generator and the move executor are tky.c's, included wholesale so
 * there is exactly one copy of game.js's semantics in this directory. This file adds the
 * policy: at every ply, evaluate the position *after* every legal move with the network
 * and sample the move from a softmax over the negated leaf values.
 *
 * Perspective. Everything the network sees is in the frame of whoever is to move in that
 * position, which is the canonicalisation unpack.canonical_board performs: black to move
 * is the board as stored, white to move is rotated 180 degrees with every colour bit
 * flipped. That is exact, not approximate -- see CLAUDE.md section 13. A child position is
 * therefore encoded for the *opponent*, so its value is the opponent's, and the policy
 * softmaxes the negation.
 *
 * Inference is ONNX Runtime's C API on the CUDA execution provider. The sub-batch size is
 * measured on the GPU actually present at startup rather than assumed.
 */
#define TKY_NO_MAIN
#include "tky.c"

#include <dlfcn.h>

#include "onnxruntime_c_api.h"

#define NN_EMBED 603              /* unpack.embedding_index range, 0..602 */
#define MEAS_S   0.25             /* per size, minimum wall clock spent measuring */

/* FNV-1a over the graph file: enough to tell two sets of weights apart in a header field,
 * and it costs one streaming pass over 12 MB at startup. */
static uint32_t hash_file(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    uint32_t h = 2166136261u;
    unsigned char buf[1 << 16];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0)
        for (ssize_t i = 0; i < n; i++) h = (h ^ buf[i]) * 16777619u;
    close(fd);
    return h;
}

/* ---- ONNX Runtime ---------------------------------------------------------- */
static const OrtApi *ort = NULL;

#define ORT_CHECK(expr) do { \
        OrtStatus *st_ = (expr); \
        if (st_) { fprintf(stderr, "onnxruntime: %s\n", ort->GetErrorMessage(st_)); \
                   ort->ReleaseStatus(st_); exit(1); } } while (0)

typedef struct {
    OrtEnv           *env;
    OrtSessionOptions*opts;
    OrtSession       *sess;
    OrtMemoryInfo    *meminfo;
    const char       *in_name, *out_name[2];
    char             *in_name_owned, *out_name_owned[2];
    OrtAllocator     *alloc;
    int               fp16_out;    /* the value output comes back as float16 */
    int               on_cuda;     /* the CUDA provider was actually enabled */
    int32_t          *probe_buf;   /* scratch boards for the startup probe only */
    int               sub_batch;
    double            probe_rate;   /* boards/s at sub_batch, when the probe ran */
    int               cap;         /* probe_buf capacity in boards */
} Net;

static float half_to_float(uint16_t h)
{
    uint32_t s = (uint32_t)(h >> 15), e = (uint32_t)((h >> 10) & 0x1F), m = (uint32_t)(h & 0x3FF);
    if (e == 0) {                            /* zero or subnormal, value m * 2^-24 */
        float f = (float)m * 5.9604645e-8f;
        return s ? -f : f;
    }
    uint32_t bits = (s << 31) | (e == 31 ? 0x7F800000u | (m << 13)
                                         : ((e + 127 - 15) << 23) | (m << 13));
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Runs `nb` boards from `src`, writing nb values to `dst`. Returns an OrtStatus rather
 * than exiting so the startup probe can see OOM. `src` is borrowed, not copied. */
static OrtStatus *net_run_raw(Net *n, const int32_t *src, int nb, float *dst)
{
    int64_t shape[2] = { nb, NSQ };
    OrtValue *in = NULL, *out[2] = { NULL, NULL };

    OrtStatus *st = ort->CreateTensorWithDataAsOrtValue(
            n->meminfo, (void *)src, (size_t)nb * NSQ * sizeof(int32_t),
            shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &in);
    if (st) return st;

    st = ort->Run(n->sess, NULL, &n->in_name, (const OrtValue *const *)&in, 1, n->out_name, 2, out);
    ort->ReleaseValue(in);
    if (!st) {
        void *raw = NULL;
        st = ort->GetTensorMutableData(out[0], &raw);
        if (!st) {
            if (n->fp16_out) { const uint16_t *h = (const uint16_t *)raw;
                               for (int i = 0; i < nb; i++) dst[i] = half_to_float(h[i]); }
            else             { memcpy(dst, raw, (size_t)nb * sizeof(float)); }
        }
    }
    for (int i = 0; i < 2; i++) if (out[i]) ort->ReleaseValue(out[i]);
    return st;
}

/* Evaluates `nboards` boards laid out contiguously in `boards`, sub-batching. */
static void net_eval(Net *n, const int32_t *boards, int nboards, float *values)
{
    for (int at = 0; at < nboards; ) {
        int nb = nboards - at;
        if (nb > n->sub_batch) nb = n->sub_batch;
        ORT_CHECK(net_run_raw(n, boards + (size_t)at * NSQ, nb, values + at));
        at += nb;
    }
}

/* Resident set in bytes, or 0 if it cannot be read. */
static size_t rss_bytes(void)
{
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long total = 0, res = 0;
    int got = fscanf(f, "%lu %lu", &total, &res);
    fclose(f);
    return got == 2 ? (size_t)res * (size_t)sysconf(_SC_PAGESIZE) : 0;
}

/* MemAvailable in bytes, or 0 if it cannot be read. */
static size_t mem_available(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[64];
    unsigned long kb = 0;
    size_t out = 0;
    while (fscanf(f, "%63s %lu", key, &kb) == 2) {
        if (!strcmp(key, "MemAvailable:")) { out = (size_t)kb * 1024; break; }
        int c; while ((c = fgetc(f)) != EOF && c != '\n') { }
    }
    fclose(f);
    return out;
}

/* Measures the sub-batch size on the GPU that is actually present: doubles the batch and
 * keeps the fastest. The machine this was developed on saturates at batch 2; a bigger card
 * will not, which is why this is measured rather than assumed.
 *
 * Three things stop the doubling, and none of them is "the allocator refused". That was
 * the original stopping rule and it is only safe on a device allocator: the CPU provider's
 * host allocator does not refuse, it overcommits, and since the one unfused attention layer
 * holds an 8 x 1296 x 1296 score matrix per board -- 26.9 MB in fp16 -- a probe to 8192
 * asks for 220 GB and takes the machine down with it. So the probe now stops when
 *
 *   - throughput falls more than 10% below the best seen, twice in a row. It has to be a
 *     real regression rather than merely a small gain, because the curve rises gently
 *     (266 to 271 boards/s from batch 1 to 2 on the development GPU); and it has to happen
 *     twice, because on a fast card a batch is a couple of milliseconds and one allocation
 *     stall otherwise vetoes every size above it;
 *   - on the CPU provider only, the next batch's predicted host memory, from the marginal
 *     RSS growth per board, exceeding a third of MemAvailable. That is where the whole
 *     problem lives: the host allocator overcommits instead of refusing. CUDA is excluded
 *     because its allocator does refuse, and because the RSS estimate is wrong there
 *     anyway -- on a 170 GB host it predicted 74 GB for sub-batch 1024 and stopped a walk
 *     that had plateaued six doublings earlier;
 *   - a single batch took longer than `budget` seconds.
 *
 * The allocator refusing still stops it, but is now a backstop rather than the mechanism.
 */
static int net_probe(Net *n, int cap, int reps, int verbose)
{
    double budget = 10.0;
    size_t avail = mem_available(), per_board = 0, prev_rss = 0;
    int prev_b = 0;
    if (!n->on_cuda && cap > 8) {
        cap = 8;
        if (verbose) fprintf(stderr, "  running on the CPU: probe capped at %d\n", cap);
    }
    int best = 1, bad = 0;
    double best_rate = 0.0;
    for (int b = 1; b <= cap; b *= 2) {
        n->probe_buf = (int32_t *)realloc(n->probe_buf, (size_t)b * NSQ * sizeof(int32_t));
        if (!n->probe_buf) { fprintf(stderr, "oom in probe buffer\n"); exit(1); }
        n->cap = b;
        for (size_t i = 0; i < (size_t)b * NSQ; i++) n->probe_buf[i] = (int32_t)(i % NN_EMBED);
        float *tmp = (float *)malloc((size_t)b * sizeof(float));

        /* Warm up twice and discard: the first call on a new shape pays for cuDNN
         * algorithm selection and arena growth, and on a fast card that one-off is several
         * times the steady-state cost. Measuring it made the probe stop at sub-batch 4 on
         * a card that wanted a far bigger one. */
        OrtStatus *st = NULL;
        for (int w = 0; w < 2 && !st; w++) st = net_run_raw(n, n->probe_buf, b, tmp);
        if (st) {
            if (verbose) fprintf(stderr, "  sub-batch %5d : %s\n", b, ort->GetErrorMessage(st));
            ort->ReleaseStatus(st); free(tmp);
            break;
        }

        /* Median of N, over at least MEAS_S of wall clock. Not the mean, which one
         * allocation stall can veto a batch size with -- that is what stopped the probe at
         * sub-batch 4 on a card that wanted far more. Not the minimum either: small batches
         * have the higher variance, so best-of rewards them for it and inverted the curve
         * on the development GPU. */
        double ts[32], t_start = now_s();
        int r = 0;
        while (r < 32 && (r < reps || now_s() - t_start < MEAS_S)) {
            for (size_t i = 0; i < (size_t)b * NSQ; i++) n->probe_buf[i] = (int32_t)((i + r) % NN_EMBED);
            double t0 = now_s();
            st = net_run_raw(n, n->probe_buf, b, tmp);
            double dt1 = now_s() - t0;
            if (st) break;
            ts[r++] = dt1;
        }
        double best_t = 0.0;
        if (r) {
            for (int i = 1; i < r; i++) {           /* insertion sort, r <= 32 */
                double v = ts[i]; int j = i - 1;
                while (j >= 0 && ts[j] > v) { ts[j + 1] = ts[j]; j--; }
                ts[j + 1] = v;
            }
            best_t = ts[r / 2];
        }
        free(tmp);
        if (st) { if (verbose) fprintf(stderr, "  sub-batch %5d : %s\n", b, ort->GetErrorMessage(st));
                  ort->ReleaseStatus(st); break; }
        double rate = (double)b / best_t;
        if (verbose) fprintf(stderr, "  sub-batch %5d : %8.1f boards/s  (%.2f ms/batch, median of %d)\n",
                             b, rate, best_t * 1000.0, r);

        /* Marginal cost, not average: the first batch also pays for loading the provider
         * and building the CUDA context, and charging that per board would over-estimate
         * by orders of magnitude and stop the probe almost immediately. */
        size_t rss = rss_bytes();
        if (prev_b && rss > prev_rss) {
            size_t pb = (rss - prev_rss) / (size_t)(b - prev_b);
            if (pb > per_board) per_board = pb;
        }
        prev_rss = rss;
        prev_b = b;

        /* One regression is not evidence -- it takes two in a row. The memory, time and
         * allocator guards below are what actually bound the probe, so this rule is free
         * to be generous. */
        if (rate < best_rate * 0.9) bad++; else bad = 0;
        if (rate > best_rate) { best_rate = rate; best = b; }
        if (bad >= 2) {
            if (verbose) fprintf(stderr, "  stopping: throughput down twice running\n");
            break;
        }
        if (best_t > budget) {
            if (verbose) fprintf(stderr, "  stopping: a batch already takes %.1fs\n", best_t);
            break;
        }
        if (!n->on_cuda && per_board && avail && per_board * (size_t)b * 2 > avail / 3) {
            if (verbose)
                fprintf(stderr, "  stopping: sub-batch %d would need ~%.1f GB of %.1f GB available\n",
                        b * 2, per_board * (size_t)b * 2 / 1e9, avail / 1e9);
            break;
        }
    }
    if (verbose) fprintf(stderr, "  chosen sub-batch %d (%.1f boards/s)\n", best, best_rate);
    n->probe_rate = best_rate;
    return best;
}

static void net_open(Net *n, const char *model, int device_id, int sub_batch,
                     int probe_cap, int tf32, int allow_cpu, int verbose)
{
    memset(n, 0, sizeof(*n));
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    /* FATAL: every failure that matters comes back as an OrtStatus and is printed here;
     * ORT's own log lines are ANSI-coloured noise, which is worse still in a notebook. */
    ORT_CHECK(ort->CreateEnv(ORT_LOGGING_LEVEL_FATAL, "nnplay", &n->env));
    ORT_CHECK(ort->CreateSessionOptions(&n->opts));
    ORT_CHECK(ort->SetSessionGraphOptimizationLevel(n->opts, ORT_ENABLE_ALL));

    /* A CPU-only runtime fails at CreateCUDAProviderOptions, a GPU one missing a CUDA
     * library fails at the append. Both stop here unless --allow-cpu: a silent CPU
     * fallback is ~100x slower, looks like a hang, and has twice been mistaken for a
     * working setup. */
    OrtCUDAProviderOptionsV2 *cuda = NULL;
    OrtStatus *st = ort->CreateCUDAProviderOptions(&cuda);
    if (!st) {
        char devbuf[16];
        snprintf(devbuf, sizeof(devbuf), "%d", device_id);
        const char *keys[] = { "device_id", "arena_extend_strategy", "use_tf32" };
        const char *vals[] = { devbuf, "kSameAsRequested", tf32 ? "1" : "0" };
        st = ort->UpdateCUDAProviderOptions(cuda, keys, vals, 3);
        if (!st) st = ort->SessionOptionsAppendExecutionProvider_CUDA_V2(n->opts, cuda);
        ort->ReleaseCUDAProviderOptions(cuda);
    }
    n->on_cuda = (st == NULL);
    if (st) {
        const char *why = ort->GetErrorMessage(st);
        if (!allow_cpu) {
            fprintf(stderr, "\nERROR: the CUDA execution provider did not load:\n  %s\n\n", why);
            if (strstr(why, "not enabled in this build"))
                fprintf(stderr,
                    "  This libonnxruntime is a CPU-only build: the 'onnxruntime' wheel rather than\n"
                    "  'onnxruntime-gpu'. Re-run setup.sh, which puts the GPU runtime in the\n"
                    "  bundle's own ortlib/ directory, where no package manager can swap it back.\n");
            else
                fprintf(stderr,
                    "  Usually a CUDA library the provider needs is missing, and the message above\n"
                    "  names it. setup.sh lists every missing one when its self-check fails.\n");
            fprintf(stderr, "  To run on the CPU anyway (~100x slower), pass --allow-cpu.\n");
            exit(2);
        }
        fprintf(stderr, "WARNING: no CUDA execution provider (%s)\n"
                        "         running on the CPU because --allow-cpu was given; ~100x slower\n", why);
        ort->ReleaseStatus(st);
    }

    ORT_CHECK(ort->CreateSession(n->env, model, n->opts, &n->sess));
    ORT_CHECK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &n->meminfo));
    ORT_CHECK(ort->GetAllocatorWithDefaultOptions(&n->alloc));

    size_t nin = 0, nout = 0;
    ORT_CHECK(ort->SessionGetInputCount(n->sess, &nin));
    ORT_CHECK(ort->SessionGetOutputCount(n->sess, &nout));
    if (nin != 1 || nout != 2) {
        fprintf(stderr, "model must have 1 input and 2 outputs (value, material); got %zu/%zu\n", nin, nout);
        exit(1);
    }
    ORT_CHECK(ort->SessionGetInputName(n->sess, 0, n->alloc, &n->in_name_owned));
    n->in_name = n->in_name_owned;
    for (int i = 0; i < 2; i++) {
        ORT_CHECK(ort->SessionGetOutputName(n->sess, i, n->alloc, &n->out_name_owned[i]));
        n->out_name[i] = n->out_name_owned[i];
    }

    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *tsi = NULL;
    ONNXTensorElementDataType et;
    ORT_CHECK(ort->SessionGetOutputTypeInfo(n->sess, 0, &ti));
    ORT_CHECK(ort->CastTypeInfoToTensorInfo(ti, &tsi));
    ORT_CHECK(ort->GetTensorElementType(tsi, &et));
    n->fp16_out = (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    ort->ReleaseTypeInfo(ti);

    if (verbose)
        fprintf(stderr, "model %s (hash %08x): input '%s', outputs '%s','%s', value is %s\n",
                model, hash_file(model), n->in_name, n->out_name[0], n->out_name[1],
                n->fp16_out ? "float16" : "float32");

    n->sub_batch = sub_batch > 0 ? sub_batch : net_probe(n, probe_cap, 3, verbose);
    free(n->probe_buf);
    n->probe_buf = NULL;
}

static void net_close(Net *n)
{
    for (int i = 0; i < 2; i++)
        if (n->out_name_owned[i]) { OrtStatus *st = ort->AllocatorFree(n->alloc, n->out_name_owned[i]); if (st) ort->ReleaseStatus(st); }
    if (n->in_name_owned) { OrtStatus *st = ort->AllocatorFree(n->alloc, n->in_name_owned); if (st) ort->ReleaseStatus(st); }
    if (n->meminfo) ort->ReleaseMemoryInfo(n->meminfo);
    if (n->sess) ort->ReleaseSession(n->sess);
    if (n->opts) ort->ReleaseSessionOptions(n->opts);
    if (n->env) ort->ReleaseEnv(n->env);
    free(n->probe_buf);
}

/* ---- GPU utilisation ------------------------------------------------------- */
/* NVML through dlopen, so there is no link-time dependency and no CUDA headers: the
 * binary still builds and runs on a box without it, just without the numbers.
 *
 * Read `util` for what it is. NVML reports the percentage of the sample period during
 * which at least one kernel was resident -- not how much of the card's arithmetic those
 * kernels used. 100% here means "never idle", not "saturated". The sub-batch curve is the
 * better saturation evidence: if the GPU had headroom, throughput would still be climbing
 * with batch size instead of going flat.
 */
typedef struct { unsigned int gpu, mem; } NvUtil;
typedef struct { unsigned long long total, free, used; } NvMem;

typedef struct {
    void      *lib, *dev;
    int      (*util)(void *, NvUtil *);
    int      (*meminfo)(void *, NvMem *);
    int      (*shutdown)(void);
    pthread_t th;
    pthread_mutex_t mu;
    volatile int stop;
    double    sum_gpu, sum_mem;
    long      n;
    unsigned long long used, total;
} GpuMon;

static void *gpumon_thread(void *arg)
{
    GpuMon *m = (GpuMon *)arg;
    while (!m->stop) {
        NvUtil u;
        NvMem mem;
        int ok_u = m->util(m->dev, &u) == 0;
        int ok_m = m->meminfo(m->dev, &mem) == 0;
        if (ok_u || ok_m) {
            pthread_mutex_lock(&m->mu);
            if (ok_u) { m->sum_gpu += u.gpu; m->sum_mem += u.mem; m->n++; }
            if (ok_m) { m->used = mem.used; m->total = mem.total; }
            pthread_mutex_unlock(&m->mu);
        }
        struct timespec ts = { 0, 100 * 1000 * 1000 };   /* 100 ms */
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static int gpumon_start(GpuMon *m, int device_id)
{
    memset(m, 0, sizeof(*m));
    m->lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!m->lib) return 0;
    int (*init)(void) = (int (*)(void))dlsym(m->lib, "nvmlInit_v2");
    int (*handle)(unsigned int, void **) =
        (int (*)(unsigned int, void **))dlsym(m->lib, "nvmlDeviceGetHandleByIndex_v2");
    m->util     = (int (*)(void *, NvUtil *))dlsym(m->lib, "nvmlDeviceGetUtilizationRates");
    m->meminfo  = (int (*)(void *, NvMem *))dlsym(m->lib, "nvmlDeviceGetMemoryInfo");
    m->shutdown = (int (*)(void))dlsym(m->lib, "nvmlShutdown");
    if (!init || !handle || !m->util || !m->meminfo || init() != 0 || handle(device_id, &m->dev) != 0) {
        dlclose(m->lib); m->lib = NULL; return 0;
    }
    pthread_mutex_init(&m->mu, NULL);
    pthread_create(&m->th, NULL, gpumon_thread, m);
    return 1;
}

/* Means since the previous call, then resets. */
static void gpumon_take(GpuMon *m, double *gpu, double *mem, double *used_gb, double *total_gb)
{
    pthread_mutex_lock(&m->mu);
    *gpu = m->n ? m->sum_gpu / m->n : -1.0;
    *mem = m->n ? m->sum_mem / m->n : -1.0;
    *used_gb = m->used / 1e9;
    *total_gb = m->total / 1e9;
    m->sum_gpu = m->sum_mem = 0.0;
    m->n = 0;
    pthread_mutex_unlock(&m->mu);
}

static void gpumon_stop(GpuMon *m)
{
    if (!m->lib) return;
    m->stop = 1;
    pthread_join(m->th, NULL);
    if (m->shutdown) m->shutdown();
    dlclose(m->lib);
    m->lib = NULL;
}

/* ---- board encoding -------------------------------------------------------- */
/* One board as NSQ int32 embedding indices, canonicalised to p's side to move. Mirrors
 * unpack.canonical_board followed by unpack.embedding_index; run_tests checks the two
 * against each other square for square. */
static void encode_stm(const Pos *p, int32_t *out)
{
    if (p->turn == SIDE_BLACK) {
        for (int sq = 0; sq < NSQ; sq++)
            out[sq] = (int32_t)(p->code[SQ2P[sq]] >> 1) - 1;
    } else {
        for (int sq = 0; sq < NSQ; sq++) {
            uint16_t c = p->code[SQ2P[NSQ - 1 - sq]];
            if (C_IDX(c) != EMPTY_IDX) c ^= 2;         /* colour bit; empties keep colour 1 */
            out[sq] = (int32_t)(c >> 1) - 1;
        }
    }
}

/* The absolute board, as the .tkn record stores it: packed codes, x*36 + y, black's frame. */
static void snapshot(const Pos *p, uint16_t *out)
{
    for (int sq = 0; sq < NSQ; sq++) out[sq] = p->code[SQ2P[sq]];
}

/* ---- output format --------------------------------------------------------- */
#define TKN_MAGIC   "TKYNNSP"
#define TKN_VERSION 1u
#define TKN_FILE_HDR 128
#define TKN_GAME_HDR 16
#define TKN_BOARD   (NSQ * 2)
#define TKN_REC     (TKN_BOARD + 16)

/* Flags on a ply record. */
enum { PF_CAPTURE = 1, PF_PROMOTION = 2, PF_FORCED_KING = 4, PF_TERMINAL = 8 };

typedef struct {
    uint16_t board[NSQ];    /* absolute, after the move */
    uint32_t move;          /* special << 28 | origin << 17 | target << 6 */
    float    value;         /* the network's value for the chosen leaf, opponent's view */
    uint16_t n_legal;       /* legal moves in the position the move was played from */
    int16_t  material;      /* after the move, black minus white, mat_shift units */
    uint8_t  side;          /* the mover: 1 black, 0 white */
    uint8_t  flags;
    uint16_t pad;
} PlyRecNN;
_Static_assert(sizeof(PlyRecNN) == TKN_REC, "PlyRecNN must match the on-disk record exactly");

/* ---- policy ---------------------------------------------------------------- */
/* Softmax over the negated leaf values. The child was encoded for the opponent, so its
 * value is the opponent's; negating puts it back in the mover's frame. temperature 0 is
 * greedy. */
static int sample_softmax(const float *v, int n, double temperature, Rng *rng, double *w)
{
    if (temperature <= 0.0) {
        int best = 0;
        for (int i = 1; i < n; i++) if (v[i] < v[best]) best = i;
        return best;
    }
    double mx = -1e30;
    for (int i = 0; i < n; i++) { double l = -(double)v[i] / temperature; if (l > mx) mx = l; }
    double sum = 0.0;
    for (int i = 0; i < n; i++) { w[i] = exp(-(double)v[i] / temperature - mx); sum += w[i]; }

    double r = (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0) * sum, acc = 0.0;
    for (int i = 0; i < n; i++) { acc += w[i]; if (r < acc) return i; }
    return n - 1;
}

/* ---- Elo against a random baseline ----------------------------------------- */
/* The random mover is anchored at Elo 0 and the bot's rating is the posterior over the
 * difference, in BayesElo's model: Rao-Kupper with an explicit draw rating, plus a
 * first-move advantage, since black moves first here.
 *
 *   P(win)  = 1 / (1 + 10^((-delta + draw_elo) / 400))
 *   P(loss) = 1 / (1 + 10^(( delta + draw_elo) / 400))
 *   P(draw) = 1 - P(win) - P(loss)
 *
 * delta is the bot's rating plus the advantage when it moves first, minus it when it does
 * not, so the two colours are two different likelihood terms over the same rating. The
 * posterior is a flat prior times that likelihood on a grid; the reported interval is the
 * 95% credible interval of the rating's marginal, which is what makes it an interval about
 * the rating rather than a normal approximation to a win rate.
 */
#define ELO_LO   (-1500)
#define ELO_HI   ( 1500)
#define ELO_N    (ELO_HI - ELO_LO + 1)
#define ED_HI    400
#define ED_STEP  5
#define ED_N     (ED_HI / ED_STEP + 1)
#define AD_HI    200
#define AD_STEP  5
#define AD_N     (2 * AD_HI / AD_STEP + 1)
#define DELTA_LO (ELO_LO - AD_HI)
#define DELTA_N  (ELO_HI + AD_HI - DELTA_LO + 1)

typedef struct { long w, l, d; } Rec;          /* from the bot's point of view */

typedef struct {
    double elo, lo, hi, draw_elo, advantage;
    int    saturated;                          /* the posterior ran into the grid edge */
} EloFit;

/* Log-likelihood of one colour's record at every (delta, draw_elo) on the grid. */
static void elo_terms(const Rec *r, double *out)
{
    for (int di = 0; di < DELTA_N; di++) {
        double delta = DELTA_LO + di;
        for (int ei = 0; ei < ED_N; ei++) {
            double ed = ei * ED_STEP;
            double pw = 1.0 / (1.0 + pow(10.0, (-delta + ed) / 400.0));
            double pl = 1.0 / (1.0 + pow(10.0, ( delta + ed) / 400.0));
            double pd = 1.0 - pw - pl;
            if (pw < 1e-15) pw = 1e-15;
            if (pl < 1e-15) pl = 1e-15;
            if (pd < 1e-15) pd = 1e-15;
            out[di * ED_N + ei] = r->w * log(pw) + r->l * log(pl) + r->d * log(pd);
        }
    }
}

static void elo_fit(const Rec *as_black, const Rec *as_white, double *marg, EloFit *f)
{
    static double *A = NULL, *B = NULL;
    if (!A) { A = malloc(sizeof(double) * DELTA_N * ED_N); B = malloc(sizeof(double) * DELTA_N * ED_N); }
    if (!A || !B) { fprintf(stderr, "oom in elo grid\n"); exit(1); }
    elo_terms(as_black, A);
    elo_terms(as_white, B);

    /* pass 1: the maximum, so the exponentials below cannot overflow */
    double best = -1e300;
    for (int i = 0; i < ELO_N; i++)
        for (int ai = 0; ai < AD_N; ai++) {
            int adv = -AD_HI + ai * AD_STEP;
            int db = (ELO_LO + i) + adv - DELTA_LO, dw = (ELO_LO + i) - adv - DELTA_LO;
            for (int ei = 0; ei < ED_N; ei++) {
                double ll = A[db * ED_N + ei] + B[dw * ED_N + ei];
                if (ll > best) best = ll;
            }
        }

    /* pass 2: marginals */
    double total = 0.0, ed_num = 0.0, ad_num = 0.0;
    for (int i = 0; i < ELO_N; i++) marg[i] = 0.0;
    for (int i = 0; i < ELO_N; i++)
        for (int ai = 0; ai < AD_N; ai++) {
            int adv = -AD_HI + ai * AD_STEP;
            int db = (ELO_LO + i) + adv - DELTA_LO, dw = (ELO_LO + i) - adv - DELTA_LO;
            for (int ei = 0; ei < ED_N; ei++) {
                double p = exp(A[db * ED_N + ei] + B[dw * ED_N + ei] - best);
                marg[i] += p;
                total += p;
                ed_num += p * (ei * ED_STEP);
                ad_num += p * adv;
            }
        }

    int mode = 0;
    for (int i = 1; i < ELO_N; i++) if (marg[i] > marg[mode]) mode = i;
    double acc = 0.0;
    int lo = 0, hi = ELO_N - 1;
    for (int i = 0; i < ELO_N; i++) {
        acc += marg[i];
        if (acc <= 0.025 * total) lo = i;
        if (acc <= 0.975 * total) hi = i;
    }
    f->elo = ELO_LO + mode;
    f->lo = ELO_LO + lo;
    f->hi = ELO_LO + hi;
    f->draw_elo = total > 0 ? ed_num / total : 0.0;
    f->advantage = total > 0 ? ad_num / total : 0.0;
    f->saturated = (mode == 0 || mode == ELO_N - 1);
}

/* ---- self-play ------------------------------------------------------------- */
typedef struct {
    int      max_plies, rep_limit, no_progress_limit, stalemate_loses, mat_shift, regicide;
    int      progress;      /* print a line every this many plies; 0 is off */
    int      stats;         /* per-ply wall-clock breakdown and GPU utilisation */
    double   temperature;
    uint64_t seed_base;
    const char *out_path;
    int      verbose;
} Cfg;

static void put_file_header(Sink *s, const Cfg *c, const Net *net, const char *model,
                            uint32_t model_hash)
{
    unsigned char hdr[TKN_FILE_HDR];
    uint32_t v;
    memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, TKN_MAGIC, 8);
    v = TKN_VERSION;                  memcpy(hdr + 8,  &v, 4);
    v = TKN_FILE_HDR;                 memcpy(hdr + 12, &v, 4);
    v = TKN_GAME_HDR;                 memcpy(hdr + 16, &v, 4);
    v = TKN_REC;                      memcpy(hdr + 20, &v, 4);
    v = VALUE_FP;                     memcpy(hdr + 24, &v, 4);
    v = (uint32_t)c->mat_shift;       memcpy(hdr + 28, &v, 4);
    memcpy(hdr + 32, &c->seed_base, 8);
    /* 40: games, 48: plies -- patched at close */
    double t = c->temperature;        memcpy(hdr + 56, &t, 8);
    v = (uint32_t)c->max_plies;       memcpy(hdr + 64, &v, 4);
    v = (uint32_t)c->rep_limit;       memcpy(hdr + 68, &v, 4);
    v = (uint32_t)c->no_progress_limit; memcpy(hdr + 72, &v, 4);
    v = (uint32_t)c->stalemate_loses; memcpy(hdr + 76, &v, 4);
    v = (uint32_t)c->regicide;        memcpy(hdr + 80, &v, 4);
    v = 36;                           memcpy(hdr + 84, &v, 4);
    v = NPIECE;                       memcpy(hdr + 88, &v, 4);
    v = NSQ;                          memcpy(hdr + 92, &v, 4);
    v = (uint32_t)net->sub_batch;     memcpy(hdr + 96, &v, 4);
    v = (uint32_t)net->fp16_out;      memcpy(hdr + 100, &v, 4);
    /* 104..127: "<basename>@<8 hex of the graph's hash>", NUL padded. The hash is what
     * makes a shard attributable to a specific set of weights: in a self-play loop the
     * file name never changes, so the name alone cannot tell one generation from the
     * next. Readers take the field as a NUL-terminated string either way. */
    const char *base = strrchr(model, '/');
    base = base ? base + 1 : model;
    char tag[32];
    snprintf(tag, sizeof(tag), "%.14s@%08x", base, model_hash);
    memcpy(hdr + 104, tag, strlen(tag) < 24 ? strlen(tag) : 24);
    sink_put(s, hdr, TKN_FILE_HDR);
}

/* The start-position reference export_onnx.py writes into net_manifest.json. A string
 * search rather than a JSON parser: the file is ours, and each key occurs exactly once. */
static int manifest_ref(const char *path, double *val, double *tol)
{
    static char buf[1 << 16];
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    char *p = strstr(buf, "\"start_position_value\"");
    if (!p || !(p = strchr(p, ':'))) return 0;
    *val = strtod(p + 1, NULL);
    *tol = 3e-3;
    char *q = strstr(buf, "\"tolerance\"");
    if (q && (q = strchr(q, ':'))) *tol = strtod(q + 1, NULL);
    return 1;
}

/* Pure C, so proving a setup needs no Python, numpy or unpack.py on the far end. */
static int mode_selfcheck(Net *net, const char *manifest)
{
    double want, tol;
    if (!manifest_ref(manifest, &want, &tol)) {
        fprintf(stderr, "selfcheck: no start_position_value in %s -- re-run export_onnx.py\n", manifest);
        return 1;
    }
    Pos *p = malloc(sizeof(Pos));
    pos_init(p);                               /* black to move: already the canonical frame */
    int32_t *b = malloc(NSQ * sizeof(int32_t));
    encode_stm(p, b);
    float v = 0.0f;
    net_eval(net, b, 1, &v);
    double d = fabs((double)v - want);
    int ok = d < tol;
    printf("\nselfcheck\n"
           "  provider        %s\n"
           "  start position  %+.9f   expected %+.9f   deviation %.2e (tolerance %.0e)   %s\n",
           net->on_cuda ? "CUDA" : "CPU (--allow-cpu)", (double)v, want, d, tol, ok ? "PASS" : "FAIL");
    if (net->probe_rate > 0)
        printf("  throughput      %.0f boards/s at sub-batch %d; pass --sub-batch %d to skip the probe\n",
               net->probe_rate, net->sub_batch, net->sub_batch);
    if (!ok)
        printf("  This runtime does not reproduce the exported graph. Re-fuse it here with:\n"
               "    pip install onnx && python3 fuse.py net_fp16_raw.onnx net_fp16.onnx\n");
    fflush(stdout);
    free(b);
    free(p);
    return ok ? 0 : 1;
}

static const char USAGE[] =
    "usage: nnplay <play|elo|selfcheck|eval|encode> [options]\n"
    "  play    self-play games and write a .tkn shard\n"
    "    --out FILE          where to write (omit to play without recording)\n"
    "    --games N           games to play (default 1)\n"
    "    --first-game N      index of the first game, so shards can be split across runs\n"
    "    --temperature T     softmax temperature over the negated leaf values (default 1.0,\n"
    "                        0 is greedy)\n"
    "    --max-plies N       ply cap, scored as a draw (default 3000)\n"
    "    --rep-limit N       repetition limit (default 4, 0 disables)\n"
    "    --no-progress N     plies without a capture or promotion before a draw (0 disables)\n"
    "    --no-regicide       do not force an available King capture\n"
    "    --seed S            base seed\n"
    "  elo     rate the network against a baseline anchored at 0. Colours alternate and\n"
    "          regicide is forced for both sides. --temperature defaults to 0 here\n"
    "          (playing strength) rather than the sampling policy's value.\n"
    "    --baseline FILE     opponent graph, default baseline_<precision>.onnx, which\n"
    "                        export_onnx.py writes from baseline.pt\n"
    "    --baseline random   the uniform mover instead: an absolute anchor, but it stops\n"
    "                        discriminating once the bot beats it every game\n"
    "  selfcheck  prove the setup: the CUDA provider loads, the start position evaluates\n"
    "          to the value export_onnx.py recorded in net_manifest.json, and the probe\n"
    "          reports this GPU's throughput. Exit 0 only if all of that holds.\n"
    "  eval    values for boards read from stdin: whitespace ints, 1296 per board\n"
    "  encode  encode_stm for a position read from stdin, for the canonicalisation test\n"
    "\n  --model FILE / --precision fp16|fp32   which graph to run (default fp16)\n"
    "  --device N            CUDA device (default 0)\n"
    "  --sub-batch N         boards per inference call; omit to measure it at startup\n"
    "  --probe-cap N         largest sub-batch the startup probe will try (default 8192)\n"
    "  --allow-cpu           run on the CPU when CUDA is unavailable, instead of stopping\n"
    "  --manifest FILE       where selfcheck finds the reference (default net_manifest.json)\n"
    "  --no-tf32             full fp32 matmuls instead of TF32; only meaningful with\n"
    "                        --precision fp32, where it is what closes the last ~1e-3\n"
    "                        against torch\n"
    "  --progress-every N    a line every N plies within a game (default 25, 0 off)\n"
    "  --stats               add where each ply's wall clock went, and GPU utilisation\n"
    "                        sampled from NVML at 10 Hz if libnvidia-ml is present\n"
    "  --quiet               no progress output at all\n";

/* Reads a position as `clear`, `turn N` and `sq idx colour promoted` lines, and prints
 * encode_stm of it. run_tests.sh diffs this against unpack.canonical_board followed by
 * unpack.embedding_index, which is the claim that the C and python frames agree. */
static int mode_encode(void)
{
    Pos *p = malloc(sizeof(Pos));
    pos_init(p);
    int32_t *e = malloc(NSQ * 4);
    char line[64];
    while (fgets(line, sizeof(line), stdin)) {
        int a, b, c, d;
        if (sscanf(line, "turn %d", &a) == 1) { p->turn = a; continue; }
        if (!strncmp(line, "clear", 5)) { for (int q = 0; q < NSQ; q++) sq_set(p, SQ2P[q], EMPTY_CODE); continue; }
        if (sscanf(line, "%d %d %d %d", &a, &b, &c, &d) == 4) sq_set(p, SQ2P[a], (uint16_t)CODE(b, c, d));
    }
    encode_stm(p, e);
    for (int q = 0; q < NSQ; q++) printf("%d\n", e[q]);
    free(e); free(p);
    return 0;
}

/* Boards in, values out: the same path self-play uses, so a parity check against torch
 * covers the export, the fusion and the C plumbing at once. */
static int mode_eval(Net *net)
{
    int32_t *boards = NULL;
    int n = 0, cap = 0, at = 0, v;
    while (scanf("%d", &v) == 1) {
        if (at == cap) { cap = cap ? cap * 2 : NSQ * 64; boards = realloc(boards, (size_t)cap * 4); }
        boards[at++] = v;
    }
    if (at % NSQ) { fprintf(stderr, "eval: %d ints is not a whole number of %d-square boards\n", at, NSQ); return 1; }
    n = at / NSQ;
    float *values = malloc((size_t)n * sizeof(float));
    net_eval(net, boards, n, values);
    for (int i = 0; i < n; i++) printf("%.9g\n", values[i]);
    free(boards); free(values);
    return 0;
}

/* Bot versus a uniformly random mover. Regicide is forced for both sides, exactly as in
 * self-play, so games end at a few hundred plies instead of running forever. Colours
 * alternate so the first-move advantage is measured rather than absorbed into the rating.
 */
static int mode_elo(Net *net, Net *opp, const char *opp_name, Cfg *cfg,
                    uint64_t games, int first_game)
{
    Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos)), *child = malloc(sizeof(Pos));
    pos_init(base);
    Gen g = { malloc(8192 * 4), 0, 8192, calloc(MAXMOVE, 2), 0, calloc(NSQ, 2), 0, -1, 0, 0 };
    RepTab *rep = calloc(1, sizeof(RepTab));
    int32_t *boards = NULL;
    float *values = NULL;
    double *weights = NULL, *marg = malloc(sizeof(double) * ELO_N);
    int bcap = 0;

    Rec as_black = { 0, 0, 0 }, as_white = { 0, 0, 0 };
    EloFit fit = { 0, 0, 0, 0, 0, 0 };
    double t_start = now_s();
    uint64_t plies_total = 0;

    fprintf(stderr, "bot vs %s, regicide forced, temperature %.3g%s\n",
            opp_name, cfg->temperature, cfg->temperature == 0.0 ? " (greedy)" : "");
    if (opp && cfg->temperature == 0.0)
        fprintf(stderr,
            "WARNING: both players are deterministic at temperature 0, so every game with a\n"
            "         given colour is the same game. The record will be N copies of two\n"
            "         results and the interval will be far too narrow to believe. Pass\n"
            "         --temperature 0.3 or so to make the games actually differ.\n");

    for (uint64_t gi = 0; gi < games; gi++) {
        uint64_t seed = cfg->seed_base ^ (((uint64_t)first_game + gi) * 0x9E3779B97F4A7C15ull);
        Rng rng; rng_seed(&rng, seed);
        memcpy(p, base, sizeof(Pos));
        rep->epoch++;
        rep_bump(rep, p->hash);

        int bot_side = (gi & 1) ? SIDE_WHITE : SIDE_BLACK;   /* alternate who moves first */
        int nply = 0, result = RES_DRAW;
        double t_game = now_s();

        for (;;) {
            gen_all(p, &g);
            if (g.n == 0) { result = cfg->stalemate_loses ? (1 - p->turn) : RES_DRAW; break; }
            if (nply >= cfg->max_plies) { result = RES_DRAW; break; }

            uint32_t forced = 0;
            if (cfg->regicide) { king_ctx_set(&g, p); forced = find_king_capture(&g); }

            uint32_t pick;
            float chosen = 0.0f;
            /* Whoever is to move picks: the bot's net, the baseline net, or -- when there
             * is no baseline net -- uniformly at random. */
            Net *mover_net = (p->turn == bot_side) ? net : opp;
            if (forced) {
                pick = forced;
            } else if (!mover_net) {
                pick = g.buf[rng_below(&rng, (uint32_t)g.n)];
            } else {
                if (g.n > bcap) {
                    bcap = g.n + 256;
                    boards = realloc(boards, (size_t)bcap * NSQ * sizeof(int32_t));
                    values = realloc(values, (size_t)bcap * sizeof(float));
                    weights = realloc(weights, (size_t)bcap * sizeof(double));
                    if (!boards || !values || !weights) { fprintf(stderr, "oom in candidates\n"); return 1; }
                }
                for (int i = 0; i < g.n; i++) {
                    memcpy(child, p, sizeof(Pos));
                    do_move(child, SQ2P[UNPACK_O(g.buf[i])], UNPACK_M(g.buf[i]));
                    encode_stm(child, boards + (size_t)i * NSQ);
                }
                net_eval(mover_net, boards, g.n, values);
                int k = sample_softmax(values, g.n, cfg->temperature, &rng, weights);
                pick = g.buf[k];
                if (mover_net == net) chosen = values[k];
            }

            int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
            do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
            nply++;

            if (cfg->progress && nply % cfg->progress == 0)
                fprintf(stderr, "    game %-4llu ply %4d  %s to move  moves %5d  material %+8.1f  "
                                "value %+0.4f%s",
                        (unsigned long long)(first_game + gi), nply,
                        p->turn == bot_side ? "bot " : "base", g.n,
                        p->material / (double)VALUE_FP * (bot_side == SIDE_BLACK ? 1 : -1),
                        (double)chosen, isatty(STDERR_FILENO) ? "\r" : "\n");

            if (p->royals[1 - mover] < rb[1 - mover])      { result = mover;     break; }
            if (p->royals[mover] < rb[mover])              { result = 1 - mover; break; }
            int seen = rep_bump(rep, p->hash);
            if (cfg->rep_limit > 0 && seen >= cfg->rep_limit) { result = RES_DRAW; break; }
        }

        plies_total += (uint64_t)nply;
        Rec *r = (bot_side == SIDE_BLACK) ? &as_black : &as_white;
        const char *tag;
        if (result == RES_DRAW)          { r->d++; tag = "draw"; }
        else if (result == bot_side)     { r->w++; tag = "win "; }
        else                             { r->l++; tag = "loss"; }

        if (cfg->progress && isatty(STDERR_FILENO))
            fprintf(stderr, "\r%*s\r", 118, "");                    /* wipe the ply line */
        elo_fit(&as_black, &as_white, marg, &fit);
        long wins = as_black.w + as_white.w, losses = as_black.l + as_white.l,
             draws = as_black.d + as_white.d;
        double n = (double)(wins + losses + draws);
        double score = n > 0 ? (wins + 0.5 * draws) / n : 0.0;
        double el = now_s() - t_start;

        fprintf(stderr,
            "game %-4llu bot=%-5s %s in %4d plies (%5.1fs) | %4ldW %4ldL %4ldD  score %.3f | "
            "elo %+6.0f [%+.0f, %+.0f]%s | first-move %+.0f | %.1f min elapsed\n",
            (unsigned long long)(first_game + gi), bot_side == SIDE_BLACK ? "black" : "white",
            tag, nply, now_s() - t_game, wins, losses, draws, score,
            fit.elo, fit.lo, fit.hi, fit.saturated ? " (at the grid edge)" : "",
            fit.advantage, el / 60.0);
    }

    long wins = as_black.w + as_white.w, losses = as_black.l + as_white.l,
         draws = as_black.d + as_white.d;
    double n = (double)(wins + losses + draws);
    fprintf(stderr,
        "\n%ld games, %llu plies (%.0f/game), %.1f min\n"
        "  as black : %ldW %ldL %ldD\n"
        "  as white : %ldW %ldL %ldD\n"
        "  overall  : %ldW %ldL %ldD   score %.4f\n"
        "  Elo %+.0f, 95%% credible interval [%+.0f, %+.0f]   (%s = 0)\n"
        "  draw rating %.0f, first-move advantage %+.0f\n",
        (long)n, (unsigned long long)plies_total, n > 0 ? plies_total / n : 0.0,
        (now_s() - t_start) / 60.0,
        as_black.w, as_black.l, as_black.d, as_white.w, as_white.l, as_white.d,
        wins, losses, draws, n > 0 ? (wins + 0.5 * draws) / n : 0.0,
        fit.elo, fit.lo, fit.hi, opp_name, fit.draw_elo, fit.advantage);
    if (fit.saturated)
        fprintf(stderr, "  the posterior is against the grid edge: the bot did not lose enough "
                        "games to bound its rating from that side\n");

    free(boards); free(values); free(weights); free(marg);
    free(g.buf); free(g.stamp); free(g.kline); free(rep); free(base); free(p); free(child);
    return 0;
}

int main(int argc, char **argv)
{
    tables_init();
    const char *mode = argc > 1 ? argv[1] : "help";
    if (!strcmp(mode, "encode")) return mode_encode();
    if (!strcmp(mode, "elotest")) {           /* the rating estimator alone, on given counts */
        if (argc < 8) { fprintf(stderr, "usage: nnplay elotest Wb Lb Db Ww Lw Dw\n"); return 1; }
        Rec b = { atol(argv[2]), atol(argv[3]), atol(argv[4]) };
        Rec w = { atol(argv[5]), atol(argv[6]), atol(argv[7]) };
        double *marg = malloc(sizeof(double) * ELO_N);
        EloFit f;
        elo_fit(&b, &w, marg, &f);
        printf("elo %+.0f [%+.0f, %+.0f] draw_elo %.1f advantage %+.1f saturated %d\n",
               f.elo, f.lo, f.hi, f.draw_elo, f.advantage, f.saturated);
        free(marg);
        return 0;
    }
    if (strcmp(mode, "play") && strcmp(mode, "eval") && strcmp(mode, "elo") && strcmp(mode, "selfcheck")) {
        fputs(USAGE, stderr);
        return 1;
    }

    Cfg cfg = { 3000, 4, 0, 1, 3, 1, 25, 0, 1.0, 0x5EED5EED5EED5EEDull, NULL, 1 };
    const char *model = "net_fp16.onnx";
    uint64_t games = 1;
    int device_id = 0, sub_batch = 0, probe_cap = 8192, first_game = 0, tf32 = 1;
    int temperature_set = 0, allow_cpu = 0;
    const char *manifest = "net_manifest.json";
    const char *baseline = NULL;         /* NULL -> derive from --precision at use */

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")       && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--precision")   && i + 1 < argc) {
            const char *p = argv[++i];
            if      (!strcmp(p, "fp16")) model = "net_fp16.onnx";
            else if (!strcmp(p, "fp32")) model = "net_fp32.onnx";
            else { fprintf(stderr, "--precision must be fp16|fp32\n"); return 1; } }
        else if (!strcmp(argv[i], "--games")       && i + 1 < argc) games = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--first-game")  && i + 1 < argc) first_game = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")         && i + 1 < argc) cfg.out_path = argv[++i];
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) { cfg.temperature = atof(argv[++i]); temperature_set = 1; }
        else if (!strcmp(argv[i], "--max-plies")   && i + 1 < argc) cfg.max_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rep-limit")   && i + 1 < argc) cfg.rep_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-progress") && i + 1 < argc) cfg.no_progress_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")        && i + 1 < argc) cfg.seed_base = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--mat-shift")   && i + 1 < argc) cfg.mat_shift = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device")      && i + 1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sub-batch")   && i + 1 < argc) sub_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe-cap")   && i + 1 < argc) probe_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--progress-every") && i + 1 < argc) cfg.progress = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stats"))       cfg.stats = 1;
        else if (!strcmp(argv[i], "--baseline")     && i + 1 < argc) baseline = argv[++i];
        else if (!strcmp(argv[i], "--allow-cpu"))    allow_cpu = 1;
        else if (!strcmp(argv[i], "--manifest")     && i + 1 < argc) manifest = argv[++i];
        else if (!strcmp(argv[i], "--no-tf32"))     tf32 = 0;
        else if (!strcmp(argv[i], "--no-regicide")) cfg.regicide = 0;
        else if (!strcmp(argv[i], "--quiet"))     { cfg.verbose = 0; cfg.progress = 0; }
        else { fprintf(stderr, "unknown option %s\n", argv[i]); fputs(USAGE, stderr); return 1; }
    }

    Net net;
    net_open(&net, model, device_id, sub_batch, probe_cap, tf32, allow_cpu, cfg.verbose);
    if (!strcmp(mode, "eval")) { int rc = mode_eval(&net); net_close(&net); return rc; }
    if (!strcmp(mode, "selfcheck")) { int rc = mode_selfcheck(&net, manifest); net_close(&net); return rc; }
    if (!strcmp(mode, "elo")) {
        if (!temperature_set) cfg.temperature = 0.0;    /* strength, not the sampling policy */

        /* The baseline defaults to the graph exported from baseline.pt, matching the
         * precision the bot is running. "random" asks for the uniform mover instead, which
         * is the absolute anchor but stops discriminating once the bot beats it every
         * game. */
        char bpath[512];
        if (!baseline) {
            snprintf(bpath, sizeof(bpath), "baseline_%s.onnx", net.fp16_out ? "fp16" : "fp32");
            baseline = bpath;
        }
        Net opp;
        Net *oppp = NULL;
        const char *opp_name = "uniform-random";
        if (strcmp(baseline, "random") != 0) {
            if (access(baseline, R_OK) != 0) {
                fprintf(stderr,
                    "elo: no baseline graph '%s'.\n"
                    "     %s\n"
                    "     Then re-export:  python3 export_onnx.py\n"
                    "     Or rate against the uniform mover with:  --baseline random\n",
                    baseline,
                    access("baseline.pt", R_OK) == 0
                        ? "baseline.pt is there but has not been exported yet."
                        : "Copy the checkpoint you want to rate against to baseline.pt.");
                net_close(&net);
                return 1;
            }
            /* same sub-batch as the bot: the probe has already measured this GPU, and a
             * second session doubles the memory the graphs hold */
            net_open(&opp, baseline, device_id, net.sub_batch, probe_cap, tf32, allow_cpu, cfg.verbose);
            oppp = &opp;
            opp_name = baseline;
        }
        int rc = mode_elo(&net, oppp, opp_name, &cfg, games, first_game);
        if (oppp) net_close(oppp);
        net_close(&net);
        return rc;
    }

    Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos)), *child = malloc(sizeof(Pos));
    pos_init(base);
    Gen g = { malloc(8192 * 4), 0, 8192, calloc(MAXMOVE, 2), 0, calloc(NSQ, 2), 0, -1, 0, 0 };
    RepTab *rep = calloc(1, sizeof(RepTab));

    int32_t *boards = NULL;
    float   *values = NULL;
    double  *weights = NULL;
    int      bcap = 0;
    PlyRecNN *plybuf = malloc(sizeof(PlyRecNN) * (size_t)(cfg.max_plies + 2));

    Sink sink = { -1, NULL, 0, 1u << 22 };
    if (cfg.out_path) {
        sink.fd = open(cfg.out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (sink.fd < 0) { perror(cfg.out_path); return 1; }
        sink.buf = malloc(sink.cap);
        put_file_header(&sink, &cfg, &net, model, hash_file(model));
    }

    int mat_div = VALUE_FP >> cfg.mat_shift;
    if (mat_div < 1) mat_div = 1;

    GpuMon mon;
    memset(&mon, 0, sizeof(mon));      /* gpumon_stop runs unconditionally at exit */
    int have_nvml = cfg.stats && gpumon_start(&mon, device_id);
    if (cfg.stats && !have_nvml)
        fprintf(stderr, "  (libnvidia-ml.so.1 not available: no GPU utilisation numbers)\n");

    uint64_t total_plies = 0, total_evals = 0, wins[3] = { 0 };
    double t_start = now_s(), t_nn = 0.0, t_gen = 0.0, t_enc = 0.0;
    double w_nn = 0.0, w_gen = 0.0, w_enc = 0.0, w_t0 = now_s();   /* since the last line */
    uint64_t w_evals = 0;

    for (uint64_t gi = 0; gi < games; gi++) {
        uint64_t seed = cfg.seed_base ^ (((uint64_t)first_game + gi) * 0x9E3779B97F4A7C15ull);
        Rng rng; rng_seed(&rng, seed);
        memcpy(p, base, sizeof(Pos));
        rep->epoch++;
        rep_bump(rep, p->hash);

        int nply = 0, no_prog = 0, result = RES_DRAW, term = TERM_PLYCAP;
        double t_game = now_s();

        for (;;) {
            double tg = now_s();
            gen_all(p, &g);
            t_gen += now_s() - tg; w_gen += now_s() - tg;
            if (g.n == 0) { result = cfg.stalemate_loses ? (1 - p->turn) : RES_DRAW; term = TERM_STALEMATE; break; }
            if (nply >= cfg.max_plies) { result = RES_DRAW; term = TERM_PLYCAP; break; }

            /* Forced King capture, exactly tky.c's rule. The chosen leaf is still put
             * through the network, one board, so every ply carries a real value. */
            uint32_t forced = 0;
            if (cfg.regicide) { king_ctx_set(&g, p); forced = find_king_capture(&g); }

            int ncand = forced ? 1 : g.n;
            if (ncand > bcap) {
                bcap = ncand + 256;
                boards = realloc(boards, (size_t)bcap * NSQ * sizeof(int32_t));
                values = realloc(values, (size_t)bcap * sizeof(float));
                weights = realloc(weights, (size_t)bcap * sizeof(double));
                if (!boards || !values || !weights) { fprintf(stderr, "oom in candidate buffer\n"); return 1; }
            }
            double te = now_s();
            for (int i = 0; i < ncand; i++) {
                uint32_t v = forced ? forced : g.buf[i];
                memcpy(child, p, sizeof(Pos));
                do_move(child, SQ2P[UNPACK_O(v)], UNPACK_M(v));
                encode_stm(child, boards + (size_t)i * NSQ);     /* child->turn is the opponent */
            }
            t_enc += now_s() - te; w_enc += now_s() - te;

            double t0 = now_s();
            net_eval(&net, boards, ncand, values);
            t_nn += now_s() - t0; w_nn += now_s() - t0;
            total_evals += (uint64_t)ncand;
            w_evals += (uint64_t)ncand;

            int pick_i = forced ? 0 : sample_softmax(values, ncand, cfg.temperature, &rng, weights);
            uint32_t pick = forced ? forced : g.buf[pick_i];
            if (cfg.progress && nply % cfg.progress == 0) {
                fprintf(stderr, "    ply %4d  moves %5d  value %+0.4f  material %+8.1f  "
                                "pieces %3d  %.1f boards/s\n",
                        nply, g.n, (double)values[pick_i], p->material / (double)VALUE_FP,
                        p->npieces, total_evals / (t_nn > 0 ? t_nn : 1.0));
                if (cfg.stats) {
                    double wall = now_s() - w_t0, other = wall - w_nn - w_gen - w_enc;
                    fprintf(stderr, "              wall %6.2fs = nn %5.2f%% + movegen %5.2f%% "
                                    "+ encode %5.2f%% + other %5.2f%%   %.0f boards/s in nn",
                            wall, 100 * w_nn / wall, 100 * w_gen / wall,
                            100 * w_enc / wall, 100 * other / wall,
                            w_nn > 0 ? w_evals / w_nn : 0.0);
                    if (have_nvml) {
                        double gu, gm, used, tot;
                        gpumon_take(&mon, &gu, &gm, &used, &tot);
                        if (gu >= 0)
                            fprintf(stderr, "\n              gpu busy %5.1f%%  memory bus %5.1f%%  "
                                            "vram %.2f/%.2f GB", gu, gm, used, tot);
                    }
                    fputc('\n', stderr);
                    w_nn = w_gen = w_enc = 0.0; w_evals = 0; w_t0 = now_s();
                }
            }

            int origin36 = UNPACK_O(pick), mv = UNPACK_M(pick);
            int mover = p->turn;
            int royals_before[2] = { p->royals[0], p->royals[1] };
            int pieces_before = p->npieces;

            int promoted_now = do_move(p, SQ2P[origin36], mv);
            int captured = pieces_before - p->npieces;

            PlyRecNN *r = &plybuf[nply];
            snapshot(p, r->board);
            r->move = ((uint32_t)(mv / 1296) << 28) | ((uint32_t)origin36 << 17) | ((uint32_t)(mv % 1296) << 6);
            r->value = values[pick_i];
            r->n_legal = (uint16_t)(g.n > 65535 ? 65535 : g.n);
            r->material = (int16_t)quantise(p->material, mat_div);
            r->side = (uint8_t)mover;
            r->flags = (uint8_t)((captured > 0 ? PF_CAPTURE : 0) | (promoted_now ? PF_PROMOTION : 0)
                                 | (forced ? PF_FORCED_KING : 0));
            r->pad = 0;
            nply++;
            no_prog = (captured > 0 || promoted_now) ? 0 : no_prog + 1;

            if (p->royals[1 - mover] < royals_before[1 - mover])      { result = mover;     term = TERM_ROYAL; }
            else if (p->royals[mover] < royals_before[mover])         { result = 1 - mover; term = TERM_ROYAL; }
            else {
                int seen = rep_bump(rep, p->hash);
                if (cfg.rep_limit > 0 && seen >= cfg.rep_limit)                       { result = RES_DRAW; term = TERM_REPETITION; }
                else if (cfg.no_progress_limit > 0 && no_prog >= cfg.no_progress_limit) { result = RES_DRAW; term = TERM_NOPROGRESS; }
                else continue;
            }
            break;
        }
        if (nply) plybuf[nply - 1].flags |= PF_TERMINAL;

        total_plies += (uint64_t)nply;
        wins[result]++;

        if (cfg.out_path) {
            unsigned char gh[TKN_GAME_HDR];
            uint32_t np = (uint32_t)nply;
            memcpy(gh, &np, 4);
            /* the user-facing score: 1 black, -1 white, 0 any draw or ply cap */
            gh[4] = (unsigned char)(int8_t)(result == RES_BLACK ? 1 : result == RES_WHITE ? -1 : 0);
            gh[5] = (unsigned char)term;
            gh[6] = 0; gh[7] = 0;
            memcpy(gh + 8, &seed, 8);
            sink_put(&sink, gh, TKN_GAME_HDR);
            sink_put(&sink, plybuf, (size_t)nply * TKN_REC);
            sink_flush(&sink);          /* a game is hours; do not hold it in a buffer */
        }
        if (cfg.verbose) {
            static const char *tn[] = { "royal", "stalemate", "repetition", "no-progress", "ply-cap" };
            fprintf(stderr, "game %llu: %d plies, %s by %s, %.1fs (%.0f boards/s)\n",
                    (unsigned long long)(first_game + gi), nply,
                    result == RES_BLACK ? "black" : result == RES_WHITE ? "white" : "draw",
                    tn[term], now_s() - t_game,
                    total_evals / (t_nn > 0 ? t_nn : 1.0));
        }
    }

    if (cfg.out_path) {
        sink_flush(&sink);
        uint64_t v64 = games;      if (pwrite(sink.fd, &v64, 8, 40) != 8) perror("pwrite");
        v64 = total_plies;         if (pwrite(sink.fd, &v64, 8, 48) != 8) perror("pwrite");
        close(sink.fd);
    }

    double dt = now_s() - t_start;
    fprintf(stderr,
        "games=%llu plies=%llu (%.1f/game) evals=%llu\n"
        "  %.1fs total, %.1fs in the network (%.0f%%), %.0f boards/s, %.2f s/ply\n"
        "  movegen %.3fs (%.2f%%)  encode %.3fs (%.2f%%)  other %.3fs (%.2f%%)\n"
        "  black=%llu white=%llu draw=%llu\n",
        (unsigned long long)games, (unsigned long long)total_plies,
        games ? (double)total_plies / games : 0.0, (unsigned long long)total_evals,
        dt, t_nn, 100.0 * t_nn / (dt > 0 ? dt : 1.0),
        t_nn > 0 ? total_evals / t_nn : 0.0, total_plies ? dt / total_plies : 0.0,
        t_gen, 100 * t_gen / dt, t_enc, 100 * t_enc / dt,
        dt - t_nn - t_gen - t_enc, 100 * (dt - t_nn - t_gen - t_enc) / dt,
        (unsigned long long)wins[RES_BLACK], (unsigned long long)wins[RES_WHITE],
        (unsigned long long)wins[RES_DRAW]);
    if (have_nvml) {
        double gu, gm, used, tot;
        gpumon_take(&mon, &gu, &gm, &used, &tot);
        if (gu >= 0)
            fprintf(stderr, "  gpu busy %.1f%% of the last interval, memory bus %.1f%%, "
                            "vram %.2f/%.2f GB\n"
                            "  (NVML \"busy\" is the share of time a kernel was resident, not how much\n"
                            "   arithmetic it used -- the sub-batch curve going flat is the saturation test)\n",
                    gu, gm, used, tot);
    }
    gpumon_stop(&mon);

    free(boards); free(values); free(weights); free(plybuf); free(g.buf); free(g.stamp); free(g.kline);
    free(rep); free(base); free(p); free(child); free(sink.buf);
    net_close(&net);
    return 0;
}
