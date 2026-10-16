/* Neural-network self-play for Taikyoku Shogi.
 *
 * The rules, the move generator and the move executor are tky.c's, included wholesale so
 * there is exactly one copy of game.js's semantics in this directory. This file adds the
 * players: a PUCT search over the network's policy priors and value head, run for many
 * games in lockstep so their leaves share GPU calls, and the older one-ply search (--sims
 * 0), which scores the children of the policy's top moves and samples from a softmax over
 * their values.
 *
 * Perspective. Everything the network sees is in the frame of whoever is to move in that
 * position, which is the canonicalisation unpack.canonical_board performs: black to move
 * is the board as stored, white to move is rotated 180 degrees with every colour bit
 * flipped. That is exact, not approximate -- see CLAUDE.md section 13. A child position is
 * therefore encoded for the *opponent*, so its value is the opponent's.
 *
 * Inference is ONNX Runtime's C API on the CUDA execution provider. The sub-batch size is
 * measured on the GPU actually present at startup rather than assumed.
 */
#define TKY_NO_MAIN
#include "tky.c"

#include <dlfcn.h>
#include <pthread.h>
#include <sys/random.h>

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

/* The model graph, value and policy heads over one pass of the trunk (export_onnx.py):
 *   board  int32 (B, 1296) -> value  (B,)  for each board's side to move
 *   moves  int32 (M, 4)    -> logits (M,)  rows (board row, from, to, special), each in
 *                                          its own board's canonical frame */
typedef struct {
    OrtEnv           *env;
    OrtSessionOptions*opts;
    OrtSession       *sess;
    OrtMemoryInfo    *meminfo;
    OrtAllocator     *alloc;
    const char       *in_name[2], *out_name[2];   /* [0] board / value, [1] moves / logits */
    char             *owned[4];
    int               fp16[2];     /* value, logits come back as float16 */
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

/* Round to nearest even; values here are network outputs in [-1, 1], but the full range
 * is handled so nothing silently wraps. */
static uint16_t float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u, mant = x & 0x7FFFFFu;
    int exp = (int)((x >> 23) & 0xFF);
    if (exp == 0xFF) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    int e = exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {                                  /* subnormal half, or zero */
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - e;
        uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (mant >> 13), rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;   /* may carry into the exponent */
    return (uint16_t)(sign | h);
}

/* One call: nb boards and nr move rows in, nb values and nr logits out (logits NULL
 * drops them). Returns an OrtStatus rather than exiting so the startup probe can see OOM.
 * The inputs are borrowed, not copied. */
static OrtStatus *net_run_raw(Net *n, const int32_t *boards, int nb, const int32_t *rows,
                              int nr, float *values, float *logits)
{
    int64_t bs[2] = { nb, NSQ }, ms[2] = { nr, 4 };
    OrtValue *in[2] = { NULL, NULL }, *out[2] = { NULL, NULL };
    OrtStatus *st = ort->CreateTensorWithDataAsOrtValue(
            n->meminfo, (void *)boards, (size_t)nb * NSQ * sizeof(int32_t),
            bs, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &in[0]);
    if (!st) st = ort->CreateTensorWithDataAsOrtValue(
            n->meminfo, (void *)rows, (size_t)nr * 4 * sizeof(int32_t),
            ms, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &in[1]);
    if (!st) st = ort->Run(n->sess, NULL, n->in_name, (const OrtValue *const *)in, 2,
                           n->out_name, 2, out);
    float *dst[2] = { values, logits };
    int cnt[2] = { nb, nr };
    for (int k = 0; k < 2 && !st; k++) {
        void *raw = NULL;
        if (!dst[k] || (st = ort->GetTensorMutableData(out[k], &raw))) continue;
        if (n->fp16[k]) { const uint16_t *h = (const uint16_t *)raw;
                          for (int i = 0; i < cnt[k]; i++) dst[k][i] = half_to_float(h[i]); }
        else            memcpy(dst[k], raw, (size_t)cnt[k] * sizeof(float));
    }
    for (int k = 0; k < 2; k++) {
        if (in[k]) ort->ReleaseValue(in[k]);
        if (out[k]) ort->ReleaseValue(out[k]);
    }
    return st;
}

/* Values only. The moves input cannot be empty, so one dummy row rides along. */
static const int32_t NO_MOVE[4] = { 0, 0, 0, 0 };
static OrtStatus *net_values_raw(Net *n, const int32_t *boards, int nb, float *values)
{
    return net_run_raw(n, boards, nb, NO_MOVE, 1, values, NULL);
}

/* Evaluates `nboards` boards laid out contiguously in `boards`, sub-batching. */
static void net_eval(Net *n, const int32_t *boards, int nboards, float *values)
{
    for (int at = 0; at < nboards; ) {
        int nb = nboards - at;
        if (nb > n->sub_batch) nb = n->sub_batch;
        ORT_CHECK(net_values_raw(n, boards + (size_t)at * NSQ, nb, values + at));
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
        for (int w = 0; w < 2 && !st; w++) st = net_values_raw(n, n->probe_buf, b, tmp);
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
            st = net_values_raw(n, n->probe_buf, b, tmp);
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
            else if (strstr(why, "driver version is insufficient"))
                fprintf(stderr,
                    "  The GPU driver is older than the CUDA runtime this onnxruntime-gpu was\n"
                    "  built against. Pin one built for an older CUDA and set up again:\n"
                    "    ORT_SPEC='onnxruntime-gpu==1.20.2' sh setup.sh\n");
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
    if (nin != 2 || nout != 2) {
        fprintf(stderr, "%s: the model graph takes (board, moves) and returns (value, logits); this "
                        "one has %zu inputs and %zu outputs.\n  A value-only graph beside a "
                        "_policy.onnx is the old export: re-run export_onnx.py.\n", model, nin, nout);
        exit(1);
    }
    for (int k = 0; k < 2; k++) {
        ORT_CHECK(ort->SessionGetInputName(n->sess, (size_t)k, n->alloc, &n->owned[k]));
        ORT_CHECK(ort->SessionGetOutputName(n->sess, (size_t)k, n->alloc, &n->owned[2 + k]));
    }
    int si = !strcmp(n->owned[0], "moves"), so = !strcmp(n->owned[2], "logits");
    n->in_name[0] = n->owned[si];      n->in_name[1] = n->owned[1 - si];
    n->out_name[0] = n->owned[2 + so]; n->out_name[1] = n->owned[3 - so];
    for (int k = 0; k < 2; k++) {
        OrtTypeInfo *ti = NULL;
        const OrtTensorTypeAndShapeInfo *tsi = NULL;
        ONNXTensorElementDataType et;
        ORT_CHECK(ort->SessionGetOutputTypeInfo(n->sess, (size_t)(k ^ so), &ti));
        ORT_CHECK(ort->CastTypeInfoToTensorInfo(ti, &tsi));
        ORT_CHECK(ort->GetTensorElementType(tsi, &et));
        n->fp16[k] = et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        ort->ReleaseTypeInfo(ti);
    }
    if (verbose)
        fprintf(stderr, "model %s (hash %08x): inputs '%s','%s', outputs '%s','%s', %s\n",
                model, hash_file(model), n->in_name[0], n->in_name[1], n->out_name[0],
                n->out_name[1], n->fp16[0] ? "float16" : "float32");

    n->sub_batch = sub_batch > 0 ? sub_batch : net_probe(n, probe_cap, 3, verbose);
    free(n->probe_buf);
    n->probe_buf = NULL;
}

static void net_close(Net *n)
{
    for (int i = 0; i < 4; i++)
        if (n->owned[i]) { OrtStatus *st = ort->AllocatorFree(n->alloc, n->owned[i]); if (st) ort->ReleaseStatus(st); }
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
#define TKN_VERSION 7u
#define TKN_FILE_HDR 256
#define TKN_GAME_HDR 16
#define TKN_BOARD   (NSQ * 2)
#define TKN_REC     (TKN_BOARD + 16)

/* Flags on a ply record. */
enum { PF_CAPTURE = 1, PF_PROMOTION = 2, PF_FORCED_KING = 4, PF_TERMINAL = 8,
       PF_KING_SAFETY = 16,      /* King safety changed the pick on this ply */
       PF_OPPONENT = 32 };       /* elo: the baseline or random mover played this move */

typedef struct {
    uint16_t board[NSQ];    /* absolute, after the move */
    uint32_t move;          /* special << 28 | origin << 17 | target << 6 */
    float    value;         /* the board's value for the side now to move: the searched
                             * Q, or the network's value at --sims 0 */
    uint16_t n_legal;       /* legal moves in the position the move was played from */
    int16_t  material;      /* after the move, black minus white, mat_shift units */
    uint8_t  side;          /* the mover: 1 black, 0 white */
    uint8_t  flags;
    uint16_t n_scored;      /* network evaluations the move cost; 0 in files written
                             * before version 2, which means "every child" */
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

/* ---- choosing a move ------------------------------------------------------- */
typedef struct {
    Gen      g;             /* the mover's moves */
    Gen      rg;            /* the opponent's replies, for the King-safety check */
    int32_t *boards;
    float   *values;
    double  *weights;
    int     *perm;
    int      cap;
    Pos     *child;
    uint64_t checks;        /* King-safety: moves played out and checked */
    uint64_t overridden;    /* plies where the rule changed the pick */
    uint64_t hopeless;      /* plies where every move lost the King */
    double   t_enc, t_nn, t_safe, t_pol;
    int32_t *parent;        /* the position itself, encoded, for its policy logits */
    int32_t *mv4;           /* its moves as graph rows (0, from, to, special) */
    float   *logits;
    struct Ranked { float l; int i; } *rk;
} Scratch;

/* The one-ply search scores the children of the policy's top POLICY_TOP of the legal
 * moves plus POLICY_EXPLORE more drawn at random from the rest, so the policy's misses
 * still get evaluated -- and recorded, so the next policy can learn them. */
#define POLICY_TOP     0.05
#define POLICY_EXPLORE 0.02

static void gen_init(Gen *g)
{
    memset(g, 0, sizeof(*g));
    g->cap = 8192;
    g->buf = malloc((size_t)g->cap * 4);
    g->stamp = calloc(MAXMOVE, 2);
    g->kline = calloc(NSQ, 2);
    g->ksq = -1;
}

static void scratch_init(Scratch *s)
{
    memset(s, 0, sizeof(*s));
    gen_init(&s->g);
    gen_init(&s->rg);
    s->child = malloc(sizeof(Pos));
    s->parent = malloc(NSQ * sizeof(int32_t));
}

static void scratch_free(Scratch *s)
{
    free(s->g.buf); free(s->g.stamp); free(s->g.kline);
    free(s->rg.buf); free(s->rg.stamp); free(s->rg.kline);
    free(s->boards); free(s->values); free(s->weights); free(s->perm); free(s->child);
    free(s->parent); free(s->mv4); free(s->logits); free(s->rk);
}

static void scratch_reserve(Scratch *s, int n)
{
    if (n <= s->cap) return;
    s->cap = n + 256;
    s->boards  = realloc(s->boards,  (size_t)s->cap * NSQ * sizeof(int32_t));
    s->values  = realloc(s->values,  (size_t)s->cap * sizeof(float));
    s->weights = realloc(s->weights, (size_t)s->cap * sizeof(double));
    s->perm    = realloc(s->perm,    (size_t)s->cap * sizeof(int));
    s->mv4     = realloc(s->mv4,     (size_t)s->cap * 4 * sizeof(int32_t));
    s->logits  = realloc(s->logits,  (size_t)s->cap * sizeof(float));
    s->rk      = realloc(s->rk,      (size_t)s->cap * sizeof(*s->rk));
    if (!s->boards || !s->values || !s->weights || !s->perm || !s->mv4 || !s->logits || !s->rk) {
        fprintf(stderr, "oom in candidate buffers\n");
        exit(1);
    }
}

/* King safety, the mirror of forced regicide: never play a move that leaves your own King
 * capturable while some move does not. Without it the bot hands games away -- the side to
 * move escaped a threatened King only ~14% of the time at T=0.3, because the one or two
 * defences are outweighed by the hundreds of moves that ignore the threat.
 *
 * Exact, not learned: the move is played on a copy and the opponent's replies go through
 * the same detector as forced regicide. A move that removes its own King -- a Free Eagle
 * sweeping over it -- also counts as hanging it. */
static int hangs_king(Scratch *s, const Pos *p, uint32_t v)
{
    int me = p->turn;
    Pos *c = s->child;
    memcpy(c, p, sizeof(Pos));
    do_move(c, SQ2P[UNPACK_O(v)], UNPACK_M(v));
    s->checks++;
    if (c->royals[me] < p->royals[me]) return 1;
    if (c->royals[1 - me] < p->royals[1 - me]) return 0;     /* takes the enemy King */
    gen_all(c, &s->rg);
    king_ctx_set(&s->rg, c);
    return find_king_capture(&s->rg) != 0;
}

/* Softmax over values[0..n), rejecting a pick that hangs the King and drawing again from
 * the rest. Removing rejected entries does not change the relative weights of the others,
 * so this is exactly the softmax restricted to the safe moves, and it costs one check on
 * the usual ply where the first pick is already safe. Rejected entries are swapped to the
 * tail. Returns the index, or -1 if all n hang. */
static int sample_safe(Scratch *s, const Pos *p, int n, double T, Rng *rng)
{
    for (int live = n; live > 0; live--) {
        int k = sample_softmax(s->values, live, T, rng, s->weights);
        if (!hangs_king(s, p, s->g.buf[s->perm[k]])) return k;
        int ti = s->perm[k]; s->perm[k] = s->perm[live - 1]; s->perm[live - 1] = ti;
        float tv = s->values[k]; s->values[k] = s->values[live - 1]; s->values[live - 1] = tv;
    }
    return -1;
}

static int rank_desc(const void *a, const void *b)
{
    float x = ((const struct Ranked *)a)->l, y = ((const struct Ranked *)b)->l;
    return x < y ? 1 : x > y ? -1 : 0;
}

/* Puts the children to score first in s->perm, and returns how many. One pass over the
 * parent ranks every legal move; the value head then sees the top few and a few at
 * random. Everything unscored stays in s->perm after them, in policy order, for the
 * King-safety fallback. */
static int pick_children(Net *net, const Pos *p, Scratch *s, Rng *rng)
{
    int n = s->g.n;
    for (int i = 0; i < n; i++) s->perm[i] = i;
    double t0 = now_s();
    encode_stm(p, s->parent);
    int flip = p->turn != SIDE_BLACK;          /* encode_stm rotates white to move */
    for (int i = 0; i < n; i++) {
        uint32_t v = s->g.buf[i];
        int o = UNPACK_O(v), mv = UNPACK_M(v), sp = mv / 1296, t = mv % 1296;
        if (flip) {
            o = NSQ - 1 - o;
            t = NSQ - 1 - t;
            if (sp >= 1 && sp <= 8) sp = (sp - 1 + 4) % 8 + 1;   /* DIRS[i] -> DIRS[i+4] */
        }
        s->mv4[4 * i] = 0; s->mv4[4 * i + 1] = o; s->mv4[4 * i + 2] = t; s->mv4[4 * i + 3] = sp;
    }
    float pv;
    ORT_CHECK(net_run_raw(net, s->parent, 1, s->mv4, n, &pv, s->logits));
    for (int i = 0; i < n; i++) { s->rk[i].l = s->logits[i]; s->rk[i].i = i; }
    qsort(s->rk, (size_t)n, sizeof(*s->rk), rank_desc);
    for (int i = 0; i < n; i++) s->perm[i] = s->rk[i].i;
    int top = (int)ceil(POLICY_TOP * n), ex = (int)(POLICY_EXPLORE * n + 0.5);
    if (top < 1) top = 1;
    if (top > n) top = n;
    if (ex > n - top) ex = n - top;
    for (int i = 0; i < ex; i++) {             /* exploration: random picks from the rest */
        int j = top + i + (int)rng_below(rng, (uint32_t)(n - top - i));
        int t = s->perm[top + i]; s->perm[top + i] = s->perm[j]; s->perm[j] = t;
    }
    s->t_pol += now_s() - t0;
    return top + ex;
}

/* The non-forced move for the side to move, from the moves already in s->g: score the
 * children pick_children chooses, softmax at temperature T over the negated values, King
 * safety on top when asked. If every scored child hangs the King, the unscored ones are
 * checked and only the safe ones are sent to the network, so pruning can never hide the
 * defence. Reports the chosen leaf's value, how many children were scored, and whether the
 * rule changed the pick. */
static uint32_t choose(Net *net, int king_safety, const Pos *p,
                       Scratch *s, Rng *rng, double T, float *val, int *n_scored,
                       int *overridden)
{
    int n = s->g.n;
    scratch_reserve(s, n);
    int scored = pick_children(net, p, s, rng);
    double t0 = now_s();
    for (int i = 0; i < scored; i++) {
        uint32_t v = s->g.buf[s->perm[i]];
        memcpy(s->child, p, sizeof(Pos));
        do_move(s->child, SQ2P[UNPACK_O(v)], UNPACK_M(v));
        encode_stm(s->child, s->boards + (size_t)i * NSQ);     /* child->turn is the opponent */
    }
    double t1 = now_s();
    net_eval(net, s->boards, scored, s->values);
    double t2 = now_s();
    s->t_enc += t1 - t0;
    s->t_nn += t2 - t1;

    int k, changed = 0;
    if (!king_safety) {
        k = sample_softmax(s->values, scored, T, rng, s->weights);
    } else {
        uint64_t before = s->checks;
        k = sample_safe(s, p, scored, T, rng);
        if (k < 0 && scored < n) {
            int m = scored;
            for (int i = scored; i < n; i++)
                if (!hangs_king(s, p, s->g.buf[s->perm[i]])) {
                    int t = s->perm[i]; s->perm[i] = s->perm[m]; s->perm[m] = t;
                    m++;
                }
            if (m > scored) {
                double t3 = now_s();
                for (int i = scored; i < m; i++) {
                    uint32_t v = s->g.buf[s->perm[i]];
                    memcpy(s->child, p, sizeof(Pos));
                    do_move(s->child, SQ2P[UNPACK_O(v)], UNPACK_M(v));
                    encode_stm(s->child, s->boards + (size_t)i * NSQ);
                }
                double t4 = now_s();
                net_eval(net, s->boards + (size_t)scored * NSQ, m - scored, s->values + scored);
                double t5 = now_s();
                s->t_enc += t4 - t3;
                s->t_nn += t5 - t4;
                t2 += t5 - t3;          /* counted above, not as King-safety time */
                k = scored + sample_softmax(s->values + scored, m - scored, T, rng,
                                            s->weights);
                scored = m;
            }
        }
        if (k < 0) {                    /* every move loses the King: nothing to steer */
            s->hopeless++;
            k = sample_softmax(s->values, scored, T, rng, s->weights);
        } else {
            changed = s->checks - before > 1;
        }
        s->t_safe += now_s() - t2;
    }
    s->overridden += changed;
    if (val) *val = s->values[k];
    if (n_scored) *n_scored = scored;
    if (overridden) *overridden = changed;
    return s->g.buf[s->perm[k]];
}

/* The uniform-random mover under the same rule: a random order, the first safe move. */
static uint32_t choose_random(int king_safety, const Pos *p, Scratch *s, Rng *rng)
{
    int n = s->g.n;
    if (!king_safety) return s->g.buf[rng_below(rng, (uint32_t)n)];
    scratch_reserve(s, n);
    for (int i = 0; i < n; i++) s->perm[i] = i;
    double t0 = now_s();
    for (int i = 0; i < n; i++) {
        int j = i + (int)rng_below(rng, (uint32_t)(n - i));
        int t = s->perm[i]; s->perm[i] = s->perm[j]; s->perm[j] = t;
        if (!hangs_king(s, p, s->g.buf[s->perm[i]])) {
            s->overridden += i > 0;
            s->t_safe += now_s() - t0;
            return s->g.buf[s->perm[i]];
        }
    }
    s->hopeless++;
    s->t_safe += now_s() - t0;
    return s->g.buf[s->perm[0]];
}

/* ---- Elo ------------------------------------------------------------------- */
/* The baseline is anchored at Elo 0 and the bot's rating is the posterior over the
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

/* ---- SPRT ------------------------------------------------------------------ */
/* Sequential probability ratio test: play until the evidence separates H0 "the bot is at
 * most elo0 above the baseline" from H1 "at least elo1", then stop. Expected cost is a
 * fraction of a fixed-length match at the same error rates, and a clear result -- a net
 * crushing a random mover -- stops after a few dozen games.
 *
 * The hypotheses are in logistic Elo, the rating a score implies: s = 1/(1+10^(-elo/400)).
 * The draw rate is a nuisance parameter, maximised out under each hypothesis separately --
 * the generalised SPRT, as fishtest runs it. That maximisation is done exactly rather than
 * with the usual normal approximation LLR ~ N (s1-s0)(2s-s0-s1) / 2 var: against a random
 * baseline the bot wins every game, the observed variance is zero, and the approximation's
 * LLR blows up and accepts H1 after two or three wins. Exactly, ten straight wins is only
 * 10 log(s1/s0) -- the binomial answer, ~1.3 for 0 vs +50.
 *
 * Under a hypothesis with score s the most likely win/draw/loss distribution given counts
 * n is p_i = f_i / (1 + lambda (a_i - s)), f the observed frequencies, a = (1, 1/2, 0),
 * lambda the root of sum f_i (a_i - s) / (1 + lambda (a_i - s)) = 0. A zero count becomes
 * 1e-3, as fishtest does, so the root exists. */
static double sprt_score(double elo) { return 1.0 / (1.0 + pow(10.0, -elo / 400.0)); }

static double sprt_loglik(const double n[3], double s)
{
    static const double a[3] = { 1.0, 0.5, 0.0 };
    double N = n[0] + n[1] + n[2], f[3], d[3];
    for (int i = 0; i < 3; i++) { f[i] = n[i] / N; d[i] = a[i] - s; }
    /* every 1 + lambda d_i must stay positive: lambda in (-1/(1-s), 1/s) */
    double lo = -1.0 / d[0], hi = -1.0 / d[2];
    for (int it = 0; it < 64; it++) {          /* 64 halvings reach the last bit of a double */
        double mid = 0.5 * (lo + hi), g = 0.0;
        for (int i = 0; i < 3; i++) g += f[i] * d[i] / (1.0 + mid * d[i]);
        if (g > 0) lo = mid; else hi = mid;      /* g falls from +inf to -inf */
    }
    double lam = 0.5 * (lo + hi), ll = 0.0;
    for (int i = 0; i < 3; i++) ll += n[i] * log(f[i] / (1.0 + lam * d[i]));
    return ll;
}

static double sprt_llr(long w, long d, long l, double elo0, double elo1)
{
    double n[3] = { w ? (double)w : 1e-3, d ? (double)d : 1e-3, l ? (double)l : 1e-3 };
    return sprt_loglik(n, sprt_score(elo1)) - sprt_loglik(n, sprt_score(elo0));
}

typedef struct {
    int    on;
    double elo0, elo1, alpha, beta;
    double lower, upper;       /* log(beta/(1-alpha)), log((1-beta)/alpha) */
} Sprt;

static void sprt_init(Sprt *s)
{
    s->lower = log(s->beta / (1.0 - s->alpha));
    s->upper = log((1.0 - s->beta) / s->alpha);
}

/* 1 when H1 is accepted, -1 for H0, 0 to keep playing. */
static int sprt_decide(const Sprt *s, double llr)
{
    return llr >= s->upper ? 1 : llr <= s->lower ? -1 : 0;
}
/* ---- end SPRT -------------------------------------------------------------- */

/* ---- settings ------------------------------------------------------------- */
typedef struct {
    int      max_plies, rep_limit, no_progress_limit, stalemate_loses, mat_shift, regicide;
    int      progress;      /* seconds between status lines; 0 is off */
    int      stats;         /* GPU utilisation in the status lines */
    double   temperature;
    uint64_t seed_base;
    const char *out_path;
    const char *children_path;     /* --children: every scored child per ply, for the policy */
    int      verbose;
    int      opening_plies;        /* plies played hot before `temperature` takes over */
    double   opening_temperature;
    int      king_safety;          /* never leave your own King capturable if avoidable */
    int      resign_pawns;         /* adjudicate a lead this large ... (0 disables) */
    int      resign_plies;         /* ... held this many plies in a row */
    int      cap_by_material;      /* score the ply cap for the side ahead, not as a draw */
    int      drop_capped;          /* play: write no game that reached the ply cap */
    int      endgames;             /* play: random sparse starts, only royal captures kept */
    int      eg_max_pieces;        /* fewer pieces than this on the board, Kings included */
    int      eg_imbalance;         /* the side ahead leads by at least this many pawns */
    int      sims;                 /* simulations per move; 0 is the one-ply search */
    int      opp_sims;             /* elo: the opponent's */
    int      leaves;               /* leaves a tree gathers per network call */
    int      games_par;            /* games in lockstep */
    double   cpuct;
    double   noise_frac;           /* root Dirichlet noise mixed into the priors; 0 is none */
    double   noise_alpha;          /* alpha = noise_alpha / legal moves */
    double   contempt;             /* a draw costs the searching side up to this much */
    int      contempt_flat;        /* play: ... always that much, whatever the root's value */
    double   soft_scale;           /* play: a capped or repeated game's label, on material */
} Cfg;

/* A step schedule, as AlphaZero uses: a hot opening so games diverge, then the playing
 * temperature so the outcome follows from the position rather than from coin flips.
 * opening_plies 0 leaves the temperature constant, which is the old behaviour. */
static inline double temp_at(const Cfg *c, int ply)
{
    return ply < c->opening_plies ? c->opening_temperature : c->temperature;
}

/* Material adjudication. The outcome is otherwise decided almost entirely by who first
 * fails to see a King capture, which is close to a coin flip given the position, so the
 * value head has next to nothing to learn. Scoring a large, lasting material lead as a win
 * makes the label follow something the position does determine.
 *
 * Measured over 27k self-play games: a 300-pawn lead held for 10 plies was later reversed
 * -- the other side went 300 ahead -- in 0.1-0.7% of games, where a single ply can swing
 * material by up to ~370 pawns (a Great General trample). Holding it for 10 plies is what
 * keeps one trample from ending a game. 300 pawns is ~13% of a side's starting material.
 *
 * *run counts consecutive plies with the same side at least resign_pawns ahead, signed:
 * positive for black. Returns the winner once the count reaches resign_plies, else -1. */
#define TERM_RESIGN 5
static int resign_check(const Cfg *c, const Pos *p, int *run)
{
    if (c->resign_pawns <= 0) return -1;
    int64_t thr = (int64_t)c->resign_pawns * VALUE_FP;
    int s = p->material >= thr ? 1 : p->material <= -thr ? -1 : 0;
    *run = s == 0 ? 0 : (*run * s > 0 ? *run + s : s);
    if (*run >= c->resign_plies) return RES_BLACK;
    if (-*run >= c->resign_plies) return RES_WHITE;
    return -1;
}

/* The ply cap, scored for the side ahead on material, or a draw when the lead is under
 * CAP_DRAW_MARGIN pawns -- or always with --cap-draw. A capped game used to be written as
 * a draw it never was. The margin is fixed by the file version (5 on), not stored. */
#define CAP_DRAW_MARGIN 50
static int cap_result(const Cfg *c, const Pos *p)
{
    int64_t margin = (int64_t)CAP_DRAW_MARGIN * VALUE_FP;
    if (!c->cap_by_material || (p->material < margin && p->material > -margin)) return RES_DRAW;
    return p->material > 0 ? RES_BLACK : RES_WHITE;
}

/* ---- --endgames: random sparse starting positions --------------------------------- */
/* Self-play from the opening almost never reaches a bare King, so the network never sees
 * a mate delivered. These starts do: one King each, fewer than eg_max_pieces pieces in
 * all, the side ahead by at least eg_imbalance pawns holding a ranging piece.
 *
 * A piece is drawn uniformly over the 300 non-King ids. One that only arises by promotion
 * is placed promoted; one from the starting setup is placed as itself or, if it has a
 * promotion, as that promotion, 50/50. An unpromoted piece that could promote is kept out
 * of its own promotion zone, where play could never leave it standing.
 *
 * Splitting pieces evenly between the sides meets a 50-pawn lead in ~9% of draws (the
 * mean piece is worth ~8 pawns), so the side meant to be ahead gets the ranging piece and
 * at least half the rest: ~35% of draws pass at the defaults. */
#define EG_MAX_FAILS 100
static uint8_t EG_RANGING[NPIECE];     /* an unlimited slide in some direction */
static uint8_t EG_IN_SETUP[NPIECE];

typedef struct { uint64_t made, short_lead, king_hangs; } EgStats;

static void eg_tables(void)
{
    for (int i = 0; i < NSQ; i++) EG_IN_SETUP[C_IDX(INIT_BOARD[i])] = 1;
    for (int i = 1; i < NPIECE; i++) {
        const PieceDef *d = &DEFS[i * 2 + SIDE_BLACK];
        for (int k = 0; k < 8; k++) if (d->dydx[k] == 99) EG_RANGING[i] = 1;
        for (int t = 0; t < d->tp_n; t++)
            for (int k = 0; k < 8; k++) if (TP_POOL[d->tp_off + t].slide[k] == 99) EG_RANGING[i] = 1;
    }
}

static void eg_draw(Rng *rng, int *idx, int *prom)
{
    int i = 1 + (int)rng_below(rng, KING_IDX - 1);
    if (!EG_IN_SETUP[i])                                       { *idx = i; *prom = 1; }
    else if (PROMOTE_IDX[i] >= 0 && rng_below(rng, 2))         { *idx = PROMOTE_IDX[i]; *prom = 1; }
    else                                                       { *idx = i; *prom = 0; }
}

static void eg_place(Pos *p, Rng *rng, int idx, int prom, int color)
{
    for (;;) {
        int s = (int)rng_below(rng, NSQ), q = SQ2P[s], x = s / 36;
        if (p->side[q] != SIDE_EMPTY) continue;
        if (!prom && PROMOTE_IDX[idx] >= 0 && (color == SIDE_BLACK ? x >= 25 : x <= 10)) continue;
        sq_set(p, q, (uint16_t)CODE(idx, color, prom));
        return;
    }
}

/* 0, or -1 once EG_MAX_FAILS draws in a row have been rejected */
static int eg_generate(Pos *p, const Cfg *c, Rng *rng, Gen *g, EgStats *st)
{
    for (int fails = 0; fails <= EG_MAX_FAILS; fails++) {
        memset(p, 0, sizeof(*p));
        p->royal_sq[0] = p->royal_sq[1] = -1;
        memset(p->side, SIDE_OFF, sizeof(p->side));
        for (int i = 0; i < NPAD; i++) p->pat[i] = -1;
        for (int i = 0; i < NSQ; i++) {
            int q = SQ2P[i];
            p->side[q] = SIDE_EMPTY;
            p->code[q] = EMPTY_CODE;
            p->hash ^= zkey(q, EMPTY_CODE);
        }
        int n = 3 + (int)rng_below(rng, (uint32_t)(c->eg_max_pieces - 3));
        int att = (int)rng_below(rng, 2), idx, prom;
        eg_place(p, rng, KING_IDX, 0, SIDE_BLACK);
        eg_place(p, rng, KING_IDX, 0, SIDE_WHITE);
        do eg_draw(rng, &idx, &prom); while (!EG_RANGING[idx]);
        eg_place(p, rng, idx, prom, att);
        int extra = n - 3, nd = (int)rng_below(rng, (uint32_t)(extra / 2 + 1));
        for (int j = 0; j < extra; j++) {
            eg_draw(rng, &idx, &prom);
            eg_place(p, rng, idx, prom, j < nd ? 1 - att : att);
        }
        p->turn = (int)rng_below(rng, 2);
        if (p->turn) p->hash ^= TURN_KEY;

        int64_t lead = (int64_t)c->eg_imbalance * VALUE_FP;
        int ahead = p->material >= lead ? SIDE_BLACK : p->material <= -lead ? SIDE_WHITE : -1;
        int armed = 0;
        for (int k = 0; ahead >= 0 && k < p->pn[ahead]; k++)
            armed |= EG_RANGING[C_IDX(p->code[p->plist[ahead][k]])];
        if (!armed) { st->short_lead++; continue; }
        /* Either King capturable: the side to move takes it at once, or its own is lost
         * before it can escape. Since only royal captures are kept, those 1-2 ply games
         * would crowd out the mates. */
        int hangs = 0;
        for (int k = 0; k < 2 && !hangs; k++, p->turn ^= 1) {
            gen_all(p, g);
            king_ctx_set(g, p);
            hangs = find_king_capture(g) != 0;
        }
        if (hangs) { st->king_hangs++; continue; }
        st->made++;
        return 0;
    }
    return -1;
}

/* Play's training label, black's view: the result, except that a game cut off by the ply
 * cap or by repetition is scored on material, soft_scale * tanh(pawns / SOFT_PAWNS). */
#define SOFT_PAWNS 300.0
static double soft_label(const Cfg *c, const Pos *p, int result, int term)
{
    if (c->soft_scale > 0.0 && (term == TERM_PLYCAP || term == TERM_REPETITION))
        return c->soft_scale * tanh((double)p->material / VALUE_FP / SOFT_PAWNS);
    return result == RES_BLACK ? 1.0 : result == RES_WHITE ? -1.0 : 0.0;
}

static void put_file_header(Sink *s, const Cfg *c, const Net *net, const char *model,
                            uint32_t model_hash)
{
    unsigned char hdr[TKN_FILE_HDR];
    uint32_t v;
    float f;
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
    /* 80, the rules: bit 0 forced regicide; version 3 on, bit 1 King safety; version 4 on,
     * bit 2 ply cap scored by material, bits 8..23 resignation lead in pawns (0 off),
     * bits 24..31 the plies it must be held */
    v = (uint32_t)(c->regicide ? 1 : 0) | (c->king_safety ? 2u : 0u)
      | (c->cap_by_material ? 4u : 0u)
      | ((uint32_t)(c->resign_pawns > 0xFFFF ? 0xFFFF : c->resign_pawns) << 8)
      | ((uint32_t)(c->resign_plies > 0xFF ? 0xFF : c->resign_plies) << 24);
    memcpy(hdr + 80, &v, 4);
    v = 36;                           memcpy(hdr + 84, &v, 4);
    v = NPIECE;                       memcpy(hdr + 88, &v, 4);
    /* 92..95, version 2 onward: the opening schedule. Offset 92 used to hold NSQ, which
     * no reader ever looked at, so version gates the reinterpretation. */
    uint16_t o16 = (uint16_t)(c->opening_plies > 65535 ? 65535 : c->opening_plies);
    memcpy(hdr + 92, &o16, 2);
    o16 = (uint16_t)(c->opening_temperature * 1000.0 + 0.5);
    memcpy(hdr + 94, &o16, 2);
    v = (uint32_t)net->sub_batch;     memcpy(hdr + 96, &v, 4);
    /* 100: bit 0 fp16; bit 1 games that reached the ply cap were dropped; bit 2 games
     * started from random --endgames positions, only royal captures kept; version 6 on,
     * bits 8..15 policy_explore and bits 16..31 policy_top of the one-ply search, both in
     * thousandths of the legal moves, and 0 under search */
    int oneply = c->sims == 0;
    v = (uint32_t)(net->fp16[0] ? 1 : 0) | (c->drop_capped ? 2u : 0u) | (c->endgames ? 4u : 0u)
      | (oneply ? ((uint32_t)(POLICY_EXPLORE * 1000.0 + 0.5) & 0xFFu) << 8
                | ((uint32_t)(POLICY_TOP * 1000.0 + 0.5) & 0xFFFFu) << 16 : 0u);
    memcpy(hdr + 100, &v, 4);
    /* 104..127: "<basename>@<8 hex of the graph's hash>", NUL padded. The hash is what
     * makes a shard attributable to a specific set of weights: in a self-play loop the
     * file name never changes, so the name alone cannot tell one generation from the
     * next. Readers take the field as a NUL-terminated string either way. */
    const char *base = strrchr(model, '/');
    base = base ? base + 1 : model;
    char tag[32];
    snprintf(tag, sizeof(tag), "%.14s@%08x", base, model_hash);
    memcpy(hdr + 104, tag, strlen(tag) < 24 ? strlen(tag) : 24);
    /* 128..255, version 7 on: the search. 128 simulations per move (0: the one-ply
     * search), 132 leaves a tree gathers per network call, 136 f32 c_puct, 140 f32 the
     * root noise's share, 144 f32 its alpha numerator (alpha = it / legal moves), 148
     * games in lockstep, 152 the opponent's simulations (elo), 156 bits: 0 tree reuse,
     * 1 min-max normalised Q, 2 unvisited children at the parent's value, 3 flat contempt
     * (self-play; else scaled by the root's value as in tree_contempt), 160 f32 the
     * contempt (0 in files written before it existed), 164 f32 the soft label's scale and
     * 168 f32 its pawns (0: none; then each game header's bytes 6..7 hold the label) */
    v = (uint32_t)c->sims;            memcpy(hdr + 128, &v, 4);
    v = (uint32_t)c->leaves;          memcpy(hdr + 132, &v, 4);
    f = (float)c->cpuct;              memcpy(hdr + 136, &f, 4);
    f = (float)c->noise_frac;         memcpy(hdr + 140, &f, 4);
    f = (float)c->noise_alpha;        memcpy(hdr + 144, &f, 4);
    v = (uint32_t)c->games_par;       memcpy(hdr + 148, &v, 4);
    v = (uint32_t)c->opp_sims;        memcpy(hdr + 152, &v, 4);
    v = c->sims ? 7u | (c->contempt_flat ? 8u : 0u) : 0u; memcpy(hdr + 156, &v, 4);
    f = (float)c->contempt;           memcpy(hdr + 160, &f, 4);
    f = (float)c->soft_scale;         memcpy(hdr + 164, &f, 4);
    f = c->soft_scale > 0.0 ? (float)SOFT_PAWNS : 0.0f; memcpy(hdr + 168, &f, 4);
    sink_put(s, hdr, TKN_FILE_HDR);
}

/* .tkn writing, shared by play and elo: a provisional header, then one game at a time,
 * flushed, so a killed run keeps every finished game; the counts are patched at close. */
static int tkn_open(Sink *s, const Cfg *c, const Net *net, const char *model)
{
    s->fd = open(c->out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (s->fd < 0) { perror(c->out_path); return -1; }
    s->buf = malloc(s->cap);
    put_file_header(s, c, net, model, hash_file(model));
    return 0;
}

/* nnplay's internal move (origin36 << 14 | special * 1296 + target) in the on-disk packing */
static inline uint32_t file_move(uint32_t v)
{
    int origin36 = UNPACK_O(v), mv = UNPACK_M(v);
    return ((uint32_t)(mv / 1296) << 28) | ((uint32_t)origin36 << 17) | ((uint32_t)(mv % 1296) << 6);
}

/* One ply record: the board after the move, and the network's value of it for the side
 * now to move. */
static void tkn_fill(PlyRecNN *r, const Pos *p, uint32_t pick, float value, int n_legal,
                     int mover, int flags, int n_scored, int mat_div)
{
    snapshot(p, r->board);
    r->move = file_move(pick);
    r->value = value;
    r->n_legal = (uint16_t)(n_legal > 65535 ? 65535 : n_legal);
    r->material = (int16_t)quantise(p->material, mat_div);
    r->side = (uint8_t)mover;
    r->flags = (uint8_t)flags;
    r->n_scored = (uint16_t)(n_scored > 65535 ? 65535 : n_scored);
}

static void tkn_put_game(Sink *s, PlyRecNN *plies, int nply, int result, int term,
                         double label, uint64_t seed)
{
    if (nply) plies[nply - 1].flags |= PF_TERMINAL;
    unsigned char gh[TKN_GAME_HDR];
    uint32_t np = (uint32_t)nply;
    memcpy(gh, &np, 4);
    /* the user-facing score: 1 black, -1 white, 0 a draw */
    gh[4] = (unsigned char)(int8_t)(result == RES_BLACK ? 1 : result == RES_WHITE ? -1 : 0);
    gh[5] = (unsigned char)term;
    int16_t l16 = (int16_t)lrint(label * 10000.0);  /* the training label, in 1/10000 */
    memcpy(gh + 6, &l16, 2);
    memcpy(gh + 8, &seed, 8);
    sink_put(s, gh, TKN_GAME_HDR);
    sink_put(s, plies, (size_t)nply * TKN_REC);
    sink_flush(s);              /* a game is hours; do not hold it in a buffer */
}

static void tkn_close(Sink *s, uint64_t games, uint64_t plies)
{
    sink_flush(s);
    if (pwrite(s->fd, &games, 8, 40) != 8) perror("pwrite");
    if (pwrite(s->fd, &plies, 8, 48) != 8) perror("pwrite");
    close(s->fd);
    free(s->buf);
    s->buf = NULL;
    s->fd = -1;
}

/* ---- .kids: what the search saw at each ply -------------------------------------- */
/* The .tkn keeps only the move played; this side file keeps what policy training learns
 * from, at no extra network cost.
 *
 *   file header, 32 bytes: "TKYKIDS\0", u32 version, u32 header size 32, u32 child size,
 *       u32 game header size 24, u64 games (patched at close)
 *   per game, 24 bytes: u32 n_plies, u32 reserved, u64 seed, u64 FNV-1a 64 of the game's
 *       ply-record move words, in order -- (seed, n_plies, hash) finds the matching .tkn
 *       game even after files are merged or reordered
 *   per ply: u16 n, then n children
 *     version 1 (--sims 0), 6 bytes each: { u32 move, u16 value as IEEE half } -- every
 *       child the value head scored
 *     version 2 (search), 8 bytes each: { u32 move, u16 visits, u16 value as half } --
 *       every root child the search visited, with its mean value
 *
 * Ply k's children belong to the position *before* move k: the .tkn record k-1's board,
 * or the start position for k = 0. A move uses the ply record's packing, special << 28 |
 * origin << 17 | target << 6, absolute frame. A value is for the side to move after that
 * child: the mover prefers low values. A forced King capture is stored as its single
 * child. Games in lockstep fill their own buffers, so a file's games never interleave. */
#define KIDS_MAGIC "TKYKIDS"
#define KIDS_FILE_HDR 32
#define KIDS_GAME_HDR 24

typedef struct { unsigned char *b; size_t len, cap; } Bytes;
typedef struct { Sink sink; uint64_t games; int version; } Kids;

static void bytes_put(Bytes *x, const void *d, size_t n)
{
    if (x->len + n > x->cap) {
        x->cap = (x->len + n) * 2;
        x->b = realloc(x->b, x->cap);
        if (!x->b) { fprintf(stderr, "oom in --children buffer\n"); exit(1); }
    }
    memcpy(x->b + x->len, d, n);
    x->len += n;
}

static int kids_open(Kids *k, const char *path, int version)
{
    memset(k, 0, sizeof(*k));
    k->version = version;
    k->sink.cap = 1u << 22;
    k->sink.fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (k->sink.fd < 0) { perror(path); return -1; }
    k->sink.buf = malloc(k->sink.cap);
    unsigned char h[KIDS_FILE_HDR];
    memset(h, 0, sizeof(h));
    memcpy(h, KIDS_MAGIC, 8);
    uint32_t v = (uint32_t)version; memcpy(h + 8, &v, 4);
    v = KIDS_FILE_HDR;              memcpy(h + 12, &v, 4);
    v = version == 2 ? 8 : 6;       memcpy(h + 16, &v, 4);
    v = KIDS_GAME_HDR;              memcpy(h + 20, &v, 4);
    sink_put(&k->sink, h, sizeof(h));
    return 0;
}

/* One ply: n children, moves[i] in nnplay's internal packing, indexed through perm (NULL
 * for the identity), with their values, and with their visits for version 2. */
static void kids_ply(Bytes *x, const uint32_t *moves, const int *perm, const float *values,
                     const int *visits, int n)
{
    if (n > 65535) n = 65535;
    uint16_t c = (uint16_t)n;
    bytes_put(x, &c, 2);
    for (int i = 0; i < n; i++) {
        uint32_t m = file_move(moves[perm ? perm[i] : i]);
        uint16_t h = float_to_half(values[i]);
        bytes_put(x, &m, 4);
        if (visits) { uint16_t vv = (uint16_t)(visits[i] > 65535 ? 65535 : visits[i]); bytes_put(x, &vv, 2); }
        bytes_put(x, &h, 2);
    }
}

static void kids_game(Kids *k, Bytes *x, const PlyRecNN *plies, int nply, uint64_t seed)
{
    uint64_t hash = 0xcbf29ce484222325ull;
    for (int i = 0; i < nply; i++) {
        uint32_t m = plies[i].move;
        for (int b = 0; b < 4; b++) { hash ^= (m >> (8 * b)) & 0xFF; hash *= 0x100000001b3ull; }
    }
    unsigned char gh[KIDS_GAME_HDR];
    uint32_t np = (uint32_t)nply, zero = 0;
    memcpy(gh, &np, 4);
    memcpy(gh + 4, &zero, 4);
    memcpy(gh + 8, &seed, 8);
    memcpy(gh + 16, &hash, 8);
    sink_put(&k->sink, gh, sizeof(gh));
    sink_put(&k->sink, x->b, x->len);
    sink_flush(&k->sink);
    x->len = 0;
    k->games++;
}

static void kids_close(Kids *k)
{
    sink_flush(&k->sink);
    if (pwrite(k->sink.fd, &k->games, 8, 24) != 8) perror("pwrite");
    close(k->sink.fd);
    free(k->sink.buf);
}

/* Seconds as a short human string: 45s, 3m41s, 2h14m. */
static void fmt_dur(double secs, char *buf, size_t n)
{
    long t = secs > 0 ? (long)(secs + 0.5) : 0;
    if (t < 60)        snprintf(buf, n, "%lds", t);
    else if (t < 3600) snprintf(buf, n, "%ldm%02lds", t / 60, t % 60);
    else               snprintf(buf, n, "%ldh%02ldm", t / 3600, (t % 3600) / 60);
}

/* The start-position reference export_onnx.py writes into net_manifest.json. A string
 * search rather than a JSON parser: the file is ours, and each key occurs exactly once. */
static char manifest_buf[1 << 18];

/* The numbers in a flat JSON array under `key`, up to cap of them; how many, or -1. */
static int manifest_array(const char *key, double *out, int cap)
{
    char pat[96];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    char *p = strstr(manifest_buf, pat);
    if (!p || !(p = strchr(p, '['))) return -1;
    p++;
    int n = 0;
    while (n < cap) {
        while (*p == ' ' || *p == '\n' || *p == ',' || *p == '\r' || *p == '\t') p++;
        if (*p == ']' || !*p) break;
        char *e;
        double x = strtod(p, &e);
        if (e == p) break;
        out[n++] = x;
        p = e;
    }
    return n;
}

static int manifest_ref(const char *path, double *val, double *tol)
{
    char *buf = manifest_buf;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t n = fread(buf, 1, sizeof(manifest_buf) - 1, f);
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

    /* The value and the policy in one call, as the search makes it: every legal move of
     * the start position through the C move generator and the rows nnplay builds, against
     * what export_onnx.py computed in torch for the same moves. */
    Gen g;
    memset(&g, 0, sizeof(g));
    g.cap = 8192; g.buf = malloc(8192 * 4); g.stamp = calloc(MAXMOVE, 2); g.kline = calloc(NSQ, 2);
    gen_all(p, &g);
    int32_t *rows = malloc((size_t)g.n * 4 * sizeof(int32_t));
    float *got = malloc((size_t)g.n * sizeof(float)), v = 0.0f;
    for (int i = 0; i < g.n; i++) {
        int mvv = UNPACK_M(g.buf[i]);
        rows[4 * i] = 0; rows[4 * i + 1] = UNPACK_O(g.buf[i]);
        rows[4 * i + 2] = mvv % 1296; rows[4 * i + 3] = mvv / 1296;
    }
    ORT_CHECK(net_run_raw(net, b, 1, rows, g.n, &v, got));
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

    static double mv[3 * 8192], lg[8192];
    int nm = manifest_array("start_position_policy_moves", mv, 3 * 8192);
    int nl = manifest_array("start_position_policy_logits", lg, 8192);
    if (nm > 0 && nl * 3 == nm) {
        double ptol = 0.05;
        char *q = strstr(manifest_buf, "\"policy_tolerance\"");
        if (q && (q = strchr(q, ':'))) ptol = strtod(q + 1, NULL);
        int matched = 0;
        double worst = 0.0;
        for (int i = 0; i < g.n; i++)
            for (int j = 0; j < nl; j++)
                if (mv[3 * j] == rows[4 * i + 1] && mv[3 * j + 1] == rows[4 * i + 2]
                    && mv[3 * j + 2] == rows[4 * i + 3]) {
                    matched++;
                    double dd = fabs((double)got[i] - lg[j]);
                    if (dd > worst) worst = dd;
                    break;
                }
        int pok = matched == g.n && g.n == nl && worst < ptol;
        printf("  policy          %d moves, %d matched the export's, largest logit deviation %.2e "
               "(tolerance %.0e)   %s\n", g.n, matched, worst, ptol, pok ? "PASS" : "FAIL");
        ok = ok && pok;
    } else {
        printf("  policy          the manifest has no start-position policy -- re-run export_onnx.py\n");
        ok = 0;
    }
    fflush(stdout);
    free(rows); free(got); free(g.buf); free(g.stamp); free(g.kline);
    free(b);
    free(p);
    return ok ? 0 : 1;
}

/* ---- search ---------------------------------------------------------------------- */
/* PUCT over the policy's priors and the value head, as AlphaZero, with these choices:
 *
 *   - Q is min-max normalised over the tree before it meets c * P * sqrt(N) / (1 + n)
 *     (MuZero). Children's values differ by ~0.01, so raw Q would barely steer the visits.
 *   - An unvisited child is scored at its parent's network value.
 *   - Priors are softmax(logits) over every legal move.
 *   - A tree walks `leaves` times before the network sees the leaves, each pending visit
 *     counting as a loss on its path (virtual loss) so the next walk goes elsewhere. The
 *     games in lockstep pool their leaves into one call.
 *   - The searched subtree under the move played is kept for the next move.
 *
 * The tree models three rules itself: a position where the side to move can take the King
 * is a win for it (forced regicide), one seen rep_limit times is a draw, and with King
 * safety a move that leaves the mover's own King capturable is never played while another
 * is not -- such an edge is pruned when first reached, and a node whose every edge is
 * pruned is lost. Without the last, a King threat with a handful of escapes among hundreds
 * of replies averaged out near a forced win. The ply cap and resignation are left to the
 * value head, whose training labels already carry them.
 *
 * A node's w sums the values backed up through it from the view of the side that moved
 * into it, so a parent prefers the child with the highest w / n. */
typedef struct { uint32_t mv; float p; int32_t c; } Edge;     /* c: child, -1 none yet */
#define PRUNED (-2)                                               /* c: hangs the King */
typedef struct {
    uint64_t h;
    int32_t  e0, ne;        /* its edges, once expanded */
    int32_t  n, vl;         /* visits, and how many of them are still pending */
    float    w, v;          /* value sum, the mover's view; network value, the side to move's */
    float    tv;            /* terminal value, the side to move's view */
    int8_t   term, pend;    /* term: 1 decided, tv; 2 a draw, valued by the contempt */
    int16_t  npr;           /* edges pruned */
} Node;

typedef struct Tree Tree;

/* Leaves from every tree, for one network call. */
typedef struct {
    int32_t *boards, *rows;
    float   *val, *lgt;
    Tree   **tr;
    int     *leaf, *roff;
    int      nb, cb, nr, cr;
} Batch;

struct Tree {
    Node    *nd;  int nn, cn;
    Edge    *ed;  int ne, cne;
    int      root;
    Pos      pos, walk;       /* the root position; a scratch copy for each walk */
    double   qmin, qmax;
    double   noise_frac, noise_alpha;
    double   contempt_c, contempt;  /* --contempt, and what a draw costs this move */
    int      contempt_flat;
    Rng     *rng;
    int      todo;            /* simulations still to run for this move */
    int      evals;           /* leaves sent to the network for this move */
    int      noise;           /* root noise still to apply */
    int      np;              /* leaves waiting for the network */
    int     *pnode, *poff, *plen, *pmoff, *pmcnt;
    int     *path; int npath, cpath;
    uint32_t *pmv; int npmv, cpmv;
};

static void *grow(void *p, int *cap, int need, size_t elt)
{
    if (need <= *cap) return p;
    *cap = need * 2 > 1024 ? need * 2 : 1024;
    p = realloc(p, (size_t)*cap * elt);
    if (!p) { fprintf(stderr, "oom in the search\n"); exit(1); }
    return p;
}

static int tree_node(Tree *t, uint64_t h)
{
    t->nd = grow(t->nd, &t->cn, t->nn + 1, sizeof(Node));
    Node *x = &t->nd[t->nn];
    memset(x, 0, sizeof(*x));
    x->h = h;
    return t->nn++;
}

static void tree_reset(Tree *t, const Pos *p)
{
    t->nn = t->ne = 0;
    memcpy(&t->pos, p, sizeof(Pos));
    t->root = tree_node(t, p->hash);
}

static void tree_init(Tree *t, int leaves, Rng *rng)
{
    memset(t, 0, sizeof(*t));
    t->rng = rng;
    t->pnode = malloc(sizeof(int) * (size_t)leaves);
    t->poff  = malloc(sizeof(int) * (size_t)leaves);
    t->plen  = malloc(sizeof(int) * (size_t)leaves);
    t->pmoff = malloc(sizeof(int) * (size_t)leaves);
    t->pmcnt = malloc(sizeof(int) * (size_t)leaves);
}

static void tree_free(Tree *t)
{
    free(t->nd); free(t->ed); free(t->path); free(t->pmv);
    free(t->pnode); free(t->poff); free(t->plen); free(t->pmoff); free(t->pmcnt);
}

static inline void path_push(Tree *t, int s)
{
    t->path = grow(t->path, &t->cpath, t->npath + 1, sizeof(int));
    t->path[t->npath++] = s;
}

/* How often h has occurred in the game so far, without counting this look. */
static int rep_peek(const RepTab *r, uint64_t h)
{
    size_t i = (size_t)(h >> (64 - REP_BITS));
    for (;;) {
        if (r->ep[i] != r->epoch) return 0;
        if (r->key[i] == h) return r->cnt[i];
        i = (i + 1) & REP_MASK;
    }
}

static double unif(Rng *r) { return ((rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

/* Marsaglia-Tsang, with the a < 1 boost. */
static double gamma_draw(Rng *r, double a)
{
    if (a < 1.0) return gamma_draw(r, a + 1.0) * pow(unif(r), 1.0 / a);
    double d = a - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = sqrt(-2.0 * log(unif(r))) * cos(2.0 * M_PI * unif(r));
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        double u = unif(r);
        if (u < 1.0 - 0.0331 * x * x * x * x || log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v;
    }
}

/* Dirichlet noise on the root's priors, alpha = noise_alpha / legal moves (AlphaZero's
 * scaling: 0.3 at chess's ~35 moves, 0.03 at Go's ~250, ~0.01 here). */
static void tree_noise(Tree *t)
{
    Node *r = &t->nd[t->root];
    t->noise = 0;
    if (r->ne < 2 || t->noise_frac <= 0.0) return;
    double alpha = t->noise_alpha / r->ne, sum = 0.0;
    double *eta = malloc(sizeof(double) * (size_t)r->ne);
    for (int i = 0; i < r->ne; i++) sum += eta[i] = gamma_draw(t->rng, alpha);
    if (sum > 0.0)
        for (int i = 0; i < r->ne; i++) {
            Edge *e = &t->ed[r->e0 + i];
            e->p = (float)((1.0 - t->noise_frac) * e->p + t->noise_frac * eta[i] / sum);
        }
    free(eta);
}

/* Contempt: a draw is worth -c * sigmoid(20 * (v_root - 0.1)) to the side searching, v_root
 * the network's value of the root. A side that thinks it is ahead is pushed off draws; the
 * offset keeps near-level positions from flipping between the two attitudes. Self-play
 * uses a flat -c instead: level positions shuffled into repetition at ~0.012. */
#define CONTEMPT_MID   0.1
#define CONTEMPT_SLOPE 20.0
static void tree_contempt(Tree *t)
{
    t->contempt = t->contempt_flat ? t->contempt_c
                : t->contempt_c / (1.0 + exp(-CONTEMPT_SLOPE * (t->nd[t->root].v - CONTEMPT_MID)));
}

static void tree_begin(Tree *t, int sims, double noise_frac, double noise_alpha, double contempt,
                       int flat)
{
    t->todo = sims;
    t->evals = 0;
    t->qmin = 1e30; t->qmax = -1e30;
    t->noise_frac = noise_frac;
    t->noise_alpha = noise_alpha;
    t->noise = noise_frac > 0.0;
    t->contempt_c = contempt;
    t->contempt_flat = flat;
    t->contempt = 0.0;
    if (t->nd[t->root].ne) tree_contempt(t);
    if (t->noise && t->nd[t->root].ne) tree_noise(t);
}

/* A terminal's value for its side to move. A draw is -contempt where the searching side is
 * to move -- an even number of plies below the root -- and +contempt where its opponent is. */
static inline double terminal_value(const Tree *t, const Node *nd, int depth)
{
    if (nd->term != 2) return nd->tv;
    return depth & 1 ? t->contempt : -t->contempt;
}

static inline double tree_norm(const Tree *t, double q)
{
    if (t->qmax <= t->qmin) return q;
    double x = (q - t->qmin) / (t->qmax - t->qmin);
    return x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;
}

/* The value v (for the leaf's side to move) up the path from `start`, undoing the
 * virtual loss each walk put there; the root's w is never read. */
static void tree_backup(Tree *t, int start, int len, double v)
{
    double x = -v;
    for (int i = start + len - 1; i >= start; i--) {
        Node *nd = &t->nd[t->path[i]];
        nd->vl--;
        if (i > start) {
            nd->w += (float)(x + 1.0);
            int real = nd->n - nd->vl;
            if (real > 0) {
                double q = (nd->w + nd->vl) / real;
                if (q < t->qmin) t->qmin = q;
                if (q > t->qmax) t->qmax = q;
            }
        } else {
            nd->w += (float)x;
        }
        x = -x;
    }
}

static void tree_undo(Tree *t, int start, int len)
{
    for (int i = start; i < start + len; i++) {
        Node *nd = &t->nd[t->path[i]];
        nd->n--; nd->vl--;
        if (i > start) nd->w += 1.0f;
    }
}

static int tree_select(const Tree *t, int s, double cpuct)
{
    const Node *nd = &t->nd[s];
    double sq = sqrt((double)nd->n), fpu = tree_norm(t, nd->v), best = -1e300;
    int arg = 0;
    for (int i = 0; i < nd->ne; i++) {
        const Edge *e = &t->ed[nd->e0 + i];
        double q = fpu, n = 0.0;
        if (e->c == PRUNED) continue;
        if (e->c >= 0) { const Node *c = &t->nd[e->c]; n = c->n; q = tree_norm(t, c->w / c->n); }
        double u = q + cpuct * e->p * sq / (1.0 + n);
        if (u > best) { best = u; arg = i; }
    }
    return arg;
}

/* Whether a position just reached is decided without the network: 1 won or lost, with its
 * value for the side to move in tv, 2 drawn, 0 not. Leaves the legal moves in g if 0. */
static int leaf_terminal(const Pos *p, const Cfg *c, const RepTab *hist, const Tree *t,
                         int start, Gen *g, float *tv)
{
    int stm = p->turn;
    if (!p->royals[stm])     { *tv = -1.0f; return 1; }
    if (!p->royals[1 - stm]) { *tv =  1.0f; return 1; }     /* the mover swept its own King */
    gen_all(p, g);
    if (!g->n) { *tv = c->stalemate_loses ? -1.0f : 0.0f; return c->stalemate_loses ? 1 : 2; }
    if (c->regicide) {
        king_ctx_set(g, p);
        if (find_king_capture(g)) { *tv = 1.0f; return 1; }
    }
    if (c->rep_limit > 0) {
        int k = rep_peek(hist, p->hash) + 1;
        for (int i = start + 1; i < t->npath - 1; i++) k += t->nd[t->path[i]].h == p->hash;
        if (k >= c->rep_limit) { *tv = 0.0f; return 2; }
    }
    return 0;
}

static void batch_reserve(Batch *b, int boards, int rows)
{
    if (b->nb + boards + 1 > b->cb) {
        b->cb = (b->nb + boards + 1) * 2;
        b->boards = realloc(b->boards, (size_t)b->cb * NSQ * sizeof(int32_t));
        b->val  = realloc(b->val,  (size_t)b->cb * sizeof(float));
        b->tr   = realloc(b->tr,   (size_t)b->cb * sizeof(Tree *));
        b->leaf = realloc(b->leaf, (size_t)b->cb * sizeof(int));
        b->roff = realloc(b->roff, (size_t)b->cb * sizeof(int));
    }
    if (b->nr + rows > b->cr) {
        b->cr = (b->nr + rows) * 2;
        b->rows = realloc(b->rows, (size_t)b->cr * 4 * sizeof(int32_t));
        b->lgt  = realloc(b->lgt,  (size_t)b->cr * sizeof(float));
    }
    if (!b->boards || !b->val || !b->tr || !b->leaf || !b->roff || !b->rows || !b->lgt) {
        fprintf(stderr, "oom in the search batch\n");
        exit(1);
    }
}

/* Queues node s at position p, whose legal moves are in g, for the network. */
static void tree_queue(Tree *t, Batch *b, int s, const Pos *p, const Gen *g, int start)
{
    int k = t->np++, n = g->n;
    t->pnode[k] = s; t->poff[k] = start; t->plen[k] = t->npath - start;
    t->pmv = grow(t->pmv, &t->cpmv, t->npmv + n, sizeof(uint32_t));
    t->pmoff[k] = t->npmv; t->pmcnt[k] = n;
    memcpy(t->pmv + t->npmv, g->buf, (size_t)n * sizeof(uint32_t));
    t->npmv += n;

    batch_reserve(b, 1, n);
    int i = b->nb++;
    encode_stm(p, b->boards + (size_t)i * NSQ);
    b->tr[i] = t; b->leaf[i] = k; b->roff[i] = b->nr;
    int flip = p->turn != SIDE_BLACK;              /* encode_stm rotates white to move */
    int32_t *r = b->rows + (size_t)b->nr * 4;
    for (int j = 0; j < n; j++, r += 4) {
        uint32_t v = g->buf[j];
        int o = UNPACK_O(v), mv = UNPACK_M(v), sp = mv / 1296, to = mv % 1296;
        if (flip) {
            o = NSQ - 1 - o;
            to = NSQ - 1 - to;
            if (sp >= 1 && sp <= 8) sp = (sp - 1 + 4) % 8 + 1;   /* DIRS[i] -> DIRS[i+4] */
        }
        r[0] = i; r[1] = o; r[2] = to; r[3] = sp;
    }
    b->nr += n;
    t->nd[s].pend = 1;
    t->evals++;
}

/* One simulation: 0 finished on a terminal, 1 queued a leaf, 2 ran into a pending one, 3
 * pruned a move that hangs the King and counts for nothing. */
static int tree_walk(Tree *t, const Cfg *c, const RepTab *hist, Gen *g, Batch *b)
{
    Pos *p = &t->walk;
    memcpy(p, &t->pos, sizeof(Pos));
    int start = t->npath, s = t->root;
    path_push(t, s);
    t->nd[s].n++; t->nd[s].vl++;
    for (;;) {
        Node *nd = &t->nd[s];
        if (nd->term) {
            tree_backup(t, start, t->npath - start, terminal_value(t, nd, t->npath - 1 - start));
            t->npath = start;
            return 0;
        }
        if (nd->pend) { tree_undo(t, start, t->npath - start); t->npath = start; return 2; }
        if (!nd->ne) {                                   /* the fresh root */
            gen_all(p, g);
            tree_queue(t, b, s, p, g, start);
            return 1;
        }
        Edge *e = &t->ed[nd->e0 + tree_select(t, s, c->cpuct)];
        do_move(p, SQ2P[UNPACK_O(e->mv)], UNPACK_M(e->mv));
        int ch = e->c, fresh = ch < 0;
        if (fresh) ch = e->c = tree_node(t, p->hash);    /* t->nd may move; t->ed does not */
        path_push(t, ch);
        Node *cn = &t->nd[ch];
        cn->n++; cn->vl++; cn->w -= 1.0f;
        if (fresh) {
            float tv;
            int r = leaf_terminal(p, c, hist, t, start, g, &tv);
            if (r == 1 && tv > 0.0f && c->king_safety) {
                tree_undo(t, start, t->npath - start);
                t->npath = start;
                e->c = PRUNED;
                Node *pn = &t->nd[s];
                if (++pn->npr == pn->ne) { pn->term = 1; pn->tv = -1.0f; }
                return 3;
            }
            if (r) {
                cn->term = (int8_t)r; cn->tv = tv;
                tree_backup(t, start, t->npath - start, terminal_value(t, cn, t->npath - 1 - start));
                t->npath = start;
                return 0;
            }
            tree_queue(t, b, ch, p, g, start);
            return 1;
        }
        s = ch;
    }
}

/* Up to `leaves` walks; terminals are backed up on the spot. */
static void tree_gather(Tree *t, const Cfg *c, const RepTab *hist, Gen *g, Batch *b)
{
    int collisions = 0;
    while (t->np < c->leaves && t->todo > t->np && collisions < c->leaves) {
        int r = tree_walk(t, c, hist, g, b);
        if (r == 0) t->todo--;
        else if (r == 2) collisions++;
    }
}

/* Leaf k of tree t back from the network: its edges, then the value up its path. */
static void tree_expand(Tree *t, int k, float v, const float *lg)
{
    int s = t->pnode[k], n = t->pmcnt[k];
    t->ed = grow(t->ed, &t->cne, t->ne + n, sizeof(Edge));
    int e0 = t->ne;
    t->ne += n;
    float mx = lg[0];
    for (int i = 1; i < n; i++) if (lg[i] > mx) mx = lg[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += t->ed[e0 + i].p = expf(lg[i] - mx);
    for (int i = 0; i < n; i++) {
        t->ed[e0 + i].p = (float)(t->ed[e0 + i].p / sum);
        t->ed[e0 + i].mv = t->pmv[t->pmoff[k] + i];
        t->ed[e0 + i].c = -1;
    }
    Node *nd = &t->nd[s];
    nd->e0 = e0; nd->ne = n; nd->v = v; nd->pend = 0;
    if (s == t->root) tree_contempt(t);
    if (s == t->root && t->noise) tree_noise(t);
    tree_backup(t, t->poff[k], t->plen[k], v);
    t->todo--;
}

/* One network call per sub-batch, then every leaf back into its tree. Row board indices
 * are rebased per sub-batch. */
static double batch_run(Net *net, Batch *b)
{
    if (!b->nb) return 0.0;
    double t0 = now_s();
    b->roff[b->nb] = b->nr;
    for (int at = 0; at < b->nb; ) {
        int nb = b->nb - at < net->sub_batch ? b->nb - at : net->sub_batch;
        int r0 = b->roff[at], r1 = b->roff[at + nb];
        for (int r = r0; r < r1; r++) b->rows[(size_t)r * 4] -= at;
        ORT_CHECK(net_run_raw(net, b->boards + (size_t)at * NSQ, nb, b->rows + (size_t)r0 * 4,
                              r1 - r0, b->val + at, b->lgt + r0));
        at += nb;
    }
    double dt = now_s() - t0;
    for (int i = 0; i < b->nb; i++) tree_expand(b->tr[i], b->leaf[i], b->val[i], b->lgt + b->roff[i]);
    for (int i = 0; i < b->nb; i++) { Tree *t = b->tr[i]; t->np = 0; t->npath = 0; t->npmv = 0; }
    b->nb = b->nr = 0;
    return dt;
}

/* The move to play, as a root edge: visits ** (1 / T), T = 0 the most visited (ties to
 * the higher Q). King safety on top: a pick that hangs the King is struck out and the
 * draw repeated, which is exactly the distribution restricted to the safe moves; if every
 * visited move hangs, the unvisited are tried in prior order. *q gets the child's mean
 * value for the side then to move. */
static int tree_pick(Tree *t, Scratch *s, Rng *rng, double T, int king_safety, int *changed,
                     float *q)
{
    const Node *r = &t->nd[t->root];
    scratch_reserve(s, r->ne);
    int live = 0, k = -1;
    for (int i = 0; i < r->ne; i++) {
        const Edge *e = &t->ed[r->e0 + i];
        if (e->c >= 0 && t->nd[e->c].n > 0) s->perm[live++] = i;
    }
    *changed = 0;
    uint64_t before = s->checks;
    while (live > 0) {
        int nmax = 0;
        for (int j = 0; j < live; j++) {
            int n = t->nd[t->ed[r->e0 + s->perm[j]].c].n;
            if (n > nmax) nmax = n;
        }
        if (T <= 0.0) {
            double bq = -1e30;
            for (int j = 0; j < live; j++) {
                const Node *c = &t->nd[t->ed[r->e0 + s->perm[j]].c];
                if (c->n == nmax && c->w / c->n > bq) { bq = c->w / c->n; k = j; }
            }
        } else {
            double sum = 0.0;
            for (int j = 0; j < live; j++)
                sum += s->weights[j] = exp(log((double)t->nd[t->ed[r->e0 + s->perm[j]].c].n / nmax) / T);
            double x = unif(rng) * sum;
            for (k = 0; k < live - 1 && (x -= s->weights[k]) > 0.0; k++) { }
        }
        if (!king_safety || !hangs_king(s, &t->pos, t->ed[r->e0 + s->perm[k]].mv)) break;
        s->perm[k] = s->perm[--live];
        k = -1;
    }
    int e;
    if (k >= 0) {
        e = s->perm[k];
    } else {                                    /* every visited move hangs the King */
        e = -1;
        for (int i = 0; i < r->ne; i++) s->perm[i] = i;
        for (int i = 0; i < r->ne && e < 0; i++) {
            int bi = i;
            for (int j = i + 1; j < r->ne; j++)
                if (t->ed[r->e0 + s->perm[j]].p > t->ed[r->e0 + s->perm[bi]].p) bi = j;
            int tmp = s->perm[i]; s->perm[i] = s->perm[bi]; s->perm[bi] = tmp;
            if (!hangs_king(s, &t->pos, t->ed[r->e0 + s->perm[i]].mv)) e = s->perm[i];
        }
        if (e < 0) {                            /* nothing saves it: the most visited */
            s->hopeless++;
            int nmax = -1;
            for (int i = 0; i < r->ne; i++) {
                int c = t->ed[r->e0 + i].c, n = c >= 0 ? t->nd[c].n : 0;
                if (n > nmax) { nmax = n; e = i; }
            }
        }
    }
    *changed = king_safety && s->checks - before > 1;
    s->overridden += *changed;
    int c = t->ed[r->e0 + e].c;
    *q = c >= 0 && t->nd[c].n > 0 ? -t->nd[c].w / t->nd[c].n : -r->v;
    return e;
}

/* The root's visited children, for .kids version 2. */
static void tree_kids(const Tree *t, Bytes *x, Scratch *s)
{
    const Node *r = &t->nd[t->root];
    scratch_reserve(s, r->ne);
    uint32_t *mv = (uint32_t *)s->mv4;
    int n = 0;
    for (int i = 0; i < r->ne; i++) {
        const Edge *e = &t->ed[r->e0 + i];
        if (e->c < 0 || t->nd[e->c].n <= 0) continue;
        mv[n] = e->mv;
        s->perm[n] = t->nd[e->c].n;
        s->values[n] = -t->nd[e->c].w / t->nd[e->c].n;
        n++;
    }
    kids_ply(x, mv, NULL, s->values, s->perm, n);
}

/* The move was played and p is the new position: keep the subtree below it, compacted. */
static void tree_advance(Tree *t, uint32_t mv, const Pos *p)
{
    const Node *r = &t->nd[t->root];
    int ch = -1;
    for (int i = 0; i < r->ne; i++)
        if (t->ed[r->e0 + i].mv == mv) { ch = t->ed[r->e0 + i].c; break; }
    if (ch < 0 || t->nd[ch].term || !t->nd[ch].ne) { tree_reset(t, p); return; }
    Node *nd = malloc(sizeof(Node) * (size_t)t->cn);
    Edge *ed = malloc(sizeof(Edge) * (size_t)t->cne);
    if (!nd || !ed) { fprintf(stderr, "oom in the search\n"); exit(1); }
    int nn = 1, ne = 0;
    nd[0] = t->nd[ch];
    for (int head = 0; head < nn; head++) {      /* breadth first, renumbering as it goes */
        Node *x = &nd[head];
        int e0 = ne;
        for (int i = 0; i < x->ne; i++) {
            Edge e = t->ed[x->e0 + i];
            if (e.c >= 0) { nd[nn] = t->nd[e.c]; e.c = nn++; }
            ed[ne++] = e;
        }
        x->e0 = e0;
    }
    free(t->nd); free(t->ed);
    t->nd = nd; t->ed = ed; t->nn = nn; t->ne = ne;
    t->root = 0;
    memcpy(&t->pos, p, sizeof(Pos));
}

static const char USAGE[] =
    "usage: nnplay <play|elo|variance|selfcheck|eval|encode> [options]\n"
    "  play    self-play games and write a .tkn shard\n"
    "    --out FILE          where to write (omit to play without recording)\n"
    "    --games N           games to keep (default 1)\n"
    "    --positions N       play whole games until at least N positions are written,\n"
    "                        however many games that takes. Games in play when it is\n"
    "                        reached are finished, so the count overshoots by up to\n"
    "                        --parallel-games games, and every game keeps a real result.\n"
    "                        With --games as well, that becomes a cap on the games.\n"
    "    --first-game N      index of the first game, so shards can be split across runs\n"
    "    --sims N            simulations of the search per move (default 128). 0 is the\n"
    "                        one-ply search: the value head scores the children of the\n"
    "                        policy's top 5% of the moves plus 2% at random\n"
    "    --temperature T     the move is drawn by visits^(1/T) after a search, or by a\n"
    "                        softmax over the children's values at --sims 0 (default 1.0;\n"
    "                        0 plays the most visited, or the best scored)\n"
    "    --opening-plies N   play the first N plies at --opening-temperature, then switch\n"
    "                        to --temperature (default 0: one temperature all game)\n"
    "    --opening-temperature T   temperature for those plies (default 1.0)\n"
    "    --leaves N          leaves a tree gathers, under virtual loss, before the network\n"
    "                        sees them (default 8)\n"
    "    --parallel-games N  games played in lockstep, their leaves sharing each network\n"
    "                        call (default 32)\n"
    "    --cpuct C           exploration constant, on min-max normalised Q (default 1.25)\n"
    "    --noise-frac F      share of Dirichlet noise in the root's priors (default 0.25,\n"
    "                        0 disables)...\n"
    "    --noise-alpha A     ...with alpha = A / legal moves (default 10)\n"
    "    --contempt C        in the search, a draw (repetition, or stalemate with\n"
    "                        stalemate a draw) is worth -C * sigmoid(20 * (v_root - 0.1))\n"
    "                        to the side searching, v_root the network's value of the\n"
    "                        root: a side that thinks it is ahead avoids draws (default\n"
    "                        0.1, 0 off; elo applies it to both sides). play makes it\n"
    "                        flat: a draw is worth -C to the side searching, always\n"
    "    --max-plies N       ply cap (default 3000)\n"
    "    --rep-limit N       repetition limit (default 4, 0 disables)\n"
    "    --no-progress N     plies without a capture or promotion before a draw (0 disables)\n"
    "    --no-regicide       do not force an available King capture\n"
    "    --no-king-safety    allow moves that leave your own King capturable. By default\n"
    "                        such a move is never played while another move avoids it,\n"
    "                        the mirror of forced regicide; this applies in every mode\n"
    "    --resign-lead P     a side ahead by P pawns of material for --resign-plies plies\n"
    "                        in a row is scored the winner, as if the other had resigned\n"
    "                        (default 300, about 13% of the starting material; 0 disables)\n"
    "    --resign-plies K    how long the lead must hold (default 10), so one trample\n"
    "                        cannot end a game\n"
    "    --cap-draw          score the ply cap as a draw. By default it goes to the side\n"
    "                        ahead on material, and is a draw under a 50-pawn lead\n"
    "    --selfplay-soft-target-scale S   label a game cut off by the ply cap or by\n"
    "                        repetition S * tanh(material / 300 pawns) for training, black's\n"
    "                        view, instead of its result (default 0.75, 0 off). The result\n"
    "                        itself, and so the reported score, is unchanged\n"
    "    --drop-capped       write no game that reached the ply cap, to the .tkn or the\n"
    "                        .kids. --games and --positions then count the games and\n"
    "                        positions written, so the run plays more than that\n"
    "    --endgames          start every game from a random sparse position with a\n"
    "                        material lead, and keep only games won by capturing the King.\n"
    "                        Adjudication is off: no resignation, and a capped game is\n"
    "                        dropped like any other non-royal ending\n"
    "    --endgames-max-pieces N   fewer than N pieces in all, Kings included (default 20)\n"
    "    --endgames-imbalance P    the side ahead leads by at least P pawns and holds a\n"
    "                        ranging piece (default 50). A draw that misses is redrawn;\n"
    "                        101 misses in a row stops the run as unsatisfiable\n"
    "    --children FILE     also write what the search saw at every ply to a .kids side\n"
    "                        file paired with --out: the visited root children with their\n"
    "                        visits and values, or at --sims 0 every scored child. It is\n"
    "                        what policy training learns from\n"
    "    --seed S            base seed. Default: a fresh random one every run, printed at\n"
    "                        start and stored in the file header, so sessions never\n"
    "                        replay each other's games and any run can still be repeated\n"
    "  elo     rate the network against a baseline anchored at 0. Colours alternate and\n"
    "          regicide is forced for both sides. No root noise, and --temperature\n"
    "          defaults to 0 (the most visited move) -- or to 0.3 with --out, since the\n"
    "          games are then kept as training data. --sims, --leaves, --parallel-games\n"
    "          and --cpuct as for play\n"
    "    --baseline FILE     opponent graph, default baseline_<precision>.onnx, which\n"
    "                        export_onnx.py writes from baseline.pt. The bot's own graph\n"
    "                        plays the net against itself, e.g. at different --sims\n"
    "    --baseline random   the uniform mover instead: an absolute anchor, but it stops\n"
    "                        discriminating once the bot beats it every game\n"
    "    --baseline-sims N   the opponent's simulations per move (default: --sims)\n"
    "    --sprt E0,E1        sequential test, on by default at 0,50: stop as soon as the\n"
    "                        games show the bot is at least E1 better than the baseline\n"
    "                        (H1) or at most E0 (H0), in logistic Elo -- the rating a score\n"
    "                        implies. Draws are handled as a nuisance parameter\n"
    "    --alpha A, --beta B the test's error rates, default 0.05 each\n"
    "    --games N           with SPRT, the most games before stopping undecided\n"
    "                        (default 1000); with --no-sprt, the exact number to play\n"
    "    --no-sprt           play exactly --games games\n"
    "    --out FILE          also keep every position, in the same .tkn as self-play. Each\n"
    "                        ply's value is the rated net's; the opponent's moves are\n"
    "                        flagged (32 in the ply flags)\n"
    "  variance   the ceiling on what any value head could learn from this generator,\n"
    "          with the one-ply search. Snapshots positions along a game and replays each\n"
    "          many times, so the outcome variance splits into the part the position\n"
    "          decides and the part decided by luck afterwards.\n"
    "    --branch-every N    snapshot every N plies, from a random phase per game so\n"
    "                        every ply is equally likely (default 100, 0 disables)\n"
    "    --branches K        playouts from each snapshot (default 8, minimum 2)\n"
    "    --max-points N      at most N snapshots per game (0 = no limit)\n"
    "    --csv FILE          one row per branch point\n"
    "  safetytest  check the King-safety rule against tky.c's slow detector; no GPU\n"
    "  selfcheck  prove the setup: the CUDA provider loads, the start position's value and\n"
    "          policy match what export_onnx.py recorded in net_manifest.json, and the\n"
    "          probe reports this GPU's throughput. Exit 0 only if all of that holds.\n"
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
    "  --progress-every S    a status line every S seconds (default 30, 0 off)\n"
    "  --stats               add GPU utilisation to the status lines, sampled from NVML\n"
    "                        at 10 Hz if libnvidia-ml is present\n"
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

/* King safety without a network: hangs_king against tky.c's slow detector on every legal
 * move of positions from random games, the Great General sweep, and the sampling
 * distribution of sample_safe against the softmax restricted to safe moves. */
static int mode_safetytest(void)
{
    Pos *p = malloc(sizeof(Pos)), *c = malloc(sizeof(Pos));
    Scratch s;
    scratch_init(&s);
    Gen ref;
    gen_init(&ref);
    Rng rng;
    rng_seed(&rng, 12345);
    long positions = 0, moves = 0, hanging = 0, mismatches = 0;
    double t_check = 0.0;
    int fail = 0;

    /* 1. exactness, over uniform-random games with forced regicide */
    for (int game = 0; game < 40; game++) {
        pos_init(p);
        for (int ply = 0; ply < 600; ply++) {
            gen_all(p, &s.g);
            if (s.g.n == 0) break;
            king_ctx_set(&s.g, p);
            uint32_t forced = find_king_capture(&s.g);
            if (ply % 7 == 3 && !forced) {
                positions++;
                int n = s.g.n;
                uint32_t *mv = malloc((size_t)n * 4);
                memcpy(mv, s.g.buf, (size_t)n * 4);
                for (int i = 0; i < n; i++) {
                    double t0 = now_s();
                    int fast = hangs_king(&s, p, mv[i]);
                    t_check += now_s() - t0;
                    memcpy(c, p, sizeof(Pos));
                    do_move(c, SQ2P[UNPACK_O(mv[i])], UNPACK_M(mv[i]));
                    int slow;
                    if (c->royals[p->turn] < p->royals[p->turn]) slow = 1;
                    else {
                        gen_all(c, &ref);
                        int kp = c->royal_sq[p->turn];
                        slow = kp >= 0 && find_king_capture_slow(&ref, P2SQ[kp]) != 0;
                    }
                    moves++;
                    hanging += slow;
                    if (fast != slow) {
                        if (mismatches < 5)
                            fprintf(stderr, "  MISMATCH game %d ply %d move %u: fast %d slow %d\n",
                                    game, ply, mv[i], fast, slow);
                        mismatches++;
                    }
                }
                free(mv);
            }
            uint32_t pick = forced ? forced : s.g.buf[rng_below(&rng, (uint32_t)s.g.n)];
            int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
            do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
            if (p->royals[0] < rb[0] || p->royals[1] < rb[1]) break;
            (void)mover;
        }
    }
    printf("1. exactness: %ld positions, %ld moves, %ld of them hang the King (%.2f%%), "
           "%ld mismatches -- %s\n", positions, moves, hanging, 100.0 * hanging / moves,
           mismatches, mismatches ? "FAIL" : "ok");
    printf("   %.1f us per check\n", 1e6 * t_check / moves);
    fail |= mismatches != 0 || hanging == 0;

    /* 2. the Great General sweep: black 3,17 -> 34,17 next to the white King on 35,18 */
    pos_init(p);
    gen_all(p, &s.g);
    uint32_t sweep = 0;
    for (int i = 0; i < s.g.n; i++) {
        uint32_t v = s.g.buf[i];
        if (UNPACK_O(v) == 3 * 36 + 17 && UNPACK_M(v) == 9 * 1296 + 34 * 36 + 17) sweep = v;
    }
    if (!sweep) { printf("2. sweep: the move is not in the opening move list -- FAIL\n"); fail = 1; }
    else {
        do_move(p, SQ2P[UNPACK_O(sweep)], UNPACK_M(sweep));
        gen_all(p, &s.g);
        int n = s.g.n, safe = 0, safe_takes = 0;
        for (int i = 0; i < n; i++) {
            uint32_t v = s.g.buf[i];
            if (!hangs_king(&s, p, v)) {
                safe++;
                int mvv = UNPACK_M(v);
                int emptied = (mvv % 1296) == 34 * 36 + 17;
                safe_takes += emptied;
                if (safe <= 12)
                    printf("   safe: %-24s %d,%d -> %d,%d special %d%s\n",
                           PIECE_NAME[C_IDX(p->code[SQ2P[UNPACK_O(v)]])],
                           UNPACK_O(v) / 36, UNPACK_O(v) % 36, (mvv % 1296) / 36,
                           (mvv % 1296) % 36, mvv / 1296, emptied ? "  (takes the General)" : "");
            }
        }
        printf("2. sweep: white has %d legal moves, %d safe, %d of those take the General "
               "-- %s\n", n, safe, safe_takes, safe > 0 && safe < n / 10 ? "ok" : "FAIL");
        fail |= !(safe > 0 && safe < n / 10);

        /* 3. sample_safe draws from the softmax restricted to the safe moves */
        scratch_reserve(&s, n);
        float *v0 = malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) v0[i] = (float)((rng_next(&rng) >> 11) * (1.0 / 9007199254740992.0) * 0.6 - 0.3);
        char *is_safe = malloc((size_t)n);
        double z = 0.0, T = 0.3;
        for (int i = 0; i < n; i++) {
            is_safe[i] = !hangs_king(&s, p, s.g.buf[i]);
            if (is_safe[i]) z += exp(-v0[i] / T);
        }
        long *hits = calloc((size_t)n, sizeof(long));
        int draws = 20000;
        uint64_t c0 = s.checks;
        for (int d = 0; d < draws; d++) {
            for (int i = 0; i < n; i++) { s.perm[i] = i; s.values[i] = v0[i]; }
            int k = sample_safe(&s, p, n, T, &rng);
            if (k < 0) { fail = 1; break; }
            hits[s.perm[k]]++;
        }
        double worst = 0.0, unsafe = 0;
        for (int i = 0; i < n; i++) {
            double want = is_safe[i] ? exp(-v0[i] / T) / z : 0.0;
            double got = (double)hits[i] / draws;
            if (!is_safe[i]) unsafe += hits[i];
            double sd = sqrt(want * (1 - want) / draws) + 1e-12;
            double dev = fabs(got - want) / sd;
            if (want > 0 && dev > worst) worst = dev;
        }
        printf("3. sampling: %d draws over %d safe of %d moves, %.0f unsafe picks, largest "
               "deviation %.1f sd, %.1f checks per draw -- %s\n",
               draws, safe, n, unsafe, worst, (double)(s.checks - c0) / draws,
               unsafe == 0 && worst < 4.5 ? "ok" : "FAIL");
        fail |= !(unsafe == 0 && worst < 4.5);
        free(v0); free(is_safe); free(hits);
    }
    free(ref.buf); free(ref.stamp); free(ref.kline);
    scratch_free(&s);
    free(p); free(c);
    return fail;
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

/* ---- games: self-play and elo share one loop ------------------------------------ */
/* games_par games run in lockstep. Each round, every game whose mover searches walks its
 * tree for up to `leaves` leaves; the leaves of all games go to the network in one call
 * per net; then every game whose search is done plays its move. Moves that need no search
 * -- a forced King capture, the one-ply search, the uniform mover -- are played as soon as
 * they come up. Games end at different times and their slots start new ones, so a file's
 * games come out in the order they finish.
 *
 * A player picks one side's moves: a network with its search (sims > 0) or the one-ply
 * search (sims 0), or the uniform mover (net NULL). Self-play has one player and one tree
 * per game, shared by both sides. elo has the rated net and its opponent, each with its
 * own tree per game, and both trees follow every move so each stays rooted at the game. */
typedef struct { Net *net; int sims; double noise; Batch batch; } Player;

typedef struct {
    int       on, searching, n_legal;
    uint64_t  gi, seed;
    Rng       rng;
    Pos       pos;
    RepTab   *rep;
    PlyRecNN *ply;
    int       nply, no_prog, lead_run, bot_side;
    Tree      tree[2];
    Bytes     kids;
    double    t0;
} Slot;

typedef struct {
    Cfg      *cfg;
    int       elo;
    Player    pl[2];
    Slot     *sl;
    Scratch   sc;
    Gen       wg;               /* move lists for tree walks */
    Pos       base;
    Sink      sink;
    Kids      kids;
    int       kids_on, mat_div;
    uint64_t  games, target_positions, started, games_done, plies, kept_games, kept_plies;
    uint64_t  wins[3], evals[2], moves[2];
    int       games_set, first_game, stop, eg_failed;
    EgStats   eg;
    double    t_start, t_nn, t_status;
    uint64_t  calls, call_boards;
    /* elo */
    const char *opp_name;
    const Sprt *sprt;
    Rec       as_black, as_white;
    EloFit    fit;
    double   *marg, llr;
    int       verdict;
} Run;

static void game_start(Run *R, Slot *s)
{
    const Cfg *c = R->cfg;
    s->gi = R->started++;
    s->seed = c->seed_base ^ (((uint64_t)R->first_game + s->gi) * 0x9E3779B97F4A7C15ull);
    rng_seed(&s->rng, s->seed);
    if (!c->endgames) memcpy(&s->pos, &R->base, sizeof(Pos));
    else if (eg_generate(&s->pos, c, &s->rng, &R->sc.g, &R->eg)) {
        fprintf(stderr,
            "--endgames: %d random positions in a row failed the requirements (so far %llu "
            "short of a %d-pawn lead held by a side with a ranging piece, %llu with a King "
            "capturable at once), so they look unsatisfiable with fewer than %d pieces. "
            "Raise --endgames-max-pieces or lower --endgames-imbalance.\n",
            EG_MAX_FAILS + 1, (unsigned long long)R->eg.short_lead, c->eg_imbalance,
            (unsigned long long)R->eg.king_hangs, c->eg_max_pieces);
        R->eg_failed = R->stop = 1;
        return;
    }
    s->rep->epoch++;
    rep_bump(s->rep, s->pos.hash);
    s->nply = s->no_prog = s->lead_run = s->searching = 0;
    s->bot_side = (s->gi & 1) ? SIDE_WHITE : SIDE_BLACK;    /* elo: alternate who moves first */
    tree_reset(&s->tree[0], &s->pos);
    tree_reset(&s->tree[1], &s->pos);
    s->kids.len = 0;
    s->t0 = now_s();
    s->on = 1;
}

static int may_start(const Run *R)
{
    if (R->stop) return 0;
    if (R->elo) return R->started < R->games;
    uint64_t in_play = R->started - R->games_done;
    if (R->target_positions)
        return R->kept_plies < R->target_positions && !(R->games_set && R->kept_games + in_play >= R->games);
    return R->kept_games + in_play < R->games;
}

static void game_end(Run *R, Slot *s, int result, int term)
{
    const Cfg *c = R->cfg;
    static const char *tn[] = { "royal", "stalemate", "repetition", "no-progress", "ply-cap",
                                "resignation" };
    s->on = 0;
    R->games_done++;
    R->plies += (uint64_t)s->nply;
    R->wins[result]++;
    if (!R->elo) {
        int keep = c->endgames ? term == TERM_ROYAL : !(c->drop_capped && term == TERM_PLYCAP);
        if (keep) {
            R->kept_games++;
            R->kept_plies += (uint64_t)s->nply;
            if (R->kids_on) kids_game(&R->kids, &s->kids, s->ply, s->nply, s->seed);
            if (c->out_path)
                tkn_put_game(&R->sink, s->ply, s->nply, result, term,
                             soft_label(c, &s->pos, result, term), s->seed);
        }
        if (!c->verbose) return;
        char head[96], eta[48] = "";
        if (R->target_positions)
            snprintf(head, sizeof(head), "Game %llu, Positions %llu/%llu",
                     (unsigned long long)R->games_done, (unsigned long long)R->kept_plies,
                     (unsigned long long)R->target_positions);
        else
            snprintf(head, sizeof(head), "Game %llu, Kept %llu/%llu",
                     (unsigned long long)R->games_done, (unsigned long long)R->kept_games,
                     (unsigned long long)R->games);
        double el = now_s() - R->t_start, remain = 0.0;
        if (R->target_positions && R->kept_plies && R->kept_plies < R->target_positions)
            remain = el / (double)R->kept_plies * (double)(R->target_positions - R->kept_plies);
        else if (!R->target_positions && R->kept_games && R->kept_games < R->games)
            remain = el / (double)R->kept_games * (double)(R->games - R->kept_games);
        if (remain > 0.0) {
            char d[32];
            fmt_dur(remain, d, sizeof(d));
            snprintf(eta, sizeof(eta), "  ETA %s", d);
        }
        fprintf(stderr, "%s (index %llu): %d plies, %s by %s, %.1fs%s%s\n",
                head, (unsigned long long)(R->first_game + s->gi), s->nply,
                result == RES_BLACK ? "black" : result == RES_WHITE ? "white" : "draw",
                tn[term], now_s() - s->t0, eta, keep ? "" : "  (dropped)");
        return;
    }
    if (c->out_path)
        tkn_put_game(&R->sink, s->ply, s->nply, result, term, soft_label(c, &s->pos, result, term),
                     s->seed);
    Rec *r = (s->bot_side == SIDE_BLACK) ? &R->as_black : &R->as_white;
    static const char *why[] = { "king", "stalemate", "repetition", "no progress", "ply cap",
                                 "adjudication" };
    const char *tag;
    if (result == RES_DRAW)          { r->d++; tag = "draw"; }
    else if (result == s->bot_side)  { r->w++; tag = "win"; }
    else                             { r->l++; tag = "loss"; }
    elo_fit(&R->as_black, &R->as_white, R->marg, &R->fit);
    long wins = R->as_black.w + R->as_white.w, losses = R->as_black.l + R->as_white.l,
         draws = R->as_black.d + R->as_white.d;
    double n = (double)(wins + losses + draws);
    char sp[48] = "";
    if (R->sprt->on) {
        R->llr = sprt_llr(wins, draws, losses, R->sprt->elo0, R->sprt->elo1);
        R->verdict = sprt_decide(R->sprt, R->llr);
        snprintf(sp, sizeof(sp), " | LLR %+.2f [%.2f, %.2f]", R->llr, R->sprt->lower, R->sprt->upper);
    }
    fprintf(stderr,
        "game %-4llu bot=%-5s %-4s by %-12s in %4d plies (%5.1fs) | %4ldW %4ldL %4ldD  score %.3f | "
        "elo %+6.0f [%+.0f, %+.0f]%s | first-move %+.0f%s | %.1f min elapsed\n",
        (unsigned long long)(R->first_game + s->gi), s->bot_side == SIDE_BLACK ? "black" : "white",
        tag, why[term], s->nply, now_s() - s->t0, wins, losses, draws, (wins + 0.5 * draws) / n,
        R->fit.elo, R->fit.lo, R->fit.hi, R->fit.saturated ? " (at the grid edge)" : "",
        R->fit.advantage, sp, (now_s() - R->t_start) / 60.0);
    if (R->verdict) R->stop = 1;
}

/* Plays mv in slot s. value is for the side to move after it; scored, the network
 * evaluations the move cost. */
static void game_move(Run *R, Slot *s, uint32_t mv, float value, int scored, int flags)
{
    const Cfg *c = R->cfg;
    Pos *p = &s->pos;
    int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] }, before = p->npieces;
    int pi = R->elo && mover != s->bot_side;
    R->evals[pi] += (uint64_t)scored;
    R->moves[pi]++;
    int promoted = do_move(p, SQ2P[UNPACK_O(mv)], UNPACK_M(mv));
    int captured = before - p->npieces;
    if (pi && c->out_path && !(flags & PF_FORCED_KING)) {   /* the rated net's view, always */
        scratch_reserve(&R->sc, 1);
        encode_stm(p, R->sc.boards);
        net_eval(R->pl[0].net, R->sc.boards, 1, &value);
    }
    tkn_fill(&s->ply[s->nply], p, mv, value, s->n_legal, mover,
             flags | (captured > 0 ? PF_CAPTURE : 0) | (promoted ? PF_PROMOTION : 0)
             | (pi ? PF_OPPONENT : 0), scored, R->mat_div);
    s->nply++;
    s->no_prog = (captured > 0 || promoted) ? 0 : s->no_prog + 1;
    tree_advance(&s->tree[0], mv, p);
    if (R->elo) tree_advance(&s->tree[1], mv, p);

    int result = -1, term = 0, adj;
    if (p->royals[1 - mover] < rb[1 - mover])      { result = mover;     term = TERM_ROYAL; }
    else if (p->royals[mover] < rb[mover])         { result = 1 - mover; term = TERM_ROYAL; }
    else {
        int seen = rep_bump(s->rep, p->hash);
        if (c->rep_limit > 0 && seen >= c->rep_limit)                       { result = RES_DRAW; term = TERM_REPETITION; }
        else if (c->no_progress_limit > 0 && s->no_prog >= c->no_progress_limit) { result = RES_DRAW; term = TERM_NOPROGRESS; }
        else if ((adj = resign_check(c, p, &s->lead_run)) >= 0)              { result = adj;      term = TERM_RESIGN; }
    }
    if (result >= 0) game_end(R, s, result, term);
}

/* The start of a move: the game may already be over, the move may need no search, or a
 * search begins. */
static void game_decide(Run *R, Slot *s)
{
    const Cfg *c = R->cfg;
    Scratch *sc = &R->sc;
    Pos *p = &s->pos;
    gen_all(p, &sc->g);
    s->n_legal = sc->g.n;
    if (!sc->g.n) { game_end(R, s, c->stalemate_loses ? 1 - p->turn : RES_DRAW, TERM_STALEMATE); return; }
    if (s->nply >= c->max_plies) { game_end(R, s, cap_result(c, p), TERM_PLYCAP); return; }
    int pi = R->elo && p->turn != s->bot_side;
    Player *P = &R->pl[pi];
    uint32_t forced = 0;
    if (c->regicide) { king_ctx_set(&sc->g, p); forced = find_king_capture(&sc->g); }
    if (forced) {                       /* exact: the side then to move has lost its King */
        float lost = -1.0f;
        int one = 1;
        if (R->kids_on) kids_ply(&s->kids, &forced, NULL, &lost, P->sims ? &one : NULL, 1);
        game_move(R, s, forced, lost, 0, PF_FORCED_KING);
    } else if (!P->net) {
        game_move(R, s, choose_random(c->king_safety, p, sc, &s->rng), 0.0f, 0, 0);
    } else if (!P->sims) {
        float v;
        int scored, changed;
        double t0 = sc->t_nn + sc->t_pol;
        uint32_t mv = choose(P->net, c->king_safety, p, sc, &s->rng, temp_at(c, s->nply), &v,
                             &scored, &changed);
        R->t_nn += sc->t_nn + sc->t_pol - t0;
        if (R->kids_on) kids_ply(&s->kids, sc->g.buf, sc->perm, sc->values, NULL, scored);
        game_move(R, s, mv, v, scored, changed ? PF_KING_SAFETY : 0);
    } else {
        tree_begin(&s->tree[pi], P->sims, P->noise, c->noise_alpha, c->contempt, c->contempt_flat);
        s->searching = 1;
    }
}

static void game_searched(Run *R, Slot *s)
{
    const Cfg *c = R->cfg;
    int pi = R->elo && s->pos.turn != s->bot_side, changed;
    Tree *t = &s->tree[pi];
    float q;
    int e = tree_pick(t, &R->sc, &s->rng, temp_at(c, s->nply), c->king_safety, &changed, &q);
    uint32_t mv = t->ed[t->nd[t->root].e0 + e].mv;
    if (R->kids_on) tree_kids(t, &s->kids, &R->sc);
    s->searching = 0;
    game_move(R, s, mv, q, t->evals, changed ? PF_KING_SAFETY : 0);
}

static void run_status(Run *R, GpuMon *mon, int have_nvml)
{
    double el = now_s() - R->t_start;
    uint64_t ev = R->evals[0] + R->evals[1], in_play = R->started - R->games_done;
    fprintf(stderr, "  [status] %llu games done, %llu in play, %llu plies | %.0f evals/s, network "
                    "%.0f%% of the wall clock",
            (unsigned long long)R->games_done, (unsigned long long)in_play,
            (unsigned long long)R->plies, el > 0 ? ev / el : 0.0, el > 0 ? 100.0 * R->t_nn / el : 0.0);
    if (R->calls) fprintf(stderr, ", %.1f boards per call", (double)R->call_boards / R->calls);
    if (have_nvml) {
        double gu, gm, used, tot;
        gpumon_take(mon, &gu, &gm, &used, &tot);
        if (gu >= 0) fprintf(stderr, " | gpu busy %.0f%%, vram %.2f/%.2f GB", gu, used, tot);
    }
    fputc('\n', stderr);
}

static int mode_games(Net *net, Net *opp, const char *opp_name, Cfg *cfg, int elo,
                      uint64_t games, int games_set, uint64_t target_positions, int first_game,
                      const Sprt *sprt, const char *model, int device_id)
{
    Run *R = calloc(1, sizeof(Run));
    R->cfg = cfg; R->elo = elo; R->games = games; R->games_set = games_set;
    R->target_positions = target_positions; R->first_game = first_game;
    R->opp_name = opp_name; R->sprt = sprt;
    R->pl[0] = (Player){ net, cfg->sims, elo ? 0.0 : cfg->noise_frac, { 0 } };
    R->pl[1] = (Player){ opp, cfg->opp_sims, 0.0, { 0 } };
    R->mat_div = VALUE_FP >> cfg->mat_shift;
    if (R->mat_div < 1) R->mat_div = 1;
    pos_init(&R->base);
    scratch_init(&R->sc);
    gen_init(&R->wg);
    if (elo) R->marg = malloc(sizeof(double) * ELO_N);
    int G = cfg->games_par;
    R->sl = calloc((size_t)G, sizeof(Slot));
    for (int i = 0; i < G; i++) {
        Slot *s = &R->sl[i];
        s->rep = calloc(1, sizeof(RepTab));
        s->ply = malloc(sizeof(PlyRecNN) * (size_t)(cfg->max_plies + 2));
        tree_init(&s->tree[0], cfg->leaves, &s->rng);
        tree_init(&s->tree[1], cfg->leaves, &s->rng);
    }

    R->sink = (Sink){ -1, NULL, 0, 1u << 22 };
    if (cfg->out_path && tkn_open(&R->sink, cfg, net, model)) return 1;
    if (cfg->children_path) {
        if (!cfg->out_path) { fprintf(stderr, "--children needs --out: its games pair with the .tkn's\n"); return 1; }
        if (kids_open(&R->kids, cfg->children_path, cfg->sims ? 2 : 1)) return 1;
        R->kids_on = 1;
    }
    GpuMon mon;
    memset(&mon, 0, sizeof(mon));
    int have_nvml = cfg->stats && gpumon_start(&mon, device_id);

    const char *how = cfg->sims ? "search" : "the one-ply search";
    if (elo) {
        fprintf(stderr, "bot (%s", how);
        if (cfg->sims) fprintf(stderr, ", %d sims", cfg->sims);
        fprintf(stderr, ") vs %s", opp_name);
        if (opp) fprintf(stderr, " (%s%s)", cfg->opp_sims ? "search, " : "the one-ply search",
                         cfg->opp_sims ? "" : "");
        if (opp && cfg->opp_sims) fprintf(stderr, " %d sims", cfg->opp_sims);
        fprintf(stderr, "; regicide forced, king safety %s, temperature %.3g, contempt %g\n",
                cfg->king_safety ? "on (both sides)" : "off", cfg->temperature, cfg->contempt);
        if (sprt->on)
            fprintf(stderr, "SPRT: H0 elo <= %+g, H1 elo >= %+g (logistic), alpha %g beta %g; stops "
                            "when LLR leaves [%.2f, %.2f], or after %llu games\n",
                    sprt->elo0, sprt->elo1, sprt->alpha, sprt->beta, sprt->lower, sprt->upper,
                    (unsigned long long)games);
        else
            fprintf(stderr, "fixed length: %llu games (--no-sprt)\n", (unsigned long long)games);
        if (opp && cfg->temperature == 0.0 && cfg->opening_plies == 0)
            fprintf(stderr,
                "WARNING: both players are deterministic at temperature 0, so every game with a\n"
                "         given colour is the same game. Pass --opening-plies 8\n"
                "         --opening-temperature 1.0 or so to make the games actually differ.\n");
    } else {
        fprintf(stderr, "self-play with %s", how);
        if (cfg->sims)
            fprintf(stderr, ": %d sims a move, %d leaves per tree per call, %d games in lockstep, "
                            "c_puct %g, root noise %g at alpha %g/moves, contempt %g%s",
                    cfg->sims, cfg->leaves, G, cfg->cpuct, cfg->noise_frac, cfg->noise_alpha,
                    cfg->contempt, cfg->contempt_flat ? " (flat)" : "");
        fputc('\n', stderr);
        if (cfg->endgames)
            fprintf(stderr, "endgames: random starts with fewer than %d pieces, the side ahead by %d+ "
                            "pawns holding a ranging piece; no adjudication; only royal captures kept\n",
                    cfg->eg_max_pieces, cfg->eg_imbalance);
        if (cfg->soft_scale > 0.0)
            fprintf(stderr, "soft labels: a game cut off by the ply cap or by repetition is labelled "
                            "%g * tanh(material / %g pawns)\n", cfg->soft_scale, SOFT_PAWNS);
    }
    if (cfg->resign_pawns > 0)
        fprintf(stderr, "adjudication: a %d-pawn lead held for %d plies wins\n",
                cfg->resign_pawns, cfg->resign_plies);

    R->t_start = R->t_status = now_s();
    for (;;) {
        for (int i = 0; i < G; i++)
            if (!R->sl[i].on && may_start(R)) game_start(R, &R->sl[i]);
        if (R->stop && R->elo) break;           /* SPRT decided: games in play are not counted */
        int active = 0;
        for (int i = 0; i < G; i++) active += R->sl[i].on;
        if (!active) break;

        for (int i = 0; i < G; i++) {
            Slot *s = &R->sl[i];
            if (!s->on) continue;
            if (!s->searching) game_decide(R, s);
            if (!s->on || !s->searching) continue;
            int pi = R->elo && s->pos.turn != s->bot_side;
            Tree *t = &s->tree[pi];
            tree_gather(t, cfg, s->rep, &R->wg, &R->pl[pi].batch);
            if (!t->todo && !t->np) game_searched(R, s);
        }
        for (int k = 0; k < 2; k++) {
            Batch *b = &R->pl[k].batch;
            if (!b->nb) continue;
            R->calls++;
            R->call_boards += (uint64_t)b->nb;
            R->t_nn += batch_run(R->pl[k].net, b);
        }
        for (int i = 0; i < G; i++) {
            Slot *s = &R->sl[i];
            if (!s->on || !s->searching) continue;
            int pi = R->elo && s->pos.turn != s->bot_side;
            if (!s->tree[pi].todo && !s->tree[pi].np) game_searched(R, s);
        }
        if (cfg->progress && now_s() - R->t_status >= cfg->progress) {
            run_status(R, &mon, have_nvml);
            R->t_status = now_s();
        }
    }

    double dt = now_s() - R->t_start;
    uint64_t ev = R->evals[0] + R->evals[1], in_play = R->started - R->games_done;
    if (elo) {
        long wins = R->as_black.w + R->as_white.w, losses = R->as_black.l + R->as_white.l,
             draws = R->as_black.d + R->as_white.d;
        double n = (double)(wins + losses + draws);
        if (in_play)
            fprintf(stderr, "\n%llu games still in play when the test stopped are not counted\n",
                    (unsigned long long)in_play);
        fprintf(stderr,
            "\n%ld games, %llu plies (%.0f/game), %.1f min\n"
            "  as black : %ldW %ldL %ldD\n"
            "  as white : %ldW %ldL %ldD\n"
            "  overall  : %ldW %ldL %ldD   score %.4f\n"
            "  Elo %+.0f, 95%% credible interval [%+.0f, %+.0f]   (%s = 0)\n"
            "  draw rating %.0f, first-move advantage %+.0f\n",
            (long)n, (unsigned long long)R->plies, n > 0 ? R->plies / n : 0.0, dt / 60.0,
            R->as_black.w, R->as_black.l, R->as_black.d, R->as_white.w, R->as_white.l, R->as_white.d,
            wins, losses, draws, n > 0 ? (wins + 0.5 * draws) / n : 0.0,
            R->fit.elo, R->fit.lo, R->fit.hi, opp_name, R->fit.draw_elo, R->fit.advantage);
        if (R->fit.saturated)
            fprintf(stderr, "  the posterior is against the grid edge: the bot did not lose enough "
                            "games to bound its rating from that side\n");
        if (sprt->on) {
            if (R->verdict > 0)
                fprintf(stderr, "  SPRT: H1 accepted, LLR %+.2f >= %.2f -- the bot is better than the "
                                "baseline by at least ~%+g logistic Elo (error rate %g)\n",
                        R->llr, sprt->upper, sprt->elo1, sprt->alpha);
            else if (R->verdict < 0)
                fprintf(stderr, "  SPRT: H0 accepted, LLR %+.2f <= %.2f -- the bot is not %+g logistic "
                                "Elo better than the baseline (error rate %g)\n",
                        R->llr, sprt->lower, sprt->elo1, sprt->beta);
            else
                fprintf(stderr, "  SPRT: undecided after %ld games, LLR %+.2f inside [%.2f, %.2f]: "
                                "the --games limit came first. The evidence does not yet separate "
                                "%+g from %+g; read the Elo interval above, or raise --games\n",
                        (long)n, R->llr, sprt->lower, sprt->upper, sprt->elo0, sprt->elo1);
        }
        if (R->moves[0] && R->moves[1] && opp)
            printf("  network evaluations per move: rated side %.1f, opponent %.1f\n",
                   (double)R->evals[0] / R->moves[0], (double)R->evals[1] / R->moves[1]);
    } else {
        fprintf(stderr,
            "games=%llu plies=%llu (%.1f/game) evals=%llu (%.1f a move)\n"
            "  %.1fs total, %.1fs in the network (%.0f%%), %.0f evals/s",
            (unsigned long long)R->games_done, (unsigned long long)R->plies,
            R->games_done ? (double)R->plies / R->games_done : 0.0, (unsigned long long)ev,
            R->plies ? (double)ev / R->plies : 0.0, dt, R->t_nn, dt > 0 ? 100.0 * R->t_nn / dt : 0.0,
            dt > 0 ? ev / dt : 0.0);
        if (R->calls) fprintf(stderr, ", %.1f boards per call", (double)R->call_boards / R->calls);
        fprintf(stderr, "\n  black=%llu white=%llu draw=%llu\n",
            (unsigned long long)R->wins[RES_BLACK], (unsigned long long)R->wins[RES_WHITE],
            (unsigned long long)R->wins[RES_DRAW]);
        if (cfg->endgames)
            fprintf(stderr, "  endgames: kept %llu royal captures of %llu games (%.1f%%), %llu plies; "
                            "%llu random positions drawn, %.1f%% accepted (%llu short of the lead, "
                            "%llu with a King capturable at once)\n",
                    (unsigned long long)R->kept_games, (unsigned long long)R->games_done,
                    R->games_done ? 100.0 * R->kept_games / R->games_done : 0.0,
                    (unsigned long long)R->kept_plies,
                    (unsigned long long)(R->eg.made + R->eg.short_lead + R->eg.king_hangs),
                    R->eg.made ? 100.0 * R->eg.made / (double)(R->eg.made + R->eg.short_lead + R->eg.king_hangs) : 0.0,
                    (unsigned long long)R->eg.short_lead, (unsigned long long)R->eg.king_hangs);
        else if (cfg->drop_capped)
            fprintf(stderr, "  dropped %llu games that reached the ply cap (%.1f%%, %llu plies); "
                            "wrote %llu games, %llu plies\n",
                    (unsigned long long)(R->games_done - R->kept_games),
                    R->games_done ? 100.0 * (R->games_done - R->kept_games) / R->games_done : 0.0,
                    (unsigned long long)(R->plies - R->kept_plies),
                    (unsigned long long)R->kept_games, (unsigned long long)R->kept_plies);
    }
    if (cfg->king_safety)
        fprintf(stderr, "  king safety changed the pick on %llu plies (%.2f%%); %llu plies had no "
                        "safe move at all\n",
                (unsigned long long)R->sc.overridden,
                R->plies ? 100.0 * R->sc.overridden / R->plies : 0.0,
                (unsigned long long)R->sc.hopeless);
    if (have_nvml) run_status(R, &mon, have_nvml);
    gpumon_stop(&mon);
    if (cfg->out_path) {
        tkn_close(&R->sink, elo ? R->games_done : R->kept_games, elo ? R->plies : R->kept_plies);
        if (elo)
            fprintf(stderr, "  wrote %llu games, %llu positions to %s\n",
                    (unsigned long long)R->games_done, (unsigned long long)R->plies, cfg->out_path);
    }
    if (R->kids_on) kids_close(&R->kids);
    for (int i = 0; i < G; i++) {
        Slot *s = &R->sl[i];
        free(s->rep); free(s->ply); free(s->kids.b);
        tree_free(&s->tree[0]); tree_free(&s->tree[1]);
    }
    for (int k = 0; k < 2; k++) {
        Batch *b = &R->pl[k].batch;
        free(b->boards); free(b->rows); free(b->val); free(b->lgt); free(b->tr); free(b->leaf); free(b->roff);
    }
    free(R->sl); free(R->marg);
    free(R->wg.buf); free(R->wg.stamp); free(R->wg.kline);
    scratch_free(&R->sc);
    int rc = R->eg_failed;
    free(R);
    return rc;
}

/* ---- how much of the outcome does the position actually decide? ---------------------- */
/* The label a value head learns is the game outcome z from position s. Its variance splits:
 *
 *     Var(z) = Var(E[z|s])  +  E[Var(z|s)]
 *              signal          noise from everything that happens after s
 *
 * and no value head can ever beat R2 = Var(E[z|s]) / Var(z). That ceiling is a property of
 * the *generator*, not of the network: a hot policy decides games by coin flip after the
 * position, which drives E[Var(z|s)] up and the ceiling down.
 *
 * E[Var(z|s)] is measurable. Snapshot a position, play it out K times under the schedule,
 * and the spread of those K outcomes is Var(z|s) at that position. Averaging over many
 * snapshots gives the noise term; the spread of the per-position means gives the signal
 * term, once the sampling error in each mean is subtracted. Nothing has to be trained.
 *
 * It is expensive -- each branch point costs K full playouts -- which is the price of
 * measuring a ceiling instead of guessing at it.
 */
/* One move for the side to move: a forced King capture if there is one, otherwise
 * choose(). Leaves the move list in s->g, so the caller can see a stalemate as
 * s->g.n == 0. */
static uint32_t pick_move(Net *net, const Cfg *cfg, Pos *p,
                          Scratch *s, Rng *rng, int ply, float *val)
{
    gen_all(p, &s->g);
    if (s->g.n == 0) return 0;
    if (cfg->regicide) {
        king_ctx_set(&s->g, p);
        uint32_t forced = find_king_capture(&s->g);
        if (forced) { if (val) *val = 0.0f; return forced; }
    }
    return choose(net, cfg->king_safety, p, s, rng, temp_at(cfg, ply), val,
                  NULL, NULL);
}

/* Plays from `ply` to a finish, under the same rules as self-play, including the lead
 * count `run` the main line had reached. Returns RES_BLACK, RES_WHITE or RES_DRAW. */
static int playout(Net *net, const Cfg *cfg, Pos *p, RepTab *rep,
                   Scratch *s, Rng *rng, int ply, int run)
{
    for (;;) {
        uint32_t pick = pick_move(net, cfg, p, s, rng, ply, NULL);
        if (s->g.n == 0) return cfg->stalemate_loses ? (1 - p->turn) : RES_DRAW;
        if (ply >= cfg->max_plies) return cap_result(cfg, p);
        int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
        do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
        ply++;
        if (p->royals[1 - mover] < rb[1 - mover]) return mover;
        if (p->royals[mover] < rb[mover]) return 1 - mover;
        if (cfg->rep_limit > 0 && rep_bump(rep, p->hash) >= cfg->rep_limit) return RES_DRAW;
        int adj = resign_check(cfg, p, &run);
        if (adj >= 0) return adj;
    }
}

typedef struct { double zbar, s2, v0; int ply, k; } Branch;

static int mode_variance(Net *net, Cfg *cfg, uint64_t games,
                         int first_game, int branch_every, int branches, int max_points,
                         const char *csv_path)
{
    if (branches < 2) { fprintf(stderr, "--branches must be at least 2\n"); return 1; }
    Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos));
    Pos *snap = malloc(sizeof(Pos)), *run = malloc(sizeof(Pos));
    pos_init(base);
    RepTab *rep = calloc(1, sizeof(RepTab)), *rsnap = calloc(1, sizeof(RepTab));
    RepTab *rrun = calloc(1, sizeof(RepTab));
    Scratch sc;
    scratch_init(&sc);
    int32_t *one = malloc(NSQ * sizeof(int32_t));

    Branch *pts = NULL;
    int npts = 0, cappts = 0;
    double t0 = now_s();
    uint64_t playouts = 0, playout_plies = 0;

    FILE *csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { perror(csv_path); return 1; }
        fprintf(csv, "game,ply,branches,mean_outcome,var_outcome,net_value\n");
    }

    fprintf(stderr, "branching variance: %llu games, a branch point every %d plies, "
                    "%d playouts each\n", (unsigned long long)games, branch_every, branches);
    fprintf(stderr, "  schedule: %g for %d plies then %g; policy top %.1f%%; max plies %d; "
                    "king safety %s\n",
            cfg->opening_temperature, cfg->opening_plies, cfg->temperature,
            100.0 * POLICY_TOP, cfg->max_plies, cfg->king_safety ? "on" : "off");

    for (uint64_t gi = 0; gi < games; gi++) {
        uint64_t seed = cfg->seed_base ^ (((uint64_t)first_game + gi) * 0x9E3779B97F4A7C15ull);
        Rng rng; rng_seed(&rng, seed);
        memcpy(p, base, sizeof(Pos));
        rep->epoch++;
        rep_bump(rep, p->hash);
        int ply = 0, points_here = 0, lead_run = 0;
        /* a fixed stride would never sample a game's first N plies */
        int phase = branch_every > 0 ? (int)((seed >> 17) % (uint64_t)branch_every) : 0;

        for (;;) {
            float mv_val;
            uint32_t pick = pick_move(net, cfg, p, &sc, &rng, ply, &mv_val);
            if (sc.g.n == 0 || ply >= cfg->max_plies) break;

            if (branch_every > 0 && (ply + phase) % branch_every == 0
                && (max_points <= 0 || points_here < max_points)) {
                memcpy(snap, p, sizeof(Pos));
                memcpy(rsnap, rep, sizeof(RepTab));
                int side = p->turn;                       /* outcomes are from this side */
                encode_stm(p, one);
                float v0 = 0.0f;
                net_eval(net, one, 1, &v0);

                double sum = 0.0, sum2 = 0.0;
                for (int k = 0; k < branches; k++) {
                    memcpy(run, snap, sizeof(Pos));
                    memcpy(rrun, rsnap, sizeof(RepTab));
                    Rng brng;
                    rng_seed(&brng, seed ^ ((uint64_t)(ply + 1) * 0x9E3779B97F4A7C15ull)
                                         ^ ((uint64_t)(k + 1) * 0xBF58476D1CE4E5B9ull));
                    int r = playout(net, cfg, run, rrun, &sc, &brng, ply, lead_run);
                    double z = (r == RES_DRAW) ? 0.0 : (r == side ? 1.0 : -1.0);
                    sum += z; sum2 += z * z;
                    playouts++;
                }
                double mean = sum / branches;
                double var = (sum2 - branches * mean * mean) / (branches - 1);
                if (npts == cappts) {
                    cappts = cappts ? cappts * 2 : 1024;
                    pts = realloc(pts, (size_t)cappts * sizeof(Branch));
                }
                /* encode_stm encodes p for p->turn, so v0 is already the side-to-move's
                 * value -- the same frame the outcomes are scored in. No negation here,
                 * unlike self-play, which scores children from the opponent's side. */
                pts[npts++] = (Branch){ mean, var, (double)v0, ply, branches };
                points_here++;
                if (csv)
                    fprintf(csv, "%llu,%d,%d,%.6f,%.6f,%.6f\n",
                            (unsigned long long)(first_game + gi), ply, branches,
                            mean, var, (double)v0);
                memcpy(p, snap, sizeof(Pos));
                memcpy(rep, rsnap, sizeof(RepTab));
            }

            int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
            do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
            ply++;
            playout_plies++;
            if (p->royals[1 - mover] < rb[1 - mover] || p->royals[mover] < rb[mover]) break;
            if (cfg->rep_limit > 0 && rep_bump(rep, p->hash) >= cfg->rep_limit) break;
            if (resign_check(cfg, p, &lead_run) >= 0) break;
        }
        if (cfg->verbose)
            fprintf(stderr, "  game %llu: %d plies, %d branch points, %llu playouts so far "
                            "(%.1f min)\n",
                    (unsigned long long)(first_game + gi), ply, points_here,
                    (unsigned long long)playouts, (now_s() - t0) / 60.0);
    }

    if (npts < 2) { fprintf(stderr, "too few branch points (%d)\n", npts); return 1; }

    double mean_within = 0, mean_zbar = 0;
    for (int i = 0; i < npts; i++) { mean_within += pts[i].s2; mean_zbar += pts[i].zbar; }
    mean_within /= npts;
    mean_zbar /= npts;
    double var_zbar = 0;
    for (int i = 0; i < npts; i++)
        var_zbar += (pts[i].zbar - mean_zbar) * (pts[i].zbar - mean_zbar);
    var_zbar /= (npts - 1);
    /* Var(zbar) overstates the signal by the sampling error left in each mean. */
    double signal = var_zbar - mean_within / branches;
    if (signal < 0) signal = 0;
    double total = signal + mean_within;
    double ceiling = total > 0 ? signal / total : 0.0;

    /* how much of that ceiling the current net already reaches */
    double sv = 0, sz = 0, svv = 0, szz = 0, svz = 0;
    for (int i = 0; i < npts; i++) {
        sv += pts[i].v0; sz += pts[i].zbar;
        svv += pts[i].v0 * pts[i].v0; szz += pts[i].zbar * pts[i].zbar;
        svz += pts[i].v0 * pts[i].zbar;
    }
    double n = npts;
    double cov = svz / n - (sv / n) * (sz / n);
    double sdv = sqrt(svv / n - (sv / n) * (sv / n));
    double sdz = sqrt(szz / n - (sz / n) * (sz / n));
    double corr = (sdv > 0 && sdz > 0) ? cov / (sdv * sdz) : 0.0 / 0.0;

    printf("\n%d branch points, %d playouts each, %llu playouts total (%.1f min)\n",
           npts, branches, (unsigned long long)playouts, (now_s() - t0) / 60.0);
    printf("  within-position variance  E[Var(z|s)] = %.4f   (outcome decided after s)\n",
           mean_within);
    printf("  between-position variance Var(E[z|s]) = %.4f   (outcome decided by s)\n", signal);
    printf("  ACHIEVABLE R2 = %.4f -- the most any value head could reach on this corpus\n",
           ceiling);
    printf("  this net's corr(value, true value) = %+.4f over the branch points\n", corr);
    printf("  (true value is the mean of %d playouts, so it is itself noisy to about "
           "%.3f)\n", branches, sqrt(mean_within / branches));

    printf("\n  by ply       points   E[Var(z|s)]   Var(E[z|s])   achievable R2\n");
    int edges[] = { 0, 50, 100, 200, 400, 800, 1 << 30 };
    for (unsigned b = 0; b + 1 < sizeof(edges) / sizeof(edges[0]); b++) {
        double w = 0, m = 0;
        int c = 0;
        for (int i = 0; i < npts; i++)
            if (pts[i].ply >= edges[b] && pts[i].ply < edges[b + 1]) { w += pts[i].s2; m += pts[i].zbar; c++; }
        if (c < 2) continue;
        w /= c; m /= c;
        double vz = 0;
        for (int i = 0; i < npts; i++)
            if (pts[i].ply >= edges[b] && pts[i].ply < edges[b + 1])
                vz += (pts[i].zbar - m) * (pts[i].zbar - m);
        vz /= (c - 1);
        double sig = vz - w / branches;
        if (sig < 0) sig = 0;
        printf("  %4d-%-6d %8d   %11.4f   %11.4f   %11.4f\n",
               edges[b], edges[b + 1] > (1 << 29) ? 0 : edges[b + 1] - 1, c, w, sig,
               (sig + w) > 0 ? sig / (sig + w) : 0.0);
    }
    printf("\n  A schedule is better when ACHIEVABLE R2 goes up without the duplicate rate\n"
           "  going up with it. Compare schedules, not absolute numbers.\n");
    if (csv) { fclose(csv); printf("  per-branch rows in %s\n", csv_path); }

    free(pts); free(one); scratch_free(&sc);
    free(base); free(p); free(snap); free(run); free(rep); free(rsnap); free(rrun);
    return 0;
}

/* Options that no longer run, kept so an old command line says why instead of "unknown". */
static int deprecated(const char *what, const char *why)
{
    fprintf(stderr, "%s: this option is deprecated and no longer runs. %s\n", what, why);
    return 1;
}

int main(int argc, char **argv)
{
    tables_init();
    const char *mode = argc > 1 ? argv[1] : "help";
    if (!strcmp(mode, "encode")) return mode_encode();
    if (!strcmp(mode, "safetytest")) return mode_safetytest();
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
    if (!strcmp(mode, "subsample"))
        return deprecated("nnplay subsample", "It measured what scoring a random subset of the "
                          "children costs, from before the policy chose them; the search "
                          "replaces both.");
    if (strcmp(mode, "play") && strcmp(mode, "eval") && strcmp(mode, "elo")
        && strcmp(mode, "selfcheck") && strcmp(mode, "variance")) {
        fputs(USAGE, stderr);
        return 1;
    }

    Cfg cfg = { .max_plies = 3000, .rep_limit = 4, .stalemate_loses = 1, .mat_shift = 3,
                .regicide = 1, .progress = 30, .temperature = 1.0, .seed_base = 0x5EED5EED5EED5EEDull,
                .verbose = 1, .opening_temperature = 1.0, .king_safety = 1, .resign_pawns = 300,
                .resign_plies = 10, .cap_by_material = 1, .eg_max_pieces = 20, .eg_imbalance = 50,
                .sims = 128, .leaves = 8, .games_par = 32, .cpuct = 1.25, .noise_frac = 0.25,
                .noise_alpha = 10.0, .contempt = 0.1, .soft_scale = 0.75, .opp_sims = -1 };
    const char *model = "net_fp16.onnx";
    uint64_t games = 1;
    int device_id = 0, sub_batch = 0, probe_cap = 8192, first_game = 0, tf32 = 1;
    int temperature_set = 0, allow_cpu = 0, games_set = 0, seed_set = 0, soft_set = 0;
    Sprt sprt = { 1, 0.0, 50.0, 0.05, 0.05, 0.0, 0.0 };
    uint64_t target_positions = 0;
    const char *manifest = "net_manifest.json";
    const char *csv_path = NULL;
    int branch_every = 100, branches = 8, max_points = 0;
    const char *baseline = NULL;         /* NULL -> derive from --precision at use */
    static const char *PRUNE = "The one-ply search now always scores the policy's top 5% plus "
                               "2% at random; use --sims for the search.";

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--model")       && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--precision")   && i + 1 < argc) {
            const char *p = argv[++i];
            if      (!strcmp(p, "fp16")) model = "net_fp16.onnx";
            else if (!strcmp(p, "fp32")) model = "net_fp32.onnx";
            else { fprintf(stderr, "--precision must be fp16|fp32\n"); return 1; } }
        else if (!strcmp(argv[i], "--games")       && i + 1 < argc) { games = strtoull(argv[++i], NULL, 10); games_set = 1; }
        else if (!strcmp(argv[i], "--positions")   && i + 1 < argc) target_positions = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--first-game")  && i + 1 < argc) first_game = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out")         && i + 1 < argc) cfg.out_path = argv[++i];
        else if (!strcmp(argv[i], "--children")    && i + 1 < argc) cfg.children_path = argv[++i];
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) { cfg.temperature = atof(argv[++i]); temperature_set = 1; }
        else if (!strcmp(argv[i], "--max-plies")   && i + 1 < argc) cfg.max_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rep-limit")   && i + 1 < argc) cfg.rep_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-progress") && i + 1 < argc) cfg.no_progress_limit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")        && i + 1 < argc) { cfg.seed_base = strtoull(argv[++i], NULL, 10); seed_set = 1; }
        else if (!strcmp(argv[i], "--mat-shift")   && i + 1 < argc) cfg.mat_shift = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device")      && i + 1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sub-batch")   && i + 1 < argc) sub_batch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--probe-cap")   && i + 1 < argc) probe_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--progress-every") && i + 1 < argc) cfg.progress = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stats"))       cfg.stats = 1;
        else if (!strcmp(argv[i], "--baseline")     && i + 1 < argc) baseline = argv[++i];
        else if (!strcmp(argv[i], "--opening-plies")       && i + 1 < argc) cfg.opening_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--opening-temperature") && i + 1 < argc) cfg.opening_temperature = atof(argv[++i]);
        else if (!strcmp(argv[i], "--allow-cpu"))    allow_cpu = 1;
        else if (!strcmp(argv[i], "--sims")          && i + 1 < argc) cfg.sims = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--baseline-sims") && i + 1 < argc) cfg.opp_sims = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--leaves")        && i + 1 < argc) cfg.leaves = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--parallel-games") && i + 1 < argc) cfg.games_par = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpuct")         && i + 1 < argc) cfg.cpuct = atof(argv[++i]);
        else if (!strcmp(argv[i], "--noise-frac")    && i + 1 < argc) cfg.noise_frac = atof(argv[++i]);
        else if (!strcmp(argv[i], "--noise-alpha")   && i + 1 < argc) cfg.noise_alpha = atof(argv[++i]);
        else if (!strcmp(argv[i], "--contempt")      && i + 1 < argc) cfg.contempt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--selfplay-soft-target-scale") && i + 1 < argc) {
            cfg.soft_scale = atof(argv[++i]);
            soft_set = 1;
        }
        else if (!strcmp(argv[i], "--branch-every")  && i + 1 < argc) branch_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--branches")      && i + 1 < argc) branches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-points")    && i + 1 < argc) max_points = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")          && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--manifest")     && i + 1 < argc) manifest = argv[++i];
        else if (!strcmp(argv[i], "--no-tf32"))     tf32 = 0;
        else if (!strcmp(argv[i], "--no-regicide")) cfg.regicide = 0;
        else if (!strcmp(argv[i], "--no-king-safety")) cfg.king_safety = 0;
        else if (!strcmp(argv[i], "--resign-lead")  && i + 1 < argc) cfg.resign_pawns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--resign-plies") && i + 1 < argc) cfg.resign_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cap-draw"))     cfg.cap_by_material = 0;
        else if (!strcmp(argv[i], "--drop-capped"))  cfg.drop_capped = 1;
        else if (!strcmp(argv[i], "--endgames"))     cfg.endgames = 1;
        else if (!strcmp(argv[i], "--endgames-max-pieces") && i + 1 < argc) cfg.eg_max_pieces = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--endgames-imbalance")  && i + 1 < argc) cfg.eg_imbalance = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-sprt"))     sprt.on = 0;
        else if (!strcmp(argv[i], "--sprt")         && i + 1 < argc) {
            if (sscanf(argv[++i], "%lf,%lf", &sprt.elo0, &sprt.elo1) != 2 || sprt.elo1 <= sprt.elo0) {
                fprintf(stderr, "--sprt wants ELO0,ELO1 with ELO0 < ELO1, e.g. --sprt 0,50\n");
                return 1;
            }
            sprt.on = 1; }
        else if (!strcmp(argv[i], "--alpha")        && i + 1 < argc) sprt.alpha = atof(argv[++i]);
        else if (!strcmp(argv[i], "--beta")         && i + 1 < argc) sprt.beta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quiet"))     { cfg.verbose = 0; cfg.progress = 0; }
        else if (!strcmp(argv[i], "--subset-fraction") || !strcmp(argv[i], "--policy-top")
                 || !strcmp(argv[i], "--policy-explore"))
            return deprecated(argv[i], PRUNE);
        else if (!strcmp(argv[i], "--policy-model"))
            return deprecated(argv[i], "The policy is part of the model graph now; re-export with "
                                       "export_onnx.py.");
        else if (!strcmp(argv[i], "--vs-full"))
            return deprecated(argv[i], "It measured the one-ply pruning against scoring every "
                                       "child; compare --sims settings with --baseline-sims.");
        else if (!strcmp(argv[i], "--proportions") || !strcmp(argv[i], "--repeats")
                 || !strcmp(argv[i], "--stride") || !strcmp(argv[i], "--verify"))
            return deprecated(argv[i], "It belonged to the subsample mode, which is gone.");
        else { fprintf(stderr, "unknown option %s\n", argv[i]); fputs(USAGE, stderr); return 1; }
    }
    if (cfg.sims < 0 || cfg.leaves < 1 || cfg.games_par < 1 || cfg.cpuct <= 0.0
        || cfg.noise_frac < 0.0 || cfg.noise_frac > 1.0 || cfg.noise_alpha <= 0.0) {
        fprintf(stderr, "--sims >= 0, --leaves >= 1, --parallel-games >= 1, --cpuct > 0, "
                        "--noise-frac in [0, 1] and --noise-alpha > 0, please\n");
        return 1;
    }
    if (!strcmp(mode, "elo")) { if (cfg.opp_sims < 0) cfg.opp_sims = cfg.sims; }
    else cfg.opp_sims = 0;
    if (!strcmp(mode, "variance")) cfg.sims = 0;

    /* A fixed default seed made every session replay the previous one's games: game i's
     * seed is a function of the base seed and i alone, so two sessions with the same model
     * wrote ~1M identical positions. A fresh seed per run, printed and stored in the
     * header, keeps runs distinct and still reproducible. */
    if (!seed_set) {
        uint64_t r;
        if (getrandom(&r, sizeof(r), 0) != (ssize_t)sizeof(r))
            r = (uint64_t)time(NULL) * 0x9E3779B97F4A7C15ull ^ (uint64_t)getpid() << 32
                ^ (uint64_t)(now_s() * 1e9);
        cfg.seed_base = r;
    }
    if ((cfg.drop_capped || cfg.endgames || cfg.children_path) && strcmp(mode, "play")) {
        fprintf(stderr, "--drop-capped, --endgames and --children apply to play only\n");
        return 1;
    }
    if (soft_set && strcmp(mode, "play")) {
        fprintf(stderr, "--selfplay-soft-target-scale applies to play only\n");
        return 1;
    }
    if (!(cfg.soft_scale >= 0.0 && cfg.soft_scale <= 1.0)) {
        fprintf(stderr, "--selfplay-soft-target-scale must be in [0, 1]\n");
        return 1;
    }
    if (strcmp(mode, "play") || cfg.endgames) cfg.soft_scale = 0.0;   /* endgames keep only kings */
    cfg.contempt_flat = !strcmp(mode, "play");
    if (cfg.endgames) {
        if (cfg.eg_max_pieces < 4 || cfg.eg_max_pieces > 805 || cfg.eg_imbalance < 0) {
            fprintf(stderr, "--endgames-max-pieces must be 4..805 (two Kings and a ranging piece "
                            "need 3, and it is a strict bound) and --endgames-imbalance >= 0\n");
            return 1;
        }
        cfg.resign_pawns = 0;          /* a lead is the starting condition, not a result */
        cfg.cap_by_material = 0;
        eg_tables();
    }
    if (strcmp(mode, "eval") && strcmp(mode, "selfcheck"))     /* the modes that play games */
        fprintf(stderr, "seed %llu%s\n", (unsigned long long)cfg.seed_base,
                seed_set ? "" : " (random; --seed with this value repeats the run)");

    Net net;
    net_open(&net, model, device_id, sub_batch, probe_cap, tf32, allow_cpu, cfg.verbose);
    int rc;
    if (!strcmp(mode, "eval")) rc = mode_eval(&net);
    else if (!strcmp(mode, "selfcheck")) rc = mode_selfcheck(&net, manifest);
    else if (!strcmp(mode, "variance"))
        rc = mode_variance(&net, &cfg, games, first_game, branch_every, branches, max_points, csv_path);
    else if (!strcmp(mode, "play"))
        rc = mode_games(&net, NULL, NULL, &cfg, 0, games, games_set, target_positions, first_game,
                        &sprt, model, device_id);
    else {
        /* Greedy measures strength, but a kept match is training data, and greedy games
         * are the least varied there are; --out samples unless told otherwise. */
        if (!temperature_set) cfg.temperature = cfg.out_path ? 0.3 : 0.0;
        /* The baseline defaults to the graph exported from baseline.pt, matching the
         * precision the bot is running. "random" asks for the uniform mover instead, which
         * is the absolute anchor but stops discriminating once the bot beats it every
         * game. */
        char bpath[512];
        Net opp;
        int opened = 0;
        if (!baseline) {
            snprintf(bpath, sizeof(bpath), "baseline_%s.onnx", net.fp16[0] ? "fp16" : "fp32");
            baseline = bpath;
        }
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
            opened = 1;
        }
        if (sprt.alpha <= 0.0 || sprt.alpha >= 0.5 || sprt.beta <= 0.0 || sprt.beta >= 0.5) {
            fprintf(stderr, "--alpha and --beta must be in (0, 0.5)\n");
            return 1;
        }
        sprt_init(&sprt);
        /* --games is the SPRT's upper limit. 1000 by default: in simulation at the default
         * bounds, a bot sitting exactly on either hypothesis decides by then in ~99% of
         * matches (~250 games on average); at 400 it is ~85%. */
        if (sprt.on && !games_set) games = 1000;
        rc = mode_games(&net, opened ? &opp : NULL, opened ? baseline : "uniform-random", &cfg, 1,
                        games, games_set, 0, first_game, &sprt, model, device_id);
        if (opened) net_close(&opp);
    }
    net_close(&net);
    return rc;
}
