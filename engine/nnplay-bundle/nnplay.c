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

    st = ort->Run(n->sess, NULL, &n->in_name, (const OrtValue *const *)&in, 1, n->out_name, 1, out);
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
    /* value, plus material from the legacy model; only value is read */
    if (nin != 1 || nout < 1 || nout > 2) {
        fprintf(stderr, "model must have 1 input and 1-2 outputs (value[, material]); got %zu/%zu\n", nin, nout);
        exit(1);
    }
    ORT_CHECK(ort->SessionGetInputName(n->sess, 0, n->alloc, &n->in_name_owned));
    n->in_name = n->in_name_owned;
    for (int i = 0; i < (int)nout; i++) {
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
        fprintf(stderr, "model %s (hash %08x): input '%s', output '%s'%s, value is %s\n",
                model, hash_file(model), n->in_name, n->out_name[0],
                nout > 1 ? " (+ material, unused)" : "", n->fp16_out ? "float16" : "float32");

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

/* ---- the policy graph -------------------------------------------------------- */
/* One parent board and its move list in, one logit per move out: (1, 1296) and (M, 3)
 * int32, rows (from, to, special) in the board's canonical frame. Shares the value
 * graph's environment and session options, so it runs on the same provider. */
typedef struct {
    OrtSession    *sess;
    OrtMemoryInfo *meminfo;
    OrtAllocator  *alloc;
    const char    *in_name[2], *out_name;     /* in_name[0] is the board, [1] the moves */
    char          *owned[3];
    int            fp16_out;
} Pol;

/* The policy graph beside a value graph: net_fp16.onnx -> net_fp16_policy.onnx. */
static void policy_path(const char *model, char *out, size_t n)
{
    size_t l = strlen(model);
    if (l > 5 && !strcmp(model + l - 5, ".onnx"))
        snprintf(out, n, "%.*s_policy.onnx", (int)(l - 5), model);
    else
        snprintf(out, n, "%s_policy.onnx", model);
}

static int pol_open(Pol *q, const Net *base, const char *path, int verbose)
{
    memset(q, 0, sizeof(*q));
    if (access(path, R_OK) != 0) return -1;
    ORT_CHECK(ort->CreateSession(base->env, path, base->opts, &q->sess));
    ORT_CHECK(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &q->meminfo));
    ORT_CHECK(ort->GetAllocatorWithDefaultOptions(&q->alloc));
    size_t nin = 0, nout = 0;
    ORT_CHECK(ort->SessionGetInputCount(q->sess, &nin));
    ORT_CHECK(ort->SessionGetOutputCount(q->sess, &nout));
    if (nin != 2 || nout != 1) {
        fprintf(stderr, "%s: a policy graph has inputs (board, moves) and one output; got %zu/%zu\n",
                path, nin, nout);
        exit(1);
    }
    char *a, *b;
    ORT_CHECK(ort->SessionGetInputName(q->sess, 0, q->alloc, &a));
    ORT_CHECK(ort->SessionGetInputName(q->sess, 1, q->alloc, &b));
    int swap = !strcmp(a, "moves");
    q->owned[0] = a; q->owned[1] = b;
    q->in_name[0] = swap ? b : a;
    q->in_name[1] = swap ? a : b;
    ORT_CHECK(ort->SessionGetOutputName(q->sess, 0, q->alloc, &q->owned[2]));
    q->out_name = q->owned[2];
    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *tsi = NULL;
    ONNXTensorElementDataType et;
    ORT_CHECK(ort->SessionGetOutputTypeInfo(q->sess, 0, &ti));
    ORT_CHECK(ort->CastTypeInfoToTensorInfo(ti, &tsi));
    ORT_CHECK(ort->GetTensorElementType(tsi, &et));
    q->fp16_out = (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    ort->ReleaseTypeInfo(ti);
    if (verbose)
        fprintf(stderr, "policy %s (hash %08x): inputs '%s','%s', logits are %s\n", path,
                hash_file(path), q->in_name[0], q->in_name[1], q->fp16_out ? "float16" : "float32");
    return 0;
}

static void pol_eval(Pol *q, const int32_t *board, const int32_t *mv3, int m, float *out)
{
    int64_t bs[2] = { 1, NSQ }, ms[2] = { m, 3 };
    OrtValue *in[2] = { NULL, NULL }, *res = NULL;
    ORT_CHECK(ort->CreateTensorWithDataAsOrtValue(q->meminfo, (void *)board, NSQ * sizeof(int32_t),
                                                  bs, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &in[0]));
    ORT_CHECK(ort->CreateTensorWithDataAsOrtValue(q->meminfo, (void *)mv3, (size_t)m * 3 * sizeof(int32_t),
                                                  ms, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &in[1]));
    ORT_CHECK(ort->Run(q->sess, NULL, q->in_name, (const OrtValue *const *)in, 2, &q->out_name, 1, &res));
    void *raw = NULL;
    ORT_CHECK(ort->GetTensorMutableData(res, &raw));
    if (q->fp16_out) { const uint16_t *h = (const uint16_t *)raw; for (int i = 0; i < m; i++) out[i] = half_to_float(h[i]); }
    else             memcpy(out, raw, (size_t)m * sizeof(float));
    ort->ReleaseValue(res);
    ort->ReleaseValue(in[0]);
    ort->ReleaseValue(in[1]);
}

static void pol_close(Pol *q)
{
    for (int i = 0; i < 3; i++)
        if (q->owned[i]) { OrtStatus *st = ort->AllocatorFree(q->alloc, q->owned[i]); if (st) ort->ReleaseStatus(st); }
    if (q->meminfo) ort->ReleaseMemoryInfo(q->meminfo);
    if (q->sess) ort->ReleaseSession(q->sess);
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
#define TKN_VERSION 6u
#define TKN_FILE_HDR 128
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
    float    value;         /* the network's value for the chosen leaf, opponent's view */
    uint16_t n_legal;       /* legal moves in the position the move was played from */
    int16_t  material;      /* after the move, black minus white, mat_shift units */
    uint8_t  side;          /* the mover: 1 black, 0 white */
    uint8_t  flags;
    uint16_t n_scored;      /* children actually evaluated; 0 in files written before
                             * --subset-fraction existed, which means "all of them" */
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
    int32_t *parent;        /* the position itself, encoded, for the policy graph */
    int32_t *mv3;           /* its moves as policy rows (from, to, special) */
    float   *logits;
    struct Ranked { float l; int i; } *rk;
} Scratch;

/* Which children the value head scores. With a policy graph: the policy's top `top` of
 * the legal moves plus `explore` more drawn at random from the rest, so the policy's
 * misses still get evaluated -- and recorded, so the next policy can learn them. With
 * pol NULL, every child, as elo --vs-full's opponent does. */
typedef struct { Pol *pol; double top, explore; } Pick;

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
    free(s->parent); free(s->mv3); free(s->logits); free(s->rk);
}

static void scratch_reserve(Scratch *s, int n)
{
    if (n <= s->cap) return;
    s->cap = n + 256;
    s->boards  = realloc(s->boards,  (size_t)s->cap * NSQ * sizeof(int32_t));
    s->values  = realloc(s->values,  (size_t)s->cap * sizeof(float));
    s->weights = realloc(s->weights, (size_t)s->cap * sizeof(double));
    s->perm    = realloc(s->perm,    (size_t)s->cap * sizeof(int));
    s->mv3     = realloc(s->mv3,     (size_t)s->cap * 3 * sizeof(int32_t));
    s->logits  = realloc(s->logits,  (size_t)s->cap * sizeof(float));
    s->rk      = realloc(s->rk,      (size_t)s->cap * sizeof(*s->rk));
    if (!s->boards || !s->values || !s->weights || !s->perm || !s->mv3 || !s->logits || !s->rk) {
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

/* Puts the children to score first in s->perm, and returns how many. One policy pass over
 * the parent ranks every legal move; the value head then sees the top few and a few at
 * random. Everything unscored stays in s->perm after them, in policy order, for the
 * King-safety fallback. */
static int pick_children(const Pick *pk, const Pos *p, Scratch *s, Rng *rng)
{
    int n = s->g.n;
    for (int i = 0; i < n; i++) s->perm[i] = i;
    if (!pk || !pk->pol || pk->top >= 1.0) return n;
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
        s->mv3[3 * i] = o; s->mv3[3 * i + 1] = t; s->mv3[3 * i + 2] = sp;
    }
    pol_eval(pk->pol, s->parent, s->mv3, n, s->logits);
    for (int i = 0; i < n; i++) { s->rk[i].l = s->logits[i]; s->rk[i].i = i; }
    qsort(s->rk, (size_t)n, sizeof(*s->rk), rank_desc);
    for (int i = 0; i < n; i++) s->perm[i] = s->rk[i].i;
    int top = (int)ceil(pk->top * n), ex = (int)(pk->explore * n + 0.5);
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
static uint32_t choose(Net *net, int king_safety, const Pick *pk, const Pos *p,
                       Scratch *s, Rng *rng, double T, float *val, int *n_scored,
                       int *overridden)
{
    int n = s->g.n;
    scratch_reserve(s, n);
    int scored = pick_children(pk, p, s, rng);
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

/* ---- self-play ------------------------------------------------------------- */
typedef struct {
    int      max_plies, rep_limit, no_progress_limit, stalemate_loses, mat_shift, regicide;
    int      progress;      /* print a line every this many plies; 0 is off */
    int      stats;         /* per-ply wall-clock breakdown and GPU utilisation */
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
    double   policy_top;           /* the value head scores the policy's top this much... */
    double   policy_explore;       /* ...plus this much more of the legal moves at random */
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
    /* 100: bit 0 fp16; version 6 on, bits 8..15 policy_explore and bits 16..31
     * policy_top, both in thousandths of the legal moves */
    v = (uint32_t)(net->fp16_out ? 1 : 0)
      | ((uint32_t)(c->policy_explore * 1000.0 + 0.5) & 0xFFu) << 8
      | ((uint32_t)(c->policy_top * 1000.0 + 0.5) & 0xFFFFu) << 16;
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
                         uint64_t seed)
{
    if (nply) plies[nply - 1].flags |= PF_TERMINAL;
    unsigned char gh[TKN_GAME_HDR];
    uint32_t np = (uint32_t)nply;
    memcpy(gh, &np, 4);
    /* the user-facing score: 1 black, -1 white, 0 a draw */
    gh[4] = (unsigned char)(int8_t)(result == RES_BLACK ? 1 : result == RES_WHITE ? -1 : 0);
    gh[5] = (unsigned char)term;
    gh[6] = 0; gh[7] = 0;
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

/* ---- .kids: the children the search scored -------------------------------------- */
/* Policy training wants, for each position, the moves the search looked at and what the
 * value head said about each. The .tkn keeps only the move played; this side file keeps
 * the rest, at no extra network cost since the values were computed anyway.
 *
 *   file header, 32 bytes: "TKYKIDS\0", u32 version 1, u32 header size 32, u32 child size
 *       6, u32 game header size 24, u64 games (patched at close)
 *   per game, 24 bytes: u32 n_plies, u32 reserved, u64 seed, u64 FNV-1a 64 of the game's
 *       ply-record move words, in order -- (seed, n_plies, hash) finds the matching .tkn
 *       game even after files are merged or reordered
 *   per ply: u16 n, then n children of { u32 move, u16 value as IEEE half }
 *
 * Ply k's children belong to the position *before* move k: the .tkn record k-1's board,
 * or the start position for k = 0. A move uses the ply record's packing, special << 28 |
 * origin << 17 | target << 6, absolute frame. A value is the network's, for the side to
 * move after that child, exactly as the search saw it: the mover prefers low values. A
 * forced King capture is stored as its single child. */
#define KIDS_MAGIC "TKYKIDS"
#define KIDS_FILE_HDR 32
#define KIDS_GAME_HDR 24

typedef struct {
    Sink     sink;
    unsigned char *buf;            /* the current game, until its header is known */
    size_t   len, cap;
    uint64_t games;
} Kids;

static int kids_open(Kids *k, const char *path)
{
    memset(k, 0, sizeof(*k));
    k->sink.cap = 1u << 22;
    k->sink.fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (k->sink.fd < 0) { perror(path); return -1; }
    k->sink.buf = malloc(k->sink.cap);
    unsigned char h[KIDS_FILE_HDR];
    memset(h, 0, sizeof(h));
    memcpy(h, KIDS_MAGIC, 8);
    uint32_t v = 1;              memcpy(h + 8, &v, 4);
    v = KIDS_FILE_HDR;           memcpy(h + 12, &v, 4);
    v = 6;                       memcpy(h + 16, &v, 4);
    v = KIDS_GAME_HDR;           memcpy(h + 20, &v, 4);
    sink_put(&k->sink, h, sizeof(h));
    return 0;
}

static void kids_need(Kids *k, size_t more)
{
    if (k->len + more <= k->cap) return;
    k->cap = (k->len + more) * 2;
    k->buf = realloc(k->buf, k->cap);
    if (!k->buf) { fprintf(stderr, "oom in --children buffer\n"); exit(1); }
}

/* n children: moves[i] in nnplay's internal packing, indexed through perm (NULL for the
 * identity), with their values. */
static void kids_ply(Kids *k, const uint32_t *moves, const int *perm, const float *values, int n)
{
    if (n > 65535) n = 65535;
    kids_need(k, 2 + (size_t)n * 6);
    uint16_t c = (uint16_t)n;
    memcpy(k->buf + k->len, &c, 2);
    k->len += 2;
    for (int i = 0; i < n; i++) {
        uint32_t m = file_move(moves[perm ? perm[i] : i]);
        uint16_t h = float_to_half(values[i]);
        memcpy(k->buf + k->len, &m, 4);
        memcpy(k->buf + k->len + 4, &h, 2);
        k->len += 6;
    }
}

static void kids_game(Kids *k, const PlyRecNN *plies, int nply, uint64_t seed)
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
    sink_put(&k->sink, k->buf, k->len);
    sink_flush(&k->sink);
    k->len = 0;
    k->games++;
}

static void kids_close(Kids *k)
{
    sink_flush(&k->sink);
    if (pwrite(k->sink.fd, &k->games, 8, 24) != 8) perror("pwrite");
    close(k->sink.fd);
    free(k->sink.buf);
    free(k->buf);
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
static int mode_selfcheck(Net *net, Pol *pol, const char *manifest)
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

    /* The policy, move for move: every legal move of the start position through the C
     * move generator, the rows nnplay builds and the policy graph, against the logits
     * export_onnx.py computed in torch for the same moves. */
    static double mv[3 * 8192], lg[8192];
    int nm = manifest_array("start_position_policy_moves", mv, 3 * 8192);
    int nl = manifest_array("start_position_policy_logits", lg, 8192);
    if (pol && nm > 0 && nl * 3 == nm) {
        double ptol = 0.05;
        char *q = strstr(manifest_buf, "\"policy_tolerance\"");
        if (q && (q = strchr(q, ':'))) ptol = strtod(q + 1, NULL);
        Gen g;
        memset(&g, 0, sizeof(g));
        g.cap = 8192; g.buf = malloc(8192 * 4); g.stamp = calloc(MAXMOVE, 2); g.kline = calloc(NSQ, 2);
        gen_all(p, &g);
        int32_t *rows = malloc((size_t)g.n * 3 * sizeof(int32_t));
        float *got = malloc((size_t)g.n * sizeof(float));
        for (int i = 0; i < g.n; i++) {
            int mvv = UNPACK_M(g.buf[i]);
            rows[3 * i] = UNPACK_O(g.buf[i]); rows[3 * i + 1] = mvv % 1296; rows[3 * i + 2] = mvv / 1296;
        }
        pol_eval(pol, b, rows, g.n, got);
        int matched = 0;
        double worst = 0.0;
        for (int i = 0; i < g.n; i++)
            for (int j = 0; j < nl; j++)
                if (mv[3 * j] == rows[3 * i] && mv[3 * j + 1] == rows[3 * i + 1] && mv[3 * j + 2] == rows[3 * i + 2]) {
                    matched++;
                    double d = fabs((double)got[i] - lg[j]);
                    if (d > worst) worst = d;
                    break;
                }
        int pok = matched == g.n && g.n == nl && worst < ptol;
        printf("  policy          %d moves, %d matched the export's, largest logit deviation %.2e "
               "(tolerance %.0e)   %s\n", g.n, matched, worst, ptol, pok ? "PASS" : "FAIL");
        ok = ok && pok;
        free(rows); free(got); free(g.buf); free(g.stamp); free(g.kline);
    } else if (pol) {
        printf("  policy          the manifest has no start-position policy -- re-run export_onnx.py\n");
        ok = 0;
    } else {
        printf("  policy          no policy graph: play, elo and variance will refuse this model\n");
    }
    fflush(stdout);
    free(b);
    free(p);
    return ok ? 0 : 1;
}

static const char USAGE[] =
    "usage: nnplay <play|elo|variance|subsample|selfcheck|eval|encode> [options]\n"
    "  play    self-play games and write a .tkn shard\n"
    "    --out FILE          where to write (omit to play without recording)\n"
    "    --games N           games to play (default 1)\n"
    "    --positions N       play whole games until at least N positions are written,\n"
    "                        however many games that takes. The last game is finished\n"
    "                        rather than cut short, so the count overshoots a little and\n"
    "                        every game keeps a real result. With --games as well, that\n"
    "                        becomes a cap on the number of games.\n"
    "    --first-game N      index of the first game, so shards can be split across runs\n"
    "    --temperature T     softmax temperature over the negated leaf values (default 1.0,\n"
    "                        0 is greedy)\n"
    "    --max-plies N       ply cap, scored as a draw (default 3000)\n"
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
    "    --policy-top F      the value head scores the policy's top F of the legal moves\n"
    "                        (default 0.05)...\n"
    "    --policy-explore F  ...plus F more drawn at random from the rest (default 0.02), so\n"
    "                        the policy's misses still get evaluated and, with --children,\n"
    "                        recorded for the next policy to learn from. Needs the policy\n"
    "                        graph beside the value graph (net_fp16_policy.onnx);\n"
    "                        --policy-model FILE names another\n"

    "    --opening-plies N   play the first N plies at --opening-temperature, then switch\n"
    "                        to --temperature. 0 (default) keeps one temperature all game.\n"
    "                        A hot opening buys diversity; a cool middlegame makes the\n"
    "                        result follow from the position instead of from luck.\n"
    "    --opening-temperature T   temperature for those plies (default 1.0)\n"
    "    --children FILE     also write every child the search scored, with its value,\n"
    "                        to a .kids side file paired with --out. About 2 KB a\n"
    "                        position; it is what policy training learns from\n"
    "    --seed S            base seed. Default: a fresh random one every run, printed at\n"
    "                        start and stored in the file header, so sessions never\n"
    "                        replay each other's games and any run can still be repeated\n"
    "  elo     rate the network against a baseline anchored at 0. Colours alternate and\n"
    "          regicide is forced for both sides. --temperature defaults to 0 here\n"
    "          (playing strength) rather than the sampling policy's value -- or to\n"
    "          0.3 with --out, since the games are then kept as training data.\n"
    "    --baseline FILE     opponent graph, default baseline_<precision>.onnx, which\n"
    "                        export_onnx.py writes from baseline.pt\n"
    "    --baseline random   the uniform mover instead: an absolute anchor, but it stops\n"
    "                        discriminating once the bot beats it every game\n"
    "    --vs-full           rate the cost of the policy pruning: the net plays itself, the\n"
    "                        rated side scoring the policy's picks and its opponent every\n"
    "                        child. Needs no baseline graph. A baseline graph must itself\n"
    "                        have a policy graph beside it\n"
    "    --sprt E0,E1        sequential test, on by default at 0,50: stop as soon as the\n"
    "                        games show the bot is at least E1 better than the baseline\n"
    "                        (H1) or at most E0 (H0), in logistic Elo -- the rating a score\n"
    "                        implies. Draws are handled as a nuisance parameter\n"
    "    --alpha A, --beta B the test's error rates, default 0.05 each\n"
    "    --games N           with SPRT, the most games before stopping undecided\n"
    "                        (default 1000); with --no-sprt, the exact number to play\n"
    "    --no-sprt           play exactly --games games, the old behaviour\n"
    "    --out FILE          also keep every position, in the same .tkn as self-play. Each\n"
    "                        ply's value is the rated net's; the opponent's moves are\n"
    "                        flagged (32 in the ply flags)\n"
    "  variance   the ceiling on what any value head could learn from this generator.\n"
    "          Snapshots positions along a game and replays each many times, so the\n"
    "          outcome variance splits into the part the position decides and the part\n"
    "          decided by luck afterwards. Use it to compare temperature schedules.\n"
    "    --branch-every N    snapshot every N plies, from a random phase per game so\n"
    "                        every ply is equally likely (default 100, 0 disables)\n"
    "    --branches K        playouts from each snapshot (default 8, minimum 2)\n"
    "    --max-points N      at most N snapshots per game (0 = no limit)\n"
    "    --csv FILE          one row per branch point\n"
    "  subsample  how much value is lost by scoring only a fraction of the children at\n"
    "          each position. Plays self-play games, and at every position compares the\n"
    "          best child found in a random subset against the best over all of them.\n"
    "    --proportions L     comma-separated fractions (default 0.01,0.02,0.05,0.1,0.25,0.5)\n"
    "    --repeats N         random subsets per position per fraction (default 8)\n"
    "    --stride N          measure every Nth ply (default 1)\n"
    "    --csv FILE          one row per position, fraction and repeat\n"
    "    --verify N          re-run N subsets as their own GPU batch as a cross-check\n"
    "  safetytest  check the King-safety rule against tky.c's slow detector; no GPU\n"
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

/* Bot versus a uniformly random mover. Regicide is forced for both sides, exactly as in
 * self-play, so games end at a few hundred plies instead of running forever. Colours
 * alternate so the first-move advantage is measured rather than absorbed into the rating.
 */
static int mode_elo(Net *net, Net *opp, const char *opp_name, Cfg *cfg,
                    uint64_t games, int first_game, const Pick *bot_pk, const Pick *opp_pk,
                    const Sprt *sprt, const char *model)
{
    Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos));
    pos_init(base);
    Scratch sc;
    scratch_init(&sc);
    Gen *g = &sc.g;
    RepTab *rep = calloc(1, sizeof(RepTab));
    double *marg = malloc(sizeof(double) * ELO_N);
    double bot_evals = 0, opp_evals = 0, bot_plies = 0, opp_plies = 0;

    Rec as_black = { 0, 0, 0 }, as_white = { 0, 0, 0 };
    EloFit fit = { 0, 0, 0, 0, 0, 0 };
    double llr = 0.0;
    int verdict = 0;                   /* SPRT: 1 H1 accepted, -1 H0 accepted, 0 undecided */

    /* --out keeps the match's positions, in the same .tkn as self-play. Every ply's value
     * is the rated net's, whoever moved, so the file holds one network's opinions like a
     * self-play file does; moves by the baseline or the random mover carry PF_OPPONENT. */
    Sink sink = { -1, NULL, 0, 1u << 22 };
    PlyRecNN *plybuf = NULL;
    uint64_t games_written = 0;
    int mat_div = VALUE_FP >> cfg->mat_shift;
    if (mat_div < 1) mat_div = 1;
    if (cfg->out_path) {
        if (tkn_open(&sink, cfg, net, model)) return 1;
        plybuf = malloc(sizeof(PlyRecNN) * (size_t)(cfg->max_plies + 2));
        fprintf(stderr, "writing every position to %s\n", cfg->out_path);
    }
    double t_start = now_s();
    uint64_t plies_total = 0;

    fprintf(stderr, "bot vs %s, regicide forced, king safety %s, temperature %.3g%s\n",
            opp_name, cfg->king_safety ? "on (both sides)" : "off",
            cfg->temperature, cfg->temperature == 0.0 ? " (greedy)" : "");
    if (cfg->resign_pawns > 0)
        fprintf(stderr, "adjudication: a %d-pawn lead held for %d plies wins\n",
                cfg->resign_pawns, cfg->resign_plies);
    fprintf(stderr, "ply cap %d: scored %s\n", cfg->max_plies,
            cfg->cap_by_material ? "for the side ahead by 50+ pawns, else a draw" : "as a draw");
    fprintf(stderr, "the bot's value head scores its policy's top %.1f%% of the moves plus %.1f%% "
                    "at random", 100.0 * bot_pk->top, 100.0 * bot_pk->explore);
    if (opp && opp_pk->pol)
        fprintf(stderr, "; so does the baseline, with its own policy\n");
    else if (opp)
        fprintf(stderr, "; its opponent, the same network, scores every child (--vs-full)\n");
    else
        fputc('\n', stderr);
    if (sprt->on)
        fprintf(stderr, "SPRT: H0 elo <= %+g, H1 elo >= %+g (logistic), alpha %g beta %g; stops "
                        "when LLR leaves [%.2f, %.2f], or after %llu games\n",
                sprt->elo0, sprt->elo1, sprt->alpha, sprt->beta, sprt->lower, sprt->upper,
                (unsigned long long)games);
    else
        fprintf(stderr, "fixed length: %llu games (--no-sprt)\n", (unsigned long long)games);
    if (opp && cfg->temperature == 0.0 && bot_pk->explore <= 0.0)
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
        int nply = 0, result = RES_DRAW, term = TERM_PLYCAP, lead_run = 0;
        double t_game = now_s();

        for (;;) {
            gen_all(p, g);
            if (g->n == 0) { result = cfg->stalemate_loses ? (1 - p->turn) : RES_DRAW; term = TERM_STALEMATE; break; }
            if (nply >= cfg->max_plies) { result = cap_result(cfg, p); term = TERM_PLYCAP; break; }

            uint32_t forced = 0;
            if (cfg->regicide) { king_ctx_set(g, p); forced = find_king_capture(g); }

            uint32_t pick;
            float chosen = 0.0f, leaf = 0.0f;
            int scored = 1, have_leaf = 0, n_legal = g->n;
            uint64_t ov0 = sc.overridden;
            /* Whoever is to move picks: the bot's net, the baseline net, or -- when there
             * is no baseline net -- uniformly at random. Forced regicide and King safety
             * are rules, not evaluations, and apply to both sides. */
            Net *mover_net = (p->turn == bot_side) ? net : opp;
            if (forced) {
                pick = forced;
            } else if (!mover_net) {
                pick = choose_random(cfg->king_safety, p, &sc, &rng);
            } else {
                /* The rated side may score only a random subset. It really does evaluate
                 * just those, so the throughput saving is real and not simulated. */
                float v;
                pick = choose(mover_net, cfg->king_safety, p->turn == bot_side ? bot_pk : opp_pk,
                              p, &sc, &rng, temp_at(cfg, nply), &v, &scored, NULL);
                if (p->turn == bot_side) { bot_evals += scored; bot_plies++; chosen = v; }
                else                     { opp_evals += scored; opp_plies++; }
                leaf = v;
                have_leaf = mover_net == net;   /* the baseline's own value is not recorded */
            }
            int overridden = sc.overridden > ov0;

            int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
            int pieces_before = p->npieces;
            int promoted_now = do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
            if (cfg->out_path) {
                if (!have_leaf) {               /* one board through the rated net */
                    scratch_reserve(&sc, 1);
                    encode_stm(p, sc.boards);
                    net_eval(net, sc.boards, 1, &leaf);
                }
                tkn_fill(&plybuf[nply], p, pick, leaf, n_legal, mover,
                         (p->npieces < pieces_before ? PF_CAPTURE : 0)
                         | (promoted_now ? PF_PROMOTION : 0) | (forced ? PF_FORCED_KING : 0)
                         | (overridden ? PF_KING_SAFETY : 0)
                         | (mover != bot_side ? PF_OPPONENT : 0),
                         scored, mat_div);
            }
            nply++;

            if (cfg->progress && nply % cfg->progress == 0)
                fprintf(stderr, "    game %-4llu ply %4d  %s to move  moves %5d  material %+8.1f  "
                                "value %+0.4f\r",
                        (unsigned long long)(first_game + gi), nply,
                        p->turn == bot_side ? "bot " : "base", g->n,
                        p->material / (double)VALUE_FP * (bot_side == SIDE_BLACK ? 1 : -1),
                        (double)chosen);

            if (p->royals[1 - mover] < rb[1 - mover])      { result = mover;     term = TERM_ROYAL; break; }
            if (p->royals[mover] < rb[mover])              { result = 1 - mover; term = TERM_ROYAL; break; }
            int seen = rep_bump(rep, p->hash);
            if (cfg->rep_limit > 0 && seen >= cfg->rep_limit) { result = RES_DRAW; term = TERM_REPETITION; break; }
            int adj = resign_check(cfg, p, &lead_run);
            if (adj >= 0) { result = adj; term = TERM_RESIGN; break; }
        }

        plies_total += (uint64_t)nply;
        if (cfg->out_path) {
            tkn_put_game(&sink, plybuf, nply, result, term, seed);
            games_written++;
        }
        Rec *r = (bot_side == SIDE_BLACK) ? &as_black : &as_white;
        const char *tag;
        if (result == RES_DRAW)          { r->d++; tag = "draw"; }
        else if (result == bot_side)     { r->w++; tag = "win "; }
        else                             { r->l++; tag = "loss"; }

        if (cfg->progress) fprintf(stderr, "\r%*s\r", 118, "");   /* wipe the ply line */
        elo_fit(&as_black, &as_white, marg, &fit);
        long wins = as_black.w + as_white.w, losses = as_black.l + as_white.l,
             draws = as_black.d + as_white.d;
        double n = (double)(wins + losses + draws);
        double score = n > 0 ? (wins + 0.5 * draws) / n : 0.0;
        double el = now_s() - t_start;

        char sp[48] = "";
        if (sprt->on) {
            llr = sprt_llr(wins, draws, losses, sprt->elo0, sprt->elo1);
            verdict = sprt_decide(sprt, llr);
            snprintf(sp, sizeof(sp), " | LLR %+.2f [%.2f, %.2f]", llr, sprt->lower, sprt->upper);
        }
        fprintf(stderr,
            "game %-4llu bot=%-5s %s in %4d plies (%5.1fs) | %4ldW %4ldL %4ldD  score %.3f | "
            "elo %+6.0f [%+.0f, %+.0f]%s | first-move %+.0f%s | %.1f min elapsed\n",
            (unsigned long long)(first_game + gi), bot_side == SIDE_BLACK ? "black" : "white",
            tag, nply, now_s() - t_game, wins, losses, draws, score,
            fit.elo, fit.lo, fit.hi, fit.saturated ? " (at the grid edge)" : "",
            fit.advantage, sp, el / 60.0);
        if (verdict) break;
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
    if (sprt->on) {
        if (verdict > 0)
            fprintf(stderr, "  SPRT: H1 accepted, LLR %+.2f >= %.2f -- the bot is better than the "
                            "baseline by at least ~%+g logistic Elo (error rate %g)\n",
                    llr, sprt->upper, sprt->elo1, sprt->alpha);
        else if (verdict < 0)
            fprintf(stderr, "  SPRT: H0 accepted, LLR %+.2f <= %.2f -- the bot is not %+g logistic "
                            "Elo better than the baseline (error rate %g)\n",
                    llr, sprt->lower, sprt->elo1, sprt->beta);
        else
            fprintf(stderr, "  SPRT: undecided after %ld games, LLR %+.2f inside [%.2f, %.2f]: "
                            "the --games limit came first. The evidence does not yet separate "
                            "%+g from %+g; read the Elo interval above, or raise --games\n",
                    (long)n, llr, sprt->lower, sprt->upper, sprt->elo0, sprt->elo1);
    }

    if (bot_plies && opp_plies)
        printf("  children scored per ply: rated side %.0f, opponent %.0f  (%.1f%% of the work)\n",
               bot_evals / bot_plies, opp_evals / opp_plies,
               100.0 * (bot_evals / bot_plies) / (opp_evals / opp_plies));
    if (cfg->king_safety)
        fprintf(stderr, "  king safety changed the pick on %llu plies (%.2f%%), both sides\n",
                (unsigned long long)sc.overridden,
                plies_total ? 100.0 * sc.overridden / plies_total : 0.0);
    if (cfg->out_path) {
        tkn_close(&sink, games_written, plies_total);
        fprintf(stderr, "  wrote %llu games, %llu positions to %s\n",
                (unsigned long long)games_written, (unsigned long long)plies_total,
                cfg->out_path);
    }
    free(plybuf); free(marg); scratch_free(&sc);
    free(rep); free(base); free(p);
    return 0;
}

/* ---- what does evaluating only a subset of children cost? --------------------- */
/* The policy scores every child, so a ply costs the branching factor. This measures what
 * is lost by scoring only a random fraction of them.
 *
 * Values are held in the mover's frame -- the negation the policy already applies -- so
 * the best child is the largest, and a subset's loss is best_full - best_subset >= 0.
 *
 * The subset's values are read out of the full evaluation rather than run as their own
 * batch. That is exact, not an approximation: a child's value does not depend on which
 * other children shared its batch. Only fp16 batch-shape rounding does, and --verify
 * measures that against a real subset-sized GPU run. Being exact makes every proportion
 * and every repeat free, so one self-play pass yields the whole curve.
 */
#define MAX_PROPS 16

typedef struct { double best, mean, sd, spread, p90, p95; int argbest; } VStats;

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double pctile(const double *sorted, int n, double q)
{
    if (n <= 0) return 0.0;
    double pos = q * (n - 1);
    int i = (int)pos;
    double f = pos - i;
    return (i + 1 < n) ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[n - 1];
}

static double *vs_buf = NULL;
static int vs_cap = 0;

/* idx == NULL means "all k of them, in order" */
static void vstats(const float *mv, const int *idx, int k, VStats *s)
{
    if (k > vs_cap) { vs_cap = k + 64; vs_buf = realloc(vs_buf, (size_t)vs_cap * sizeof(double)); }
    double sum = 0.0, sum2 = 0.0;
    s->best = -1e30;
    s->argbest = idx ? idx[0] : 0;
    for (int j = 0; j < k; j++) {
        int i = idx ? idx[j] : j;
        double v = mv[i];
        vs_buf[j] = v;
        sum += v;
        sum2 += v * v;
        if (v > s->best) { s->best = v; s->argbest = i; }
    }
    s->mean = sum / k;
    double var = sum2 / k - s->mean * s->mean;
    s->sd = var > 0 ? sqrt(var) : 0.0;
    qsort(vs_buf, (size_t)k, sizeof(double), cmp_dbl);
    s->spread = vs_buf[k - 1] - vs_buf[0];
    s->p90 = pctile(vs_buf, k, 0.90);
    s->p95 = pctile(vs_buf, k, 0.95);
}

/* Value the sampling policy actually expects to get from this set, at temperature T.
 * The engine samples rather than taking the argmax, so this is the decision-relevant
 * number; the argmax loss is the T -> 0 case of it. */
static double policy_ev(const float *mv, const int *idx, int k, double T)
{
    if (T <= 0.0) {
        double b = -1e30;
        for (int j = 0; j < k; j++) { double v = mv[idx ? idx[j] : j]; if (v > b) b = v; }
        return b;
    }
    double mx = -1e30;
    for (int j = 0; j < k; j++) { double v = mv[idx ? idx[j] : j] / T; if (v > mx) mx = v; }
    double z = 0.0, ev = 0.0;
    for (int j = 0; j < k; j++) {
        double v = mv[idx ? idx[j] : j], w = exp(v / T - mx);
        z += w;
        ev += w * v;
    }
    return ev / z;
}

typedef struct {
    double  p;
    long    n;
    double  sum_loss, sum_loss2, max_loss, sum_pol, sum_k, sum_rank, sum_agree;
    double  sum_mean, sum_sd, sum_spread, sum_p90, sum_p95;
    double *loss;
    int     lcap;
} PropAcc;

static int mode_subsample(Net *net, Cfg *cfg, uint64_t games, int first_game,
                          const double *props, int nprops, int repeats, int stride,
                          const char *csv_path, int verify)
{
    Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos)), *child = malloc(sizeof(Pos));
    pos_init(base);
    Gen g = { malloc(8192 * 4), 0, 8192, calloc(MAXMOVE, 2), 0, calloc(NSQ, 2), 0, -1, 0, 0 };
    RepTab *rep = calloc(1, sizeof(RepTab));

    int32_t *boards = NULL, *sub_boards = NULL;
    float   *values = NULL, *mv = NULL, *sub_vals = NULL;
    double  *weights = NULL;
    int     *perm = NULL;
    int      bcap = 0;

    PropAcc acc[MAX_PROPS];
    memset(acc, 0, sizeof(acc));
    for (int q = 0; q < nprops; q++) acc[q].p = props[q];

    FILE *csv = NULL;
    if (csv_path) {
        csv = fopen(csv_path, "w");
        if (!csv) { perror(csv_path); return 1; }
        fprintf(csv, "game,ply,n_children,proportion,repeat,k,best_full,best_part,loss,rank,"
                     "agree,mean_full,sd_full,spread_full,p90_full,p95_full,"
                     "mean_part,sd_part,spread_part,p90_part,p95_part,ev_full,ev_part,policy_loss\n");
    }

    long positions = 0, skipped_forced = 0;
    double full_mean = 0, full_sd = 0, full_spread = 0, full_p90 = 0, full_p95 = 0, full_n = 0;
    double full_best = 0, full_gap = 0;
    double t0 = now_s(), verify_dev = -1.0;
    long verified = 0;

    fprintf(stderr, "subset evaluation: %llu games, stride %d, %d repeats per proportion, "
                    "temperature %.3g\n",
            (unsigned long long)games, stride, repeats, cfg->temperature);

    for (uint64_t gi = 0; gi < games; gi++) {
        uint64_t seed = cfg->seed_base ^ (((uint64_t)first_game + gi) * 0x9E3779B97F4A7C15ull);
        Rng rng;
        rng_seed(&rng, seed);
        memcpy(p, base, sizeof(Pos));
        rep->epoch++;
        rep_bump(rep, p->hash);
        int nply = 0;

        for (;;) {
            gen_all(p, &g);
            if (g.n == 0 || nply >= cfg->max_plies) break;

            uint32_t forced = 0;
            if (cfg->regicide) { king_ctx_set(&g, p); forced = find_king_capture(&g); }

            if (g.n > bcap) {
                bcap = g.n + 256;
                boards     = realloc(boards,     (size_t)bcap * NSQ * sizeof(int32_t));
                sub_boards = realloc(sub_boards, (size_t)bcap * NSQ * sizeof(int32_t));
                values     = realloc(values,     (size_t)bcap * sizeof(float));
                sub_vals   = realloc(sub_vals,   (size_t)bcap * sizeof(float));
                mv         = realloc(mv,         (size_t)bcap * sizeof(float));
                weights    = realloc(weights,    (size_t)bcap * sizeof(double));
                perm       = realloc(perm,       (size_t)bcap * sizeof(int));
                if (!boards || !sub_boards || !values || !sub_vals || !mv || !weights || !perm) {
                    fprintf(stderr, "oom in candidate buffers\n");
                    return 1;
                }
            }
            for (int i = 0; i < g.n; i++) {
                memcpy(child, p, sizeof(Pos));
                do_move(child, SQ2P[UNPACK_O(g.buf[i])], UNPACK_M(g.buf[i]));
                encode_stm(child, boards + (size_t)i * NSQ);
            }
            net_eval(net, boards, g.n, values);
            for (int i = 0; i < g.n; i++) mv[i] = -values[i];      /* the mover's frame */

            int measure = !forced && (nply % stride == 0);
            if (forced) skipped_forced++;

            if (measure) {
                VStats f;
                vstats(mv, NULL, g.n, &f);
                double ev_full = policy_ev(mv, NULL, g.n, cfg->temperature);
                positions++;
                full_mean += f.mean; full_sd += f.sd; full_spread += f.spread;
                full_p90 += f.p90; full_p95 += f.p95; full_n += g.n;
                full_best += f.best; full_gap += f.best - f.p95;

                for (int q = 0; q < nprops; q++) {
                    int k = (int)(props[q] * g.n + 0.5);
                    if (k < 1) k = 1;
                    if (k > g.n) k = g.n;
                    for (int r = 0; r < repeats; r++) {
                        for (int i = 0; i < g.n; i++) perm[i] = i;
                        for (int i = 0; i < k; i++) {          /* partial Fisher-Yates */
                            int j = i + (int)rng_below(&rng, (uint32_t)(g.n - i));
                            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
                        }
                        VStats s;
                        vstats(mv, perm, k, &s);
                        double loss = f.best - s.best;
                        double ev_part = policy_ev(mv, perm, k, cfg->temperature);
                        int rank = 0;
                        for (int i = 0; i < g.n; i++) if (mv[i] > s.best) rank++;

                        PropAcc *a = &acc[q];
                        if (a->n == a->lcap) {
                            a->lcap = a->lcap ? a->lcap * 2 : 4096;
                            a->loss = realloc(a->loss, (size_t)a->lcap * sizeof(double));
                        }
                        a->loss[a->n] = loss;
                        a->n++;
                        a->sum_loss += loss;
                        a->sum_loss2 += loss * loss;
                        if (loss > a->max_loss) a->max_loss = loss;
                        a->sum_pol += ev_full - ev_part;
                        a->sum_k += k;
                        a->sum_rank += rank;
                        a->sum_agree += (s.argbest == f.argbest);
                        a->sum_mean += s.mean; a->sum_sd += s.sd; a->sum_spread += s.spread;
                        a->sum_p90 += s.p90; a->sum_p95 += s.p95;

                        if (csv)
                            fprintf(csv, "%llu,%d,%d,%.4f,%d,%d,%.6f,%.6f,%.6f,%d,%d,"
                                         "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                                         "%.6f,%.6f,%.6f\n",
                                    (unsigned long long)(first_game + gi), nply, g.n,
                                    props[q], r, k, f.best, s.best, loss, rank,
                                    s.argbest == f.argbest,
                                    f.mean, f.sd, f.spread, f.p90, f.p95,
                                    s.mean, s.sd, s.spread, s.p90, s.p95,
                                    ev_full, ev_part, ev_full - ev_part);

                        /* Prove the subset really is a read-out of the full evaluation by
                         * running it as its own GPU batch, for the first few positions. */
                        if (verify > 0 && verified < verify && q == nprops - 1 && r == 0) {
                            for (int i = 0; i < k; i++)
                                memcpy(sub_boards + (size_t)i * NSQ,
                                       boards + (size_t)perm[i] * NSQ, NSQ * sizeof(int32_t));
                            net_eval(net, sub_boards, k, sub_vals);
                            double dmax = 0.0;
                            for (int i = 0; i < k; i++) {
                                double d = fabs((double)(-sub_vals[i]) - (double)mv[perm[i]]);
                                if (d > dmax) dmax = d;
                            }
                            if (dmax > verify_dev) verify_dev = dmax;
                            verified++;
                        }
                    }
                }
            }

            uint32_t pick = forced ? forced
                                   : g.buf[sample_softmax(values, g.n, cfg->temperature, &rng, weights)];
            int mover = p->turn, rb[2] = { p->royals[0], p->royals[1] };
            do_move(p, SQ2P[UNPACK_O(pick)], UNPACK_M(pick));
            nply++;
            if (cfg->progress && nply % cfg->progress == 0)
                fprintf(stderr, "    game %-3llu ply %4d  children %5d  positions measured %ld"
                                "   %.1f min\r",
                        (unsigned long long)(first_game + gi), nply, g.n, positions,
                        (now_s() - t0) / 60.0);
            if (p->royals[1 - mover] < rb[1 - mover] || p->royals[mover] < rb[mover]) break;
            if (cfg->rep_limit > 0 && rep_bump(rep, p->hash) >= cfg->rep_limit) break;
        }
        if (cfg->progress) fprintf(stderr, "\r%*s\r", 100, "");
        fprintf(stderr, "  game %llu: %d plies, %ld positions measured so far (%.1f min)\n",
                (unsigned long long)(first_game + gi), nply, positions, (now_s() - t0) / 60.0);
    }

    if (!positions) { fprintf(stderr, "no positions measured\n"); return 1; }

    printf("\n%ld positions from %llu games, %.0f children each on average "
           "(%ld forced King captures skipped)\n",
           positions, (unsigned long long)games, full_n / positions, skipped_forced);
    printf("full evaluation, averaged over positions: best %+.4f  mean %+.4f  sd %.4f  "
           "spread %.4f  p90 %+.4f  p95 %+.4f\n",
           full_best / positions, full_mean / positions, full_sd / positions,
           full_spread / positions, full_p90 / positions, full_p95 / positions);
    printf("  the best child sits %+.4f above the 95th percentile on average. When that gap is\n"
           "  wide the argmax loss is all-or-nothing, and top-1 agreement says more than the mean.\n",
           full_gap / positions);
    if (verify_dev >= 0)
        printf("verify: a real subset-sized GPU batch reproduced the read-out values to %.2e "
               "over %ld positions\n", verify_dev, verified);

    printf("\n  frac   evals   top-1    mean     sd      p50     p90     p95     max    "
           "policy   rank   sd of\n");
    printf(  "         /pos    agree    loss    loss     loss    loss    loss    loss    "
           "loss    of pick subset\n");
    for (int q = 0; q < nprops; q++) {
        PropAcc *a = &acc[q];
        if (!a->n) continue;
        qsort(a->loss, (size_t)a->n, sizeof(double), cmp_dbl);
        double m = a->sum_loss / a->n;
        double var = a->sum_loss2 / a->n - m * m;
        printf("  %5.3f  %6.0f  %5.1f%%  %7.4f %7.4f  %7.4f %7.4f %7.4f %7.4f  %7.4f  %6.1f  %6.4f\n",
               a->p, a->sum_k / a->n, 100.0 * a->sum_agree / a->n, m,
               var > 0 ? sqrt(var) : 0.0,
               pctile(a->loss, (int)a->n, 0.50), pctile(a->loss, (int)a->n, 0.90),
               pctile(a->loss, (int)a->n, 0.95), a->max_loss,
               a->sum_pol / a->n, a->sum_rank / a->n, a->sum_sd / a->n);
    }
    printf("\ntop-1 agreement should track frac: the chance a uniform subset holds the best child\n"
           "is k/n, which is the fraction. Departures from that line are the interesting part.\n");
    printf("loss = best value over all children minus best over the sampled subset, in the\n"
           "mover's frame, so it is >= 0 and in the same units as the value head (tanh, [-1,1]).\n"
           "policy loss compares the expected value of sampling at temperature %.3g from the\n"
           "subset against sampling from the whole set -- the argmax loss is its T -> 0 case.\n"
           "rank of pick is how many children were genuinely better than the one the subset chose.\n",
           cfg->temperature);
    if (csv) { fclose(csv); printf("per-position rows written to %s\n", csv_path); }

    for (int q = 0; q < nprops; q++) free(acc[q].loss);
    free(boards); free(sub_boards); free(values); free(sub_vals); free(mv); free(weights);
    free(perm); free(vs_buf); vs_buf = NULL; vs_cap = 0;
    free(g.buf); free(g.stamp); free(g.kline); free(rep); free(base); free(p); free(child);
    return 0;
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
static uint32_t pick_move(Net *net, const Cfg *cfg, const Pick *pk, Pos *p,
                          Scratch *s, Rng *rng, int ply, float *val)
{
    gen_all(p, &s->g);
    if (s->g.n == 0) return 0;
    if (cfg->regicide) {
        king_ctx_set(&s->g, p);
        uint32_t forced = find_king_capture(&s->g);
        if (forced) { if (val) *val = 0.0f; return forced; }
    }
    return choose(net, cfg->king_safety, pk, p, s, rng, temp_at(cfg, ply), val,
                  NULL, NULL);
}

/* Plays from `ply` to a finish, under the same rules as self-play, including the lead
 * count `run` the main line had reached. Returns RES_BLACK, RES_WHITE or RES_DRAW. */
static int playout(Net *net, const Cfg *cfg, const Pick *pk, Pos *p, RepTab *rep,
                   Scratch *s, Rng *rng, int ply, int run)
{
    for (;;) {
        uint32_t pick = pick_move(net, cfg, pk, p, s, rng, ply, NULL);
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

static int mode_variance(Net *net, Cfg *cfg, const Pick *pk, uint64_t games,
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
            100.0 * pk->top, cfg->max_plies, cfg->king_safety ? "on" : "off");

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
            uint32_t pick = pick_move(net, cfg, pk, p, &sc, &rng, ply, &mv_val);
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
                    int r = playout(net, cfg, pk, run, rrun, &sc, &brng, ply, lead_run);
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
    if (strcmp(mode, "play") && strcmp(mode, "eval") && strcmp(mode, "elo")
        && strcmp(mode, "selfcheck") && strcmp(mode, "subsample") && strcmp(mode, "variance")) {
        fputs(USAGE, stderr);
        return 1;
    }

    Cfg cfg = { 3000, 4, 0, 1, 3, 1, 25, 0, 1.0, 0x5EED5EED5EED5EEDull, NULL, NULL, 1, 0, 1.0,
                1, 300, 10, 1, 0.05, 0.02 };
    const char *model = "net_fp16.onnx";
    uint64_t games = 1;
    int device_id = 0, sub_batch = 0, probe_cap = 8192, first_game = 0, tf32 = 1;
    int temperature_set = 0, allow_cpu = 0, games_set = 0, seed_set = 0;
    Sprt sprt = { 1, 0.0, 50.0, 0.05, 0.05, 0.0, 0.0 };
    uint64_t target_positions = 0;
    const char *manifest = "net_manifest.json";
    const char *csv_path = NULL;
    double props[MAX_PROPS] = { 0.01, 0.02, 0.05, 0.10, 0.25, 0.50 };
    int nprops = 6, repeats = 8, stride = 1, verify = 4;
    const char *policy_model = NULL;
    int vs_full = 0;
    int branch_every = 100, branches = 8, max_points = 0;
    const char *baseline = NULL;         /* NULL -> derive from --precision at use */

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
        else if (!strcmp(argv[i], "--subset-fraction")) {
            fprintf(stderr, "--subset-fraction is gone: the value head now scores the policy's top "
                            "--policy-top of the moves (default 0.05) plus --policy-explore at random "
                            "(default 0.02). In elo, --vs-full measures what that costs against "
                            "scoring every child.\n");
            return 1; }
        else if (!strcmp(argv[i], "--policy-top")     && i + 1 < argc) cfg.policy_top = atof(argv[++i]);
        else if (!strcmp(argv[i], "--policy-explore") && i + 1 < argc) cfg.policy_explore = atof(argv[++i]);
        else if (!strcmp(argv[i], "--policy-model")   && i + 1 < argc) policy_model = argv[++i];
        else if (!strcmp(argv[i], "--vs-full"))      vs_full = 1;
        else if (!strcmp(argv[i], "--branch-every")  && i + 1 < argc) branch_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--branches")      && i + 1 < argc) branches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-points")    && i + 1 < argc) max_points = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeats")      && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stride")       && i + 1 < argc) stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")          && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--verify")       && i + 1 < argc) verify = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--proportions")  && i + 1 < argc) {
            const char *v = argv[++i];
            nprops = 0;
            while (*v && nprops < MAX_PROPS) {
                char *end;
                double x = strtod(v, &end);
                if (end == v) break;
                if (x > 0.0 && x <= 1.0) props[nprops++] = x;
                v = (*end == ',') ? end + 1 : end;
            }
            if (!nprops) { fprintf(stderr, "--proportions needs values in (0, 1]\n"); return 1; } }
        else if (!strcmp(argv[i], "--manifest")     && i + 1 < argc) manifest = argv[++i];
        else if (!strcmp(argv[i], "--no-tf32"))     tf32 = 0;
        else if (!strcmp(argv[i], "--no-regicide")) cfg.regicide = 0;
        else if (!strcmp(argv[i], "--no-king-safety")) cfg.king_safety = 0;
        else if (!strcmp(argv[i], "--resign-lead")  && i + 1 < argc) cfg.resign_pawns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--resign-plies") && i + 1 < argc) cfg.resign_plies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cap-draw"))     cfg.cap_by_material = 0;
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
        else { fprintf(stderr, "unknown option %s\n", argv[i]); fputs(USAGE, stderr); return 1; }
    }

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
    if (strcmp(mode, "eval") && strcmp(mode, "selfcheck"))     /* the modes that play games */
        fprintf(stderr, "seed %llu%s\n", (unsigned long long)cfg.seed_base,
                seed_set ? "" : " (random; --seed with this value repeats the run)");

    Net net;
    net_open(&net, model, device_id, sub_batch, probe_cap, tf32, allow_cpu, cfg.verbose);
    if (cfg.policy_top <= 0.0 || cfg.policy_top > 1.0 || cfg.policy_explore < 0.0 || cfg.policy_explore > 0.255) {
        fprintf(stderr, "--policy-top must be in (0, 1] and --policy-explore in [0, 0.255]\n");
        return 1;
    }
    /* The policy graph beside the value graph. play, elo and variance choose moves with
     * it and refuse without it; eval, selfcheck and subsample measure the value graph and
     * use it only if present. */
    char ppath[512];
    policy_path(model, ppath, sizeof(ppath));
    if (policy_model) snprintf(ppath, sizeof(ppath), "%s", policy_model);
    Pol pol;
    int have_pol = pol_open(&pol, &net, ppath, cfg.verbose) == 0;
    int needs_pol = !strcmp(mode, "play") || !strcmp(mode, "elo") || !strcmp(mode, "variance");
    if (needs_pol && !have_pol) {
        fprintf(stderr,
            "%s: no policy graph '%s'.\n"
            "     play, elo and variance pick the children to score with the policy head, so they\n"
            "     need a model_policy.py checkpoint exported with export_onnx.py, which writes it\n"
            "     beside the value graph. A legacy model has no policy head and cannot be used.\n",
            mode, ppath);
        net_close(&net);
        return 1;
    }
    Pick pick_cfg = { have_pol ? &pol : NULL, cfg.policy_top, cfg.policy_explore };
    if (!strcmp(mode, "eval")) { int rc = mode_eval(&net); net_close(&net); return rc; }
    if (!strcmp(mode, "selfcheck")) {
        int rc = mode_selfcheck(&net, have_pol ? &pol : NULL, manifest);
        if (have_pol) pol_close(&pol);
        net_close(&net);
        return rc;
    }
    if (!strcmp(mode, "variance")) {
        int rc = mode_variance(&net, &cfg, &pick_cfg, games, first_game,
                               branch_every, branches, max_points, csv_path);
        pol_close(&pol);
        net_close(&net);
        return rc;
    }
    if (!strcmp(mode, "subsample")) {
        int rc = mode_subsample(&net, &cfg, games, first_game, props, nprops,
                                repeats, stride, csv_path, verify);
        net_close(&net);
        return rc;
    }
    if (!strcmp(mode, "elo")) {
        /* Greedy measures strength, but a kept match is training data, and greedy games
         * are the least varied there are; --out samples unless told otherwise. */
        if (!temperature_set) cfg.temperature = cfg.out_path ? 0.3 : 0.0;

        /* The baseline defaults to the graph exported from baseline.pt, matching the
         * precision the bot is running. "random" asks for the uniform mover instead, which
         * is the absolute anchor but stops discriminating once the bot beats it every
         * game. */
        char bpath[512];
        Net opp;
        Net *oppp = NULL;
        Pol opol;
        int opp_opened = 0, opol_opened = 0;
        const char *opp_name = "uniform-random";
        Pick opp_pick = { NULL, 1.0, 0.0 };       /* every child: the --vs-full opponent */

        if (vs_full) {
            /* The net against itself, one session driving both sides: the rated side
             * scores the policy's picks, the opponent scores everything. No baseline graph
             * is involved, so this isolates what the pruning costs from any change in
             * weights. */
            if (baseline) { fprintf(stderr, "elo: --vs-full plays the net against itself; drop --baseline\n"); return 1; }
            oppp = &net;
            opp_name = "itself scoring every child";
        } else if (1) {
        if (!baseline) {
            snprintf(bpath, sizeof(bpath), "baseline_%s.onnx", net.fp16_out ? "fp16" : "fp32");
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
            oppp = &opp;
            opp_opened = 1;
            opp_name = baseline;
            char bp[512];
            policy_path(baseline, bp, sizeof(bp));
            if (pol_open(&opol, &opp, bp, cfg.verbose) != 0) {
                fprintf(stderr, "elo: the baseline has no policy graph '%s'. A legacy model cannot be "
                                "a baseline; export a model_policy.py checkpoint as baseline.pt, or "
                                "use --baseline random or --vs-full.\n", bp);
                net_close(&opp);
                pol_close(&pol);
                net_close(&net);
                return 1;
            }
            opol_opened = 1;
            opp_pick = (Pick){ &opol, cfg.policy_top, cfg.policy_explore };
        }
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
        int rc = mode_elo(&net, oppp, opp_name, &cfg, games, first_game, &pick_cfg, &opp_pick, &sprt,
                          model);
        if (opol_opened) pol_close(&opol);
        if (opp_opened) net_close(oppp);
        pol_close(&pol);
        net_close(&net);
        return rc;
    }

    Pos *base = malloc(sizeof(Pos)), *p = malloc(sizeof(Pos));
    pos_init(base);
    Scratch sc;
    scratch_init(&sc);
    Gen *g = &sc.g;
    RepTab *rep = calloc(1, sizeof(RepTab));
    PlyRecNN *plybuf = malloc(sizeof(PlyRecNN) * (size_t)(cfg.max_plies + 2));

    Sink sink = { -1, NULL, 0, 1u << 22 };
    if (cfg.out_path && tkn_open(&sink, &cfg, &net, model)) return 1;
    Kids kids;
    if (cfg.children_path) {
        if (!cfg.out_path) { fprintf(stderr, "--children needs --out: its games pair with the .tkn's\n"); return 1; }
        if (kids_open(&kids, cfg.children_path)) return 1;
    }

    int mat_div = VALUE_FP >> cfg.mat_shift;
    if (mat_div < 1) mat_div = 1;

    GpuMon mon;
    memset(&mon, 0, sizeof(mon));      /* gpumon_stop runs unconditionally at exit */
    int have_nvml = cfg.stats && gpumon_start(&mon, device_id);
    if (cfg.stats && !have_nvml)
        fprintf(stderr, "  (libnvidia-ml.so.1 not available: no GPU utilisation numbers)\n");

    uint64_t total_plies = 0, total_evals = 0, games_done = 0, wins[3] = { 0 };
    double t_start = now_s(), t_nn = 0.0, t_gen = 0.0, t_enc = 0.0, t_safe = 0.0, t_pol = 0.0;
    double w_nn = 0.0, w_gen = 0.0, w_enc = 0.0, w_t0 = now_s();   /* since the last line */
    uint64_t w_evals = 0;

    for (uint64_t gi = 0; ; gi++) {
        /* --positions plays whole games until the target is met, so it overshoots rather
         * than truncating the last one: a cut-short game would be written with a result it
         * never reached, and the value target would be a lie. */
        if (target_positions) {
            if (total_plies >= target_positions) break;
            if (games_set && gi >= games) break;
        } else if (gi >= games) {
            break;
        }
        uint64_t seed = cfg.seed_base ^ (((uint64_t)first_game + gi) * 0x9E3779B97F4A7C15ull);
        Rng rng; rng_seed(&rng, seed);
        memcpy(p, base, sizeof(Pos));
        rep->epoch++;
        rep_bump(rep, p->hash);

        int nply = 0, no_prog = 0, result = RES_DRAW, term = TERM_PLYCAP, lead_run = 0, adj;
        double t_game = now_s();

        for (;;) {
            double tg = now_s();
            gen_all(p, g);
            t_gen += now_s() - tg; w_gen += now_s() - tg;
            if (g->n == 0) { result = cfg.stalemate_loses ? (1 - p->turn) : RES_DRAW; term = TERM_STALEMATE; break; }
            if (nply >= cfg.max_plies) { result = cap_result(&cfg, p); term = TERM_PLYCAP; break; }

            /* Forced King capture, exactly tky.c's rule. The chosen leaf is still put
             * through the network, one board, so every ply carries a real value. */
            uint32_t forced = 0;
            if (cfg.regicide) { king_ctx_set(g, p); forced = find_king_capture(g); }

            /* --subset-fraction scores only a random slice of the children and picks from
             * that slice, so the saving is real. Both sides subset: self-play has no rated
             * side. King safety sits on top: see choose(). */
            double enc0 = sc.t_enc, nn0 = sc.t_nn, safe0 = sc.t_safe, pol0 = sc.t_pol;
            uint32_t pick;
            float chosen;
            int ncand, overridden = 0;
            if (forced) {
                scratch_reserve(&sc, 1);
                double te = now_s();
                memcpy(sc.child, p, sizeof(Pos));
                do_move(sc.child, SQ2P[UNPACK_O(forced)], UNPACK_M(forced));
                encode_stm(sc.child, sc.boards);
                double t0 = now_s();
                net_eval(&net, sc.boards, 1, sc.values);
                sc.t_enc += t0 - te;
                sc.t_nn += now_s() - t0;
                pick = forced;
                chosen = sc.values[0];
                ncand = 1;
            } else {
                pick = choose(&net, cfg.king_safety, &pick_cfg, p, &sc, &rng,
                              temp_at(&cfg, nply), &chosen, &ncand, &overridden);
            }
            if (cfg.children_path) {
                if (forced) kids_ply(&kids, &pick, NULL, &chosen, 1);
                else        kids_ply(&kids, sc.g.buf, sc.perm, sc.values, ncand);
            }
            t_enc += sc.t_enc - enc0; w_enc += sc.t_enc - enc0;
            t_nn += sc.t_nn - nn0;    w_nn += sc.t_nn - nn0;
            t_safe += sc.t_safe - safe0;
            t_pol += sc.t_pol - pol0;
            total_evals += (uint64_t)ncand;
            w_evals += (uint64_t)ncand;

            if (cfg.progress && nply % cfg.progress == 0) {
                fprintf(stderr, "    ply %4d  moves %5d  value %+0.4f  material %+8.1f  "
                                "pieces %3d  %.1f boards/s\n",
                        nply, g->n, (double)chosen, p->material / (double)VALUE_FP,
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

            tkn_fill(&plybuf[nply], p, pick, chosen, g->n, mover,
                     (captured > 0 ? PF_CAPTURE : 0) | (promoted_now ? PF_PROMOTION : 0)
                     | (forced ? PF_FORCED_KING : 0) | (overridden ? PF_KING_SAFETY : 0),
                     ncand, mat_div);
            nply++;
            no_prog = (captured > 0 || promoted_now) ? 0 : no_prog + 1;

            if (p->royals[1 - mover] < royals_before[1 - mover])      { result = mover;     term = TERM_ROYAL; }
            else if (p->royals[mover] < royals_before[mover])         { result = 1 - mover; term = TERM_ROYAL; }
            else {
                int seen = rep_bump(rep, p->hash);
                if (cfg.rep_limit > 0 && seen >= cfg.rep_limit)                       { result = RES_DRAW; term = TERM_REPETITION; }
                else if (cfg.no_progress_limit > 0 && no_prog >= cfg.no_progress_limit) { result = RES_DRAW; term = TERM_NOPROGRESS; }
                else if ((adj = resign_check(&cfg, p, &lead_run)) >= 0)                   { result = adj;      term = TERM_RESIGN; }
                else continue;
            }
            break;
        }
        total_plies += (uint64_t)nply;
        games_done++;
        wins[result]++;

        if (cfg.children_path) kids_game(&kids, plybuf, nply, seed);
        if (cfg.out_path) tkn_put_game(&sink, plybuf, nply, result, term, seed);
        if (cfg.verbose) {
            static const char *tn[] = { "royal", "stalemate", "repetition", "no-progress", "ply-cap",
                                        "resignation" };
            /* Count against whichever limit is in force. --first-game only shifts the
             * per-game seed and is not stored anywhere, so name the absolute index when
             * it is set or a game cannot be reproduced from the log alone. */
            char head[96], from[48] = "";
            if (first_game)
                snprintf(from, sizeof(from), " (game index %llu)",
                         (unsigned long long)(first_game + gi));
            if (target_positions)
                snprintf(head, sizeof(head), "Game %llu, Positions %llu/%llu",
                         (unsigned long long)(gi + 1), (unsigned long long)total_plies,
                         (unsigned long long)target_positions);
            else
                snprintf(head, sizeof(head), "Game %llu/%llu",
                         (unsigned long long)(gi + 1), (unsigned long long)games);
            /* Extrapolate from the whole run so far, against whichever limit is in force.
             * Per position rather than per game when counting positions, since game
             * lengths vary by an order of magnitude and the position rate is far steadier. */
            char eta[48] = "";
            double elapsed = now_s() - t_start, remain = 0.0;
            if (target_positions && total_plies && total_plies < target_positions)
                remain = elapsed / (double)total_plies * (double)(target_positions - total_plies);
            else if (!target_positions && gi + 1 < games)
                remain = elapsed / (double)(gi + 1) * (double)(games - gi - 1);
            if (remain > 0.0) {
                char d[32];
                fmt_dur(remain, d, sizeof(d));
                snprintf(eta, sizeof(eta), "  ETA %s", d);
            }
            fprintf(stderr, "%s%s: %d plies, %s by %s, %.1fs (%.0f boards/s)%s\n",
                    head, from, nply,
                    result == RES_BLACK ? "black" : result == RES_WHITE ? "white" : "draw",
                    tn[term], now_s() - t_game,
                    total_evals / (t_nn > 0 ? t_nn : 1.0), eta);
        }
    }

    if (cfg.out_path) tkn_close(&sink, games_done, total_plies);
    if (cfg.children_path) kids_close(&kids);

    double dt = now_s() - t_start;
    fprintf(stderr,
        "games=%llu plies=%llu (%.1f/game) evals=%llu\n"
        "  %.1fs total, %.1fs in the network (%.0f%%), %.0f boards/s, %.2f s/ply\n"
        "  policy %.3fs (%.2f%%)  movegen %.3fs (%.2f%%)  encode %.3fs (%.2f%%)  king safety "
        "%.3fs (%.2f%%)  other %.3fs (%.2f%%)\n"
        "  black=%llu white=%llu draw=%llu\n",
        (unsigned long long)games_done, (unsigned long long)total_plies,
        games_done ? (double)total_plies / games_done : 0.0, (unsigned long long)total_evals,
        dt, t_nn, 100.0 * t_nn / (dt > 0 ? dt : 1.0),
        t_nn > 0 ? total_evals / t_nn : 0.0, total_plies ? dt / total_plies : 0.0,
        t_pol, 100 * t_pol / dt, t_gen, 100 * t_gen / dt, t_enc, 100 * t_enc / dt, t_safe,
        100 * t_safe / dt, dt - t_nn - t_gen - t_enc - t_safe - t_pol,
        100 * (dt - t_nn - t_gen - t_enc - t_safe - t_pol) / dt,
        (unsigned long long)wins[RES_BLACK], (unsigned long long)wins[RES_WHITE],
        (unsigned long long)wins[RES_DRAW]);
    if (cfg.king_safety)
        fprintf(stderr, "  king safety changed the pick on %llu plies (%.2f%%); %llu plies had no "
                        "safe move at all\n",
                (unsigned long long)sc.overridden,
                total_plies ? 100.0 * sc.overridden / total_plies : 0.0,
                (unsigned long long)sc.hopeless);
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

    pol_close(&pol);
    free(plybuf); scratch_free(&sc);
    free(rep); free(base); free(p); free(sink.buf);
    net_close(&net);
    return 0;
}
