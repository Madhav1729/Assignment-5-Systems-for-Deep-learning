#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static float *transpose_keys(const float *K, int N, int d)
{
    float *Kt = malloc((size_t)N * d * sizeof(float));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < d; k++)
            Kt[(size_t)k * N + j] = K[(size_t)j * d + k];
    return Kt;
}

static void compute_score_row(const float *restrict q, const float *restrict Kt, float *restrict row, int N, int d,
                              float scale)
{
    for (int j = 0; j < N; j++)
        row[j] = 0.0f;

    for (int k = 0; k < d; k++) {
        const float query = q[k] * scale;
        const float *restrict keys = Kt + (size_t)k * N;
        for (int j = 0; j < N; j++)
            row[j] += query * keys[j];
    }
}

static void softmax_row(float *restrict row, int N)
{
    float max = row[0];
    for (int j = 1; j < N; j++)
        if (row[j] > max)
            max = row[j];

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        row[j] = expf(row[j] - max);
        sum += row[j];
    }

    const float inverse_sum = 1.0f / sum;
    for (int j = 0; j < N; j++)
        row[j] *= inverse_sum;
}

static void compute_output_row(const float *restrict row, const float *restrict V, float *restrict o, int N, int d)
{
    for (int k = 0; k < d; k++)
        o[k] = 0.0f;

    for (int j = 0; j < N; j++) {
        const float weight = row[j];
        const float *restrict v = V + (size_t)j * d;
        for (int k = 0; k < d; k++)
            o[k] += weight * v[k];
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const float scale = 1.0f / sqrtf((float)d);
    float *S = malloc((size_t)N * N * sizeof(float));
    float *Kt = transpose_keys(K, N, d);

    for (int i = 0; i < N; i++)
        compute_score_row(Q + (size_t)i * d, Kt, S + (size_t)i * N, N, d, scale);

    for (int i = 0; i < N; i++)
        softmax_row(S + (size_t)i * N, N);

    for (int i = 0; i < N; i++)
        compute_output_row(S + (size_t)i * N, V, O + (size_t)i * d, N, d);

    free(S);
    free(Kt);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
