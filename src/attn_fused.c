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

#include <immintrin.h>

#define TILE_ROWS 6
#define TILE_COLS 16

typedef struct {
    int N;
    int d;
    int key_panels;
    int value_panels;
    int score_stride;
    int padded_d;
} Shape;

static int min_int(int a, int b)
{
    return a < b ? a : b;
}

static Shape make_shape(int N, int d)
{
    Shape shape;
    shape.N = N;
    shape.d = d;
    shape.key_panels = (N + TILE_COLS - 1) / TILE_COLS;
    shape.value_panels = (d + TILE_COLS - 1) / TILE_COLS;
    shape.score_stride = (shape.key_panels + 1) * TILE_COLS;
    shape.padded_d = shape.value_panels * TILE_COLS;
    return shape;
}

static float *allocate_floats(size_t count)
{
    void *memory;
    if (posix_memalign(&memory, 64, count * sizeof(float)))
        abort();
    return memory;
}

static float *allocate_keys(Shape shape)
{
    return allocate_floats((size_t)shape.key_panels * shape.d * TILE_COLS);
}

static float *allocate_values(Shape shape)
{
    return allocate_floats((size_t)shape.value_panels * shape.N * TILE_COLS);
}

static const float *key_panel(const float *keys, Shape shape, int panel)
{
    return keys + (size_t)panel * shape.d * TILE_COLS;
}

static const float *value_panel(const float *values, Shape shape, int panel, int first_key)
{
    return values + ((size_t)panel * shape.N + first_key) * TILE_COLS;
}

static void pack_key_panel(const float *K, float *keys, Shape shape, int panel)
{
    float *packed = keys + (size_t)panel * shape.d * TILE_COLS;
    for (int c = 0; c < TILE_COLS; c++) {
        const int j = panel * TILE_COLS + c;
        for (int k = 0; k < shape.d; k++)
            packed[k * TILE_COLS + c] = j < shape.N ? K[(size_t)j * shape.d + k] : 0.0f;
    }
}

static void pack_value_panel(const float *V, float *values, Shape shape, int panel)
{
    float *packed = values + (size_t)panel * shape.N * TILE_COLS;
    for (int j = 0; j < shape.N; j++)
        for (int c = 0; c < TILE_COLS; c++) {
            const int column = panel * TILE_COLS + c;
            packed[(size_t)j * TILE_COLS + c] = column < shape.d ? V[(size_t)j * shape.d + column] : 0.0f;
        }
}

static inline __attribute__((always_inline)) void tile_kernel(const int rows, const float *restrict A, int lda,
                                                              const float *restrict B, int depth,
                                                              float *restrict C, int ldc, int accumulate)
{
    __m256 left[TILE_ROWS], right[TILE_ROWS];

    for (int r = 0; r < rows; r++) {
        left[r] = accumulate ? _mm256_loadu_ps(C + (size_t)r * ldc) : _mm256_setzero_ps();
        right[r] = accumulate ? _mm256_loadu_ps(C + (size_t)r * ldc + 8) : _mm256_setzero_ps();
    }

    for (int k = 0; k < depth; k++) {
        const __m256 b_left = _mm256_loadu_ps(B + (size_t)k * TILE_COLS);
        const __m256 b_right = _mm256_loadu_ps(B + (size_t)k * TILE_COLS + 8);
        for (int r = 0; r < rows; r++) {
            const __m256 a = _mm256_broadcast_ss(A + (size_t)r * lda + k);
            left[r] = _mm256_fmadd_ps(a, b_left, left[r]);
            right[r] = _mm256_fmadd_ps(a, b_right, right[r]);
        }
    }

    for (int r = 0; r < rows; r++) {
        _mm256_storeu_ps(C + (size_t)r * ldc, left[r]);
        _mm256_storeu_ps(C + (size_t)r * ldc + 8, right[r]);
    }
}

static void multiply_tile(int rows, const float *A, int lda, const float *B, int depth, float *C, int ldc,
                          int accumulate)
{
    switch (rows) {
    case 6: tile_kernel(6, A, lda, B, depth, C, ldc, accumulate); break;
    case 5: tile_kernel(5, A, lda, B, depth, C, ldc, accumulate); break;
    case 4: tile_kernel(4, A, lda, B, depth, C, ldc, accumulate); break;
    case 3: tile_kernel(3, A, lda, B, depth, C, ldc, accumulate); break;
    case 2: tile_kernel(2, A, lda, B, depth, C, ldc, accumulate); break;
    default: tile_kernel(1, A, lda, B, depth, C, ldc, accumulate); break;
    }
}

static float *pack_keys(const float *K, Shape shape)
{
    float *keys = allocate_keys(shape);
    for (int panel = 0; panel < shape.key_panels; panel++)
        pack_key_panel(K, keys, shape, panel);
    return keys;
}

static float *pack_values(const float *V, Shape shape)
{
    float *values = allocate_values(shape);
    for (int panel = 0; panel < shape.value_panels; panel++)
        pack_value_panel(V, values, shape, panel);
    return values;
}

static void softmax_row(float *row, int N, float scale)
{
    float max = row[0];
    for (int j = 1; j < N; j++)
        if (row[j] > max)
            max = row[j];

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        row[j] = expf((row[j] - max) * scale);
        sum += row[j];
    }

    const float inverse_sum = 1.0f / sum;
    for (int j = 0; j < N; j++)
        row[j] *= inverse_sum;
}

#define KEYS_PER_BLOCK 256

static void compute_row_scores(const float *Q, const float *keys, float *scores, int rows, Shape shape)
{
    for (int panel = 0; panel < shape.key_panels; panel++)
        multiply_tile(rows, Q, shape.d, key_panel(keys, shape, panel), shape.d, scores + panel * TILE_COLS,
                      shape.score_stride, 0);
}

static void apply_row_softmax(float *scores, int rows, Shape shape)
{
    const float scale = 1.0f / sqrtf((float)shape.d);
    for (int r = 0; r < rows; r++)
        softmax_row(scores + (size_t)r * shape.score_stride, shape.N, scale);
}

static void compute_row_output(const float *scores, const float *values, float *output, int rows, Shape shape)
{
    for (int first_key = 0; first_key < shape.N; first_key += KEYS_PER_BLOCK) {
        const int keys = min_int(KEYS_PER_BLOCK, shape.N - first_key);
        for (int panel = 0; panel < shape.value_panels; panel++)
            multiply_tile(rows, scores + first_key, shape.score_stride,
                          value_panel(values, shape, panel, first_key), keys, output + panel * TILE_COLS,
                          shape.padded_d, first_key > 0);
    }
}

static void write_rows(const float *output, float *O, int rows, Shape shape)
{
    for (int r = 0; r < rows; r++)
        memcpy(O + (size_t)r * shape.d, output + (size_t)r * shape.padded_d, (size_t)shape.d * sizeof(float));
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const Shape shape = make_shape(N, d);
    float *scores = allocate_floats((size_t)TILE_ROWS * shape.score_stride);
    float *output = allocate_floats((size_t)TILE_ROWS * shape.padded_d);
    float *keys = pack_keys(K, shape);
    float *values = pack_values(V, shape);

    for (int i = 0; i < N; i += TILE_ROWS) {
        const int rows = min_int(TILE_ROWS, N - i);
        compute_row_scores(Q + (size_t)i * d, keys, scores, rows, shape);
        apply_row_softmax(scores, rows, shape);
        compute_row_output(scores, values, output, rows, shape);
        write_rows(output, O + (size_t)i * d, rows, shape);
    }

    free(scores);
    free(output);
    free(keys);
    free(values);
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
