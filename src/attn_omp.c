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

#define BLOCK_ROWS 48
#define CHUNK_KEYS 128
#define SCORE_STRIDE (CHUNK_KEYS + TILE_COLS)
#define LOG2_E 1.44269504f

typedef struct {
    const float *queries;
    const float *keys;
    const float *values;
    float *O;
    Shape shape;
} Problem;

typedef struct {
    float *scores;
    float *output;
    float row_max[BLOCK_ROWS];
    float row_sum[BLOCK_ROWS];
} Workspace;

static Workspace create_workspace(Shape shape)
{
    Workspace workspace;
    workspace.scores = allocate_floats((size_t)BLOCK_ROWS * SCORE_STRIDE);
    workspace.output = allocate_floats((size_t)BLOCK_ROWS * shape.padded_d);
    return workspace;
}

static void destroy_workspace(Workspace *workspace)
{
    free(workspace->scores);
    free(workspace->output);
}

static void reset_workspace(Workspace *workspace, int rows, Shape shape)
{
    for (int r = 0; r < rows; r++) {
        workspace->row_max[r] = -1e30f;
        workspace->row_sum[r] = 0.0f;
    }
    memset(workspace->output, 0, (size_t)rows * shape.padded_d * sizeof(float));
}

static inline __m256 exp2_vector(__m256 x)
{
    x = _mm256_max_ps(x, _mm256_set1_ps(-126.0f));
    const __m256 whole = _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 fraction = _mm256_sub_ps(x, whole);

    __m256 result = _mm256_set1_ps(1.5403530e-4f);
    result = _mm256_fmadd_ps(result, fraction, _mm256_set1_ps(1.3333558e-3f));
    result = _mm256_fmadd_ps(result, fraction, _mm256_set1_ps(9.6181291e-3f));
    result = _mm256_fmadd_ps(result, fraction, _mm256_set1_ps(5.5504109e-2f));
    result = _mm256_fmadd_ps(result, fraction, _mm256_set1_ps(2.4022651e-1f));
    result = _mm256_fmadd_ps(result, fraction, _mm256_set1_ps(6.9314718e-1f));
    result = _mm256_fmadd_ps(result, fraction, _mm256_set1_ps(1.0f));

    const __m256i biased = _mm256_add_epi32(_mm256_cvtps_epi32(whole), _mm256_set1_epi32(127));
    const __m256i exponent = _mm256_slli_epi32(biased, 23);
    return _mm256_mul_ps(result, _mm256_castsi256_ps(exponent));
}

static inline float horizontal_max(__m256 v)
{
    __m128 x = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_max_ps(x, _mm_movehl_ps(x, x));
    return _mm_cvtss_f32(_mm_max_ss(x, _mm_shuffle_ps(x, x, 1)));
}

static inline float horizontal_sum(__m256 v)
{
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    return _mm_cvtss_f32(_mm_add_ss(x, _mm_shuffle_ps(x, x, 1)));
}

static inline float row_maximum(const float *row, int count, float start)
{
    const int vector_end = count & ~7;
    __m256 maximum = _mm256_set1_ps(start);
    for (int j = 0; j < vector_end; j += 8)
        maximum = _mm256_max_ps(maximum, _mm256_loadu_ps(row + j));

    float result = horizontal_max(maximum);
    for (int j = vector_end; j < count; j++)
        result = row[j] > result ? row[j] : result;
    return result;
}

static inline float exponentiate_row(float *row, int count, float maximum)
{
    const int vector_end = count & ~7;
    const __m256 shift = _mm256_set1_ps(maximum);
    __m256 sums = _mm256_setzero_ps();
    for (int j = 0; j < vector_end; j += 8) {
        const __m256 value = exp2_vector(_mm256_sub_ps(_mm256_loadu_ps(row + j), shift));
        _mm256_storeu_ps(row + j, value);
        sums = _mm256_add_ps(sums, value);
    }

    float sum = horizontal_sum(sums);
    for (int j = vector_end; j < count; j++) {
        row[j] = exp2f(row[j] - maximum);
        sum += row[j];
    }
    return sum;
}

static inline void scale_row(float *row, int count, float factor)
{
    const __m256 scale = _mm256_set1_ps(factor);
    for (int k = 0; k < count; k += 8)
        _mm256_storeu_ps(row + k, _mm256_mul_ps(scale, _mm256_loadu_ps(row + k)));
}

static void compute_chunk_scores(const Problem *problem, Workspace *workspace, int first_row, int rows,
                                 int first_key, int keys)
{
    const Shape shape = problem->shape;
    const int panels = (keys + TILE_COLS - 1) / TILE_COLS;

    for (int panel = 0; panel < panels; panel++) {
        const float *keys_panel = key_panel(problem->keys, shape, first_key / TILE_COLS + panel);
        for (int r = 0; r < rows; r += TILE_ROWS)
            multiply_tile(min_int(TILE_ROWS, rows - r), problem->queries + (size_t)(first_row + r) * shape.d,
                          shape.d, keys_panel, shape.d,
                          workspace->scores + (size_t)r * SCORE_STRIDE + panel * TILE_COLS, SCORE_STRIDE, 0);
    }
}

static void update_row_softmax(Workspace *workspace, int r, int keys, Shape shape)
{
    float *row = workspace->scores + (size_t)r * SCORE_STRIDE;
    const float old_max = workspace->row_max[r];
    const float new_max = row_maximum(row, keys, old_max);
    const float correction = exp2f(old_max - new_max);

    workspace->row_max[r] = new_max;
    workspace->row_sum[r] = workspace->row_sum[r] * correction + exponentiate_row(row, keys, new_max);

    if (correction != 1.0f)
        scale_row(workspace->output + (size_t)r * shape.padded_d, shape.padded_d, correction);
}

static void accumulate_chunk_output(const Problem *problem, Workspace *workspace, int rows, int first_key, int keys)
{
    const Shape shape = problem->shape;

    for (int panel = 0; panel < shape.value_panels; panel++) {
        const float *values_panel = value_panel(problem->values, shape, panel, first_key);
        for (int r = 0; r < rows; r += TILE_ROWS)
            multiply_tile(min_int(TILE_ROWS, rows - r), workspace->scores + (size_t)r * SCORE_STRIDE, SCORE_STRIDE,
                          values_panel, keys, workspace->output + (size_t)r * shape.padded_d + panel * TILE_COLS,
                          shape.padded_d, 1);
    }
}

static void write_block_output(const Problem *problem, const Workspace *workspace, int first_row, int rows)
{
    const Shape shape = problem->shape;

    for (int r = 0; r < rows; r++) {
        const float inverse_sum = 1.0f / workspace->row_sum[r];
        for (int k = 0; k < shape.d; k++)
            problem->O[(size_t)(first_row + r) * shape.d + k] =
                workspace->output[(size_t)r * shape.padded_d + k] * inverse_sum;
    }
}

static void process_block(const Problem *problem, Workspace *workspace, int first_row, int rows)
{
    const Shape shape = problem->shape;
    reset_workspace(workspace, rows, shape);

    for (int first_key = 0; first_key < shape.N; first_key += CHUNK_KEYS) {
        const int keys = min_int(CHUNK_KEYS, shape.N - first_key);

        compute_chunk_scores(problem, workspace, first_row, rows, first_key, keys);
        for (int r = 0; r < rows; r++)
            update_row_softmax(workspace, r, keys, shape);
        accumulate_chunk_output(problem, workspace, rows, first_key, keys);
    }

    write_block_output(problem, workspace, first_row, rows);
}

static float *scale_queries(const float *Q, Shape shape)
{
    const float factor = LOG2_E / sqrtf((float)shape.d);
    const size_t count = (size_t)shape.N * shape.d;
    float *queries = allocate_floats(count);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i++)
        queries[i] = Q[i] * factor;
    return queries;
}

static float *pack_keys(const float *K, Shape shape)
{
    float *keys = allocate_keys(shape);
#pragma omp parallel for schedule(static)
    for (int panel = 0; panel < shape.key_panels; panel++)
        pack_key_panel(K, keys, shape, panel);
    return keys;
}

static float *pack_values(const float *V, Shape shape)
{
    float *values = allocate_values(shape);
#pragma omp parallel for schedule(static)
    for (int panel = 0; panel < shape.value_panels; panel++)
        pack_value_panel(V, values, shape, panel);
    return values;
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    const Shape shape = make_shape(N, d);
    float *queries = scale_queries(Q, shape);
    float *keys = pack_keys(K, shape);
    float *values = pack_values(V, shape);
    const Problem problem = {queries, keys, values, O, shape};

#pragma omp parallel
    {
        Workspace workspace = create_workspace(shape);

#pragma omp for schedule(dynamic, 1)
        for (int first_row = 0; first_row < N; first_row += BLOCK_ROWS)
            process_block(&problem, &workspace, first_row, min_int(BLOCK_ROWS, N - first_row));

        destroy_workspace(&workspace);
    }

    free(queries);
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
