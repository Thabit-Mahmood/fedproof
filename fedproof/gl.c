/* Goldilocks kernels: p = 2^64 - 2^32 + 1. Inputs and outputs are canonical (< p). */
#include <stdint.h>
#include <stddef.h>
typedef unsigned __int128 u128;
#define P 0xFFFFFFFF00000001ULL
#define EPS 0xFFFFFFFFULL

static inline uint64_t red(u128 x) {
    uint64_t lo = (uint64_t)x, hi = (uint64_t)(x >> 64);
    uint64_t hh = hi >> 32, hl = hi & EPS, t0, r;
    if (__builtin_sub_overflow(lo, hh, &t0)) t0 -= EPS;
    uint64_t t1 = hl * EPS;
    if (__builtin_add_overflow(t0, t1, &r)) r += EPS;
    if (r >= P) r -= P;
    return r;
}
static inline uint64_t mulm(uint64_t a, uint64_t b) { return red((u128)a * b); }
static inline uint64_t addm(uint64_t a, uint64_t b) {
    uint64_t s;
    if (__builtin_add_overflow(a, b, &s)) s += EPS;
    if (s >= P) s -= P;
    return s;
}
static inline uint64_t subm(uint64_t a, uint64_t b) { return a >= b ? a - b : a - b + P; }

void gl_mul(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n) {
    #pragma omp parallel for schedule(static) if (n > 65536)
    for (size_t i = 0; i < n; i++) out[i] = mulm(a[i], b[i]);
}
void gl_mul_scalar(const uint64_t *a, uint64_t c, uint64_t *out, size_t n) {
    #pragma omp parallel for schedule(static) if (n > 65536)
    for (size_t i = 0; i < n; i++) out[i] = mulm(a[i], c);
}
/* rows x n matrix times a length-n vector, elementwise per row */
void gl_mul_rowvec(const uint64_t *a, const uint64_t *v, uint64_t *out, size_t rows, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; r++)
        for (size_t i = 0; i < n; i++) out[r * n + i] = mulm(a[r * n + i], v[i]);
}
/* in-place NTT of each row. tw[m + j] = w_{2m}^j for j < m, m = 1, 2, 4, ..., n/2. rev = bit reversal. */
void gl_ntt(uint64_t *x, size_t rows, size_t n, const uint64_t *tw, const int64_t *rev, uint64_t scale) {
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; r++) {
        uint64_t *a = x + r * n;
        for (size_t i = 0; i < n; i++) {
            size_t j = (size_t)rev[i];
            if (i < j) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
        }
        for (size_t m = 1; m < n; m <<= 1) {
            const uint64_t *w = tw + m;
            for (size_t s = 0; s < n; s += 2 * m)
                for (size_t j = 0; j < m; j++) {
                    uint64_t u = a[s + j], v = mulm(a[s + j + m], w[j]);
                    a[s + j] = addm(u, v);
                    a[s + j + m] = subm(u, v);
                }
        }
        if (scale != 1) for (size_t i = 0; i < n; i++) a[i] = mulm(a[i], scale);
    }
}
/* out[j] = sum_i c[i] * M[i][j] */
void gl_lincomb(const uint64_t *M, const uint64_t *c, uint64_t *out, size_t rows, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n; j++) {
        uint64_t acc = 0;
        for (size_t i = 0; i < rows; i++) acc = addm(acc, mulm(c[i], M[i * n + j]));
        out[j] = acc;
    }
}
/* acc[j] += sum_i A[i][j] * B[i][j] */
void gl_muladd_rows(const uint64_t *A, const uint64_t *B, uint64_t *acc, size_t rows, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < n; j++) {
        uint64_t s = acc[j];
        for (size_t i = 0; i < rows; i++) s = addm(s, mulm(A[i * n + j], B[i * n + j]));
        acc[j] = s;
    }
}
/* out[idx[i]] += vals[i] */
void gl_scatter(uint64_t *out, const int64_t *idx, const uint64_t *vals, size_t n) {
    for (size_t i = 0; i < n; i++) out[idx[i]] = addm(out[idx[i]], vals[i]);
}
/* out[idx[i]] += a[i] * b[i] */
void gl_scatter_mul(uint64_t *out, const int64_t *idx, const uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) out[idx[i]] = addm(out[idx[i]], mulm(a[i], b[i]));
}
/* out[i] = sum_j A[i][j] * v[j]  (matrix-vector) */
void gl_matvec(const uint64_t *A, const uint64_t *v, uint64_t *out, size_t rows, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; i++) {
        uint64_t s = 0;
        for (size_t j = 0; j < n; j++) s = addm(s, mulm(A[i * n + j], v[j]));
        out[i] = s;
    }
}
