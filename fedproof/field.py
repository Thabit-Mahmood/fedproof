"""Goldilocks field arithmetic (p = 2^64 - 2^32 + 1) on numpy uint64 arrays, NTT, coset evaluation.

All functions take and return canonical representatives in [0, p).
"""
import hashlib
import os
import numpy as np

P = 2**64 - 2**32 + 1
GEN = 7                         # generator of the multiplicative group
_P = np.uint64(P)
_EPS = np.uint64(2**32 - 1)     # 2^64 mod p
_M32 = np.uint64(0xFFFFFFFF)
_S32 = np.uint64(32)
_HALF = np.uint64(P // 2)
U64 = np.uint64

np.seterr(over='ignore')

# ---------------------------------------------------------------- optional C kernel
import ctypes
_lib = None


def _load_kernel():
    global _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, 'libgl.so')
    src = os.path.join(here, 'gl.c')
    if os.environ.get('FEDPROOF_NO_C'):
        return
    try:
        if not os.path.exists(so) and os.path.exists(src):
            import subprocess
            subprocess.run(['gcc', '-O3', '-march=native', '-fopenmp', '-shared', '-fPIC', '-o', so, src], check=True)
        lib = ctypes.CDLL(so)
        vp, sz, u64 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64
        lib.gl_mul.argtypes = [vp, vp, vp, sz]
        lib.gl_mul_scalar.argtypes = [vp, u64, vp, sz]
        lib.gl_mul_rowvec.argtypes = [vp, vp, vp, sz, sz]
        lib.gl_ntt.argtypes = [vp, sz, sz, vp, vp, u64]
        lib.gl_lincomb.argtypes = [vp, vp, vp, sz, sz]
        lib.gl_muladd_rows.argtypes = [vp, vp, vp, sz, sz]
        lib.gl_scatter.argtypes = [vp, vp, vp, sz]
        lib.gl_scatter_mul.argtypes = [vp, vp, vp, vp, sz]
        lib.gl_matvec.argtypes = [vp, vp, vp, sz, sz]
        for f in ('gl_mul', 'gl_mul_scalar', 'gl_mul_rowvec', 'gl_ntt', 'gl_lincomb', 'gl_muladd_rows',
                  'gl_scatter', 'gl_scatter_mul', 'gl_matvec'):
            getattr(lib, f).restype = None
        _lib = lib
    except Exception:                                            # pragma: no cover
        _lib = None


_load_kernel()


def have_kernel():
    return _lib is not None


def _c(a):
    return np.ascontiguousarray(a, dtype=np.uint64)


def _ptr(a):
    return a.ctypes.data


def arr(x):
    return np.asarray(x, dtype=np.uint64)


def from_int(x):
    """Signed integers (python ints, int64 arrays, object arrays) -> field elements."""
    a = np.asarray(x)
    if a.dtype == np.uint64:
        return a
    if a.dtype == object or a.dtype.kind not in 'iub':
        flat = [int(v) % P for v in a.ravel()]
        return np.array(flat, dtype=np.uint64).reshape(a.shape)
    a64 = a.astype(np.int64)
    u = a64.astype(np.uint64)
    return np.where(a64 < 0, u - _EPS, u)


def to_signed(x):
    """Field element -> signed int64 in (-p/2, p/2). Valid only for |value| < 2^63."""
    x = arr(x)
    return np.where(x > _HALF, (x - _P).astype(np.int64), x.astype(np.int64))


def fadd(a, b):
    s = a + b
    s = np.where(s < a, s + _EPS, s)
    return np.where(s >= _P, s - _P, s)


def fsub(a, b):
    d = a - b
    return np.where(a < b, d + _P, d)


def fneg(a):
    a = arr(a)
    return np.where(a == 0, a, _P - a)


def _reduce128(hi, lo):
    hh = hi >> _S32
    hl = hi & _M32
    t0 = lo - hh
    t0 = np.where(lo < hh, t0 - _EPS, t0)
    t1 = hl * _EPS
    r = t0 + t1
    r = np.where(r < t1, r + _EPS, r)
    return np.where(r >= _P, r - _P, r)


def fmul(a, b):
    a = arr(a); b = arr(b)
    if _lib is not None and a.size >= 256:
        if a.shape == b.shape:
            ac, bc = _c(a), _c(b)
            out = np.empty(a.shape, dtype=np.uint64)
            _lib.gl_mul(_ptr(ac), _ptr(bc), _ptr(out), a.size)
            return out
        if b.ndim == 0 or b.size == 1:
            ac = _c(a)
            out = np.empty(a.shape, dtype=np.uint64)
            _lib.gl_mul_scalar(_ptr(ac), ctypes.c_uint64(int(b.reshape(-1)[0])), _ptr(out), a.size)
            return out
        if a.ndim == 2 and b.ndim == 1 and b.shape[0] == a.shape[1]:
            ac, bc = _c(a), _c(b)
            out = np.empty(a.shape, dtype=np.uint64)
            _lib.gl_mul_rowvec(_ptr(ac), _ptr(bc), _ptr(out), a.shape[0], a.shape[1])
            return out
    return _fmul_np(a, b)


def _fmul_np(a, b):
    a0 = a & _M32; a1 = a >> _S32
    b0 = b & _M32; b1 = b >> _S32
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1
    mid = p01 + (p00 >> _S32)
    mid2 = (mid & _M32) + p10
    lo = (mid2 << _S32) | (p00 & _M32)
    hi = p11 + (mid >> _S32) + (mid2 >> _S32)
    return _reduce128(hi, lo)


def fsum(x, axis=None):
    """Exact modular sum along an axis (fewer than 2^31 terms)."""
    x = arr(x)
    lo = (x & _M32).sum(axis=axis, dtype=np.uint64)
    hi = (x >> _S32).sum(axis=axis, dtype=np.uint64)
    lo = np.where(lo >= _P, lo - _P, lo)
    hi = np.where(hi >= _P, hi - _P, hi)
    return fadd(lo, fmul(hi, U64(1 << 32)))


def fpow(a, e):
    """Elementwise a**e for a python int exponent e >= 0."""
    a = arr(a)
    result = np.ones_like(a)
    base = a.copy()
    while e:
        if e & 1:
            result = fmul(result, base)
        e >>= 1
        if e:
            base = fmul(base, base)
    return result


def finv(a):
    return fpow(a, P - 2)


def inv_int(a):
    return pow(int(a) % P, P - 2, P)


def fdot(a, b):
    """sum_i a_i b_i for 1-D arrays."""
    return fsum(fmul(a, b))


def scatter_add(n, idx, vals):
    """Return length-n array out with out[idx[i]] += vals[i] (mod p)."""
    vals = arr(vals)
    if _lib is not None:
        out = np.zeros(n, dtype=np.uint64)
        ic = np.ascontiguousarray(idx, dtype=np.int64); vc = _c(vals)
        _lib.gl_scatter(_ptr(out), ic.ctypes.data, _ptr(vc), vc.size)
        return out
    lo = np.zeros(n, dtype=np.uint64)
    hi = np.zeros(n, dtype=np.uint64)
    np.add.at(lo, idx, vals & _M32)
    np.add.at(hi, idx, vals >> _S32)
    lo = np.where(lo >= _P, lo - _P, lo)
    hi = np.where(hi >= _P, hi - _P, hi)
    return fadd(lo, fmul(hi, U64(1 << 32)))


# ---------------------------------------------------------------- NTT
_root_cache = {}
_tw_cache = {}
_rev_cache = {}


def root_of_unity(n):
    """Primitive n-th root of unity, n a power of two up to 2^32."""
    if n not in _root_cache:
        assert n & (n - 1) == 0 and n <= 2**32
        _root_cache[n] = pow(GEN, (P - 1) // n, P)
    return _root_cache[n]


def powers(base, n):
    """[base^0, ..., base^(n-1)] as a field array (doubling construction)."""
    out = np.ones(n, dtype=np.uint64)
    if n == 1:
        return out
    b = U64(int(base) % P)
    out[1] = b
    filled = 2
    step = int(base) % P
    while filled < n:
        m = min(filled, n - filled)
        mult = U64(pow(step, filled, P))
        out[filled:filled + m] = fmul(out[:m], mult)
        filled += m
    return out


def _twiddles(n, inverse):
    key = (n, inverse)
    if key not in _tw_cache:
        w = root_of_unity(n)
        if inverse:
            w = inv_int(w)
        tws = {}
        m = 1
        while m < n:
            tws[m] = powers(pow(w, n // (2 * m), P), m)
            m *= 2
        _tw_cache[key] = tws
    return _tw_cache[key]


_twflat_cache = {}


def _tw_flat(n, inverse):
    key = (n, inverse)
    if key not in _twflat_cache:
        tws = _twiddles(n, inverse)
        flat = np.zeros(n, dtype=np.uint64)
        m = 1
        while m < n:
            flat[m:2 * m] = tws[m]
            m *= 2
        _twflat_cache[key] = (flat, np.ascontiguousarray(_bitrev(n), dtype=np.int64))
    return _twflat_cache[key]


def lincomb_rows(M, c):
    """out[j] = sum_i c[i] M[i, j]."""
    M = arr(M); c = arr(c)
    if _lib is not None:
        Mc, cc = _c(M), _c(c)
        out = np.empty(M.shape[1], dtype=np.uint64)
        _lib.gl_lincomb(_ptr(Mc), _ptr(cc), _ptr(out), M.shape[0], M.shape[1])
        return out
    return fsum(_fmul_np(M, c[:, None]), axis=0)


def muladd_rows(A, B, acc=None):
    """acc[j] + sum_i A[i, j] B[i, j]."""
    A = arr(A); B = arr(B)[:np.shape(A)[0]]
    assert A.shape == B.shape
    if acc is None:
        acc = np.zeros(A.shape[1], dtype=np.uint64)
    if _lib is not None:
        Ac, Bc = _c(A), _c(B)
        out = np.array(acc, dtype=np.uint64, copy=True)
        _lib.gl_muladd_rows(_ptr(Ac), _ptr(Bc), _ptr(out), A.shape[0], A.shape[1])
        return out
    return fadd(acc, fsum(_fmul_np(A, B), axis=0))


def matvec(A, v):
    """out[i] = sum_j A[i, j] v[j]."""
    A = arr(A); v = arr(v)
    if _lib is not None:
        Ac, vc = _c(A), _c(v)
        out = np.empty(A.shape[0], dtype=np.uint64)
        _lib.gl_matvec(_ptr(Ac), _ptr(vc), _ptr(out), A.shape[0], A.shape[1])
        return out
    return fsum(_fmul_np(A, v[None, :]), axis=1)


def scatter_mul(n, idx, a, b):
    """out[idx[i]] += a[i] * b[i]."""
    if _lib is not None:
        out = np.zeros(n, dtype=np.uint64)
        ic = np.ascontiguousarray(idx, dtype=np.int64); ac, bc = _c(a), _c(b)
        _lib.gl_scatter_mul(_ptr(out), ic.ctypes.data, _ptr(ac), _ptr(bc), ac.size)
        return out
    return scatter_add(n, idx, fmul(a, b))


def _bitrev(n):
    if n not in _rev_cache:
        bits = n.bit_length() - 1
        idx = np.arange(n, dtype=np.int64)
        rev = np.zeros(n, dtype=np.int64)
        for i in range(bits):
            rev |= ((idx >> i) & 1) << (bits - 1 - i)
        _rev_cache[n] = rev
    return _rev_cache[n]


def ntt(a, inverse=False):
    """NTT along the last axis. Forward: coefficients -> evaluations at w^i. Inverse undoes it."""
    a = arr(a)
    n = a.shape[-1]
    if n == 1:
        return a.copy()
    if _lib is not None:
        x = np.array(a, dtype=np.uint64, order='C', copy=True)
        tw, rev = _tw_flat(n, inverse)
        _lib.gl_ntt(_ptr(x), x.size // n, n, _ptr(tw), rev.ctypes.data,
                    ctypes.c_uint64(inv_int(n) if inverse else 1))
        return x
    tws = _twiddles(n, inverse)
    shape = a.shape
    x = a[..., _bitrev(n)]
    m = 1
    while m < n:
        x = x.reshape(shape[:-1] + (n // (2 * m), 2 * m))
        u = x[..., :m]
        v = fmul(x[..., m:], tws[m])
        x = np.concatenate([fadd(u, v), fsub(u, v)], axis=-1)
        m *= 2
    x = x.reshape(shape)
    if inverse:
        x = fmul(x, U64(inv_int(n)))
    return x


_shift_cache = {}


def coset_shift(n, inverse=False):
    key = (n, inverse)
    if key not in _shift_cache:
        g = inv_int(GEN) if inverse else GEN
        _shift_cache[key] = powers(g, n)
    return _shift_cache[key]


def coset_eval(coeffs, n):
    """Evaluate polynomials (coeffs along last axis, length <= n) on the coset GEN * H_n."""
    coeffs = arr(coeffs)
    k = coeffs.shape[-1]
    assert k <= n
    padded = np.zeros(coeffs.shape[:-1] + (n,), dtype=np.uint64)
    padded[..., :k] = fmul(coeffs, coset_shift(n)[:k])
    return ntt(padded)


def coset_interp(evals):
    """Inverse of coset_eval: evaluations on GEN * H_n -> n coefficients."""
    n = evals.shape[-1]
    return fmul(ntt(evals, inverse=True), coset_shift(n, inverse=True))


def coset_points(n):
    """The points GEN * w_n^i."""
    return fmul(powers(root_of_unity(n), n), U64(GEN))


def poly_eval_int(coeffs, x):
    """Horner with python ints (slow reference)."""
    acc = 0
    for c in reversed([int(v) for v in coeffs]):
        acc = (acc * x + c) % P
    return acc


# ---------------------------------------------------------------- randomness
def bytes_to_field(buf, count):
    """Rejection-sample field elements from a byte string; returns (array, bytes_used) or None if not enough."""
    v = np.frombuffer(buf, dtype='<u8')
    v = v[v < _P]
    if len(v) < count:
        return None
    return v[:count].copy()


def rand_field(shape, rng=None):
    """Uniform field elements. rng=None uses the OS generator; otherwise a Drbg instance."""
    count = int(np.prod(shape)) if not isinstance(shape, int) else shape
    need = 8 * (count + 64)
    while True:
        buf = os.urandom(need) if rng is None else rng.read(need)
        out = bytes_to_field(buf, count)
        if out is not None:
            return out.reshape(shape)


class Drbg:
    """Deterministic byte generator (SHAKE256 in counter mode) for reproducible tests."""

    def __init__(self, seed):
        self.seed = seed if isinstance(seed, bytes) else repr(seed).encode()
        self.ctr = 0

    def read(self, n):
        h = hashlib.shake_256(self.seed + self.ctr.to_bytes(8, 'little'))
        self.ctr += 1
        return h.digest(n)
