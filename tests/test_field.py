import sys, time, random
sys.path.insert(0, '.')
import numpy as np
from fedproof import field as F
P = F.P
rnd = random.Random(1)
def rints(n):
    edge = [0, 1, P-1, P-2, 2**32-1, 2**32, 2**32+1, 2**63, 2**64-2**32, P//2, P//2+1]
    return edge + [rnd.randrange(P) for _ in range(n)]
a = rints(4000); b = rints(4000); rnd.shuffle(b)
A = np.array(a, dtype=np.uint64); B = np.array(b, dtype=np.uint64)
assert [int(x) for x in F.fadd(A, B)] == [(x+y) % P for x, y in zip(a, b)]
assert [int(x) for x in F.fsub(A, B)] == [(x-y) % P for x, y in zip(a, b)]
assert [int(x) for x in F.fmul(A, B)] == [(x*y) % P for x, y in zip(a, b)]
assert [int(x) for x in F.fneg(A)] == [(-x) % P for x in a]
# all edge pairs
E = [0, 1, P-1, P-2, 2**32-1, 2**32, 2**32+1, 2**63, 2**64-2**32, P//2]
for x in E:
    for y in E:
        assert int(F.fmul(np.uint64(x), np.uint64(y))) == x*y % P
        assert int(F.fadd(np.array([x],dtype=np.uint64), np.array([y],dtype=np.uint64))[0]) == (x+y) % P
        assert int(F.fsub(np.array([x],dtype=np.uint64), np.array([y],dtype=np.uint64))[0]) == (x-y) % P
inv = F.finv(A[A != 0]); assert all(int(x)*int(y) % P == 1 for x, y in zip(A[A != 0][:200], inv[:200]))
assert int(F.fsum(A)) == sum(a) % P
M = np.array([[rnd.randrange(P) for _ in range(7)] for _ in range(50)], dtype=np.uint64)
assert [int(v) for v in F.fsum(M, axis=0)] == [sum(int(M[i, j]) for i in range(50)) % P for j in range(7)]
# signed conversion
s = np.array([-5, 0, 7, -2**62, 2**62], dtype=np.int64)
assert [int(v) for v in F.from_int(s)] == [int(v) % P for v in s]
assert list(F.to_signed(F.from_int(s))) == list(s)
assert [int(v) for v in F.from_int(np.array([-3, 10**30], dtype=object))] == [P-3, 10**30 % P]
# scatter
idx = np.array([rnd.randrange(13) for _ in range(5000)]); vals = np.array([rnd.randrange(P) for _ in range(5000)], dtype=np.uint64)
ref = [0]*13
for i, v in zip(idx, vals): ref[i] = (ref[i] + int(v)) % P
assert [int(v) for v in F.scatter_add(13, idx, vals)] == ref
# ntt vs naive
for n in (1, 2, 8, 64):
    c = np.array([rnd.randrange(P) for _ in range(n)], dtype=np.uint64)
    w = F.root_of_unity(n)
    ev = F.ntt(c)
    assert [int(v) for v in ev] == [F.poly_eval_int(c, pow(w, i, P)) for i in range(n)]
    assert np.array_equal(F.ntt(ev, inverse=True), c)
    ce = F.coset_eval(c, 4*n)
    pts = F.coset_points(4*n)
    assert [int(v) for v in ce] == [F.poly_eval_int(c, int(x)) for x in pts]
    assert np.array_equal(F.coset_interp(ce)[:n], c) and not F.coset_interp(ce)[n:].any()
# 2D batch
c = F.rand_field((5, 256)); assert np.array_equal(F.ntt(F.ntt(c), inverse=True), c)
assert pow(F.root_of_unity(2**32), 2**31, P) == P - 1
assert [int(v) for v in F.powers(3, 11)] == [pow(3, i, P) for i in range(11)]
print('field tests passed')
x = F.rand_field((256, 16384)); t = time.time(); y = F.ntt(x); print('ntt 256x16384: %.2fs' % (time.time()-t))
t = time.time(); z = F.fmul(x, y); print('mul 4.2M: %.3fs' % (time.time()-t))
# ---- C kernel vs numpy path vs python ints
print('kernel:', F.have_kernel())
if F.have_kernel():
    x = F.rand_field((7, 512)); y = F.rand_field((7, 512))
    assert np.array_equal(F.fmul(x, y), F._fmul_np(x, y))
    assert np.array_equal(F.fmul(x, np.uint64(12345678901234567)), F._fmul_np(x, np.uint64(12345678901234567)))
    assert np.array_equal(F.fmul(x, y[0]), F._fmul_np(x, y[0]))
    lib = F._lib; F._lib = None
    n1 = F.ntt(x); n1i = F.ntt(x, inverse=True); lc = F.lincomb_rows(x, y[:, 0]); ma = F.muladd_rows(x, y, y[0]); mv = F.matvec(x, y[0])
    idx = np.array([rnd.randrange(13) for _ in range(512)]); sc = F.scatter_add(13, idx, x[0]); sm = F.scatter_mul(13, idx, x[0], y[0])
    F._lib = lib
    assert np.array_equal(F.ntt(x), n1) and np.array_equal(F.ntt(x, inverse=True), n1i)
    assert np.array_equal(F.lincomb_rows(x, y[:, 0]), lc) and np.array_equal(F.muladd_rows(x, y, y[0]), ma)
    assert np.array_equal(F.matvec(x, y[0]), mv)
    assert np.array_equal(F.scatter_add(13, idx, x[0]), sc) and np.array_equal(F.scatter_mul(13, idx, x[0], y[0]), sm)
    assert [int(v) for v in lc[:5]] == [sum(int(y[i, 0]) * int(x[i, j]) for i in range(7)) % P for j in range(5)]
    print('kernel cross-checks passed')
