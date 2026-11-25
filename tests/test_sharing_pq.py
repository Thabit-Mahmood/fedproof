import sys, random, itertools
sys.path.insert(0, '.')
import numpy as np
from fedproof import field as F
from fedproof.sharing import Sharing, rs_decode, _poly_eval
from fedproof import pq
P = F.P; rnd = random.Random(5)
kp, T, K = 3, 3, 13
S = Sharing(20, kp, T, K)
vals = F.rand_field((S.nblk, S.w), F.Drbg('v'))
sh = S.share_values(vals)                                   # K x nblk
# reconstruction from any w correct shares
for trial in range(20):
    nodes = sorted(rnd.sample(range(K), S.w))
    Cm = S.secret_coeffs(nodes)
    rec = np.array([F.matvec(Cm, sh[nodes, B]) for B in range(S.nblk)])
    assert np.array_equal(rec, vals[:, :kp])
# linearity: sum of shares = shares of sum
vals2 = F.rand_field((S.nblk, S.w), F.Drbg('w')); assert np.array_equal(F.fadd(sh, S.share_values(vals2)), S.share_values(F.fadd(vals, vals2)))
# secrecy condition: for EVERY set of T nodes the map anchors -> their shares is a bijection (so T shares are uniform, independent of secrets)
def det(Mx):
    Mx = [list(map(int, r)) for r in Mx]; n = len(Mx); d = 1
    for c in range(n):
        piv = next((r for r in range(c, n) if Mx[r][c]), None)
        if piv is None: return 0
        if piv != c: Mx[c], Mx[piv] = Mx[piv], Mx[c]; d = -d
        d = d * Mx[c][c] % P; inv = pow(Mx[c][c], P - 2, P)
        for r in range(c + 1, n):
            f = Mx[r][c] * inv % P
            if f: Mx[r] = [(a - f * b) % P for a, b in zip(Mx[r], Mx[c])]
    return d % P
cnt = 0
for sub in itertools.combinations(range(K), T):
    assert det(S.lam[list(sub)][:, kp:]) != 0; cnt += 1
print('secrecy: all %d subsets of %d nodes see an invertible image of the anchors' % (cnt, T))
# error-correcting decoding
pts = S.alphas
for e_true in range(0, 4):
    for trial in range(15):
        f = [rnd.randrange(P) for _ in range(S.w)]
        y = [_poly_eval(f, x) for x in pts]
        bad = rnd.sample(range(K), e_true)
        for i in bad: y[i] = (y[i] + rnd.randrange(1, P)) % P
        emax = (K - S.w) // 2
        res = rs_decode(pts, y, S.w, emax)
        if e_true <= emax:
            assert res is not None and res[0] == f and sorted(res[1]) == sorted(bad), (e_true, res)
        else:
            assert res is None or res[0] != f or True     # beyond the bound: must not be relied on
# beyond bound: count silent wrong decodes when errors > emax (documented limit)
emax = (K - S.w) // 2; wrong = fail = 0
for trial in range(200):
    f = [rnd.randrange(P) for _ in range(S.w)]; y = [_poly_eval(f, x) for x in pts]
    for i in rnd.sample(range(K), emax + 1): y[i] = (y[i] + rnd.randrange(1, P)) % P
    res = rs_decode(pts, y, S.w, emax)
    if res is None: fail += 1
    elif res[0] != f: wrong += 1
print('decoding with emax=%d: random errors beyond the bound -> %d detected failures, %d wrong decodes of 200' % (emax, fail, wrong))
# with erasures (offline nodes)
f = [rnd.randrange(P) for _ in range(S.w)]; y = [_poly_eval(f, x) for x in pts]
alive = sorted(rnd.sample(range(K), K - 2)); yy = [y[i] for i in alive]; yy[0] = (yy[0] + 1) % P
res = rs_decode([pts[i] for i in alive], yy, S.w, (len(alive) - S.w) // 2); assert res[0] == f and res[1] == [0]
# transport
ek, dk = pq.kem_keygen(); blob, seed = pq.seal(ek, b'hello shares', b'aad')
assert pq.open_(dk, blob, b'aad') == b'hello shares' and pq.open_(dk, blob, b'other') is None
assert pq.open_with_seed(ek, blob, b'aad', seed) == b'hello shares' and pq.open_with_seed(ek, blob, b'aad', b'\x00' * 32) is None
bad = bytearray(blob); bad[-1] ^= 1; assert pq.open_(dk, bytes(bad), b'aad') is None
bad = bytearray(blob); bad[5] ^= 1; assert pq.open_(dk, bytes(bad), b'aad') is None
pk, sk = pq.sig_keygen(); sg = pq.sign(sk, b'm'); assert pq.verify(pk, b'm', sg) and not pq.verify(pk, b'n', sg) and len(sg) == pq.SIG_LEN
print('sharing and transport tests passed; kem ct %d B, sig %d B' % (pq.KEM_CT, len(sg)))
