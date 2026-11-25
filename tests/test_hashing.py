import sys
import random
sys.path.insert(0, '.')
from fedproof.hashing import *
rnd = random.Random(2)
for n in (2, 4, 64, 1024):
    leaves = [H(bytes([i % 256, i // 256])) for i in range(n)]
    tr = MerkleTree(leaves)
    for trial in range(30):
        t = rnd.randrange(1, min(n, 40) + 1)
        idx = sorted(rnd.sample(range(n), t))
        pf = tr.open(idx)
        assert merkle_verify(tr.root, n, idx, [leaves[i] for i in idx], pf)
        bad = [leaves[i] for i in idx]; bad[rnd.randrange(t)] = H(b'x')
        assert not merkle_verify(tr.root, n, idx, bad, pf)
        if pf:
            pf2 = list(pf); pf2[rnd.randrange(len(pf2))] = H(b'y'); assert not merkle_verify(tr.root, n, idx, [leaves[i] for i in idx], pf2)
            assert not merkle_verify(tr.root, n, idx, [leaves[i] for i in idx], pf[:-1])
        assert not merkle_verify(tr.root, n, idx, [leaves[i] for i in idx], pf + [H(b'z')])
        if t < n:
            j = next(i for i in range(n) if i not in idx)
            idx2 = sorted(idx[:-1] + [j]) if j not in idx[:-1] else idx
            if idx2 != idx: assert not merkle_verify(tr.root, n, idx2, [leaves[i] for i in idx], pf)
    assert not merkle_verify(tr.root, n, [0, 0], [leaves[0], leaves[0]], tr.open([0]))
a = Transcript(b'x'); b = Transcript(b'x'); c = Transcript(b'y')
a.absorb('m', b'1'); b.absorb('m', b'1'); c.absorb('m', b'1')
fa = a.chal_field('r', 1000); fb = b.chal_field('r', 1000); fc = c.chal_field('r', 1000)
assert (fa == fb).all() and not (fa == fc).all() and (fa < np.uint64(F.P)).all()
assert not (a.chal_field('r', 10) == fa[:10]).all()
i1 = a.chal_indices('q', 50, 256); assert len(set(i1.tolist())) == 50 and i1.max() < 256
d = Transcript(b'x'); d.absorb('m', b'12'); e = Transcript(b'x'); e.absorb('m1', b'2')
assert d.state != e.state
print('hashing tests passed')
