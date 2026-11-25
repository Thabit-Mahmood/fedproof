import sys, time, copy
sys.path.insert(0, '.')
import numpy as np
from fedproof import field as F
from fedproof.constraints import CS, LC, BIT, FREE
from fedproof import ligero as L
from fedproof.hashing import Transcript

def build(prm, prover=True, seed=1, cheat=None):
    """Toy statement with two pre-committed blocks (P: FREE half-filled, N: BIT), a main block and an aux block."""
    rng = F.Drbg(('toy', seed)) if prover else None
    cs = CS(prm.ell, prover, rng)
    cs.add_block('P', free_fill=prm.ell // 2); cs.add_block('N'); cs.add_block('M'); cs.add_block('A')
    npv, nbits = 300, 700
    pv = cs.alloc('P', FREE, npv); nb = cs.alloc('N', BIT, nbits)
    g = np.random.default_rng(seed)
    if prover:
        cs.set(pv, g.integers(0, 200, npv)); cs.set(nb, g.integers(0, 2, nbits))
    cs.freeze('P'); cs.freeze('N')
    # main: decompose pv into 9 bits; products pv[i]*pv[i+1]; sum of N bits in groups of 7 equals a 3-bit number
    b9 = cs.bits('M', LC.cells(pv), 9, signed=False)
    z = cs.mul('M', LC.cells(pv[:-1]), LC.cells(pv[1:]) + 5)
    grp = nb.reshape(100, 7)
    cnt = cs.bits('M', LC.cells(grp, np.ones(7, dtype=np.int64)), 3, signed=False)
    tot = cs.alloc('M', FREE, 1)
    if prover:
        cs.set(tot, F.fsum(cs.get(z)))
    cs.eq(LC.cells(z.reshape(1, -1)), LC.cells(tot))
    cs.freeze('M')
    # aux: random-coefficient check  (sum a_i pv_i) * (sum c_i bit_i) = y
    a = fe_rand = F.rand_field(npv, F.Drbg(('ch', seed)))
    c = F.rand_field(nbits, F.Drbg(('ch2', seed)))
    y = cs.mul('A', LC.cells(pv.reshape(1, -1), a), LC.cells(nb.reshape(1, -1), c))
    pub = None
    if prover:
        pub = int(cs.get(y)[0]); 
    cs.freeze('A')
    return cs, dict(pv=pv, nb=nb, y=y, tot=tot, z=z, b9=b9), pub

def run(prm, seed=1, mutate=None, strict=True, verify_ctx=b'ctx', pub_override=None):
    cs, ids, pub = build(prm, True, seed)
    cs.eq(LC.cells(ids['y']), int(pub))                       # public output
    if mutate: mutate(cs, ids)
    rng = cs.rng
    sysd = cs.system()
    blocks = [L.commit_block(cs.message_matrix('P'), prm, rng), L.commit_block(cs.message_matrix('N'), prm, rng),
              L.commit_block(cs.message_matrix('M'), prm, rng), L.commit_block(cs.message_matrix('A'), prm, rng, with_masks=True)]
    fails = cs.check(sysd)
    T = Transcript(b'toy'); T.absorb('ctx', b'ctx')
    for b in blocks: T.absorb('root', b.root)
    t0 = time.time(); proof = L.prove(prm, blocks, sysd, T, strict=strict); tp = time.time() - t0
    # verifier side
    vcs, vids, _ = build(prm, False, seed)
    vcs.eq(LC.cells(vids['y']), int(pub if pub_override is None else pub_override))
    vsys = vcs.system()
    metas = [(b.root, b.nrows) for b in blocks]
    def ver(pf, metas=metas, ctx=verify_ctx):
        Tv = Transcript(b'toy'); Tv.absorb('ctx', ctx)
        for r, _ in metas: Tv.absorb('root', r)
        return L.verify(prm, metas, vsys, Tv, pf)
    return proof, ver, fails, tp, blocks, sysd

prm = L.Params(256, 24, 24, 2)
print('params', prm.soundness())
proof, ver, fails, tp, blocks, sysd = run(prm)
assert fails == [], fails
t0 = time.time(); ok = ver(proof); tv = time.time() - t0
assert ok
print('honest proof verifies; rows=%d ncon=%d nnz=%d prove %.2fs verify %.2fs size %d B' % (sysd['m'], sysd['ncon'], len(sysd['con']), tp, tv, L.proof_size(proof)))
assert len(L.proof_bytes(proof)) == L.proof_size(proof)
# decode check
M0 = L.decode_rows(blocks[0].coeffs, prm)
# --- tampering with the proof object
def tam(f):
    p2 = copy.deepcopy(proof); f(p2); return ver(p2)
assert not tam(lambda p: p['q_prox'].__setitem__((0, 3), (int(p['q_prox'][0, 3]) + 1) % F.P))
assert not tam(lambda p: p['q_lin'].__setitem__((1, 0), (int(p['q_lin'][1, 0]) + 1) % F.P))
assert not tam(lambda p: p['q_quad'].__setitem__((0, 5), (int(p['q_quad'][0, 5]) + 1) % F.P))
for bi in range(4):
    assert not tam(lambda p: p['opens'][bi]['vals'].__setitem__((0, 0), (int(p['opens'][bi]['vals'][0, 0]) + 1) % F.P))
    assert not tam(lambda p: p['opens'][bi]['salts'].__setitem__(0, b'\x00' * 32))
    assert not tam(lambda p: p['opens'][bi]['path'].__setitem__(0, b'\x00' * 48))
assert not tam(lambda p: p['q_prox'].__setitem__((0, 0), np.uint64(F.P)))            # non-canonical element
assert not tam(lambda p: p.__setitem__('q_lin', p['q_lin'][:, :-1]))
assert not tam(lambda p: p['opens'].pop())
assert not ver(proof, ctx=b'other context')
m2 = list(ver.__defaults__[0]); 
bad_metas = [(b'\x11' * 48, blocks[0].nrows)] + [(b.root, b.nrows) for b in blocks[1:]]
assert not ver(proof, metas=bad_metas)
# --- statement mismatch: different public output
proofX, verX, _, _, _, _ = run(prm, pub_override=12345)
assert not verX(proofX)
# --- cheating provers (assertions off): each violated constraint class must be rejected
def cheat_lin(cs, ids): cs.set(ids['tot'], F.fadd(cs.get(ids['tot']), F.U64(1)))
def cheat_bit(cs, ids): cs.set(ids['nb'][:1], 2)
def cheat_mul(cs, ids): cs.set(ids['z'][:1], F.fadd(cs.get(ids['z'][:1]), F.U64(1)))
def cheat_pre(cs, ids): cs.set(ids['pv'][:1], F.fadd(cs.get(ids['pv'][:1]), F.U64(1)))
def cheat_b9(cs, ids): cs.set(ids['b9'][0, :1], F.fsub(F.U64(1), cs.get(ids['b9'][0, :1])))
for name, f in (('lin', cheat_lin), ('bit', cheat_bit), ('mul', cheat_mul), ('pre', cheat_pre), ('b9', cheat_b9)):
    p3, v3, fails3, _, _, _ = run(prm, mutate=f, strict=False)
    assert fails3 != [], name
    assert not v3(p3), name
print('tamper and cheating-prover tests passed')
# --- ZK sanity: opened values of a block do not depend on its message (same randomness, different message)
r1 = F.Drbg('same'); r2 = F.Drbg('same')
msg1 = np.zeros((3, prm.ell), dtype=np.uint64); msg2 = F.rand_field((3, prm.ell), F.Drbg('m'))
c1 = L.encode_rows(msg1, prm, r1); c2 = L.encode_rows(msg2, prm, r2)
assert np.array_equal(L.decode_rows(c2, prm), msg2)
# rank test: evaluations at any t' coset points are an invertible function of the randomiser (so they are uniform)
pts = [int(x) for x in F.coset_points(prm.n)[:prm.tprime]]
V = [[(pow(x, prm.ell, F.P) - 1) * pow(x, j, F.P) % F.P for j in range(prm.tprime)] for x in pts]
def rank(Mx):
    Mx = [row[:] for row in Mx]; rk = 0; ncol = len(Mx[0])
    for c in range(ncol):
        piv = next((r for r in range(rk, len(Mx)) if Mx[r][c]), None)
        if piv is None: continue
        Mx[rk], Mx[piv] = Mx[piv], Mx[rk]; inv = pow(Mx[rk][c], F.P - 2, F.P)
        Mx[rk] = [v * inv % F.P for v in Mx[rk]]
        for r in range(len(Mx)):
            if r != rk and Mx[r][c]:
                f = Mx[r][c]; Mx[r] = [(a - f * b) % F.P for a, b in zip(Mx[r], Mx[rk])]
        rk += 1
    return rk
assert rank(V) == prm.tprime
print('zk rank test passed (t\' = %d opened evaluations are a bijective image of the randomiser)' % prm.tprime)
