"""Cheating provers aimed at the changed parts: smoothed-hinge output stage (sign bit, saturation flag, quantisation),
centred inputs, clipping at the kappa boundary. Each cheat changes prover-side VALUES only (the statement structure is
untouched), recomputes downstream honestly, then the message is checked by cs.check and by site_verify.
Usage: python3 tests/test_output_stage_cheats.py"""
import sys, contextlib
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, ligero as L, step as ST, message as SP, constraints as CSm
from test_step import setup

prm = L.Params(1024, 64, 64, 2)

def build(cfg, W, X, y, seed=0):
    pub, wit, d0, nz, keys, rng = setup(cfg, prm, K=5, kp=2, T=1, seed=seed)
    pub.weights = W; wit['X'], wit['y'] = X, y
    d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
    return pub, wit, d0, nz, [k[0] for k in keys]

@contextlib.contextmanager
def patched(obj, name, fn):
    old = getattr(obj, name); setattr(obj, name, fn(old))
    try: yield
    finally: setattr(obj, name, old)

def run(label, pub, wit, d0, nz, eks, hook=None, ctxm=None):
    rng = F.Drbg(('atk', label))
    with (ctxm or contextlib.nullcontext()):
        msg, priv = SP.site_prove(pub, wit, d0, nz, eks, rng, hook=hook or (lambda *a: None), strict=False)
    cs = priv['cs']; cs.mats['D0'] = L.decode_rows(d0.coeffs, prm); cs.mats['NZ'] = L.decode_rows(nz.coeffs, prm)
    fails = cs.check(priv['sysd'])
    ok = SP.site_verify(pub, d0.root, nz.root, msg)[0]
    ref = M.plain_step(pub.cfg, pub.weights, wit['X'], wit['y'], wit['rho'], pub.beta)
    differs = not np.array_equal(priv['u'], ref['u'])
    print('  %-60s u differs: %-5s constraint failures: %-22s verifier: %s' % (label, differs, ','.join('%s x%d' % (k, len(v)) for k, v in fails) or 'none', 'ACCEPT' if ok else 'reject'))
    if ok and differs: raise SystemExit('BREAK: ' + label)
    return ok, differs

# a model whose output stage covers all three regions (negative e, middle, saturated) for every record
cfg = M.Config(arch=(64, 12, 10), b=8, center=True)
g = np.random.default_rng(4)
W = [(g.integers(-cfg.Wmax, cfg.Wmax + 1, (12, 64)), g.integers(-cfg.Wmax, cfg.Wmax + 1, 12)),
     (g.integers(-150, 151, (10, 12)), g.integers(-cfg.Wmax, cfg.Wmax + 1, 10))]
X = g.integers(0, 16, (8, 64)); y = g.integers(0, 10, 8)
ref = M.plain_step(cfg, W, X, y)
e = ref['zs'][2] + (2 << 16); v = e >> 8
print('output regions over 80 units: e<0 %d, middle %d, saturated %d; clipped records %d of 8' % (
    (e < 0).sum(), ((v >= 0) & (v <= 1024)).sum(), (v > 1024).sum(), (ref['kappa'] == 0).sum()))
pub, wit, d0, nz, eks = build(cfg, W, X, y)
run('honest (centred, all output regions)', pub, wit, d0, nz, eks)

calls = {'n': 0}
def nz_cheat(mode):
    def mk(old):
        def nonzero(cs, blk, lc):
            z = old(cs, blk, lc)
            calls['n'] += 1
            if cs.prover and lc.n == cfg.b * cfg.arch[-1]:            # the output-stage flag
                vz = cs.get(z).copy()
                if mode == 'off': vz[:] = 0                            # claim "not saturated" everywhere
                else: vz[:] = 1                                        # claim "saturated" everywhere
                cs.set(z, vz)
            return z
        return nonzero
    return mk
run('saturation flag forced to 0 (p not clamped at 1)', pub, wit, d0, nz, eks, ctxm=patched(ST, 'nonzero', nz_cheat('off')))
run('saturation flag forced to 1 (p clamped although below 1)', pub, wit, d0, nz, eks, ctxm=patched(ST, 'nonzero', nz_cheat('on')))

Be = cfg.ranges()['Be']
def bits_cheat(old):
    def bits(self, name, lc, nbits, signed=True):
        ids = old(self, name, lc, nbits, signed)
        if self.prover and nbits == Be and lc.n == cfg.b * cfg.arch[-1] and not getattr(self, '_done', False):
            self._done = True
            b = self.get(ids).copy(); b[:, Be - 1] = 1 - b[:, Be - 1]; self.set(ids, b)   # malicious sign bit
        return ids
    return bits
run('output sign bit flipped (negative e treated as positive)', pub, wit, d0, nz, eks, ctxm=patched(CSm.CS, 'bits', bits_cheat))

def p_round(name, cs, ids):
    if name == 'p':
        v = F.to_signed(cs.get(ids)); cs.set(ids, np.minimum(v + 1, 1 << cfg.Fd))      # round up instead of floor
run('p rounded up instead of floored', pub, wit, d0, nz, eks, hook=p_round)
def p_neg(name, cs, ids):
    if name == 'p':
        v = F.to_signed(cs.get(ids)); cs.set(ids, np.where(v == 0, -1, v))           # p below 0 where it should clamp
run('p = -1 instead of clamping at 0', pub, wit, d0, nz, eks, hook=p_neg)

# centring: the first-layer activations are the committed x cells plus a public constant; try to use the
# uncentred value 16x in the norm stage by under-reporting c1 for records whose centred norm is larger
def c_unc(name, cs, ids):
    if name == 'c1':
        h0 = (F.to_signed(cs.get(ST_X[0])) << 4)
        want = M.ceilsqrt1((h0 ** 2).sum(1) + (1 << 16))
        cs.set(ids, ((want[:, None] >> np.arange(ids.shape[1])) & 1).astype(np.uint64))
def grab(name, cs, ids):
    if name == 'data': ST_X[0] = ids[0]
    c_unc(name, cs, ids)
ST_X = [None]
run('c1 computed from uncentred inputs', pub, wit, d0, nz, eks, hook=grab)

# kappa boundary: choose Cg equal to record 0's norm bound n, so n == Cg exactly (kappa must be 1)
for dCg in (0, -1):
    cfgB = M.Config(arch=(64, 12, 10), b=8, center=True, Cg=int(ref['n'][0]) + dCg)
    refB = M.plain_step(cfgB, W, X, y)
    pubB, witB, d0B, nzB, eksB = build(cfgB, W, X, y)
    print('Cg=%d n[0]=%d kappa[0]=%d' % (cfgB.Cg, refB['n'][0], refB['kappa'][0]))
    run('honest at boundary n - Cg = %d' % (-dCg), pubB, witB, d0B, nzB, eksB)
    def kflip(name, cs, ids):
        if name == 'kappa': cs.set(ids[0, -1], 1 - int(cs.get(ids[0, -1])))
    run('kappa of record 0 flipped at the boundary', pubB, witB, d0B, nzB, eksB, hook=kflip)
    def sp_up(name, cs, ids):
        if name == 'sp':
            v = (F.to_signed(cs.get(ids)) << np.arange(ids.shape[1])).sum(1) + 1
            cs.set(ids, ((v[:, None] >> np.arange(ids.shape[1])) & 1).astype(np.uint64))
    run('clip quotient s\' = floor(num/n) + 1', pubB, witB, d0B, nzB, eksB, hook=sp_up)

# the existing hooks 'zbits1' and 'kappa' fire after the dependent cells were computed; here the sign bit is flipped
# inside the decomposition itself so that every downstream value is recomputed from the cheated bit
def flip_top(match):
    def mk(old):
        def bits(self, name, lc, nbits, signed=True):
            ids = old(self, name, lc, nbits, signed)
            if self.prover and match(self, nbits, lc) and not getattr(self, '_done2', False):
                self._done2 = True
                b = self.get(ids).copy(); b[0, nbits - 1] = 1 - b[0, nbits - 1]; self.set(ids, b)
            return ids
        return bits
    return mk
R0 = cfg.ranges()
run('hidden ReLU sign bit of one unit flipped, downstream recomputed', pub, wit, d0, nz, eks,
    ctxm=patched(CSm.CS, 'bits', flip_top(lambda cs, nb, lc: nb == R0['Bz'][0] and lc.n == cfg.b * cfg.arch[1])))
for dCg in (0, -1):
    cfgB = M.Config(arch=(64, 12, 10), b=8, center=True, Cg=int(ref['n'][0]) + dCg)
    RB = cfgB.ranges()
    pubB, witB, d0B, nzB, eksB = build(cfgB, W, X, y)
    run('kappa of record 0 flipped inside decomposition (n-Cg=%d)' % (-dCg), pubB, witB, d0B, nzB, eksB,
        ctxm=patched(CSm.CS, 'bits', flip_top(lambda cs, nb, lc: nb == RB['Bn'] + 1 and lc.n == cfgB.b)))
print('consistent cheats on the output stage, centring and clipping: all rejected')
