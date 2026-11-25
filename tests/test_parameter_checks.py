"""Parameter checks: configurations that cannot be built are refused, parameter search never fails, and the
statement soundness accounting adds up."""
import sys, math
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, ligero as L, step as ST, message as SP, pq
from fedproof.sharing import Sharing

# configurations that cannot be built are now refused by ranges(), or made buildable
try:
    M.Config(arch=(8, 4, 3), b=2, g=0).ranges(); raise SystemExit('g = 0 accepted')
except AssertionError:
    pass
cfg = M.Config(arch=(4, 2, 1), b=2, Bw=2, Fw=12, Fd=2)          # previously Bt - 1 < Fw
R = cfg.ranges()
assert all(R['Bt'][l] - 1 >= cfg.Fw for l in range(1, cfg.L))
prm = L.Params(256, 32, 32, 1)
g = np.random.default_rng(0)
X = g.integers(0, 16, (cfg.b, 4)); y = g.integers(0, 1, cfg.b)
rho = g.integers(0, 2, (cfg.d, cfg.J, cfg.r)); beta = g.integers(0, 2, (cfg.d, cfg.J, cfg.r))
W = M.proof_weights(cfg, M.init_model(cfg))
shr = Sharing(cfg.d, 1, 1, 3); rng = F.Drbg(('l1',))
d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
nz = L.commit_block(ST.noise_block_matrix(cfg, prm.ell, rho), prm, rng)
keys = [pq.kem_keygen() for _ in range(3)]
pub = SP.Public(cfg, prm, shr, W, beta, b'l1')
msg, priv = SP.site_prove(pub, dict(X=X, y=y, rho=rho), d0, nz, [k[0] for k in keys], rng)
assert np.array_equal(priv['u'], M.plain_step(cfg, W, X, y, rho, beta)['u']) and SP.site_verify(pub, d0.root, nz.root, msg)[0]
print('  ok  g = 0 refused; the narrow-weight-scale configuration now builds, proves and verifies')

# find_params returns a valid parameter set or None, never raises
for ell in (256, 512, 1024, 2048):
    for bits in (64, 128, 256):
        for blowup in (2, 4, 8):
            p = L.find_params(ell, bits, blowup)
            assert p is None or p.soundness()['rbr_bits'] >= bits
print('  ok  find_params never raises over 36 settings')

# the digest term counts the union over nodes; the algebraic term counts (n + 2)/|F|
p = L.find_params(1024, 128, 8)
s1 = L.statement_soundness(p, 3, n_nodes=1); s13 = L.statement_soundness(p, 3, n_nodes=13)
assert abs((s1['digest_bits'] - s13['digest_bits']) - math.log2(13)) < 1e-9
assert abs(p.soundness()['algebraic_bits'] - p.sigma * (64 - math.log2(p.n + 2) - math.log2(1 + 2**-32))) < 1e-9
print('  ok  statement soundness %.1f bits (query %.1f, algebraic %.1f, Freivalds %.1f, digest %.1f with 13 nodes)' % (
    s13['total_bits'], s13['query_bits'], s13['algebraic_bits'], s13['freivalds_bits'], s13['digest_bits']))
print('parameter checks passed')
