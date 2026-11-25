"""The proved vector equals the reference semantics for other shapes: centred inputs, a three-layer network, and extremes."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import model as M, ligero as L, step as ST, message as SP
from test_step import setup

prm = L.Params(1024, 64, 64, 2)
def check(cfg, label, weights=None, X=None, y=None):
    pub, wit, d0, nz, keys, rng = setup(cfg, prm, K=5, kp=2, T=1)
    if weights is not None: pub.weights = weights
    if X is not None:
        wit['X'], wit['y'] = X, y
        d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
    msg, priv = SP.site_prove(pub, wit, d0, nz, [k[0] for k in keys], rng)
    ref = M.plain_step(cfg, pub.weights, wit['X'], wit['y'], wit['rho'], pub.beta)
    assert np.array_equal(priv['u'], ref['u']), label
    assert SP.site_verify(pub, d0.root, nz.root, msg)[0], label
    print('  ok  %-58s clipped records %d of %d' % (label, int((ref['kappa'] == 0).sum()), cfg.b))

check(M.Config(center=True), 'centred inputs')
check(M.Config(arch=(64, 16, 12, 10), b=8), 'three-layer network')
check(M.Config(arch=(64, 12, 10), b=8, Cg=1 << 16, sh=21), 'threshold so large that no record is clipped')
check(M.Config(arch=(64, 12, 10), b=8, Cg=4, sh=7), 'threshold so small that every record is clipped hard')
# extremes: all weights at the public clamp (both signs), inputs at their maximum and at zero
for sign, xval, name in ((1, 15, 'weights +max, inputs max'), (-1, 15, 'weights -max, inputs max'), (1, 0, 'weights +max, inputs zero')):
    cfg = M.Config(arch=(64, 12, 10), b=8)
    W = [(np.full((cfg.arch[l], cfg.arch[l - 1]), sign * cfg.Wmax, dtype=np.int64), np.full(cfg.arch[l], sign * cfg.Wmax, dtype=np.int64)) for l in range(1, cfg.L + 1)]
    check(cfg, 'extreme: ' + name, weights=W, X=np.full((8, 64), xval, dtype=np.int64), y=np.arange(8) % 10)
cfg = M.Config(arch=(64, 12, 10), b=8); g = np.random.default_rng(3)
W = [(g.choice([-cfg.Wmax, cfg.Wmax], size=(cfg.arch[l], cfg.arch[l - 1])), g.choice([-cfg.Wmax, cfg.Wmax], size=cfg.arch[l])) for l in range(1, cfg.L + 1)]
check(cfg, 'extreme: random +-max weights, random inputs', weights=W, X=g.integers(0, 16, (8, 64)), y=g.integers(0, 10, 8))
# random differential test: random models (weights anywhere inside the clamp), random data, centred and not
g = np.random.default_rng(11); n_rand = 0
for center in (False, True):
    for trial in range(8):
        cfg = M.Config(arch=(64, 12, 10), b=8, center=center)
        scale = int(g.choice([8, 64, cfg.Wmax]))
        W = [(g.integers(-scale, scale + 1, size=(cfg.arch[l], cfg.arch[l - 1])), g.integers(-scale, scale + 1, size=cfg.arch[l])) for l in range(1, cfg.L + 1)]
        pub, wit, d0, nz, keys, rng = setup(cfg, prm, K=5, kp=2, T=1, seed=100 + trial)
        pub.weights = W
        X, y = g.integers(0, 16, (8, 64)), g.integers(0, 10, 8)
        wit['X'], wit['y'] = X, y
        d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
        msg, priv = SP.site_prove(pub, wit, d0, nz, [k[0] for k in keys], rng)
        ref = M.plain_step(cfg, W, X, y, wit['rho'], pub.beta)
        assert np.array_equal(priv['u'], ref['u']) and SP.site_verify(pub, d0.root, nz.root, msg)[0]
        n_rand += 1
print('  ok  %d random models and data sets, centred and not: proved vector equals the reference' % n_rand)
print('step variants passed')
