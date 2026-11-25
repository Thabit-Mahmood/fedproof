"""The proved vector equals the reference semantics for the model shapes of the industrial datasets:
8-bit features, 41 to 61 inputs, 2 to 15 classes, two hidden layers, chunks of 16 and 32 records."""
import sys
sys.path.insert(0, '.')
import numpy as np
from fedproof import field as F, model as M, ligero as L, step as ST, message as SP, pq
from fedproof.sharing import Sharing


def run(cfg, seed, prm, K=5, kp=2, T=1):
    g = np.random.default_rng(seed)
    X = g.integers(0, 1 << cfg.Bx, (cfg.b, cfg.arch[0]))
    X[0] = (1 << cfg.Bx) - 1; X[1] = 0                       # one record at each input extreme
    y = g.integers(0, cfg.arch[-1], cfg.b)
    rho = g.integers(0, 2, (cfg.d, cfg.J, cfg.r)); beta = g.integers(0, 2, (cfg.d, cfg.J, cfg.r))
    W = M.proof_weights(cfg, M.init_model(cfg))
    if seed % 2:                                               # a model far from initialisation
        W = [(g.integers(-cfg.Wmax, cfg.Wmax + 1, w.shape), g.integers(-cfg.Wmax, cfg.Wmax + 1, b.shape)) for w, b in W]
    shr = Sharing(cfg.d, kp, T, K)
    rng = F.Drbg(('iiot', seed))
    d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
    nz = L.commit_block(ST.noise_block_matrix(cfg, prm.ell, rho), prm, rng)
    keys = [pq.kem_keygen() for _ in range(K)]
    pub = SP.Public(cfg, prm, shr, W, beta, b'round=1|site=0|shape-test')
    msg, priv = SP.site_prove(pub, dict(X=X, y=y, rho=rho), d0, nz, [k[0] for k in keys], rng)
    ref = M.plain_step(cfg, W, X, y, rho, beta)
    assert np.array_equal(priv['u'], ref['u'])
    assert SP.site_verify(pub, d0.root, nz.root, msg)[0]
    return int((ref['kappa'] == 0).sum())


prm = L.Params(1024, 64, 64, 2)
for arch, Bx, b in [((61, 64, 32, 15), 8, 16), ((61, 64, 32, 15), 8, 32), ((41, 64, 32, 5), 8, 16), ((43, 64, 32, 10), 8, 16), ((59, 64, 32, 10), 8, 16), ((61, 128, 64, 15), 8, 16), ((41, 32, 2), 8, 32), ((61, 64, 32, 15), 4, 16)]:
    for seed in (1, 2):
        cfg = M.Config(arch=arch, Bx=Bx, b=b)
        nclip = run(cfg, seed, prm)
        print('  ok  arch %-16s Bx %d b %2d seed %d  d=%5d  clipped %d of %d' % (arch, Bx, b, seed, cfg.d, nclip, b))
print('industrial shapes passed')
