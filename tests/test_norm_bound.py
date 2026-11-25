"""Exact check of the proven sensitivity bound: for every record, ||s * G_record||_2 < Chat, computed with python
integers from the reference intermediates (which the fuzz shows equal the proved ones), over adversarial models/data.
Also an independent exact recomputation of Params.soundness query bits. Usage: python3 tests/test_norm_bound.py"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
from fractions import Fraction
from fedproof import model as M, ligero as L
g = np.random.default_rng(9); worst = 0.0; n_rec = 0
for trial in range(400):
    center = bool(trial % 2)
    cfg = M.Config(arch=(int(g.integers(2, 40)), int(g.integers(1, 20)), int(g.integers(2, 11))), b=4, center=center,
                   Cg=int(g.choice([1, 5, 300, 1 << 10, 1 << 14, 1 << 18])), g=int(g.integers(1, 12)), Bs=int(g.integers(0, 10)))
    try:
        R = cfg.ranges()
    except AssertionError:
        continue
    if cfg.Cg >= (1 << R['Bn']): continue
    mode = trial % 3
    W = []
    for l in range(1, cfg.L + 1):
        sh = (cfg.arch[l], cfg.arch[l - 1])
        if mode == 0: W.append((g.integers(-cfg.Wmax, cfg.Wmax + 1, sh), g.integers(-cfg.Wmax, cfg.Wmax + 1, cfg.arch[l])))
        else: W.append((g.choice([-cfg.Wmax, cfg.Wmax], sh), g.choice([-cfg.Wmax, cfg.Wmax], cfg.arch[l])))
    X = g.integers(0, 16, (4, cfg.arch[0])) if mode != 2 else np.full((4, cfg.arch[0]), int(g.choice([0, 15])))
    y = g.integers(0, cfg.arch[-1], 4)
    r = M.plain_step(cfg, W, X, y)
    for j in range(4):
        sq = 0
        for l in range(1, cfg.L + 1):
            dl = [int(v) for v in r['deltas'][l][j]]; hl = [int(v) for v in r['hs'][l - 1][j]]
            sq += sum(d * d for d in dl) * (sum(h * h for h in hl) + (1 << (2 * cfg.Fa)))
        s = int(r['s'][j]); n_rec += 1
        ratio = math.sqrt(s * s * sq) / cfg.Chat
        assert s * s * sq < cfg.Chat ** 2, ('bound violated', cfg, j)
        worst = max(worst, ratio)
print('norm bound: %d records, max ||sG||/Chat = %.4f (must be < 1)' % (n_rec, worst))

def lC(a, b): return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)
for prm in (L.find_params(1024, 128, blowup=8), L.Params(1024, 64, 64, 2), L.Params(2048, 300, 300, 3, blowup=8)):
    n, k, t = prm.n, prm.k, prm.t
    best = min(max(math.comb(n - e - 1, t), math.comb(min(n, e + 2 * k - 2), t)) for e in range(0, (n - k) // 2 + 1))
    exact = -math.log2(Fraction(best, math.comb(n, t)))
    P = 2**64 - 2**32 + 1
    alg = -prm.sigma * math.log2((n + 2) / P)          # prox n/p plus lin 1/p plus quad 1/p, per repetition
    print(prm.ident().decode(), 'calculator', {k_: round(v, 3) for k_, v in prm.soundness().items() if isinstance(v, float)},
          '| independent exact query bits %.3f, algebraic with (n+2)/p: %.3f' % (exact, alg))
