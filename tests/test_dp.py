import sys, time, math
sys.path.insert(0, '.')
import numpy as np
from dataclasses import replace
from fedproof import dp, model as M
# exactness of the distribution on a small case against brute force
import itertools
D = dp.UniformSum(3, 2)
ref = [0] * 10
for t in itertools.product(range(4), repeat=3): ref[sum(t)] += 1
assert D.counts == ref and abs(D.var - 3 * 15 / 12) < 1e-12
D = dp.UniformSum(5, 3); assert sum(D.counts) == 8 ** 5 and D.counts == D.counts[::-1]
mean = sum(z * c for z, c in enumerate(D.counts)) / 8 ** 5; var = sum((z - mean) ** 2 * c for z, c in enumerate(D.counts)) / 8 ** 5
assert abs(mean - 5 * 3.5) < 1e-9 and abs(var - D.var) < 1e-9
# Renyi divergence routine against a direct computation on a small case (no conditioning: L = 0 keeps the whole overlap)
p = np.array(D.counts, dtype=float) / 8 ** 5
for s in (1, 3, 7):
    for a in (1.5, 2, 4):
        ov = slice(s, D.zmax + 1); P_ = p[ov] / p[ov].sum(); Q_ = p[:D.zmax + 1 - s] / p[:D.zmax + 1 - s].sum()
        direct = math.log((P_ ** a * Q_ ** (1 - a)).sum()) / (a - 1)
        assert abs(direct - D.h(a, s, 0)) < 1e-9
print('distribution and divergence routines agree with brute force')
t0 = time.time()
cfg = M.Config()
for r in (11, 13):
    c = replace(cfg, r=r)
    res = dp.account(c, H_min=8, n_uses=1, stride=37)          # coarse scan picks the order, then every shift is checked for it
    assert res['exact']
    print('r=%d: sigma %.0f, Delta %.0f, noise multiplier z=%.2f, eps(1 use)=%.3f at alpha=%s [Gaussian with same variance: %.3f], cut L=%d, edge delta %.1e' % (
        r, res['sigma'], res['Delta'], res['z'], res['eps'], res['alpha'], res['eps_gauss'], res['L'], res['delta_edge']))
    print('   kappa / Gaussian value per order:', ' '.join('%g:%.4f(s=%d)' % (a, k, arg) for a, k, arg, _ in res['table']))
    # the exact one-release privacy profile (single-coordinate shift of the full sensitivity) must lie below the accountant
    dist = dp.UniformSum(8 * c.J, c.r)
    ex = dist.hockey_stick(res['S'], res['eps'])
    print('   exact delta at the accounted eps for the concentrated shift: %.3e (must be <= 1e-5)' % ex)
    assert ex <= 1e-5
print('%.0fs' % (time.time() - t0))
