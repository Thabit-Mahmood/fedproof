"""Record-level differential privacy accounting for the distributed integer noise.

Each site adds, per coordinate, the sum of J independent uniform r-bit integers. The opened aggregate therefore carries
at least M = H_min * J such uniforms from honest sites. Replacing one record shifts the truncated sum by an integer
vector D with ||D||_2 <= Delta (model.Config.sens) and |D_i| <= S.

The noise has bounded support, so the Renyi divergence of the shifted distributions is infinite. We therefore work, as in
approximate concentrated DP (Bun and Steinke), with the distributions conditioned on a central event E whose complement
has probability at most delta0, and add delta0 once per release:
  h_alpha(s)   Renyi divergence of order alpha between P|E and (P shifted by s)|E, one coordinate
  kappa_alpha  max over 1 <= s <= S of h_alpha(s) / s^2,  so a release costs at most kappa_alpha * Delta^2
The exact probability mass function is computed with integer arithmetic.
"""
import math
from itertools import accumulate
import numpy as np


class UniformSum:
    """Exact distribution of the sum of M independent uniform integers on {0, ..., 2^r - 1}."""

    def __init__(self, M, r):
        self.M, self.r = M, r
        W = 1 << r
        c = [1] * W
        for _ in range(M - 1):
            pre = [0] + list(accumulate(c))                   # pre[i] = sum c[:i]
            n_new = len(c) + W - 1
            c = [pre[min(z, len(c) - 1) + 1] - pre[max(0, z - W + 1)] for z in range(n_new)]
        self.counts = c
        self.zmax = len(c) - 1
        self.total_bits = M * r
        assert sum(c) == 1 << self.total_bits
        ln2 = math.log(2.0)
        self.logp = np.array([math.log(v) for v in c]) - self.total_bits * ln2
        self.cum = list(accumulate(c))                        # cum[z] = number of outcomes <= z
        self.var = M * (4.0 ** r - 1) / 12.0
        self.sigma = math.sqrt(self.var)

    def cdf(self, z):
        """P(Z <= z) as a float (exact rational, correctly rounded)."""
        if z < 0:
            return 0.0
        z = min(z, self.zmax)
        return self.cum[z] / (1 << self.total_bits) if self.cum[z].bit_length() > self.total_bits - 1000 else 0.0

    def log_cdf(self, z):
        if z < 0:
            return -math.inf
        return math.log(self.cum[min(z, self.zmax)]) - self.total_bits * math.log(2.0)

    def central_cut(self, S, delta0, d):
        """Largest L with 2 d P(Z < L + S) <= delta0: the event E_s = {L + s <= z <= zmax - L}."""
        target = math.log(delta0 / (2.0 * d))
        lo, hi = 0, self.zmax // 2 - S
        if self.log_cdf(lo + S - 1) > target:
            return None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.log_cdf(mid + S - 1) <= target:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def h(self, alpha, s, L):
        """Renyi divergence of order alpha between P and P(. - s), both conditioned on {L + s <= z <= zmax - L}."""
        a = self.logp[L + s: self.zmax - L + 1]
        b = self.logp[L: self.zmax - L - s + 1]
        t = alpha * a + (1.0 - alpha) * b
        m = t.max()
        lse = m + math.log(np.exp(t - m).sum())
        logPE = math.log(np.exp(a).sum())
        return (lse - logPE) / (alpha - 1.0)

    def kappa(self, alpha, S, L, stride=1):
        best, arg = 0.0, 1
        for s in range(1, S + 1, stride):
            v = self.h(alpha, s, L) / (s * s)
            if v > best:
                best, arg = v, s
        if stride > 1:
            v = self.h(alpha, S, L) / (S * S)
            if v > best:
                best, arg = v, S
        return best, arg

    def hockey_stick(self, s, eps):
        """Exact delta(eps) of one release with a single-coordinate shift s (both directions are equal by symmetry)."""
        p = np.exp(self.logp)
        q = np.zeros_like(p)
        q[s:] = p[:-s]
        return float(np.maximum(p - math.exp(eps) * q, 0.0).sum())


def rdp_to_dp(rdp, alpha, delta):
    """Standard conversion (Mironov 2017): (alpha, rdp)-RDP implies (rdp + log(1/delta)/(alpha-1), delta)-DP."""
    return rdp + math.log(1.0 / delta) / (alpha - 1.0)


ALPHAS = (1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64)


def account(cfg, H_min, n_uses, delta=1e-5, alphas=ALPHAS, stride=16, dist=None, edge_frac=0.01, exact=True):
    """Privacy of one record that is used in n_uses rounds, when at least H_min honest sites add noise each time.

    The order alpha is chosen with a coarse scan over shifts (every stride-th shift). With exact=True the constant
    kappa of the chosen order is then recomputed over EVERY integer shift 1..S, so the returned epsilon is a rigorous
    upper bound (any single order gives a valid bound).
    """
    M = H_min * cfg.J
    dist = dist or UniformSum(M, cfg.r)
    Delta = cfg.sens
    S = int(2 * cfg.Chat // (1 << cfg.sh)) + 1
    delta0 = edge_frac * delta / n_uses
    L = dist.central_cut(S, delta0, cfg.d)
    if L is None:
        return dict(eps=math.inf, eps_gauss=math.inf, alpha=None, z=dist.sigma / Delta, sigma=dist.sigma, Delta=Delta, S=S,
                    reason='noise support too small for this sensitivity')
    dconv = delta * (1 - edge_frac)
    best = None
    table = []
    for a in alphas:
        k, arg = dist.kappa(a, S, L, stride)
        eps = rdp_to_dp(n_uses * k * Delta * Delta, a, dconv)
        table.append((a, k / (a / (2.0 * dist.var)), arg, eps))
        if best is None or eps < best[0]:
            best = (eps, a, k, arg)
    eps, a, k, arg = best
    if exact and stride > 1:
        k, arg = dist.kappa(a, S, L, 1)
        eps = rdp_to_dp(n_uses * k * Delta * Delta, a, dconv)
    return dict(eps=eps, alpha=a, kappa=k, kappa_argmax=arg, delta=delta, sigma=dist.sigma, Delta=Delta, S=S, z=dist.sigma / Delta,
                L=L, delta_edge=n_uses * delta0, table=table, M=M, exact=bool(exact or stride == 1),
                eps_gauss=min(rdp_to_dp(n_uses * al * Delta * Delta / (2 * dist.var), al, delta) for al in alphas))
