"""Packed Shamir sharing over the Goldilocks field, with error-correcting reconstruction.

A block of kp secrets plus T random anchors defines a polynomial of degree < kp + T through the points
x = 1..kp (secrets) and x = kp+1..kp+T (anchors). Node m holds its value at alpha_m = kp + T + 1 + m.
Any T shares are independent of the secrets; any kp + T correct shares determine them; with K' shares of
which at most e are wrong, decoding succeeds when K' >= kp + T + 2e.
"""
import numpy as np
from . import field as F

P = F.P


def _lagrange_row(xs, x):
    """Coefficients c with f(x) = sum_c c[i] f(xs[i]) for deg f < len(xs)."""
    out = []
    for i, xi in enumerate(xs):
        num, den = 1, 1
        for j, xj in enumerate(xs):
            if j != i:
                num = num * ((x - xj) % P) % P
                den = den * ((xi - xj) % P) % P
        out.append(num * pow(den, P - 2, P) % P)
    return out


class Sharing:
    def __init__(self, d, kp, T, K):
        assert kp >= 1 and T >= 1 and K >= kp + T
        self.d, self.kp, self.T, self.K = d, kp, T, K
        self.w = kp + T
        self.nblk = -(-d // kp)
        self.xs = list(range(1, self.w + 1))
        self.alphas = [self.w + 1 + m for m in range(K)]
        self.lam = np.array([_lagrange_row(self.xs, a) for a in self.alphas], dtype=np.uint64)       # K x w

    def ident(self):
        return ('packed-shamir|d=%d|kp=%d|T=%d|K=%d' % (self.d, self.kp, self.T, self.K)).encode()

    def share_values(self, vals):
        """vals: (rows, w) field array -> shares (K, rows)."""
        return np.array([F.matvec(vals, self.lam[m]) for m in range(self.K)], dtype=np.uint64)

    def secret_coeffs(self, nodes):
        """Matrix (kp x w) mapping the shares of the first w listed nodes to the kp secrets."""
        pts = [self.alphas[m] for m in nodes[:self.w]]
        return np.array([_lagrange_row(pts, x) for x in self.xs[:self.kp]], dtype=np.uint64)


def _solve(Mx, ncols):
    """Gaussian elimination mod p on an augmented matrix (list of rows); returns one solution or None."""
    Mx = [r[:] for r in Mx]
    piv_cols, rk = [], 0
    for c in range(ncols):
        piv = next((r for r in range(rk, len(Mx)) if Mx[r][c]), None)
        if piv is None:
            continue
        Mx[rk], Mx[piv] = Mx[piv], Mx[rk]
        inv = pow(Mx[rk][c], P - 2, P)
        Mx[rk] = [v * inv % P for v in Mx[rk]]
        for r in range(len(Mx)):
            if r != rk and Mx[r][c]:
                f = Mx[r][c]
                Mx[r] = [(a - f * b) % P for a, b in zip(Mx[r], Mx[rk])]
        piv_cols.append(c)
        rk += 1
    if any(row[ncols] for row in Mx[rk:]):
        return None
    sol = [0] * ncols
    for r, c in enumerate(piv_cols):
        sol[c] = Mx[r][ncols]
    return sol


def _poly_eval(co, x):
    acc = 0
    for c in reversed(co):
        acc = (acc * x + c) % P
    return acc


def rs_decode(points, values, deg, emax):
    """Berlekamp-Welch. Find f with deg f < deg agreeing with all but <= emax of (points, values).

    Returns (coefficients of f, list of positions in error) or None when no such polynomial exists.
    """
    npts = len(points)
    values = [int(v) % P for v in values]
    for e in range(0, emax + 1):
        if npts < deg + 2 * e:
            break
        # unknowns: q_0..q_{deg+e-1}, E_0..E_{e-1} (E monic of degree e):  Q(x) - y (E(x)) = y x^e
        rows = []
        for x, y in zip(points, values):
            xp = [pow(x, i, P) for i in range(deg + e)]
            row = xp + [(-y * xp[i]) % P for i in range(e)] + [y * pow(x, e, P) % P]
            rows.append(row)
        sol = _solve(rows, deg + 2 * e)
        if sol is None:
            continue
        Q, E = sol[:deg + e], sol[deg + e:] + [1]
        # divide Q by E
        Qw = Q[:]
        quot = [0] * max(len(Q) - e, 1)
        for i in range(len(Q) - 1, e - 1, -1):
            c = Qw[i]
            quot[i - e] = c
            for j in range(e + 1):
                Qw[i - e + j] = (Qw[i - e + j] - c * E[j]) % P
        if any(Qw[:e]):
            continue
        f = quot[:deg] + [0] * (deg - len(quot))
        if any(quot[deg:]):
            continue
        bad = [i for i, (x, y) in enumerate(zip(points, values)) if _poly_eval(f, x) != y]
        if len(bad) <= e:
            return f, bad
    return None
