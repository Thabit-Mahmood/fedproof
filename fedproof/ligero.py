"""Zero-knowledge argument over interleaved Reed-Solomon codes (Ligero-type), made non-interactive with Fiat-Shamir.

Every committed row is a polynomial p(X) = m(X) + (X^ell - 1) r(X) of degree < k = ell + t', where m interpolates the
row's ell message slots on the subgroup H_ell and r is a random polynomial of degree < t'. The codeword is the vector
of evaluations on the coset 7 * H_n. Columns are hashed with a fresh salt into a SHA3-384 Merkle tree.

Three tests are run sigma times on the same t opened columns:
  proximity : a random combination of all rows is a codeword;
  linear    : sum_i R_i(X) p_i(X) sums to r^T b over H_ell, where R = r^T A for the sparse system A w = b;
  quadratic : sum_i rho_i (p_i^2 - p_i) + sum_j rho_j (px_j py_j - pz_j) vanishes on H_ell.
Each test is blinded by mask rows that are themselves rows of the committed matrix (degree < k), so the proximity
test also covers them.
"""
import math
import os
import numpy as np
from . import field as F
from .hashing import MerkleTree, merkle_verify, leaf_hash, H, DIGEST

SALT = 32
MASKS_PER_REP = 5          # prox, lin_a, lin_b, quad_c1, quad_c2


class Params:
    def __init__(self, ell, tprime, t, sigma, blowup=4, name=''):
        assert ell & (ell - 1) == 0 and blowup in (2, 4, 8, 16)
        assert 2 <= tprime <= ell and 1 <= t <= tprime and sigma >= 1
        self.ell, self.tprime, self.t, self.sigma, self.blowup, self.name = ell, tprime, t, sigma, blowup, name
        self.k = ell + tprime
        self.n = blowup * ell
        assert 2 * self.k - 1 <= self.n, 'quadratic test needs 2k-1 <= n'
        assert ell // 2 + tprime >= 2 * t, 'data blocks are opened twice: need ell/2 + t\' >= 2t'
        self.deg_lin = self.k + ell - 1
        self.deg_quad = self.k + tprime - 1
        self._tab = None

    def ident(self):
        return ('ligero-v1|ell=%d|tp=%d|t=%d|sigma=%d|blowup=%d|hash=sha3-384|field=goldilocks'
                % (self.ell, self.tprime, self.t, self.sigma, self.blowup)).encode()

    def tables(self):
        if self._tab is None:
            eta = F.coset_points(self.n)
            zh = F.fsub(F.fpow(eta, self.ell), F.U64(1))
            self._tab = dict(eta=eta, zh=zh, zh_inv=F.finv(zh), e_l1=F.fpow(eta, self.ell - 1),
                             e_t1=F.fpow(eta, self.tprime - 1))
        return self._tab

    def soundness(self):
        """Round-by-round soundness errors (log2) from the unique-decoding analysis."""
        n, k, t = self.n, self.k, self.t

        def lchoose(a, b):
            if b < 0 or b > a:
                return -math.inf
            return (math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)) / math.log(2)

        best = None
        for e in range(0, (n - k) // 2 + 1):
            far = lchoose(n - e - 1, t) - lchoose(n, t)                    # proximity: all queries miss > e bad columns
            close = lchoose(min(n, e + 2 * k - 2), t) - lchoose(n, t)      # quadratic (dominates linear)
            v = max(far, close)
            if best is None or v < best[0]:
                best = (v, e)
        # ((n + 2)/|F|)^sigma: proximity gap n/|F| plus the linear and quadratic identity tests (1/|F| each), |F| = p > 2^64 - 2^32
        alg = self.sigma * (math.log2(n + 2) - 64 + math.log2(1 + 2**-32))
        return dict(query_bits=-best[0], e=best[1], algebraic_bits=-alg,
                    rbr_bits=min(-best[0], -alg), zk_ok=self.t <= self.tprime)


def statement_soundness(prm, n_layers, n_nodes=1):
    """Soundness of a site's whole statement (log2 of the error, as positive bits), per round of the interactive protocol.

    Adds to Params.soundness() the two checks whose challenges are drawn after the first commitment: the randomised
    matrix identity (error at most 2/|F| per layer and repetition) and, on the node side, the share digests
    (error 1/|F| per repetition).
    """
    base = prm.soundness()
    logF = math.log2(F.P)
    freivalds = prm.sigma * (logF - 1) - math.log2(max(n_layers, 1))
    digest = prm.sigma * logF - math.log2(max(n_nodes, 1))          # union over the nodes that check digests
    out = dict(base)
    out.update(freivalds_bits=freivalds, digest_bits=digest, total_bits=min(base['rbr_bits'], freivalds, digest))
    return out


def find_params(ell, target_bits, blowup=4, name=''):
    """Smallest t (= t') and sigma reaching target_bits of round-by-round soundness, or None."""
    n = blowup * ell
    sigma = math.ceil(target_bits / (64 - math.log2(n) - 1e-9))
    for t in range(8, ell // 2 + 1, 2):                               # Params needs ell/2 + t' >= 2t, i.e. t <= ell/2
        if 2 * (ell + t) - 1 > n:
            break
        p = Params(ell, t, t, sigma, blowup, name)
        if p.soundness()['rbr_bits'] >= target_bits:
            return p
    return None


# ---------------------------------------------------------------- commitments
class Committed:
    """Prover-side commitment to a block of rows."""

    def __init__(self, coeffs, prm, rng=None):
        self.coeffs = coeffs
        self.nrows = coeffs.shape[0]
        self.cw = F.coset_eval(coeffs, prm.n)
        raw = os.urandom(SALT * prm.n) if rng is None else rng.read(SALT * prm.n)
        self.salts = [raw[i * SALT:(i + 1) * SALT] for i in range(prm.n)]
        cols = np.ascontiguousarray(self.cw.T)
        self.tree = MerkleTree([leaf_hash(self.salts[j], cols[j].tobytes()) for j in range(prm.n)])
        self.root = self.tree.root


def encode_rows(msgs, prm, rng=None):
    """Message rows (rows x ell) -> coefficient rows (rows x k) with fresh randomisers."""
    msgs = F.arr(msgs)
    rows = msgs.shape[0]
    assert msgs.shape[1] == prm.ell
    co = np.zeros((rows, prm.k), dtype=np.uint64)
    if rows:
        co[:, :prm.ell] = F.ntt(msgs, inverse=True)
        r = F.rand_field((rows, prm.tprime), rng)
        co[:, :prm.tprime] = F.fsub(co[:, :prm.tprime], r)
        co[:, prm.ell:] = r
    return co


def decode_rows(coeffs, prm):
    """Coefficient rows -> message rows (evaluations on H_ell)."""
    lo = coeffs[:, :prm.ell].copy()
    lo[:, :prm.tprime] = F.fadd(lo[:, :prm.tprime], coeffs[:, prm.ell:])
    return F.ntt(lo)


def mask_rows(prm, rng=None):
    """The 5*sigma mask rows (coefficient form)."""
    co = F.rand_field((MASKS_PER_REP * prm.sigma, prm.k), rng)
    ell = prm.ell
    for s in range(prm.sigma):
        a, b = co[5 * s + 1], co[5 * s + 2]
        tot = (int(a[ell]) + int(b[1]) + int(b[ell + 1])) % F.P      # a + X^(ell-1) b must sum to zero over H_ell
        a[0] = (F.P - tot) % F.P
    return co


def commit_block(msgs, prm, rng=None, with_masks=False):
    co = encode_rows(msgs, prm, rng)
    if with_masks:
        co = np.vstack([co, mask_rows(prm, rng)])
    return Committed(co, prm, rng)


# ---------------------------------------------------------------- the argument
def _sys_digest(sysd):
    parts = [str((sysd['m'], sysd['ell'], sysd['ncon'], tuple(sysd['block_rows']))).encode()]
    for key in ('con', 'cell', 'coef', 'const', 'bit_rows', 'mx', 'my', 'mz'):
        parts.append(H(np.ascontiguousarray(sysd[key]).tobytes()))
    return H(b''.join(parts))


def _challenges(prm, sysd, T, M):
    m = sysd['m']
    nb, nm = len(sysd['bit_rows']), len(sysd['mx'])
    T.absorb('params', prm.ident())
    T.absorb('system', _sys_digest(sysd))
    ch = []
    for s in range(prm.sigma):
        rp = T.chal_field('prox%d' % s, M)
        rp[m + 5 * np.arange(prm.sigma)] = 0          # prox masks of other repetitions are left out
        rp[m + 5 * s] = 1                             # own prox mask enters with coefficient one
        rl = T.chal_field('lin%d' % s, sysd['ncon'])
        rq = T.chal_field('quad%d' % s, nb + nm)
        ch.append((rp, rl, rq))
    return ch


def _R_matrix(sysd, rl):
    R = F.scatter_mul(sysd['m'] * sysd['ell'], sysd['cell'], rl[sysd['con']], sysd['coef'])
    return R.reshape(sysd['m'], sysd['ell'])


def _lin_target(sysd, rl):
    return int(F.fdot(rl, sysd['const'])) if sysd['ncon'] else 0


def _sum_over_H(q, ell):
    """sum of q(zeta) over the subgroup H_ell = ell * (sum of coefficients at multiples of ell)."""
    return int(F.fmul(F.fsum(q[::ell]), F.U64(ell)))


CHUNK = 64


def prove(prm, blocks, sysd, T, strict=True):
    """blocks: Committed objects in global row order; the last one ends with the 5*sigma mask rows."""
    m, ell, n, k = sysd['m'], prm.ell, prm.n, prm.k
    M = m + MASKS_PER_REP * prm.sigma
    assert sum(b.nrows for b in blocks) == M and sysd['ell'] == ell
    tab = prm.tables()
    U = np.vstack([b.cw for b in blocks])
    C = np.vstack([b.coeffs for b in blocks])
    ch = _challenges(prm, sysd, T, M)
    Ub = U[sysd['bit_rows']]
    Ub2 = F.fmul(Ub, Ub) if len(Ub) else Ub
    Uxy = F.fmul(U[sysd['mx']], U[sysd['my']]) if len(sysd['mx']) else None
    Uz = U[sysd['mz']] if len(sysd['mx']) else None
    nb = len(sysd['bit_rows'])
    q_prox, q_lin, q_quad = [], [], []
    for s, (rp, rl, rq) in enumerate(ch):
        q_prox.append(F.lincomb_rows(C, rp))
        # linear test
        R = _R_matrix(sysd, rl)
        acc = np.zeros(n, dtype=np.uint64)
        for i0 in range(0, m, CHUNK):
            Re = F.coset_eval(F.ntt(R[i0:i0 + CHUNK], inverse=True), n)
            acc = F.muladd_rows(Re, U[i0:i0 + Re.shape[0]], acc)
        acc = F.fadd(acc, F.fadd(U[m + 5 * s + 1], F.fmul(tab['e_l1'], U[m + 5 * s + 2])))
        co = F.coset_interp(acc)
        if strict:
            assert not co[prm.deg_lin:].any(), 'linear test polynomial has too high degree'
            assert _sum_over_H(co[:prm.deg_lin], ell) == _lin_target(sysd, rl), 'witness violates a linear constraint'
        q_lin.append(co[:prm.deg_lin].copy())
        # quadratic test
        Q = np.zeros(n, dtype=np.uint64)
        if nb:
            Q = F.fsub(F.lincomb_rows(Ub2, rq[:nb]), F.lincomb_rows(Ub, rq[:nb]))
        if Uxy is not None:
            Q = F.fadd(Q, F.fsub(F.lincomb_rows(Uxy, rq[nb:]), F.lincomb_rows(Uz, rq[nb:])))
        he = F.fadd(F.fmul(Q, tab['zh_inv']), F.fadd(U[m + 5 * s + 3], F.fmul(tab['e_t1'], U[m + 5 * s + 4])))
        hc = F.coset_interp(he)
        if strict:
            assert not hc[prm.deg_quad:].any(), 'witness violates a quadratic constraint'
        q_quad.append(hc[:prm.deg_quad].copy())
    q_prox, q_lin, q_quad = np.array(q_prox), np.array(q_lin), np.array(q_quad)
    T.absorb('q_prox', q_prox); T.absorb('q_lin', q_lin); T.absorb('q_quad', q_quad)
    cols = T.chal_indices('columns', prm.t, n)
    opens = []
    for b in blocks:
        opens.append(dict(vals=np.ascontiguousarray(b.cw[:, cols]), salts=[b.salts[j] for j in cols],
                          path=b.tree.open(cols)))
    return dict(q_prox=q_prox, q_lin=q_lin, q_quad=q_quad, opens=opens)


def _canon(a, shape):
    return (isinstance(a, np.ndarray) and a.dtype == np.uint64 and a.shape == shape and bool((a < np.uint64(F.P)).all()))


def verify(prm, metas, sysd, T, proof):
    """metas: list of (root, nrows) per block in global row order (last block includes the mask rows)."""
    try:
        return bool(_verify(prm, metas, sysd, T, proof))
    except Exception:              # any malformed input is a rejection, never a crash
        return False


def _verify(prm, metas, sysd, T, proof):
    m, ell, n, k, t, sg = sysd['m'], prm.ell, prm.n, prm.k, prm.t, prm.sigma
    M = m + MASKS_PER_REP * sg
    if sum(nr for _, nr in metas) != M or sysd['ell'] != ell or len(proof['opens']) != len(metas):
        return False
    if not (_canon(proof['q_prox'], (sg, k)) and _canon(proof['q_lin'], (sg, prm.deg_lin))
            and _canon(proof['q_quad'], (sg, prm.deg_quad))):
        return False
    tab = prm.tables()
    ch = _challenges(prm, sysd, T, M)
    T.absorb('q_prox', proof['q_prox']); T.absorb('q_lin', proof['q_lin']); T.absorb('q_quad', proof['q_quad'])
    cols = T.chal_indices('columns', t, n)
    parts = []
    for (root, nrows), op in zip(metas, proof['opens']):
        vals = op['vals']
        if not _canon(vals, (nrows, t)) or len(op['salts']) != t:
            return False
        if any((not isinstance(sl, (bytes, bytearray))) or len(sl) != SALT for sl in op['salts']):
            return False
        colsb = np.ascontiguousarray(vals.T)
        leaves = [leaf_hash(op['salts'][j], colsb[j].tobytes()) for j in range(t)]
        if not merkle_verify(root, n, cols, leaves, op['path']):
            return False
        parts.append(vals)
    Uc = np.vstack(parts)                                    # M x t
    eta_l1, eta_t1, zh = tab['e_l1'][cols], tab['e_t1'][cols], tab['zh'][cols]
    nb = len(sysd['bit_rows'])
    Ub = Uc[sysd['bit_rows']]
    Ub2 = F.fmul(Ub, Ub) if nb else Ub
    has_mul = len(sysd['mx']) > 0
    Uxy = F.fmul(Uc[sysd['mx']], Uc[sysd['my']]) if has_mul else None
    Uz = Uc[sysd['mz']] if has_mul else None
    for s, (rp, rl, rq) in enumerate(ch):
        # proximity
        if not np.array_equal(F.coset_eval(proof['q_prox'][s], n)[cols], F.lincomb_rows(Uc, rp)):
            return False
        # linear
        if _sum_over_H(proof['q_lin'][s], ell) != _lin_target(sysd, rl):
            return False
        R = _R_matrix(sysd, rl)
        acc = np.zeros(t, dtype=np.uint64)
        for i0 in range(0, m, CHUNK):
            Re = F.coset_eval(F.ntt(R[i0:i0 + CHUNK], inverse=True), n)[:, cols]
            acc = F.muladd_rows(Re, Uc[i0:i0 + Re.shape[0]], acc)
        acc = F.fadd(acc, F.fadd(Uc[m + 5 * s + 1], F.fmul(eta_l1, Uc[m + 5 * s + 2])))
        if not np.array_equal(F.coset_eval(proof['q_lin'][s], n)[cols], acc):
            return False
        # quadratic
        Q = np.zeros(t, dtype=np.uint64)
        if nb:
            Q = F.fsub(F.lincomb_rows(Ub2, rq[:nb]), F.lincomb_rows(Ub, rq[:nb]))
        if has_mul:
            Q = F.fadd(Q, F.fsub(F.lincomb_rows(Uxy, rq[nb:]), F.lincomb_rows(Uz, rq[nb:])))
        hq = F.coset_eval(proof['q_quad'][s], n)[cols]
        rhs = F.fmul(zh, F.fsub(hq, F.fadd(Uc[m + 5 * s + 3], F.fmul(eta_t1, Uc[m + 5 * s + 4]))))
        if not np.array_equal(Q, rhs):
            return False
    return True


def proof_size(proof):
    """Size in bytes of a canonical encoding of the proof."""
    size = 8 * (proof['q_prox'].size + proof['q_lin'].size + proof['q_quad'].size)
    for op in proof['opens']:
        size += 8 * op['vals'].size + SALT * len(op['salts']) + DIGEST * len(op['path'])
    return size


def proof_bytes(proof):
    out = [proof['q_prox'].tobytes(), proof['q_lin'].tobytes(), proof['q_quad'].tobytes()]
    for op in proof['opens']:
        out += [op['vals'].tobytes(), b''.join(op['salts']), b''.join(op['path'])]
    return b''.join(out)
