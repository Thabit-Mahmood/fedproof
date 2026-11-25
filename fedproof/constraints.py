"""Constraint system: cells laid out in rows of ell slots, grouped in blocks.

Row kinds: BIT rows (every slot satisfies v*v = v), FREE rows (no quadratic constraint),
and MUL triples of rows (X, Y, Z) with X*Y = Z slot by slot. Linear constraints are sparse
equations sum coef*cell = const over any cells of any block.

The same builder code is run by the prover (which also assigns values) and by the verifier
(prover=False, structure only), so both derive an identical statement.
"""
import numpy as np
from . import field as F

BIT, FREE, MX, MY, MZ = range(5)
_KS, _BS = 40, 44
_IDX = (1 << _KS) - 1


class _Seg:
    def __init__(self):
        self.data = np.zeros(256, dtype=np.uint64)
        self.size = 0

    def grow(self, n):
        start = self.size
        self.size += n
        if self.size > len(self.data):
            new = np.zeros(max(self.size, 2 * len(self.data)), dtype=np.uint64)
            new[:start] = self.data[:start]
            self.data = new
        return start


def fe(c):
    """Coefficients (python ints, signed arrays, or field arrays) -> field array."""
    if isinstance(c, np.ndarray) and c.dtype == np.uint64:
        return c
    if isinstance(c, (int, np.integer)):
        return np.array(int(c) % F.P, dtype=np.uint64)
    return F.from_int(c)


class LC:
    """A vector of n affine combinations of cells: value[i] = sum_j coef[i, j] * cell[ids[i, j]] + const[i]."""

    def __init__(self, n, terms=(), const=None):
        self.n = n
        self.terms = list(terms)                      # (ids (n, w) int64, coef broadcastable to (n, w) uint64)
        self.const = np.zeros(n, dtype=np.uint64) if const is None else np.broadcast_to(fe(const), (n,)).copy()

    @staticmethod
    def cells(ids, coef=1):
        """ids (n,) -> one cell per instance (coef scalar or per instance);
        ids (n, w) -> w cells per instance (coef scalar, per column (w,), or full (n, w) / (n, 1) / (1, w))."""
        ids = np.asarray(ids, dtype=np.int64)
        c = fe(coef)
        if ids.ndim == 1:
            ids = ids[:, None]
            if c.ndim == 1:
                c = c[:, None]
        elif c.ndim == 1:
            assert c.shape[0] == ids.shape[1], 'a 1-D coefficient for 2-D ids is per column'
        return LC(ids.shape[0], [(ids, c)])

    @staticmethod
    def constant(n, val):
        return LC(n, [], val)

    def __add__(self, o):
        if isinstance(o, (int, np.integer)):
            return LC(self.n, self.terms, F.fadd(self.const, fe(o)))
        assert o.n == self.n
        return LC(self.n, self.terms + o.terms, F.fadd(self.const, o.const))

    def scale(self, c):
        """Multiply by a scalar or by a per-instance vector (n,)."""
        c = fe(c)
        cc = c[:, None] if c.ndim == 1 else c
        return LC(self.n, [(ids, F.fmul(np.broadcast_to(co, ids.shape), np.broadcast_to(cc, ids.shape)))
                           for ids, co in self.terms], F.fmul(self.const, c))

    def __neg__(self):
        return self.scale(F.P - 1)

    def __sub__(self, o):
        if isinstance(o, (int, np.integer)):
            return LC(self.n, self.terms, F.fsub(self.const, fe(o)))
        return self + (-o)


class CS:
    def __init__(self, ell, prover=True, rng=None):
        self.ell = ell
        self.prover = prover
        self.rng = rng
        self.blocks = []
        self.bidx = {}
        self._con, self._vid, self._coef, self._const = [], [], [], []
        self.ncon = 0
        self.mats = {}
        self.lenient = False          # test-only: a cheating prover may hold out-of-range values

    # ------------------------------------------------------------ cells
    def add_block(self, name, free_fill=None):
        self.bidx[name] = len(self.blocks)
        self.blocks.append(dict(name=name, fill=free_fill or self.ell, segs={k: _Seg() for k in range(5)},
                                frozen=False, nrows=None))

    def alloc(self, name, kind, n):
        b = self.bidx[name]
        blk = self.blocks[b]
        assert not blk['frozen'], 'block %s is frozen' % name
        assert kind in (BIT, FREE)
        start = blk['segs'][kind].grow(int(n))
        return (b << _BS) | (kind << _KS) | np.arange(start, start + int(n), dtype=np.int64)

    def alloc_mul(self, name, n):
        b = self.bidx[name]
        blk = self.blocks[b]
        assert not blk['frozen'], 'block %s is frozen' % name
        out = []
        for kind in (MX, MY, MZ):
            start = blk['segs'][kind].grow(int(n))
            out.append((b << _BS) | (kind << _KS) | np.arange(start, start + int(n), dtype=np.int64))
        return out

    def _segs_of(self, vids):
        key = vids >> _KS
        k0 = int(key.flat[0]) if key.size else 0
        if key.size == 0 or (key == k0).all():
            return [(None, self.blocks[k0 >> 4]['segs'][k0 & 15])] if key.size else []
        return [(key == k, self.blocks[int(k) >> 4]['segs'][int(k) & 15]) for k in np.unique(key)]

    def set(self, vids, vals):
        if not self.prover:
            return
        vids = np.asarray(vids, dtype=np.int64)
        vals = np.broadcast_to(fe(vals), vids.shape)
        for mask, seg in self._segs_of(vids):
            if mask is None:
                seg.data[vids & _IDX] = vals
            else:
                seg.data[vids[mask] & _IDX] = vals[mask]

    def get(self, vids):
        vids = np.asarray(vids, dtype=np.int64)
        out = np.zeros(vids.shape, dtype=np.uint64)
        for mask, seg in self._segs_of(vids):
            if mask is None:
                out = seg.data[vids & _IDX]
            else:
                out[mask] = seg.data[vids[mask] & _IDX]
        return out

    def val(self, lc):
        """Prover-side value of an LC."""
        acc = lc.const.copy()
        for ids, coef in lc.terms:
            prod = F.fmul(self.get(ids), np.broadcast_to(coef, ids.shape))
            acc = F.fadd(acc, F.fsum(prod, axis=1))
        return acc

    # ------------------------------------------------------------ constraints
    def eq(self, lc, rhs=None):
        """Add the n linear constraints lc == rhs (rhs an LC, an int, or None for zero)."""
        if rhs is not None:
            lc = lc - rhs
        base = self.ncon
        for ids, coef in lc.terms:
            con = np.broadcast_to(np.arange(base, base + lc.n, dtype=np.int64)[:, None], ids.shape)
            self._con.append(con.reshape(-1))
            self._vid.append(ids.reshape(-1))
            self._coef.append(np.broadcast_to(coef, ids.shape).reshape(-1))
        self._const.append(F.fneg(lc.const))
        self.ncon += lc.n

    def mul(self, name, a, b):
        """Allocate a multiplication gate per instance: returns ids of z = a * b."""
        x, y, z = self.alloc_mul(name, a.n)
        self.eq(LC.cells(x), a)
        self.eq(LC.cells(y), b)
        if self.prover:
            va, vb = self.val(a), self.val(b)
            self.set(x, va); self.set(y, vb); self.set(z, F.fmul(va, vb))
        return z

    def bits(self, name, lc, nbits, signed=True):
        """Decompose lc into nbits boolean cells: sum 2^k b_k = lc + offset, offset = 2^(nbits-1) if signed.

        Proves lc lies in [-2^(nbits-1), 2^(nbits-1)) (signed) or [0, 2^nbits) (unsigned). Returns ids (n, nbits).
        """
        assert 1 <= nbits <= 62
        off = (1 << (nbits - 1)) if signed else 0
        ids = self.alloc(name, BIT, lc.n * nbits).reshape(lc.n, nbits)
        pw = fe(np.array([1 << k for k in range(nbits)], dtype=np.int64))
        self.eq(LC.cells(ids, pw), lc + off)
        if self.prover:
            v = F.to_signed(self.val(lc)) + np.int64(off)
            if (v < 0).any() or (v >> np.int64(nbits)).any():
                if not self.lenient:
                    raise ValueError('bits(): value out of range for %d bits in block %s' % (nbits, name))
                v = v & np.int64((1 << nbits) - 1)
            self.set(ids, ((v[:, None] >> np.arange(nbits, dtype=np.int64)[None, :]) & 1).astype(np.uint64))
        return ids

    def mul_full(self, name, a, b):
        """Like mul() but returns the ids of all three cells (x, y, z)."""
        x, y, z = self.alloc_mul(name, a.n)
        self.eq(LC.cells(x), a)
        self.eq(LC.cells(y), b)
        if self.prover:
            va, vb = self.val(a), self.val(b)
            self.set(x, va); self.set(y, vb); self.set(z, F.fmul(va, vb))
        return x, y, z

    def bits_raw(self, name, n, nbits, vals=None):
        """Allocate n*nbits boolean cells holding the binary digits of the non-negative integers vals (no constraint)."""
        ids = self.alloc(name, BIT, n * nbits).reshape(n, nbits)
        if self.prover:
            v = np.asarray(vals, dtype=np.int64)
            if (v < 0).any() or (v >> np.int64(nbits)).any():
                if not self.lenient:
                    raise ValueError('bits_raw(): value out of range for %d bits in block %s' % (nbits, name))
                v = v & np.int64((1 << nbits) - 1)
            self.set(ids, ((v[:, None] >> np.arange(nbits, dtype=np.int64)[None, :]) & 1).astype(np.uint64))
        return ids

    @staticmethod
    def from_bits(ids, lo=0, hi=None):
        """LC equal to sum_{k=lo}^{hi-1} 2^(k-lo) * bit_k."""
        hi = ids.shape[1] if hi is None else hi
        pw = fe(np.array([1 << (k - lo) for k in range(lo, hi)], dtype=np.int64))
        return LC.cells(ids[:, lo:hi], pw)

    # ------------------------------------------------------------ layout
    def freeze(self, name):
        blk = self.blocks[self.bidx[name]]
        segs = blk['segs']
        assert segs[MX].size == segs[MY].size == segs[MZ].size
        rows = {}
        for kind in range(5):
            fill = blk['fill'] if kind == FREE else self.ell
            rows[kind] = -(-segs[kind].size // fill)
        base, acc = {}, 0
        for kind in range(5):
            base[kind] = acc
            acc += rows[kind]
        blk.update(frozen=True, nrows=acc, rows=rows, base=base)
        return acc

    def _row0(self):
        r0, acc = [], 0
        for blk in self.blocks:
            assert blk['frozen'], 'block %s not frozen' % blk['name']
            r0.append(acc)
            acc += blk['nrows']
        return r0, acc

    def gcell(self, vids):
        """Virtual ids -> global cell index (global row * ell + slot)."""
        r0, _ = self._row0()
        nkey = len(self.blocks) << 4
        rowbase = np.zeros(nkey, dtype=np.int64)
        fill = np.full(nkey, self.ell, dtype=np.int64)
        for b, blk in enumerate(self.blocks):
            for kind in range(5):
                rowbase[(b << 4) | kind] = r0[b] + blk['base'][kind]
            fill[(b << 4) | FREE] = blk['fill']
        key = vids >> _KS
        idx = vids & _IDX
        f = fill[key]
        return (rowbase[key] + idx // f) * self.ell + idx % f

    def message_matrix(self, name):
        """Prover: the block's rows (nrows x ell). FREE rows get random padding, others zero padding."""
        blk = self.blocks[self.bidx[name]]
        ell = self.ell
        M = np.zeros((blk['nrows'], ell), dtype=np.uint64)
        for kind in range(5):
            seg, rows, base = blk['segs'][kind], blk['rows'][kind], blk['base'][kind]
            if rows == 0:
                continue
            if kind == FREE:
                fill = blk['fill']
                sub = F.rand_field((rows, ell), self.rng)
                live = sub[:, :fill].copy().reshape(-1)
                live[:seg.size] = seg.data[:seg.size]
                sub[:, :fill] = live.reshape(rows, fill)
                M[base:base + rows] = sub
            else:
                flat = np.zeros(rows * ell, dtype=np.uint64)
                flat[:seg.size] = seg.data[:seg.size]
                M[base:base + rows] = flat.reshape(rows, ell)
        self.mats[name] = M
        return M

    def system(self):
        """The statement handed to the argument: sparse linear system and the quadratic row structure."""
        r0, m = self._row0()
        bit_rows, mx, my, mz = [], [], [], []
        for b, blk in enumerate(self.blocks):
            bit_rows.append(r0[b] + blk['base'][BIT] + np.arange(blk['rows'][BIT], dtype=np.int64))
            for lst, kind in ((mx, MX), (my, MY), (mz, MZ)):
                lst.append(r0[b] + blk['base'][kind] + np.arange(blk['rows'][kind], dtype=np.int64))
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dtype=dt)
        vid = cat(self._vid, np.int64)
        return dict(m=m, ell=self.ell, ncon=self.ncon, con=cat(self._con, np.int64), cell=self.gcell(vid),
                    coef=cat(self._coef, np.uint64), const=cat(self._const, np.uint64),
                    bit_rows=cat(bit_rows, np.int64), mx=cat(mx, np.int64), my=cat(my, np.int64), mz=cat(mz, np.int64),
                    block_rows=[blk['nrows'] for blk in self.blocks])

    def check(self, sysd=None, mats=None):
        """Prover-side debugging: evaluate every constraint directly on the witness. Returns list of failures."""
        sysd = sysd or self.system()
        mats = mats or self.mats
        W = np.vstack([mats[blk['name']] for blk in self.blocks])
        w = W.reshape(-1)
        resid = F.fsub(F.scatter_mul(sysd['ncon'], sysd['con'], sysd['coef'], w[sysd['cell']]), sysd['const'])
        fails = []
        if resid.any():
            fails.append(('lin', np.nonzero(resid)[0]))
        B = W[sysd['bit_rows']]
        if (F.fmul(B, B) != B).any():
            fails.append(('bit', np.nonzero((F.fmul(B, B) != B).any(axis=1))[0]))
        if len(sysd['mx']):
            bad = F.fmul(W[sysd['mx']], W[sysd['my']]) != W[sysd['mz']]
            if bad.any():
                fails.append(('mul', np.nonzero(bad.any(axis=1))[0]))
        return fails
