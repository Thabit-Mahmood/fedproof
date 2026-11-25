"""Constraints for the verified training step (see model.plain_step for the reference semantics).

Blocks, in global row order:
  D0  the data chunk as committed earlier (FREE rows, half of every row is random padding)
  NZ  the noise bits committed in the previous round (BIT rows)
  D1  the same chunk re-committed with fresh randomness for its next use
  M   main witness of this round
  A   auxiliary witness committed after the first challenge (Freivalds products), followed by the mask rows
"""
import numpy as np
from . import field as F
from . import model as M
from .constraints import CS, LC, BIT, FREE, fe

BLOCKS = ('D0', 'NZ', 'D1', 'M', 'A')


def new_cs(ell, prover, rng=None):
    cs = CS(ell, prover, rng)
    for name in BLOCKS:
        cs.add_block(name, free_fill=ell // 2 if name in ('D0', 'D1') else None)
    return cs


def data_layout(cs, name, cfg):
    xd = cs.alloc(name, FREE, cfg.b * cfg.arch[0]).reshape(cfg.b, cfg.arch[0])
    yd = cs.alloc(name, FREE, cfg.b)
    return xd, yd


def noise_layout(cs, name, cfg):
    return cs.alloc(name, BIT, cfg.d * cfg.J * cfg.r).reshape(cfg.d, cfg.J * cfg.r)


def data_block_matrix(cfg, ell, X, y, rng=None):
    """Message rows of a stand-alone data block (used at registration)."""
    cs = CS(ell, True, rng)
    cs.add_block('D', free_fill=ell // 2)
    xd, yd = data_layout(cs, 'D', cfg)
    cs.set(xd, np.asarray(X, dtype=np.int64)); cs.set(yd, np.asarray(y, dtype=np.int64))
    cs.freeze('D')
    return cs.message_matrix('D')


def noise_block_matrix(cfg, ell, rho):
    cs = CS(ell, True)
    cs.add_block('N')
    ids = noise_layout(cs, 'N', cfg)
    cs.set(ids, np.asarray(rho, dtype=np.int64).reshape(cfg.d, -1))
    cs.freeze('N')
    return cs.message_matrix('N')


def block_rows(cfg, ell):
    """Number of rows of a data block and of a noise block (public layout constants)."""
    nd = cfg.b * cfg.arch[0] + cfg.b
    return -(-nd // (ell // 2)), -(-(cfg.d * cfg.J * cfg.r) // ell)


# ---------------------------------------------------------------- gadgets
def lin_layer(ids_in, scale_in, W, const):
    """n = b*dout instances: sum_k W[i, k] * scale_in * cell[j, k] + const[i]  (instance index j*dout + i)."""
    b, din = ids_in.shape
    dout = W.shape[0]
    ids = np.broadcast_to(ids_in[:, None, :], (b, dout, din)).reshape(b * dout, din)
    coef = np.broadcast_to(fe(np.asarray(W, dtype=np.int64) * int(scale_in))[None, :, :], (b, dout, din)).reshape(b * dout, din)
    cst = np.broadcast_to(fe(np.asarray(const, dtype=np.int64))[None, :], (b, dout)).reshape(-1)
    return LC(b * dout, [(ids, coef)], cst)


def nonzero(cs, blk, lc):
    """Flag cell equal to 1 when lc != 0 and 0 when lc == 0."""
    x, y, z = cs.alloc_mul(blk, lc.n)
    cs.eq(LC.cells(x), lc)
    if cs.prover:
        vx = cs.val(lc)
        inv = F.finv(vx)                       # inverse of zero is taken as zero
        cs.set(x, vx); cs.set(y, inv); cs.set(z, F.fmul(vx, inv))
    z2 = cs.mul(blk, lc, LC.cells(z))
    cs.eq(LC.cells(z2), lc)
    return z


def ceilsqrt1(cs, blk, V, nbits, hk=None, name=''):
    """LC for s = isqrt(V) + 1, the unique s >= 1 with 0 <= s^2 - V - 1 <= 2s - 2."""
    sv = M.ceilsqrt1(np.maximum(F.to_signed(cs.val(V)), 0)) if cs.prover else None
    sb = cs.bits_raw(blk, V.n, nbits, sv)
    if hk is not None:
        hk(name, sb)
    s = cs.from_bits(sb)
    ssq = cs.mul(blk, s, s)
    lo = LC.cells(ssq) - V - 1
    cs.bits(blk, lo, nbits + 1, signed=False)
    cs.bits(blk, s.scale(2) - 2 - lo, nbits + 1, signed=False)
    return s


# ---------------------------------------------------------------- phase 1
def build_phase1(cs, cfg, shr, G, weights, beta, wit=None, hook=None):
    """Everything committed before the first challenge. wit = dict(X, y, rho) for the prover."""
    R = cfg.ranges()
    b, L, arch, d = cfg.b, cfg.L, cfg.arch, cfg.d
    Fa, Fw, Bx = cfg.Fa, cfg.Fw, cfg.Bx
    Pv = cs.prover

    def hk(name, ids):
        # test-only: lets a cheating prover overwrite cells at a named point; later values are computed from them
        if Pv and hook is not None:
            hook(name, cs, ids)

    xd, yd = data_layout(cs, 'D0', cfg)
    rho = noise_layout(cs, 'NZ', cfg)
    xd1, yd1 = data_layout(cs, 'D1', cfg)
    if Pv:
        Xw, yw = np.asarray(wit['X'], dtype=np.int64), np.asarray(wit['y'], dtype=np.int64)
        cs.set(xd, Xw); cs.set(yd, yw); cs.set(xd1, Xw); cs.set(yd1, yw)
        cs.set(rho, np.asarray(wit['rho'], dtype=np.int64).reshape(d, -1))
    hk('data', (xd, yd, xd1, yd1, rho))
    for name in ('D0', 'NZ', 'D1'):
        cs.freeze(name)
    # the refreshed chunk equals the old one
    cs.eq(LC.cells(xd1.reshape(-1)), LC.cells(xd.reshape(-1)))
    cs.eq(LC.cells(yd1), LC.cells(yd))
    # inputs are Bx-bit integers, labels are one-hot
    cs.bits('M', LC.cells(xd.reshape(-1)), Bx, signed=False)
    C = arch[-1]
    yb = cs.alloc('M', BIT, b * C).reshape(b, C)
    if Pv:
        oh = np.zeros((b, C), dtype=np.int64)
        oh[np.arange(b), yw] = 1
        cs.set(yb, oh)
    hk('onehot', yb)
    cs.eq(LC.cells(yb, np.ones((1, C), dtype=np.int64)), 1)
    cs.eq(LC.cells(yb, np.arange(C, dtype=np.int64)[None, :]), LC.cells(yd))

    # ---- forward pass
    h_list = [(xd, 1 << (Fa - Bx), -(cfg.xoff << (Fa - Bx)))]      # (cells, scale, constant): layer input = scale*cell + constant
    act_list = [None]
    delta = [None] * (L + 1)
    dsq = [None] * (L + 1)
    for l in range(1, L + 1):
        W, bv = weights[l - 1]
        dl = arch[l]
        hid, hsc, hc = h_list[l - 1]
        Wi = np.asarray(W, dtype=np.int64)
        zlc = lin_layer(hid, hsc, Wi, (np.asarray(bv, dtype=np.int64) << Fa) + hc * Wi.sum(axis=1))
        if l < L:
            Bz = R['Bz'][l - 1]
            zb = cs.bits('M', zlc, Bz)
            sign = LC.cells(zb[:, Bz - 1])
            if Bz - 1 - Fw <= cfg.Bh:
                hz = cs.mul('M', sign, cs.from_bits(zb, Fw, Bz - 1))
                act = sign
            else:
                qlo = cs.from_bits(zb, Fw, Fw + cfg.Bh)
                qhi = cs.from_bits(zb, Fw + cfg.Bh, Bz - 1)
                sat = LC.cells(nonzero(cs, 'M', qhi))
                v1 = cs.mul('M', sat, (-qlo) + cfg.Hmax)
                hz = cs.mul('M', sign, qlo + LC.cells(v1))
                ss = cs.mul('M', sign, sat)
                act = sign - LC.cells(ss)
            hk('zbits%d' % l, zb)
            hk('h%d' % l, hz)
            h_list.append((hz.reshape(b, dl), 1, 0))
            act_list.append(act)
        else:
            # one-vs-all smoothed hinge: p = clamp(z/4 + 1/2, 0, 1) at scale 2^Fd, error = p - label
            Be, she, Fd = R['Be'], R['she'], cfg.Fd
            eb = cs.bits('M', zlc + (2 << (Fw + Fa)), Be)
            sign = LC.cells(eb[:, Be - 1])
            if Be - 1 - she <= Fd:
                pz = cs.mul('M', sign, cs.from_bits(eb, she, Be - 1))
            else:
                qlo = cs.from_bits(eb, she, she + Fd)
                qhi = cs.from_bits(eb, she + Fd, Be - 1)
                sat = LC.cells(nonzero(cs, 'M', qhi))
                v1 = cs.mul('M', sat, (-qlo) + (1 << Fd))
                pz = cs.mul('M', sign, qlo + LC.cells(v1))
            hk('p', pz)
            dlc = LC.cells(pz) - LC.cells(yb.reshape(-1), 1 << Fd)
            dx, _, dz = cs.mul_full('M', dlc, dlc)
            hk('delta%d' % L, dx)
            delta[L] = dx.reshape(b, dl)
            dsq[L] = dz.reshape(b, dl)

    # ---- backward pass
    for l in range(L - 1, 0, -1):
        Wn = np.asarray(weights[l][0], dtype=np.int64)            # W_{l+1}, shape (d_{l+1}, d_l)
        Bt = R['Bt'][l]
        tlc = lin_layer(delta[l + 1], 1, Wn.T, np.zeros(arch[l], dtype=np.int64))
        tb = cs.bits('M', tlc, Bt)
        fl = cs.from_bits(tb, Fw, Bt) - (1 << (Bt - 1 - Fw))
        dcell = cs.mul('M', act_list[l], fl)
        hk('delta%d' % l, dcell)
        delta[l] = dcell.reshape(b, arch[l])
        dsq[l] = cs.mul('M', LC.cells(dcell), LC.cells(dcell)).reshape(b, arch[l])

    # ---- per-record bound on the gradient norm
    msq = []
    for l in range(1, L + 1):
        a = ceilsqrt1(cs, 'M', LC.cells(dsq[l], 1), R['Ba'][l], hk, 'a%d' % l)
        hid, hsc, hc = h_list[l - 1]
        hl = LC.cells(hid.reshape(-1), hsc) + hc
        hsq = cs.mul('M', hl, hl).reshape(b, -1)
        c = ceilsqrt1(cs, 'M', LC.cells(hsq, 1) + (1 << (2 * Fa)), R['Bc'][l], hk, 'c%d' % l)
        ac = cs.mul('M', a, c)
        mv = -((-F.to_signed(cs.get(ac))) >> cfg.g) if Pv else None
        mb = cs.bits_raw('M', b, R['Bm'][l], mv)
        hk('m%d' % l, mb)
        mlc = cs.from_bits(mb)
        cs.bits('M', mlc.scale(1 << cfg.g) - LC.cells(ac), cfg.g, signed=False)
        msq.append(cs.mul('M', mlc, mlc))
    nlc = ceilsqrt1(cs, 'M', LC.cells(np.stack(msq, axis=1), 1), R['Bn'], hk, 'n')

    # ---- clip scale
    num = cfg.Cg * cfg.Ss
    assert cfg.Cg < (1 << R['Bn'])
    spv = (num // np.maximum(F.to_signed(cs.val(nlc)), 1)) if Pv else None
    spb = cs.bits_raw('M', b, R['Bsp'], spv)
    hk('sp', spb)
    sp = cs.from_bits(spb)
    prod = cs.mul('M', sp, nlc)
    e = (-LC.cells(prod)) + num
    cs.bits('M', e, R['Bn'], signed=False)
    cs.bits('M', nlc - 1 - e, R['Bn'], signed=False)
    kb = cs.bits('M', (-nlc) + cfg.Cg, R['Bn'] + 1, signed=True)
    kappa = LC.cells(kb[:, R['Bn']])
    ksel = cs.mul('M', kappa, (-sp) + cfg.Ss)
    slc = sp + LC.cells(ksel)
    s_cell = cs.alloc('M', FREE, b)
    if Pv:
        cs.set(s_cell, cs.val(slc))
    hk('kappa', kb)
    hk('s', s_cell)
    cs.eq(LC.cells(s_cell), slc)

    # ---- scaled errors and the summed gradient U (bit-decomposed)
    BU = R['BU']
    off = 1 << (BU - 1)
    sd = [None]
    Uint = np.zeros(d, dtype=np.int64)
    for l, (o, ow, ob) in zip(range(1, L + 1), cfg.layer_slices()):
        dl = arch[l]
        sdl = cs.mul('M', LC.cells(np.repeat(s_cell, dl)), LC.cells(delta[l].reshape(-1))).reshape(b, dl)
        sd.append(sdl)
        hk('sd%d' % l, sdl)
        if Pv:
            sdv = F.to_signed(cs.get(sdl))
            hid, hsc, hc = h_list[l - 1]
            Uint[o:ow] = (sdv.T @ (F.to_signed(cs.get(hid)) * hsc + hc)).reshape(-1)
            Uint[ow:ob] = sdv.sum(0) << Fa
    Ub = cs.bits_raw('M', d, BU, Uint + off if Pv else None)
    hk('U', Ub)
    for l, (o, ow, ob) in zip(range(1, L + 1), cfg.layer_slices()):
        cs.eq(cs.from_bits(Ub[ow:ob]) - off, LC.cells(sd[l].T, 1 << Fa))            # bias coordinates

    # ---- truncation, noise, the shared vector u
    tq = cs.from_bits(Ub, cfg.sh, BU) - (1 << (BU - 1 - cfg.sh))
    JR = cfg.J * cfg.r
    pw = np.tile(1 << np.arange(cfg.r, dtype=np.int64), cfg.J)
    beta2 = np.asarray(beta, dtype=np.int64).reshape(d, JR)
    noise = LC(d, [(rho, fe(pw[None, :] * (1 - 2 * beta2)))], fe((pw[None, :] * beta2).sum(1)))
    ulc = tq + noise
    u_cell = cs.alloc('M', FREE, d)
    if Pv:
        cs.set(u_cell, cs.val(ulc))
    hk('u', u_cell)
    cs.eq(LC.cells(u_cell), ulc)

    # ---- sharing randomness
    anc = cs.alloc('M', FREE, shr.nblk * shr.T).reshape(shr.nblk, shr.T)
    msk = cs.alloc('M', FREE, G * shr.w).reshape(G, shr.w)
    shares = None
    if Pv:
        cs.set(anc, F.rand_field(anc.shape, cs.rng)); cs.set(msk, F.rand_field(msk.shape, cs.rng))
        vals = np.zeros((shr.nblk + G, shr.w), dtype=np.uint64)
        up = np.zeros(shr.nblk * shr.kp, dtype=np.uint64)
        up[:d] = cs.get(u_cell)
        vals[:shr.nblk, :shr.kp] = up.reshape(shr.nblk, shr.kp)
        vals[:shr.nblk, shr.kp:] = cs.get(anc)
        vals[shr.nblk:] = cs.get(msk)
        shares = shr.share_values(vals)                     # K x (nblk + G)
        if hook is not None:
            sh2 = hook('shares', cs, shares)
            shares = shares if sh2 is None else sh2
    cs.freeze('M')
    return dict(h=h_list, sd=sd, Ub=Ub, u=u_cell, anc=anc, msk=msk, shares=shares, R=R)


def derive_challenges(T, cfg, shr, G):
    ch = dict(a={}, c={})
    for l in range(1, cfg.L + 1):
        ch['a'][l] = [T.chal_field('fa%d_%d' % (l, g), cfg.arch[l]) for g in range(G)]
        ch['c'][l] = [T.chal_field('fc%d_%d' % (l, g), cfg.arch[l - 1]) for g in range(G)]
    ch['gamma'] = T.chal_field('gamma', G * shr.nblk).reshape(G, shr.nblk)
    return ch


# ---------------------------------------------------------------- phase 2
def build_phase2(cs, cfg, shr, G, ctx, ch, digests=None):
    """Freivalds checks of the weight-gradient sums and the public share digests. Returns digests (K x G)."""
    R = ctx['R']
    b, L, arch, d = cfg.b, cfg.L, cfg.arch, cfg.d
    BU = R['BU']
    off = 1 << (BU - 1)
    pw = fe(np.array([1 << k for k in range(BU)], dtype=np.int64))
    Ub = ctx['Ub']
    for g in range(G):
        for l, (o, ow, ob) in zip(range(1, L + 1), cfg.layer_slices()):
            a, c = ch['a'][l][g], ch['c'][l][g]
            hid, hsc, hc = ctx['h'][l - 1]
            qlc = LC.cells(hid, F.fmul(c, fe(hsc))[None, :])
            qlc = LC(qlc.n, qlc.terms, np.broadcast_to(F.fmul(F.fsum(c), fe(hc)), (qlc.n,)))
            z = cs.mul('A', LC.cells(ctx['sd'][l], a[None, :]), qlc)
            ack = F.fmul(np.repeat(a, arch[l - 1]), np.tile(c, arch[l]))                 # a_i c_k, row-major (i, k)
            coef = F.fmul(np.repeat(ack, BU), np.tile(pw, len(ack)))
            rhs = LC(1, [(Ub[o:ow].reshape(1, -1), coef.reshape(1, -1))],
                     F.fneg(F.fmul(F.fsum(ack), fe(off))).reshape(1))
            cs.eq(LC.cells(z.reshape(1, -1), 1), rhs)
    K, kp, Tn = shr.K, shr.kp, shr.T
    pos = np.arange(d) % kp
    blk = np.arange(d) // kp
    out = np.zeros((K, G), dtype=np.uint64)
    for g in range(G):
        gam = ch['gamma'][g]
        cu = F.fmul(shr.lam[:, pos], np.broadcast_to(gam[blk][None, :], (K, d)))
        ca = F.fmul(np.tile(shr.lam[:, kp:], (1, shr.nblk)), np.broadcast_to(np.repeat(gam, Tn)[None, :], (K, shr.nblk * Tn)))
        lc = LC(K, [(np.broadcast_to(ctx['u'][None, :], (K, d)), cu),
                    (np.broadcast_to(ctx['anc'].reshape(1, -1), (K, shr.nblk * Tn)), ca),
                    (np.broadcast_to(ctx['msk'][g][None, :], (K, shr.w)), shr.lam)])
        out[:, g] = cs.val(lc) if cs.prover else digests[:, g]
        cs.eq(lc, LC.constant(K, out[:, g]))
    cs.freeze('A')
    return out


def node_check(shr, G, gamma, node, share_row, digests_row):
    """What node `node` checks on its decrypted shares: share_row has nblk + G entries."""
    for g in range(G):
        v = F.fadd(F.fdot(gamma[g], share_row[:shr.nblk]), share_row[shr.nblk + g])
        if int(v) != int(digests_row[g]):
            return False
    return True
