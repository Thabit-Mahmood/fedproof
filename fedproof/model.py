"""The training step every site must prove, as exact integer arithmetic (reference semantics).

All quantities are integers. Real values are recovered by dividing by a power of two:
  inputs x in [0, 2^Bx), activation scale 2^Fa, weight scale 2^Fw, error scale 2^Fd.
The step maps (public model, private chunk of b records, private committed noise bits, public beacon bits)
to the integer vector u that the site secret-shares.
"""
import math
import hashlib
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    arch: tuple = (64, 24, 10)
    b: int = 16            # records per chunk
    Bx: int = 4            # input bits
    center: bool = False   # subtract 2^(Bx-1) from every input (zero-centred features)
    Fa: int = 8            # activation fractional bits
    Fw: int = 8            # weight fractional bits (model used inside the proof)
    Bw: int = 10           # weight clamp: |W| <= 2^(Bw-1) - 1  (about 2.0 in real units)
    Bh: int = 10           # hidden activation clamp: h <= 2^Bh - 1
    Fd: int = 10           # error fractional bits
    g: int = 8             # norm rounding shift
    Cg: int = 1 << 10      # clipping threshold on the rounded norm (real threshold C = Cg * 2^g / 2^(Fd+Fa))
    Bs: int = 8            # clip scale S_s = 2^Bs
    sh: int = 15           # truncation of the summed, scaled gradient before sharing
    J: int = 4             # uniforms per coordinate per site
    r: int = 11            # bits per uniform
    # public update rule
    Fhp: int = 16          # extra fractional bits of the public high-precision weights
    lr_num: int = 1
    lr_shift: int = 1      # real learning rate = lr_num / 2^lr_shift
    mom_num: int = 9
    mom_den: int = 10
    init_seed: int = 0
    init_out_shift: int = 3   # output layer initial weights are divided by 2^init_out_shift
    init_bias: int = 64      # initial hidden bias, in weight units 2^Fw (64/256 = 0.25)

    # ---- derived
    @property
    def L(self):
        return len(self.arch) - 1

    @property
    def d(self):
        return sum(self.arch[l] * self.arch[l - 1] + self.arch[l] for l in range(1, self.L + 1))

    @property
    def xoff(self):
        return (1 << (self.Bx - 1)) if self.center else 0

    @property
    def Wmax(self):
        return (1 << (self.Bw - 1)) - 1

    @property
    def Hmax(self):
        return (1 << self.Bh) - 1

    @property
    def Ss(self):
        return 1 << self.Bs

    @property
    def Chat(self):
        """Proven bound on the l2 norm of one record's scaled gradient s*G."""
        return (1 << self.g) * self.Cg * self.Ss

    @property
    def C_real(self):
        return self.Cg * (1 << self.g) / float(1 << (self.Fd + self.Fa))

    @property
    def sens(self):
        """l2 sensitivity of the truncated vector tq to replacing one record."""
        return 2.0 * self.Chat / (1 << self.sh) + math.sqrt(self.d)

    def layer_slices(self):
        out, o = [], 0
        for l in range(1, self.L + 1):
            nw = self.arch[l] * self.arch[l - 1]
            out.append((o, o + nw, o + nw + self.arch[l]))
            o += nw + self.arch[l]
        return out

    def ranges(self):
        """Worst-case magnitudes and bit widths of every intermediate value; asserts nothing can wrap mod p."""
        R = {}
        Wm, Fa, Fw = self.Wmax, self.Fa, self.Fw
        assert self.g >= 1 and self.Fa >= self.Bx, 'norm rounding needs g >= 1; inputs need Fa >= Bx'
        hmax = [max((1 << self.Bx) - 1 - self.xoff, self.xoff) << (Fa - self.Bx)] + [self.Hmax] * (self.L - 1)
        R['hmax'] = hmax
        R['Bz'], R['zmax'] = [], []
        for l in range(1, self.L + 1):
            zmax = self.arch[l - 1] * Wm * hmax[l - 1] + Wm * (1 << Fa)
            R['zmax'].append(zmax)
            R['Bz'].append(zmax.bit_length() + 1)
        emax = R['zmax'][-1] + (2 << (Fw + Fa))           # output stage decomposes z + 2*one
        R['Be'] = emax.bit_length() + 1
        R['she'] = Fw + Fa - self.Fd + 2                   # p = clamp(z/4 + 1/2, 0, 1) at scale 2^Fd
        assert R['she'] >= 0
        dmax = [0] * (self.L + 1)
        dmax[self.L] = 1 << self.Fd
        R['Bt'] = [0] * (self.L + 1)
        for l in range(self.L - 1, 0, -1):
            tmax = self.arch[l + 1] * Wm * dmax[l + 1]
            R['Bt'][l] = max(tmax.bit_length() + 1, Fw + 1)   # the floor by 2^Fw needs Bt - 1 >= Fw
            dmax[l] = (tmax >> Fw) + 1
        R['dmax'] = dmax
        R['Ba'], R['Bc'], R['Bm'] = [0] * (self.L + 1), [0] * (self.L + 1), [0] * (self.L + 1)
        nq = 0
        for l in range(1, self.L + 1):
            A = self.arch[l] * dmax[l] ** 2
            Cq = self.arch[l - 1] * hmax[l - 1] ** 2 + (1 << (2 * Fa))
            a, c = math.isqrt(A) + 1, math.isqrt(Cq) + 1
            mm = -((-a * c) >> self.g)
            R['Ba'][l], R['Bc'][l], R['Bm'][l] = a.bit_length(), c.bit_length(), mm.bit_length()
            nq += mm * mm
            assert A < 1 << 62 and Cq < 1 << 62 and a * c < 1 << 62 and (1 << R['Ba'][l]) ** 2 < 1 << 62
            assert (1 << R['Bc'][l]) ** 2 < 1 << 62 and (1 << (R['Bm'][l] + self.g)) < 1 << 63
        nmax = math.isqrt(nq) + 1
        R['Bn'] = max(nmax.bit_length(), self.Cg.bit_length() + 1)      # the encoder also needs Cg < 2^Bn
        assert nq < 1 << 62 and (1 << R['Bn']) ** 2 < 1 << 62
        num = self.Cg * self.Ss
        R['Bsp'] = num.bit_length()                      # bits of s' = floor(num / n) (n >= 1)
        assert (1 << R['Bsp']) * (1 << R['Bn']) < 1 << 62
        R['sdmax'] = self.Ss * max(dmax[1:])
        umax = self.b * self.Chat
        R['BU'] = umax.bit_length() + 1
        assert R['BU'] <= 62 and R['BU'] - 1 >= self.sh
        assert self.b * R['sdmax'] * max(hmax + [1 << Fa]) < 1 << 62       # Freivalds identity holds over the integers
        R['noise_max'] = self.J * ((1 << self.r) - 1)
        R['u_absmax'] = (umax >> self.sh) + 1 + R['noise_max']
        return R


# ---------------------------------------------------------------- public model state
def init_model(cfg):
    """Deterministic public initialisation: high-precision integer weights (scale 2^(Fw+Fhp))."""
    hp = []
    for l in range(1, cfg.L + 1):
        fan_in, fan_out = cfg.arch[l - 1], cfg.arch[l]
        seed = hashlib.shake_256(b'sfl-init|%d|%d' % (cfg.init_seed, l)).digest(8 * (fan_in * fan_out))
        raw = np.frombuffer(seed, dtype='<u8').astype(np.float64) / 2.0**64          # uniform [0,1)
        lim = math.sqrt(6.0 / (fan_in + fan_out))
        W = np.floor((2 * raw - 1) * lim * (1 << (cfg.Fw + cfg.Fhp))).astype(np.int64).reshape(fan_out, fan_in)
        if l == cfg.L:
            W = W >> cfg.init_out_shift
            bvec = np.zeros(fan_out, dtype=np.int64)
        else:
            bvec = np.full(fan_out, cfg.init_bias << cfg.Fhp, dtype=np.int64)
        hp.append((W, bvec))
    return dict(hp=hp, vel=np.zeros(cfg.d, dtype=np.int64), round=0)


def proof_weights(cfg, state):
    """The clamped low-precision weights used inside the proof (rounded high-precision weights)."""
    out = []
    half = 1 << (cfg.Fhp - 1)
    for W, bvec in state['hp']:
        out.append((np.clip((W + half) >> cfg.Fhp, -cfg.Wmax, cfg.Wmax), np.clip((bvec + half) >> cfg.Fhp, -cfg.Wmax, cfg.Wmax)))
    return out


def model_digest(cfg, state):
    h = hashlib.sha3_384(repr(cfg).encode())
    for W, bvec in proof_weights(cfg, state):
        h.update(np.ascontiguousarray(W, dtype='<i8').tobytes()); h.update(np.ascontiguousarray(bvec, dtype='<i8').tobytes())
    return h.digest()


def apply_update(cfg, state, agg_sum, nvalid):
    """Public deterministic update from the opened sum of the sites' vectors u (python ints or int64)."""
    agg = np.asarray(agg_sum, dtype=object)
    # twice the centred sum of truncated gradients; + nvalid compensates the mean rounding-down of the truncation
    agg2 = 2 * agg - nvalid * cfg.J * ((1 << cfg.r) - 1) + nvalid
    vel = np.array([(cfg.mom_num * int(v)) // cfg.mom_den for v in state['vel']], dtype=object) + agg2
    # real mean gradient = agg2 * 2^sh / (2 * Ss * 2^(Fd+Fa) * nvalid * b); high-precision weight units 2^(Fw+Fhp)
    shift = cfg.sh + cfg.Fw + cfg.Fhp - (1 + cfg.Bs + cfg.Fd + cfg.Fa) - cfg.lr_shift
    num = vel * cfg.lr_num * (1 << max(shift, 0))
    den = nvalid * cfg.b * (1 << max(-shift, 0))
    step = np.array([(2 * int(x) + den) // (2 * den) for x in num], dtype=object)
    lim = cfg.Wmax << cfg.Fhp
    new_hp = []
    for (W, bvec), (o, ow, ob) in zip(state['hp'], cfg.layer_slices()):
        Wn = np.clip(W - step[o:ow].astype(np.int64).reshape(W.shape), -lim, lim)
        bn = np.clip(bvec - step[ow:ob].astype(np.int64), -lim, lim)
        new_hp.append((Wn, bn))
    return dict(hp=new_hp, vel=vel.astype(np.int64), round=state['round'] + 1)


# ---------------------------------------------------------------- integer helpers
def ceilsqrt1(V):
    """Smallest s >= 1 with s^2 >= V + 1, i.e. isqrt(V) + 1, for int64 arrays V >= 0."""
    V = np.asarray(V, dtype=np.int64)
    s = np.floor(np.sqrt(V.astype(np.float64))).astype(np.int64)
    s = np.where(s * s > V, s - 1, s)
    s = np.where(s * s > V, s - 1, s)
    s = np.where((s + 1) * (s + 1) <= V, s + 1, s)
    s = np.where((s + 1) * (s + 1) <= V, s + 1, s)
    assert ((s * s <= V) & ((s + 1) * (s + 1) > V)).all()
    return s + 1


def noise_from_bits(cfg, rho, beta):
    """rho, beta: (d, J, r) arrays of bits. nu = rho XOR beta; noise_i = sum_j sum_k 2^k nu[i, j, k] (uncentred)."""
    nu = np.bitwise_xor(rho.astype(np.int64), beta.astype(np.int64))
    return (nu * (1 << np.arange(cfg.r, dtype=np.int64))[None, None, :]).sum(axis=(1, 2))


def plain_step(cfg, weights, X, y, rho=None, beta=None, clip=True):
    """Reference semantics. X: (b, d0) ints in [0, 2^Bx); y: (b,) class labels. Returns dict of intermediates."""
    Fa, Fw = cfg.Fa, cfg.Fw
    L = cfg.L
    X = np.asarray(X, dtype=np.int64)
    bsz = X.shape[0]
    h = (X - cfg.xoff) << (Fa - cfg.Bx)
    hs, acts, zs = [h], [None], [None]
    for l in range(1, L + 1):
        W, bv = weights[l - 1]
        z = h @ W.T + bv * (1 << Fa)
        zs.append(z)
        if l < L:
            v = z >> Fw
            act = ((z >= 0) & (v <= cfg.Hmax)).astype(np.int64)
            h = np.clip(v, 0, cfg.Hmax)
            hs.append(h); acts.append(act)
    onehot = np.zeros((bsz, cfg.arch[-1]), dtype=np.int64)
    onehot[np.arange(bsz), y] = 1
    # one-vs-all smoothed hinge: p = clamp(z/4 + 1/2, 0, 1), error = p - y
    v = (zs[L] + (2 << (Fw + Fa))) >> (Fw + Fa - cfg.Fd + 2)
    deltas = [None] * (L + 1)
    deltas[L] = np.clip(v, 0, 1 << cfg.Fd) - (onehot << cfg.Fd)
    for l in range(L - 1, 0, -1):
        t = deltas[l + 1] @ weights[l][0]
        deltas[l] = acts[l] * (t >> Fw)
    nq = np.zeros(bsz, dtype=np.int64)
    a_l, c_l, m_l = [None], [None], [None]
    for l in range(1, L + 1):
        a = ceilsqrt1((deltas[l] ** 2).sum(1))
        c = ceilsqrt1((hs[l - 1] ** 2).sum(1) + (1 << (2 * Fa)))
        m = -((-(a * c)) >> cfg.g)
        a_l.append(a); c_l.append(c); m_l.append(m)
        nq += m * m
    n = ceilsqrt1(nq)
    num = cfg.Cg * cfg.Ss
    sp = num // n
    kappa = (n <= cfg.Cg).astype(np.int64)
    s = np.where(kappa == 1, cfg.Ss, sp) if clip else np.full(bsz, cfg.Ss, dtype=np.int64)
    U = np.zeros(cfg.d, dtype=np.int64)
    sds = [None]
    for l, (o, ow, ob) in zip(range(1, L + 1), cfg.layer_slices()):
        sd = s[:, None] * deltas[l]
        sds.append(sd)
        U[o:ow] = (sd.T @ hs[l - 1]).reshape(-1)
        U[ow:ob] = sd.sum(0) << Fa
    tq = U >> cfg.sh
    out = dict(hs=hs, acts=acts, zs=zs, deltas=deltas, a=a_l, c=c_l, m=m_l, n=n, sp=sp, kappa=kappa, s=s, sds=sds, U=U, tq=tq)
    if rho is not None:
        out['noise'] = noise_from_bits(cfg, rho, beta)
        out['u'] = tq + out['noise']
    return out


def predict(cfg, weights, X):
    h = (np.asarray(X, dtype=np.int64) - cfg.xoff) << (cfg.Fa - cfg.Bx)
    for l in range(1, cfg.L + 1):
        W, bv = weights[l - 1]
        z = h @ W.T + bv * (1 << cfg.Fa)
        if l < cfg.L:
            h = np.clip(z >> cfg.Fw, 0, cfg.Hmax)
    return z.argmax(1)
