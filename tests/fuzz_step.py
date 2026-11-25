"""Differential fuzz: random (legal) configurations, adversarial models and data; honest prover vs reference.
A break would be: constraints satisfied / proof verifies while proved u != plain_step u, or a crash in an
accepted config. Usage: python3 tests/fuzz_step.py [n_trials] [seed]"""
import sys, time, traceback
sys.path.insert(0, '.')
import numpy as np
from fedproof import field as F, model as M, ligero as L, step as ST, message as SP, pq
from fedproof.sharing import Sharing

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
g = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
prm = L.Params(256, 32, 32, 1)
keys = [pq.kem_keygen() for _ in range(3)]

def rand_cfg():
    L_ = int(g.integers(1, 4))
    arch = [int(g.integers(1, 9))] + [int(g.integers(1, 7)) for _ in range(L_ - 1)] + [int(g.integers(1, 5))]
    Bx = int(g.integers(1, 9)); Fa = int(g.integers(Bx, 12)); Fw = int(g.integers(0, 12))
    Fd = int(g.integers(1, Fw + Fa + 3)); Bw = int(g.integers(2, 13)); Bh = int(g.integers(1, 13))
    return M.Config(arch=tuple(arch), b=int(g.integers(1, 5)), Bx=Bx, center=bool(g.integers(0, 2)), Fa=Fa, Fw=Fw,
                    Bw=Bw, Bh=Bh, Fd=Fd, g=int(g.integers(1, 10)), Cg=int(g.choice([1, 2, 3, 7, 64, 1 << 10, 1 << 14])),
                    Bs=int(g.integers(0, 10)), sh=int(g.integers(0, 20)), J=1, r=3)

def rand_w(cfg, mode):
    out = []
    for l in range(1, cfg.L + 1):
        sh = (cfg.arch[l], cfg.arch[l - 1])
        if mode == 0: W, b = g.integers(-cfg.Wmax, cfg.Wmax + 1, sh), g.integers(-cfg.Wmax, cfg.Wmax + 1, cfg.arch[l])
        elif mode == 1: W, b = g.choice([-cfg.Wmax, cfg.Wmax], sh), g.choice([-cfg.Wmax, cfg.Wmax], cfg.arch[l])
        else:
            s = int(g.choice([1, -1])); W, b = np.full(sh, s * cfg.Wmax), np.full(cfg.arch[l], s * cfg.Wmax)
        out.append((np.asarray(W, dtype=np.int64), np.asarray(b, dtype=np.int64)))
    return out

done = refused = good = 0; stats = dict(sat=0, neg=0, mid=0, clip=0, noclip=0, hsat=0); t0 = time.time()
while done < N:
    cfg = rand_cfg()
    try:
        R = cfg.ranges()
        if cfg.Cg >= (1 << R['Bn']) or any(0 < R['Bt'][l] and R['Bt'][l] - 1 < cfg.Fw for l in range(1, cfg.L)): refused += 1; continue
        shr = Sharing(cfg.d, 1, 1, 3)
    except AssertionError:
        refused += 1; continue
    W = rand_w(cfg, int(g.integers(0, 3)))
    xm = int(g.integers(0, 3))
    X = g.integers(0, 1 << cfg.Bx, (cfg.b, cfg.arch[0])) if xm == 0 else np.full((cfg.b, cfg.arch[0]), 0 if xm == 1 else (1 << cfg.Bx) - 1)
    y = g.integers(0, cfg.arch[-1], cfg.b)
    rho = g.integers(0, 2, (cfg.d, cfg.J, cfg.r)); beta = g.integers(0, 2, (cfg.d, cfg.J, cfg.r))
    rng = F.Drbg(('fz', done))
    try:
        d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
        nz = L.commit_block(ST.noise_block_matrix(cfg, prm.ell, rho), prm, rng)
        pub = SP.Public(cfg, prm, shr, W, beta, b'fuzz')
        msg, priv = SP.site_prove(pub, dict(X=X, y=y, rho=rho), d0, nz, [k[0] for k in keys], rng)
    except Exception as ex:
        print("PROVER FAILED (completeness) on", cfg.arch, "Bx",cfg.Bx,"Fa",cfg.Fa,"Fw",cfg.Fw,"Fd",cfg.Fd,"Bw",cfg.Bw,"Bh",cfg.Bh, type(ex).__name__, ex, traceback.format_exc().splitlines()[-3].strip()); done += 1; continue
    ref = M.plain_step(cfg, W, X, y, rho, beta)
    ok = SP.site_verify(pub, d0.root, nz.root, msg)[0]
    same = np.array_equal(priv['u'], ref['u'])
    if ok and not same:
        print('BREAK: verified u != reference', cfg)
    elif not ok:
        print('honest proof rejected', cfg)
    else:
        good += 1
        e = ref['zs'][cfg.L] + (2 << (cfg.Fw + cfg.Fa)); she = cfg.Fw + cfg.Fa - cfg.Fd + 2
        v = e >> she
        stats['sat'] += int((v > (1 << cfg.Fd)).sum()); stats['neg'] += int((e < 0).sum()); stats['mid'] += int(((v >= 0) & (v <= (1 << cfg.Fd))).sum())
        stats['clip'] += int((ref['kappa'] == 0).sum()); stats['noclip'] += int((ref['kappa'] == 1).sum())
        stats['hsat'] += sum(int(((z >> cfg.Fw) > cfg.Hmax).sum()) for z in ref['zs'][1:cfg.L])
    done += 1
print('fuzz: %d trials, %d honest proofs verified and equal to reference, %d configs refused; coverage %s; %.0fs' % (done, good, refused, stats, time.time() - t0))
