"""Experiments that use the exact integer simulator (bit-identical to the proved pipeline, see tests/test_step*.py):
tuning on public data, E1 detection quality, E2 detection against privacy, E4 poisoning, E9 membership inference."""
import os, sys, json, time, math, itertools
sys.path.insert(0, '.')
import numpy as np
from dataclasses import replace
from fedproof import model as M, simulator as S
from experiments import common as C

R_E1 = 2 * C.PER_PLANT // C.B   # rounds: every chunk used twice (256 chunks of 128 per plant)
TUNE_USES = 8                   # passes over the public pseudo-plants when tuning
NOISE_R = (12, 13, 14, 15, 16)  # bits per uniform; J = 4 uniforms per coordinate per plant
H_SWEEP = range(1, C.PLANTS + 1)  # epsilon against the number of honest plants in every opened sum
USES = (1, 2, 4)                # times each record is used (10 plants): 256, 512 and 1024 rounds
SCALE_USES = (4, 8)             # 50 plants of 8192 records: 256 and 512 rounds
SEEDS = (0, 1, 2)


def _rounds_for(d, uses, b=C.B):
    return uses * min(len(p[1]) for p in d['plants']) // b


# ---------------------------------------------------------------- tuning (public data only)
def tune(dataset, quick=False):
    """Clipping threshold, truncation and learning rate chosen on the public 5% alone: it is split into 10 pseudo-plants
    (80%) and a validation part (20%). First without noise; then, for every noise level, the learning rate is chosen
    again with that noise, because noise favours smaller steps. Best macro-F1 on the validation part wins."""
    d = C.load(dataset, 'iid', 0)
    meta = d['meta']
    Xp, yp = d['X_pub'], d['y_pub']
    g = np.random.default_rng(0)
    idx = g.permutation(len(yp)); cut = int(0.8 * len(yp))
    tr, va = idx[:cut], idx[cut:]
    plants = [(Xp[p], yp[p]) for p in np.array_split(tr, C.PLANTS)]
    sites = C.chunks(plants, C.B)
    rounds = 16 if quick else TUNE_USES * min(len(s) for s in sites)

    def run(cfg, noise):
        st, _h = S.run_plain(cfg, sites, rounds, Xp[va], yp[va], noise=noise, eval_every=10 ** 9)
        return C.eval_int(cfg, st, Xp[va], yp[va])['macro_f1']
    grid = [dict(Cg=Cg, sh=sh, lr_shift=lr) for (Cg, sh) in ((1024, 15), (256, 13), (64, 11)) for lr in (0, 1, 2, 3)]
    if quick:
        grid = grid[:2]
    rows = []
    for cand in grid:
        f1 = run(C.make_cfg(meta, noisy=False, **cand), False)
        rows.append(dict(cand, rounds=rounds, noise_r=0, macro_f1=f1))
        print('  tune %s %s macro-F1 %.4f' % (dataset, cand, f1), flush=True)
    best = max(rows, key=lambda r: r['macro_f1'])
    chosen = {k: best[k] for k in ('Cg', 'sh', 'lr_shift')}
    lr_noisy = {}
    for r in (NOISE_R if not quick else (C.R_DEFAULT,)):
        res = []
        for lr in ((1, 2, 3, 4) if not quick else (2,)):
            f1 = run(C.make_cfg(meta, Cg=chosen['Cg'], sh=chosen['sh'], lr_shift=lr, r=r), True)
            res.append((f1, -lr)); rows.append(dict(Cg=chosen['Cg'], sh=chosen['sh'], lr_shift=lr, rounds=rounds, noise_r=r, macro_f1=f1))
            print('  tune %s noise r=%d lr_shift %d macro-F1 %.4f' % (dataset, r, lr, f1), flush=True)
        lr_noisy[str(r)] = -max(res)[1]
    os.makedirs(C.RESULTS, exist_ok=True)
    json.dump(dict(dataset=dataset, chosen=chosen, lr_noisy=lr_noisy, grid=rows, public_rows=int(len(yp)), code=C.code_version()),
              open(C.tuning_path(dataset), 'w'), indent=1)
    return chosen


# ---------------------------------------------------------------- E1
def e1(quick=False):
    jobs = []
    for ds, split, seed in itertools.product(C.DATASETS, C.SPLITS, SEEDS):
        if not C.available(ds, split, 0):
            continue
        for system in ('float_central', 'float_fedavg', 'int_no_noise'):
            jobs.append(('%s|%s|%d|%s' % (ds, split, seed, system), ds, split, seed, system, 32 if quick else R_E1, quick))
    C.run_jobs(C.Store('E1_detection'), jobs, _e1_job, 'E1')


def _e1_job(job):
    key, ds, split, seed, system, R, quick = job
    d = C.load(ds, split, 0)
    meta = d['meta']; cfg = replace(C.make_cfg(meta, noisy=False), init_seed=seed); nc = C.normal_class(meta)
    t0 = time.time()
    if system == 'float_central':
        ev = C.evaluate(C.float_central(d, seed, max_iter=20 if quick else 40), d['y_test'], cfg.arch[-1], nc)
    elif system == 'float_fedavg':
        ev = C.evaluate(C.float_fedavg(d, cfg.arch, seed, R), d['y_test'], cfg.arch[-1], nc)
    else:
        stt, _h = S.run_plain(cfg, C.chunks(d['plants'], cfg.b), R, d['X_test'], d['y_test'], seed=seed, noise=False, eval_every=10 ** 9)
        ev = C.eval_int(cfg, stt, d['X_test'], d['y_test'], nc)
    return key, dict(dataset=ds, split=split, seed=seed, system=system, rounds=R, seconds=time.time() - t0, **ev)


# ---------------------------------------------------------------- E2 and the models for E9
def e2(quick=False):
    combos = []
    plan = [(sp, u) for sp in ('iid', 'dir0.3') for u in USES] + [(C.SCALE['tag'], u) for u in SCALE_USES]
    for ds, (split, uses), r, seed in itertools.product(C.DATASETS, plan, NOISE_R, SEEDS):
        if not C.available(ds, split, 0):              # 10 plants: USES; 50 plants (more records per round): SCALE_USES
            continue
        if quick and (r != NOISE_R[0] or uses not in (USES[0], SCALE_USES[0]) or seed):
            continue
        combos.append((ds, split, r, uses, seed))
    nan = dict(eps=float('nan'), Delta=float('nan'), sigma=float('nan'), z=float('nan'), eps_gauss=float('nan'))
    hmin = lambda split: C.SCALE['H'] if split == C.SCALE['tag'] else C.H_MIN
    cfgs = {(ds, r): C.make_cfg(C.load(ds, 'iid', 0)['meta'], r=r) for ds, _s, r, _u, _sd in combos}
    need = sorted({(ds, r, u, hmin(sp)) for ds, sp, r, u, _sd in combos})
    accs = dict(zip(need, C.epsilons([(cfgs[(ds, r)], u, H) for ds, r, u, H in need]))) if not quick else {}
    jobs = []
    for ds, split, r, uses, seed in combos:
        for system in ('ours', 'trusted_dp'):
            jobs.append(('%s|%s|r%d|u%d|%d|%s' % (ds, split, r, uses, seed, system), ds, split, r, uses, seed, system,
                         16 if quick else None, accs.get((ds, r, uses, hmin(split)), nan), hmin(split)))
    C.run_jobs(C.Store('E2_privacy'), jobs, _e2_job, 'E2')
    # epsilon for every number of honest plants (two uses), so H_MIN = A_MIN - F_MAX can be read against the others
    sh = C.Store('E2_hmin')
    items = []
    for ds in C.DATASETS:
        if quick or not C.available(ds, 'iid', 0):
            continue
        meta = C.load(ds, 'iid', 0)['meta']
        for r, H in itertools.product(NOISE_R, H_SWEEP):
            key = '%s|r%d|u2|H%d' % (ds, r, H)
            if not sh.done(key):
                items.append((key, ds, r, H, C.make_cfg(meta, r=r)))
    for (key, ds, r, H, _c), a in zip(items, C.epsilons([(it[4], 2, it[3]) for it in items])):
        sh.add(key, dict(dataset=ds, r=r, uses=2, H=H, eps=a['eps'], eps_gauss_reference=a['eps_gauss']))
        print('E2 eps %-40s %.4g' % (key, a['eps']), flush=True)


def _e2_job(job):
    key, ds, split, r, uses, seed, system, R, acc, H = job
    d = C.load(ds, split, 0)
    meta = d['meta']; nc = C.normal_class(meta)
    cfg = replace(C.make_cfg(meta, r=r), init_seed=seed)
    R = R or _rounds_for(d, uses)
    t0 = time.time()
    if system == 'ours':
        stt, _h = S.run_plain(cfg, C.chunks(d['plants'], cfg.b), R, d['X_test'], d['y_test'], seed=seed, noise=True, eval_every=10 ** 9)
        extra = dict(eps=acc['eps'], eps_gauss_reference=acc['eps_gauss'],
                     noise_sigma_per_plant=acc['sigma'] / math.sqrt(H) if acc['sigma'] == acc['sigma'] else None)
    else:
        if not (acc['eps'] == acc['eps'] and math.isfinite(acc['eps'])):
            return key, None
        sig = C.gaussian_sigma_for(acc['eps'], acc['Delta'], uses)
        stt, _h = S.run_plain(cfg, C.chunks(d['plants'], cfg.b), R, d['X_test'], d['y_test'], seed=seed, noise=True,
                              eval_every=10 ** 9, central_noise=C.trusted_noise(sig, cfg.d))
        extra = dict(eps=acc['eps'], central_sigma=sig)
    ev = C.eval_int(cfg, stt, d['X_test'], d['y_test'], nc)
    if seed == 0 and split == 'iid':                               # kept for the membership attack (E9)
        os.makedirs(os.path.join(C.RESULTS, 'models'), exist_ok=True)
        np.savez_compressed(os.path.join(C.RESULTS, 'models', key.replace('|', '_') + '.npz'),
                            **{'w%d' % i: np.asarray(w) for i, w in enumerate(_flat_state(stt))})
    return key, dict(dataset=ds, split=split, r=r, uses=uses, seed=seed, system=system, rounds=R, H_min=H,
                     plants=len(d['plants']),
                     seconds=time.time() - t0, **extra, **ev)


def _flat_state(state):
    out = []
    def rec(x):
        if isinstance(x, np.ndarray):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
        elif isinstance(x, dict):
            for k in sorted(x):
                rec(x[k])
    rec(state)
    return out


def _unflat_state(cfg, arrays):
    tmpl = M.init_model(cfg)
    it = iter(arrays)
    def rec(x):
        if isinstance(x, np.ndarray):
            return next(it).astype(x.dtype)
        if isinstance(x, list):
            return [rec(y) for y in x]
        if isinstance(x, tuple):
            return tuple(rec(y) for y in x)
        if isinstance(x, dict):
            return {k: rec(x[k]) for k in sorted(x)}
        return x
    return rec(tmpl)


# ---------------------------------------------------------------- E4
ATTACKS = ('label_flip', 'attack_to_normal', 'backdoor', 'garbage')
SHARES = (1, 2, 3)


def _trigger(d):
    """Backdoor trigger: the 3 features with the lowest spread on the public data, each set to the end of the byte
    range farther from its usual value (setting a feature that is usually 255 to 255 would change nothing)."""
    Xp = d['X_pub']
    idx = np.argsort(Xp.std(0))[:3]
    vals = np.array([0 if np.median(Xp[:, j]) >= 128 else 255 for j in idx], dtype=Xp.dtype)
    return idx, vals


def _poison(attack, X, y, ncls, normal, trig, rng):
    X = X.copy(); y = y.copy()
    if attack == 'label_flip':
        y = (y + 1 + rng.integers(0, ncls - 1, len(y))) % ncls
    elif attack == 'attack_to_normal':
        y[y != normal] = normal
    elif attack == 'backdoor':
        sel = y != normal
        X[np.ix_(sel, trig[0])] = trig[1]
        y[sel] = normal
    elif attack == 'garbage':
        X = rng.integers(0, 256, X.shape); y = rng.integers(0, ncls, len(y))
    return X, y


def _asr(cfg, state, d, attack, normal, trig):
    W = M.proof_weights(cfg, state)
    att = d['y_test'] != normal
    X = d['X_test'][att].copy()
    if attack == 'backdoor':
        X[:, trig[0]] = trig[1]
    elif attack not in ('attack_to_normal',):
        return None
    return float((M.predict(cfg, W, X) == normal).mean())       # attack traffic passed as normal


E4_SYSTEMS = ('plain_fedavg', 'norm_bound_only', 'krum', 'median', 'trimmed_mean', 'ours')


def e4(quick=False):
    jobs = []
    for ds, attack, share, seed in itertools.product(C.DATASETS, ATTACKS, SHARES, SEEDS):
        if not C.available(ds, 'iid', 0) or (quick and (share != 2 or seed)):
            continue
        for system in E4_SYSTEMS:
            jobs.append(('%s|%s|%d|%d|%s' % (ds, attack, share, seed, system), ds, attack, share, seed, system, 16 if quick else R_E1))
    C.run_jobs(C.Store('E4_poisoning'), jobs, _e4_job, 'E4')


def _e4_job(job):
    """One poisoning run. Our system: corrupted plants can only change their data (the proof fixes the computation).
    Baselines that do not verify training: corrupted plants send any vector. Plain averaging gets a boosted update
    (model replacement); norm-bound-only gets the attack direction scaled to the bound; the robust rules (which see
    every update, so they cannot hide them) get the attack direction scaled to the median honest norm, to blend in."""
    key, ds, attack, share, seed, system, R = job
    d = C.load(ds, 'iid', 0)
    meta = d['meta']; nc = C.normal_class(meta); ncls = len(meta['classes'])
    cfg = replace(C.make_cfg(meta), init_seed=seed)
    trig = _trigger(d)
    corrupt = tuple(range(share))
    sites = C.chunks(d['plants'], cfg.b)
    rng_p = np.random.default_rng(1000 + seed)
    bound = cfg.b * cfg.Chat / (1 << cfg.sh)                   # the largest norm an honest clipped chunk can have

    def transform(i, t, X, y, W):
        return _poison(attack, X, y, ncls, nc, trig, rng_p)

    def craft(i, t, W, hon):
        X, y = sites[i][t % len(sites[i])]
        Xp, yp = _poison(attack, X, y, ncls, nc, trig, rng_p)
        v = M.plain_step(cfg, W, Xp, yp, clip=False)['tq'].astype(np.float64)
        nv = np.linalg.norm(v)
        if system == 'plain_fedavg':                             # model replacement: boosted by the plant count
            v = v * len(sites)
        elif system == 'norm_bound_only':                        # scaled up to the bound
            v = v * (bound / nv) if nv > 0 else v
        else:                                                    # robust rules: blend in with the honest norms
            target = float(np.median([np.linalg.norm(h.astype(np.float64)) for h in hon])) if hon else nv
            v = v * (target / nv) if nv > 0 else v
        return np.rint(v).astype(np.int64)

    rules = dict(krum=S.agg_krum, median=S.agg_median, trimmed_mean=S.agg_trimmed_mean)
    t0 = time.time()
    if system == 'ours':
        stt, _h = S.run_plain(cfg, sites, R, d['X_test'], d['y_test'], seed=seed, noise=True, eval_every=10 ** 9,
                              transform=transform, corrupt=corrupt)
    else:
        agg = (lambda v, _r=rules[system]: _r(v, C.F_MAX)) if system in rules else None
        stt, _h = S.run_plain(cfg, sites, R, d['X_test'], d['y_test'], seed=seed, noise=True, eval_every=10 ** 9,
                              craft=craft, corrupt=corrupt, clip=(system != 'plain_fedavg'), aggregate=agg)
    ev = C.eval_int(cfg, stt, d['X_test'], d['y_test'], nc)
    return key, dict(dataset=ds, attack=attack, share=share, seed=seed, system=system, rounds=R,
                     attack_success=_asr(cfg, stt, d, attack, nc, trig), seconds=time.time() - t0, **ev)


# ---------------------------------------------------------------- E9
def e9(quick=False):
    """Loss-threshold membership inference [Yeom et al.] on the E2 models kept for seed 0, iid: members are plant
    training records, non-members test records, with the same number of each class on both sides. Training follows a
    balanced class mix and the test set the natural one, so unmatched sets would let the attack read the class mix
    instead of membership. Reported: the best advantage TPR - FPR over thresholds."""
    st_ = C.Store('E9_membership')
    mdir = os.path.join(C.RESULTS, 'models')
    if not os.path.isdir(mdir):
        return
    for f in sorted(os.listdir(mdir)):
        key = f[:-4]
        if st_.done(key):
            continue
        parts = key.split('_')
        system = parts[-1] if parts[-1] in ('ours',) else '_'.join(parts[-2:])
        ds = next(x for x in C.DATASETS if key.startswith(x))
        r = int(next(p for p in parts if p.startswith('r') and p[1:].isdigit())[1:])
        d = C.load(ds, 'iid', 0)
        cfg = C.make_cfg(d['meta'], r=r)
        z = np.load(os.path.join(mdir, f))
        state = _unflat_state(cfg, [z['w%d' % i] for i in range(len(z.files))])
        W = M.proof_weights(cfg, state)
        mem = np.concatenate([p[0] for p in d['plants']]); ymem = np.concatenate([p[1] for p in d['plants']])
        im, it = _matched(ymem, d['y_test'], 5000, np.random.default_rng(0))
        n = len(im)
        s_in = _score(cfg, W, mem[im], ymem[im]); s_out = _score(cfg, W, d['X_test'][it], d['y_test'][it])
        th = np.unique(np.concatenate([s_in, s_out]))
        adv = max(float((s_in >= t).mean() - (s_out >= t).mean()) for t in th[:: max(1, len(th) // 2000)])
        st_.add(key, dict(model=key, dataset=ds, r=r, system=system, advantage=adv, n=n))
        print('E9 %-60s advantage %.4f' % (key, adv), flush=True)


def _matched(ymem, ytest, n, g):
    """Indices of members and non-members with equal counts per class, n in total at most, spread over the classes in
    proportion to the smaller of the two supplies."""
    classes = np.union1d(np.unique(ymem), np.unique(ytest))
    supply = {c: min(int((ymem == c).sum()), int((ytest == c).sum())) for c in classes}
    tot = sum(supply.values())
    im, it = [], []
    for c in classes:
        k = int(round(supply[c] * min(1.0, n / tot))) if tot else 0
        if k:
            im.append(g.choice(np.flatnonzero(ymem == c), k, replace=False))
            it.append(g.choice(np.flatnonzero(ytest == c), k, replace=False))
    return np.concatenate(im), np.concatenate(it)


def _score(cfg, W, X, y):
    """Membership score: the margin of the true class (higher = more confident = more likely a member)."""
    h = (np.asarray(X, dtype=np.int64) - cfg.xoff) << (cfg.Fa - cfg.Bx)
    for l in range(1, cfg.L + 1):
        Wl, bv = W[l - 1]
        z = h @ Wl.T + bv * (1 << cfg.Fa)
        if l < cfg.L:
            h = np.clip(z >> cfg.Fw, 0, cfg.Hmax)
    zt = z[np.arange(len(y)), y].astype(np.float64)
    z2 = z.astype(np.float64).copy(); z2[np.arange(len(y)), y] = -np.inf
    return zt - z2.max(1)
