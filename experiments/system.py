"""Experiments with real proofs and the full protocol: E3 integrity, E5 proof cost, E8 end-to-end training with audit."""
import os
import sys
import time
import subprocess
import re
import itertools
sys.path.insert(0, '.')
import numpy as np
from fedproof import model as M, ligero as L, protocol as PR
from experiments import common as C

LEVELS = dict(P128=(4096, 128, 4), P256=(4096, 256, 8))   # (ell, target bits, blowup); P256 keeps about 128 bits post-quantum


def params(level):
    ell, target, blowup = LEVELS[level]
    return L.find_params(ell, target, blowup)


# ---------------------------------------------------------------- E3
def e3(quick=False):
    """Every integrity test: cheating strategies on the training step, changed public inputs, malformed messages, and the
    protocol attack scenarios. The counts in the paper come from this file."""
    st_ = C.Store('E3_integrity')
    key = 'suite'
    if st_.done(key) and not quick:
        return
    t0 = time.time()
    out = subprocess.run([sys.executable, 'tests/run_all.py'], capture_output=True, text=True)
    summ = open('runs/tests_summary.txt').read() if os.path.exists('runs/tests_summary.txt') else out.stdout
    lines = [l for l in summ.splitlines() if l.startswith(('PASS', 'FAIL'))]
    scen = sum(int(m.group(1)) for m in re.finditer(r'(\d+) [\w-]* ?protocol scenarios passed', summ))
    scen += sum(int(m.group(1)) for m in re.finditer(r'(\d+) erasure scenarios passed', summ))
    row = dict(files=len(lines), passed=sum(l.startswith('PASS') for l in lines), failed=[l.split()[1] for l in lines if l.startswith('FAIL')],
               protocol_scenarios=scen, seconds=time.time() - t0, summary=lines)
    st_.add(key, row)
    print('E3 %d of %d test files pass, %d protocol scenarios' % (row['passed'], row['files'], scen), flush=True)


# ---------------------------------------------------------------- E5
def e5(quick=False):
    from experiments import bench as B
    st_ = C.Store('E5_proofcost')
    shapes = []
    for ds in C.DATASETS:
        if C.available(ds, 'iid', 0):
            m = C.load(ds, 'iid', 0)['meta']
            shapes.append((ds, (m['n_features'], 64, 32, len(m['classes']))))
    if not shapes:                                   # before the data is prepared: the planned shapes
        shapes = [('edge-iiotset(planned)', (61, 64, 32, 15)), ('wustl(planned)', (41, 64, 32, 5)), ('x-iiotid(planned)', (59, 64, 32, 10))]
    shapes.append(('larger', (shapes[0][1][0], 128, 64, shapes[0][1][-1])))
    reps = 1 if quick else 5
    for (name, arch), b, level in itertools.product(shapes if not quick else shapes[:1], (16, C.B) if not quick else (C.B,), LEVELS):
        key = '%s|%s|b%d|%s' % (name, arch, b, level)
        if st_.done(key):
            continue
        ell, target, blowup = LEVELS[level]
        r = B.one(arch, b, ell, target, blowup, reps=reps)
        if r is None:
            continue
        st_.add(key, dict(model=name, level=level, **r))
        print('E5 %-50s prove %.2fs verify %.2fs message %.0f kB' % (key, r['prove_s'], r['verify_s'], r['msg_kB']), flush=True)


# ---------------------------------------------------------------- E8
def e8(quick=False, rounds=40, level='P128', K=7, kp=2, Tn=1, A_min=7):
    """Full system with real proofs. Each round checks, for every valid plant, that its proved vector equals the
    simulator's step on the same weights, chunk and noise bits (bit for bit), and that the decoded aggregate equals the
    sum of those vectors; at the end an independent audit replays the whole log."""
    st_ = C.Store('E8_endtoend')
    for ds in C.DATASETS:
        if not C.available(ds, 'iid', 0):
            continue
        key = '%s|%s|%d' % (ds, level, rounds if not quick else 3)
        if st_.done(key):
            continue
        d = C.load(ds, 'iid', 0)
        meta = d['meta']; cfg = C.make_cfg(meta); nc = C.normal_class(meta)
        prm = params(level)
        R = 3 if quick else rounds
        site_chunks = C.chunks(d['plants'], cfg.b)
        site_chunks = [ch[:max(R, 1)] for ch in site_chunks]           # only the chunks the run will use are committed
        t0 = time.time()
        reg, sites, nodes, log = PR.setup(cfg, prm, site_chunks, K, kp, Tn, A_min, max_rounds=R + 2)
        t_setup = time.time() - t0
        state = M.init_model(cfg)
        times, exact, mism = [], 0, 0
        for t in range(1, R + 1):
            W = M.proof_weights(cfg, state)
            rho = {s.idx: (s.noise[0] if s.noise is not None else None) for s in sites}
            tr = time.time()
            state, rep = PR.run_round(reg, sites, nodes, log, state, t)
            times.append(time.time() - tr)
            if rep['aborted'] is None:
                r = reg.view.history[-1]
                tot = np.zeros(cfg.d, dtype=np.int64)
                for i in rep['valid']:
                    c = PR.chunk_index(i, t, len(site_chunks[i]))
                    X, y = site_chunks[i][c]
                    beta = PR.beacon_bits(cfg, reg.view.round.beacon, i)
                    ref = M.plain_step(cfg, W, X, y, rho[i], beta)['u']
                    same = np.array_equal(ref, sites[i].last['u'])
                    exact += same; mism += (not same)
                    tot += sites[i].last['u']
                mism += (not np.array_equal(tot, rep['agg']))
            if t % 10 == 0 or t == R:
                ev = C.eval_int(cfg, state, d['X_test'], d['y_test'], nc)
                print('E8 %s round %d: %.1fs/round, macro-F1 %.4f, exact %d, mismatches %d' % (ds, t, np.mean(times), ev['macro_f1'], exact, mism), flush=True)
        ta = time.time()
        st2, hist = PR.audit(reg.gen, log.entries)
        t_audit = time.time() - ta
        ev = C.eval_int(cfg, state, d['X_test'], d['y_test'], nc)
        st_.add(key, dict(dataset=ds, level=level, rounds=R, N=len(sites), K=K, Tn=Tn, A_min=A_min, setup_s=t_setup,
                          round_s_mean=float(np.mean(times)), round_s_max=float(np.max(times)), exact_vectors=int(exact),
                          mismatches=int(mism), audit_same=M.model_digest(cfg, st2) == M.model_digest(cfg, state),
                          audit_s=t_audit, log_entries=len(log.entries), log_bytes=int(sum(len(x) for x in log.raw)),
                          aborted=[h['t'] for h in hist if h['outcome'] != 'result'], **ev))
        print('E8 %s done: audit %s, %d exact vectors, %d mismatches' % (ds, st_.rows[key]['audit_same'], exact, mism), flush=True)
