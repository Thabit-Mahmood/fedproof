"""Cost of one proof-carrying site message for the model shapes of the industrial datasets.

Two security levels: 128 bits round-by-round (classical) and 256 bits (about 128 bits against quantum attackers
after the QROM loss). For each shape and level several parameter sets are tried and all are reported.
Run alone, it writes runs/bench.json. Times are single-process wall clock on the machine that runs it.
"""
import sys, time, json, resource, platform, os
sys.path.insert(0, '.')
import numpy as np
from fedproof import field as F, model as M, ligero as L, step as ST, message as SP, pq
from fedproof.sharing import Sharing


def one(arch, b, ell, target, blowup, K=13, kp=4, T=3, reps=2, seed=0):
    cfg = M.Config(arch=arch, b=b, Bx=8)
    prm = L.find_params(ell, target, blowup)
    if prm is None:
        return None
    g = np.random.default_rng(seed)
    X = g.integers(0, 256, (b, arch[0])); y = g.integers(0, arch[-1], b)
    rho = g.integers(0, 2, (cfg.d, cfg.J, cfg.r)); beta = g.integers(0, 2, (cfg.d, cfg.J, cfg.r))
    W = M.proof_weights(cfg, M.init_model(cfg))
    shr = Sharing(cfg.d, kp, T, K)
    rng = F.Drbg(('bench', seed))
    t0 = time.time()
    d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
    nz = L.commit_block(ST.noise_block_matrix(cfg, prm.ell, rho), prm, rng)
    t_commit = time.time() - t0
    keys = [pq.kem_keygen() for _ in range(K)]
    pub = SP.Public(cfg, prm, shr, W, beta, b'bench')
    tp, tv = [], []
    for _ in range(reps):
        t0 = time.time(); msg, priv = SP.site_prove(pub, dict(X=X, y=y, rho=rho), d0, nz, [k[0] for k in keys], rng); tp.append(time.time() - t0)
        t0 = time.time(); ok, _ = SP.site_verify(pub, d0.root, nz.root, msg); tv.append(time.time() - t0)
        assert ok
    ref = M.plain_step(cfg, W, X, y, rho, beta)
    assert np.array_equal(priv['u'], ref['u'])
    snd = prm.soundness()
    sysd = priv['sysd']
    return dict(arch=list(arch), d=cfg.d, b=b, ell=ell, blowup=blowup, t=prm.t, sigma=prm.sigma, target=target,
                rbr_bits=round(snd['rbr_bits'], 1), rows=sysd['block_rows'], cells=sysd['m'] * ell,
                prove_s=round(min(tp), 2), verify_s=round(min(tv), 2), next_commit_s=round(t_commit, 2),
                proof_kB=round(L.proof_size(msg['proof']) / 1e3, 1), msg_kB=round(SP.message_size(msg) / 1e3, 1),
                share_ct_kB_per_node=round(len(msg['cts'][0]) / 1e3, 2), K=K,
                peak_rss_MB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3))


if __name__ == '__main__':
    shapes = [((61, 64, 32, 15), 16), ((61, 64, 32, 15), 32), ((41, 64, 32, 5), 16), ((43, 64, 32, 10), 16), ((61, 128, 64, 15), 16)]
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        shapes = shapes[:1]
    res = dict(machine=dict(cpu=platform.processor() or platform.machine(), cores=os.cpu_count(), python=platform.python_version()), rows=[])
    for arch, b in shapes:
        for target, choices in ((128, [(1024, 8), (2048, 4), (4096, 4)]), (256, [(2048, 8), (4096, 4), (4096, 8)])):
            for ell, blowup in choices:
                r = one(arch, b, ell, target, blowup)
                if r is not None:
                    res['rows'].append(r)
                    print(json.dumps(r), flush=True)
                os.makedirs('runs', exist_ok=True); json.dump(res, open('runs/bench.json', 'w'), indent=1)
