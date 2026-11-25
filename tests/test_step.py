import sys, time
sys.path.insert(0, '.')
import numpy as np
from fedproof import field as F, model as M, simulator as S, ligero as L, step as ST, message as SP, pq
from fedproof.sharing import Sharing

def setup(cfg, prm, K=7, kp=3, T=2, seed=0):
    Xtr, ytr, Xte, yte = S.load_digits_int()
    g = np.random.default_rng(seed)
    idx = g.choice(len(ytr), cfg.b, replace=False)
    X, y = Xtr[idx], ytr[idx]
    rho = g.integers(0, 2, (cfg.d, cfg.J, cfg.r)); beta = g.integers(0, 2, (cfg.d, cfg.J, cfg.r))
    state = M.init_model(cfg)
    # move the model away from initialisation so that all branches are exercised
    st = state
    W = M.proof_weights(cfg, st)
    shr = Sharing(cfg.d, kp, T, K)
    rng = F.Drbg(('site', seed))
    d0 = L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, rng), prm, rng)
    nz = L.commit_block(ST.noise_block_matrix(cfg, prm.ell, rho), prm, rng)
    keys = [pq.kem_keygen() for _ in range(K)]
    pub = SP.Public(cfg, prm, shr, W, beta, b'round=7|site=3|model=' + M.model_digest(cfg, st))
    return pub, dict(X=X, y=y, rho=rho), d0, nz, keys, rng

if __name__ == '__main__':
    cfg = M.Config()
    prm = L.Params(1024, 64, 64, 2)
    pub, wit, d0, nz, keys, rng = setup(cfg, prm)
    t0 = time.time(); msg, priv = SP.site_prove(pub, wit, d0, nz, [k[0] for k in keys], rng); tp = time.time() - t0
    cs, sysd = priv['cs'], priv['sysd']
    cs.mats['D0'] = L.decode_rows(d0.coeffs, prm); cs.mats['NZ'] = L.decode_rows(nz.coeffs, prm)
    assert cs.check(sysd) == []
    ref = M.plain_step(cfg, pub.weights, wit['X'], wit['y'], wit['rho'], pub.beta)
    assert np.array_equal(priv['u'], ref['u']), 'proved vector differs from the reference semantics'
    t0 = time.time(); ok, gamma = SP.site_verify(pub, d0.root, nz.root, msg); tv = time.time() - t0
    assert ok
    print('step: d=%d b=%d rows=%s cells=%d ncon=%d nnz=%d | prove %.2fs verify %.2fs | message %.1f kB (proof %.1f kB)' % (
        cfg.d, cfg.b, sysd['block_rows'], sysd['m'] * prm.ell, sysd['ncon'], len(sysd['con']), tp, tv,
        SP.message_size(msg) / 1e3, L.proof_size(msg['proof']) / 1e3))
    # nodes decrypt and check their shares; reconstruct u from shares
    shr = pub.shr; G = pub.G
    rows = []
    for m, (ek, dk) in enumerate(keys):
        pt = pq.open_(dk, msg['cts'][m], SP._aad(pub, m)); assert pt is not None
        row = np.frombuffer(pt, dtype=np.uint64); rows.append(row)
        assert ST.node_check(shr, G, gamma, m, row, msg['digests'][m])
    rows = np.array(rows)
    nodes = list(range(shr.w)); Cm = shr.secret_coeffs(nodes)
    rec = np.array([F.matvec(Cm, rows[nodes, B]) for B in range(shr.nblk)]).reshape(-1)[:cfg.d]
    assert np.array_equal(F.to_signed(rec), ref['u'])
    print('shares decrypt, pass the digest check and reconstruct the proven vector')
