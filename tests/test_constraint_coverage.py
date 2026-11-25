"""Static coverage of the step statement: cells in no linear constraint, per block/kind, and per named region."""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import model as M, ligero as L, message as SP
from fedproof.constraints import BIT, FREE, MX, MY, MZ
from test_step import setup
for cfg in (M.Config(), M.Config(center=True), M.Config(arch=(64, 16, 12, 10), b=8)):
    prm = L.Params(1024, 64, 64, 2)
    pub, wit, d0, nz, keys, rng = setup(cfg, prm, K=5, kp=2, T=1)
    msg, priv = SP.site_prove(pub, wit, d0, nz, [k[0] for k in keys], rng)
    cs, sysd = priv['cs'], priv['sysd']
    used = np.zeros(sysd['m'] * prm.ell, dtype=bool); used[sysd['cell'][sysd['coef'] != 0]] = True
    rep = []
    for blk in cs.blocks:
        for kind, kn in ((BIT, 'bit'), (FREE, 'free'), (MX, 'x'), (MY, 'y'), (MZ, 'z')):
            seg = blk['segs'][kind]
            if not seg.size: continue
            vids = (cs.bidx[blk['name']] << 44) | (kind << 40) | np.arange(seg.size, dtype=np.int64)
            unc = int((~used[cs.gcell(vids)]).sum())
            if unc: rep.append('%s/%s %d of %d' % (blk['name'], kn, unc, seg.size))
    print(cfg.arch, 'center' if cfg.center else '', 'unconstrained-by-linear:', rep)
    assert all(r.startswith('M/y ') for r in rep), rep     # only the inverse witnesses of the nonzero gadget
