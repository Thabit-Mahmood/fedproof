"""The pure-numpy path (no C kernel) proves and verifies, and interoperates with the kernel path."""
import sys, subprocess, os
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, ligero as L, message as SP
from test_step import setup
env = dict(os.environ, FEDPROOF_NO_C='1')
r = subprocess.run([sys.executable, 'tests/test_ligero.py'], capture_output=True, text=True, env=env)
assert r.returncode == 0, r.stderr[-400:]
print('toy argument tests pass without the kernel')
cfg = M.Config(arch=(64, 12, 10), b=8); prm = L.Params(512, 48, 48, 2, blowup=8)
pub, wit, d0, nz, keys, rng = setup(cfg, prm, K=5, kp=2, T=1)
eks = [k[0] for k in keys]
assert F.have_kernel()
msg_k, priv_k = SP.site_prove(pub, wit, d0, nz, eks, F.Drbg('k'))
lib = F._lib; F._lib = None
try:
    assert SP.site_verify(pub, d0.root, nz.root, msg_k)[0], 'kernel-made proof rejected by the numpy verifier'
    msg_n, priv_n = SP.site_prove(pub, wit, d0, nz, eks, F.Drbg('k'))
    assert SP.site_verify(pub, d0.root, nz.root, msg_n)[0]
finally:
    F._lib = lib
assert SP.site_verify(pub, d0.root, nz.root, msg_n)[0], 'numpy-made proof rejected by the kernel verifier'
assert np.array_equal(priv_k['u'], priv_n['u']) and msg_k['root_m'] == msg_n['root_m'], 'the two paths disagree'
print('kernel and numpy paths produce identical commitments and accept each other\'s proofs')
