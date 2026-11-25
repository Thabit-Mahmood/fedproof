"""A cheating site changes one decision of the training step, recomputes everything downstream honestly, and
tries to get its message accepted. Every such message must be rejected by public verification."""
import sys, time, copy
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, ligero as L, step as ST, message as SP, pq
from fedproof.constraints import BIT, FREE, MX, MY, MZ
from test_step import setup

cfg = M.Config(); prm = L.Params(1024, 64, 64, 2)
pub, wit, d0, nz, keys, rng0 = setup(cfg, prm)
eks = [k[0] for k in keys]
P = F.P

def setbits(cs, ids, vals):
    nb = ids.shape[1]; v = np.asarray(vals, dtype=np.int64) & ((1 << nb) - 1)
    cs.set(ids, ((v[:, None] >> np.arange(nb)[None, :]) & 1).astype(np.uint64))
def getbits(cs, ids):
    return (F.to_signed(cs.get(ids)) << np.arange(ids.shape[1])[None, :]).sum(1)

def cheat(point, fn):
    def hook(name, cs, ids):
        if name == point:
            return fn(cs, ids)
    return hook

def crafted_U(cs, ids):          # replace the summed gradient by a chosen vector (weight coordinates only)
    v = getbits(cs, ids); o, ow, ob = cfg.layer_slices()[0]
    v[o:ow] = (1 << (ids.shape[1] - 1)) + 12345        # a constant push on every first-layer weight
    setbits(cs, ids, v)
def crafted_bias(cs, ids):
    v = getbits(cs, ids); o, ow, ob = cfg.layer_slices()[1]; v[ow] += 77; setbits(cs, ids, v)
def swap_data(cs, ids):
    xd, yd, xd1, yd1, rho = ids; X2 = (F.to_signed(cs.get(xd)) + 3) % 16; cs.set(xd, X2); cs.set(xd1, X2)
def swap_label(cs, ids):
    xd, yd, xd1, yd1, rho = ids; y2 = (F.to_signed(cs.get(yd)) + 1) % 10; cs.set(yd, y2); cs.set(yd1, y2)
def fresh_noise(cs, ids):
    xd, yd, xd1, yd1, rho = ids; cs.set(rho, np.zeros(rho.shape, dtype=np.uint64))
def refresh_other(cs, ids):
    xd, yd, xd1, yd1, rho = ids; cs.set(xd1[0, :4], [1, 2, 3, 4])
def onehot_other(cs, ids):
    oh = np.zeros(ids.shape, dtype=np.int64); oh[:, 0] = 1; cs.set(ids, oh)
def flip_sign(cs, ids):
    cs.set(ids[:5, -1], F.fsub(F.U64(1), cs.get(ids[:5, -1])))
def zero_h(cs, ids): cs.set(ids[:8], 0)
def zero_delta(cs, ids): cs.set(ids[:8], 0)
def dec_bits(cs, ids): setbits(cs, ids, getbits(cs, ids) - 1)
def half_bits(cs, ids): setbits(cs, ids, getbits(cs, ids) // 2)
def big_sp(cs, ids): setbits(cs, ids, np.full(ids.shape[0], cfg.Ss * 4))
def kappa_one(cs, ids): cs.set(ids[:, -1], 1)
def no_clip(cs, ids): cs.set(ids, cfg.Ss)
def scale_sd(cs, ids): cs.set(ids, F.fmul(cs.get(ids), F.U64(2)))
def push_u(cs, ids): cs.set(ids[:10], F.fadd(cs.get(ids[:10]), F.U64(1000)))
def shift_p(cs, ids): cs.set(ids[:6], F.fadd(cs.get(ids[:6]), F.U64(5)))
def full_p(cs, ids): cs.set(ids, 1 << cfg.Fd)
def ignore_beta(cs, ids):
    pass

cheats = [('use other data than committed', 'data', swap_data), ('use another label than committed', 'data', swap_label),
          ('use other noise bits than committed', 'data', fresh_noise), ('re-commit a different chunk', 'data', refresh_other),
          ('one-hot label not matching the label', 'onehot', onehot_other), ('flip ReLU sign bits', 'zbits1', flip_sign),
          ('zero some activations', 'h1', zero_h), ('zero some output errors', 'delta2', zero_delta),
          ('shift some output values p', 'p', shift_p), ('set every output value p to one', 'p', full_p),
          ('zero some hidden errors', 'delta1', zero_delta), ('under-report error norm (layer 1)', 'a1', dec_bits),
          ('under-report input norm (layer 2)', 'c2', dec_bits), ('under-report layer bound m', 'm1', dec_bits),
          ('halve the record norm bound', 'n', half_bits), ('inflate the clip quotient', 'sp', big_sp),
          ('claim no clipping needed', 'kappa', kappa_one), ('skip clipping (scale = full)', 's', no_clip),
          ('double the scaled errors', 'sd1', scale_sd), ('submit a crafted update (weights)', 'U', crafted_U),
          ('submit a crafted update (one bias)', 'U', crafted_bias), ('shift the shared vector', 'u', push_u)]
# honest reference: clipping must be active, otherwise the clipping cheats would be vacuous
ref = M.plain_step(cfg, pub.weights, wit['X'], wit['y'], wit['rho'], pub.beta)
assert (ref['kappa'] == 0).all(), 'test precondition: every record is being clipped'
t0 = time.time(); rejected = 0
for label, point, fn in cheats:
    rng = F.Drbg(('cheat', label))
    msg, priv = SP.site_prove(pub, wit, d0, nz, eks, rng, hook=cheat(point, fn), strict=False)
    cs = priv['cs']; cs.mats['D0'] = L.decode_rows(d0.coeffs, prm); cs.mats['NZ'] = L.decode_rows(nz.coeffs, prm)
    fails = cs.check(priv['sysd'])
    ok, _ = SP.site_verify(pub, d0.root, nz.root, msg)
    assert fails, 'constraint system did not notice: ' + label
    assert not ok, 'verifier accepted: ' + label
    rejected += 1
    print('  rejected: %-42s (violated: %s)' % (label, ','.join('%s x%d' % (k, len(v)) for k, v in fails)))
assert rejected == len(cheats)
print('%d of %d cheating strategies rejected in %.0fs' % (rejected, len(cheats), time.time() - t0))

# wrong shares to one node: the proof stays valid, the node's check fails (complaint path)
def bad_share(name, cs, ids):
    if name == 'shares':
        sh = ids.copy(); sh[2, 0] = (int(sh[2, 0]) + 1) % P; return sh
msg, priv = SP.site_prove(pub, wit, d0, nz, eks, F.Drbg('bs'), hook=bad_share, strict=True)
ok, gamma = SP.site_verify(pub, d0.root, nz.root, msg); assert ok
res = []
for m, (ek, dk) in enumerate(keys):
    row = np.frombuffer(pq.open_(dk, msg['cts'][m], SP._aad(pub, m)), dtype=np.uint64)
    res.append(ST.node_check(pub.shr, pub.G, gamma, m, row, msg['digests'][m]))
assert res == [m != 2 for m in range(len(keys))]
print('wrong share to node 2: proof valid, node 2 detects it, other nodes unaffected')

# statement binding: an honest message must fail under any changed public input
msg, priv = SP.site_prove(pub, wit, d0, nz, eks, F.Drbg('hon'))
assert SP.site_verify(pub, d0.root, nz.root, msg)[0]
def variant(**kw):
    p2 = SP.Public(pub.cfg, pub.prm, pub.shr, pub.weights, pub.beta, pub.ctx)
    for k, v in kw.items(): setattr(p2, k, v)
    return p2
W2 = [(w.copy(), b.copy()) for w, b in pub.weights]; W2[0][0][0, 0] += 1
W3 = [(w.copy(), b.copy()) for w, b in pub.weights]; W3[1][1][3] -= 1
b2 = pub.beta.copy(); b2[5, 0, 0] ^= 1
binding = [('different model weight', variant(weights=W2), d0.root, nz.root, msg), ('different bias', variant(weights=W3), d0.root, nz.root, msg),
           ('different beacon bit', variant(beta=b2), d0.root, nz.root, msg), ('different round context', variant(ctx=b'round=8'), d0.root, nz.root, msg),
           ('different data commitment', pub, nz.root, nz.root, msg), ('different noise commitment', pub, d0.root, d0.root, msg)]
m2 = copy.deepcopy(msg); m2['digests'][0, 0] = (int(m2['digests'][0, 0]) + 1) % P; binding.append(('altered digest', pub, d0.root, nz.root, m2))
m3 = copy.deepcopy(msg); c = bytearray(m3['cts'][1]); c[-1] ^= 1; m3['cts'][1] = bytes(c); binding.append(('altered ciphertext', pub, d0.root, nz.root, m3))
m4 = copy.deepcopy(msg); m4['root_d1'] = msg['root_m']; binding.append(('altered refreshed-data root', pub, d0.root, nz.root, m4))
m5 = copy.deepcopy(msg); m5['cts'] = m5['cts'][::-1]; binding.append(('reordered ciphertexts', pub, d0.root, nz.root, m5))
cfg2 = M.Config(Cg=cfg.Cg * 2); binding.append(('different clipping threshold', variant(cfg=cfg2), d0.root, nz.root, msg))
cfg3 = M.Config(sh=cfg.sh - 1); binding.append(('different truncation', variant(cfg=cfg3), d0.root, nz.root, msg))
for label, p2, r0, rn, mm in binding:
    assert not SP.site_verify(p2, r0, rn, mm)[0], label
print('%d binding checks passed (an honest message fails under any changed public input)' % len(binding))

# static coverage: which allocated cells appear in no linear constraint?
cs, sysd = priv['cs'], priv['sysd']
used = np.zeros(sysd['m'] * prm.ell, dtype=bool); used[sysd['cell'][sysd['coef'] != 0]] = True
r0 = 0; report = []
for blk in cs.blocks:
    for kind, kname in ((BIT, 'bit'), (FREE, 'free'), (MX, 'x'), (MY, 'y'), (MZ, 'z')):
        seg = blk['segs'][kind]
        if seg.size == 0: continue
        vids = (cs.bidx[blk['name']] << 44) | (kind << 40) | np.arange(seg.size, dtype=np.int64)
        g = cs.gcell(vids); unc = int((~used[g]).sum())
        if unc: report.append('%s/%s: %d of %d' % (blk['name'], kname, unc, seg.size))
print('cells in no linear constraint:', report)
