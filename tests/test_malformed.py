"""Malformed messages must be rejected, never crash a verifier, a round or the audit."""
import sys, copy
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, protocol as PR, message as SP, ligero as L
from test_protocol import build

sysm, sites, nodes, log, _ = build()
cfg = sysm.cfg
st0 = M.init_model(cfg)
# an honest message of round 1 (taken from a first, separate system so that roots match that system)
sA, sitesA, nodesA, logA, _ = build()
stA, repA = PR.run_round(sA, sitesA, nodesA, logA, M.init_model(cfg), 1)
honest = [e for e in logA.entries if e['kind'] == 'site' and e['site'] == 0][0]['msg']
beacon_entry = [e for e in logA.entries if e['kind'] == 'beacon'][0]

def mutants(msg):
    out = []
    def mk(label, f):
        m = copy.deepcopy(msg); f(m); out.append((label, m))
    for key in ('root_d1', 'root_m', 'root_a', 'next_noise'):
        for val in (2**63, 2**127, 2**200, -1, None, 'text', b'short', [1, 2], 3.5, b'\x00' * 47, b'\x00' * 49):
            mk('%s=%r' % (key, val if not isinstance(val, bytes) else len(val)), lambda m, k=key, v=val: m.__setitem__(k, v))
    for val in (None, 7, 'x', [], np.zeros((2, 2)), np.zeros((7, 2), dtype=np.int64), np.full((7, 2), F.P, dtype=np.uint64), np.zeros((7, 3), dtype=np.uint64)):
        mk('digests=%s' % type(val).__name__, lambda m, v=val: m.__setitem__('digests', v))
    for val in (None, 7, [], [1] * 7, [b''] * 7, [b'x' * 10] * 7, ['s'] * 7, msg['cts'][:-1], msg['cts'] + [b'']):
        mk('cts variant', lambda m, v=val: m.__setitem__('cts', v))
    for val in (None, 7, {}, [], 'p', {'q_prox': 1}):
        mk('proof=%r' % (val,), lambda m, v=val: m.__setitem__('proof', v))
    mk('q_prox list', lambda m: m['proof'].__setitem__('q_prox', m['proof']['q_prox'].tolist()))
    mk('q_lin object', lambda m: m['proof'].__setitem__('q_lin', m['proof']['q_lin'].astype(object)))
    mk('q_quad 1-D', lambda m: m['proof'].__setitem__('q_quad', m['proof']['q_quad'].ravel()))
    mk('q_prox float', lambda m: m['proof'].__setitem__('q_prox', m['proof']['q_prox'].astype(float)))
    mk('opens None', lambda m: m['proof'].__setitem__('opens', None))
    mk('opens ints', lambda m: m['proof'].__setitem__('opens', [1, 2, 3, 4, 5]))
    mk('opens long', lambda m: m['proof'].__setitem__('opens', m['proof']['opens'] * 9))
    mk('vals None', lambda m: m['proof']['opens'][0].__setitem__('vals', None))
    mk('vals 3-D', lambda m: m['proof']['opens'][1].__setitem__('vals', m['proof']['opens'][1]['vals'][None]))
    mk('salts ints', lambda m: m['proof']['opens'][2].__setitem__('salts', [2**70] * 48))
    mk('path ints', lambda m: m['proof']['opens'][3].__setitem__('path', [2**400]))
    mk('path None', lambda m: m['proof']['opens'][3].__setitem__('path', None))
    mk('extra key', lambda m: m.__setitem__('extra', 1))
    mk('missing key', lambda m: m.pop('root_a'))
    for val in (None, 5, 2**3000, 'sig', []):
        mk('sig=%s' % type(val).__name__, lambda m, v=val: m.__setitem__('sig', v))
    for whole in (7, [], 'x', None, 2**128, {'a': 1}, b'bytes'):
        out.append(('whole message %r' % (whole,), whole))
    return out

muts = mutants(honest)
ref_sys = PR.System.from_genesis(sA.gen)
e_site = [e for e in logA.entries if e['kind'] == 'site' and e['site'] == 0][0]
view = PR.Replica(ref_sys, cached=False)
for e in logA.entries[:logA.entries.index(e_site)]:
    view.consume(e)
view._finalize_keys(view.open_round(1))
assert view.check_site(1, 0, honest)[0] == 'proven'
for label, m in muts:
    try:
        verdict, pub, gamma = view.check_site(1, 0, m)
    except Exception as e:
        raise SystemExit('check_site raised %r on %s' % (e, label))
    assert verdict != 'proven', label
    assert not PR.well_formed(ref_sys, m) or isinstance(m, dict), label
print('%d malformed site messages rejected without any exception' % len(muts))
# site_verify directly with odd roots
pubX = SP.Public(cfg, sysm.prm, sysm.shr, M.proof_weights(cfg, st0), PR.beacon_bits(cfg, b'b', 0), b'ctx')
for r in (2**127, 2**63, None, 'r', b'x'):
    try:
        assert SP.site_verify(pubX, r, b'\x00' * 48, honest) == (False, None)
        assert SP.site_verify(pubX, b'\x00' * 48, r, honest) == (False, None)
    except Exception as e:
        raise SystemExit('site_verify raised %r' % (e,))
print('site_verify rejects malformed roots without raising')
# ligero.verify with junk
for junk in (None, 5, {}, {'q_prox': 2**70, 'q_lin': None, 'q_quad': [], 'opens': 3}):
    assert L.verify(sysm.prm, [(b'\x00' * 48, 1)], dict(m=1, ell=sysm.prm.ell, ncon=0), None, junk) is False
# a site posting garbage inside a live round: the round completes, the site is struck, the audit reproduces the model
state = st0
for t, garbage in enumerate([muts[0][1], 7, muts[40][1], {'root_d1': 2**63}], start=1):
    sites[2].replay = garbage
    state, rep = PR.run_round(sysm, sites, nodes, log, state, t)
    sites[2].replay = None
    assert rep['aborted'] is None and 2 not in rep['valid'], (t, rep)
st2, out = PR.audit(sysm.gen, log.entries)
assert M.model_digest(cfg, st2) == M.model_digest(cfg, state)
print('garbage from a site inside live rounds: rounds complete, site excluded, audit reproduces the model')
# garbage from nodes: beacon reveals, sums, and a garbage seed in answer to a complaint
sysm, sites, nodes, log, _ = build()
state = M.init_model(cfg)
nodes[1].beacon_reveal = lambda t: 2**300
nodes[3].release_sum = lambda t: (np.zeros(3), b'sig')
nodes[5].release_sum = lambda t: ('junk', None)
nodes[6].false_complaint = 1
sites[1].reveal = lambda t, node: 2**64
nodes[2].complaints = lambda t: [7, (None, None), ('a', 'b', 'c')]
st0 = state
state, rep = PR.run_round(sysm, sites, nodes, log, state, 1)
# three misbehaving nodes exceed T = 1: too few good sums remain, so the round aborts and nothing is accepted
assert rep['aborted'] == 'aggregate could not be decoded' and 1 not in rep['valid'], rep
assert all(e['msg']['cts'][1] == b'' for e in log.entries if e['kind'] == 'site')          # node 1 sat the round out
assert M.model_digest(cfg, state) == M.model_digest(cfg, st0)
st2, out = PR.audit(sysm.gen, log.entries)
assert M.model_digest(cfg, st2) == M.model_digest(cfg, state)
# within the bound: one node sends a garbage sum, a site answers a complaint with a garbage seed
sysm, sites, nodes, log, _ = build()
nodes[3].release_sum = lambda t: ('junk', None)
nodes[6].false_complaint = 1
sites[1].reveal = lambda t, node: (2**64, b'sig')
state, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(cfg), 1)
assert rep['aborted'] is None and 1 not in rep['valid'], rep
st2, out = PR.audit(sysm.gen, log.entries)
assert M.model_digest(cfg, st2) == M.model_digest(cfg, state)
print('garbage from nodes and a garbage complaint answer: handled (abort beyond the bound), audit reproduces the model')
print('malformed-input tests passed')
