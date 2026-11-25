"""A malicious relay against honest sites and nodes: single-site release, duplicated messages, substituted round keys,
decoding with too few sums, handed-in copies, repeated requests, exceptions, garbage and tampered rounds."""
import sys, time, copy
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, protocol as PR, pq
from test_protocol import build

results = []
def scenario(name):
    def deco(fn):
        t0 = time.time(); fn(); results.append(name); print('  ok  %-74s %.1fs' % (name, time.time() - t0))
    return deco

def audit_ok(sysm, log, state):
    st2, out = PR.audit(sysm.gen, log.entries)
    return M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, state)

def honest_sum(sites, valid):
    return sum(sites[i].last['u'] for i in valid)


@scenario('coordinator asks honest nodes to certify and release one site alone: refused')
def _():
    sysm, sites, nodes, log, _ = build()
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1, equivocate=lambda m, v: [2])
    assert rep['aborted'] and 'no quorum' in rep['aborted'] and not any(e['kind'] == 'sums' for e in log.entries)
    assert all(nd.release_sum(1) is None for nd in nodes)                     # no certificate for any set they signed
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)                    # an honest round, then ask again
    assert rep['aborted'] is None
    assert all(nd.release_sum(2) is None for nd in nodes)                     # once per round, and shares are erased
    assert audit_ok(sysm, log, st)


@scenario('relay posts one site message twice, or three times alone: it counts once')
def _():
    sysm, sites, nodes, log, _ = build()
    orig = log.append
    def dup(e, _o=orig):
        _o(e)
        if e.get('kind') == 'site' and e.get('site') == 0:
            _o(e)
    log.append = dup
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    log.append = orig
    assert rep['valid'] == [0, 1, 2, 3, 4] and np.array_equal(rep['agg'], honest_sum(sites, rep['valid']))
    assert audit_ok(sysm, log, st)
    # the relay drops the other sites and posts site 0 three times: the floor of 3 is not met by one site
    sysm, sites, nodes, log, _ = build()
    for i in (1, 2, 3, 4):
        sites[i].offline = True
    orig = log.append
    def trip(e, _o=orig):
        _o(e)
        if e.get('kind') == 'site' and e.get('site') == 0:
            _o(e); _o(e)
    log.append = trip
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1)
    log.append = orig
    assert rep['aborted'] and rep['valid'] == [0] and not any(e['kind'] == 'sums' for e in log.entries)
    assert audit_ok(sysm, log, st)


@scenario('stolen signing key used to substitute a round key: the node disputes it')
def _():
    sysm, sites, nodes, log, _ = build()
    stolen = nodes[4].sk                                     # seized after an earlier round
    atk_ek, atk_dk = pq.kem_keygen()
    real = nodes[4].round_key
    nodes[4].round_key = lambda t, _r=real: (_r(t), (atk_ek, pq.sign(stolen, PR._hb(b'round-key', t, 4, atk_ek))))[1]
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert 4 in sysm.node_burned and rep['aborted'] is None
    assert any(e['kind'] == 'key-dispute' for e in log.entries)
    assert all(e['msg']['cts'][4] == b'' for e in log.entries if e['kind'] == 'site')
    assert audit_ok(sysm, log, st)


@scenario('if the relay also drops the dispute, the substitution works (stated limit)')
def _():
    sysm, sites, nodes, log, _ = build()
    stolen = nodes[4].sk
    atk_ek, atk_dk = pq.kem_keygen()
    real = nodes[4].round_key
    nodes[4].round_key = lambda t, _r=real: (_r(t), (atk_ek, pq.sign(stolen, PR._hb(b'round-key', t, 4, atk_ek))))[1]
    nodes[4].key_dispute = lambda t: None                     # the dispute is censored
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    blobs = [e['msg']['cts'][4] for e in log.entries if e['kind'] == 'site']
    assert all(len(b) > 0 for b in blobs)                     # sites encrypted to the attacker: node 4 counts as corrupted


@scenario('decoding with fewer than w + 2T sums is refused: no wrong aggregate')
def _():
    sysm, sites, nodes, log, _ = build(K=9, kp=2, Tn=2, A_min=3)
    shr = sysm.shr
    for m in (6, 7, 8):
        nodes[m].withhold_beacon = True
    P = F.P
    agree = [shr.alphas[m] for m in (0, 1, 2)]
    Z = lambda x: (x - agree[0]) * (x - agree[1]) * (x - agree[2])
    c = (-pow(60, P - 2, P)) % P
    for m in (4, 5):
        nd = nodes[m]; orig = nd.release_sum
        def rel(t, _nd=nd, _o=orig):
            out = _o(t)
            if out is None:
                return None
            acc = F.fadd(out[0], F.U64(c * (Z(shr.alphas[_nd.idx]) % P) % P))
            return acc, pq.sign(_nd.sk, PR._hb(b'sum', t, acc))
        nd.release_sum = rel
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1)
    assert rep['aborted'] == 'aggregate could not be decoded'
    assert M.model_digest(sysm.cfg, st) == M.model_digest(sysm.cfg, st0) and audit_ok(sysm, log, st)


@scenario('nodes read the logged copy: a different copy handed to them has no effect')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['valid'] == [0, 1, 2, 3, 4] and not any(e['kind'] == 'complaint' for e in log.entries)
    import inspect                                           # no public node method takes a message: no relay channel
    for name, fn in inspect.getmembers(PR.Node, inspect.isfunction):
        if not name.startswith('_'):
            assert not ({'msg', 'message', 'share', 'shares'} & set(inspect.signature(fn).parameters)), name


@scenario('a second request in the same round returns the same message; spent noise is never reused')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    s0 = sites[0]
    m1 = s0.round_message(1); m2 = s0.round_message(1)
    assert m1 is not None and m1 is m2
    assert s0.round_message(2) is None                              # round 2 is not open on the site's own view
    log.append(dict(kind='abort', t=1, reason='test'))
    assert s0.round_message(1) is None                              # closed: nothing more for round 1
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4]   # proven in round 1, so its next noise is fresh


@scenario('an exception mid-round still erases every round key and share')
def _():
    sysm, sites, nodes, log, _ = build()
    def boom(r, i):
        raise RuntimeError('crash during share checking')
    nodes[3]._check_share = boom
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None                                    # the other nodes carry the round
    assert all(nd._secrets_held() == dict(round_keys=[], share_rounds=[]) for nd in nodes)


@scenario('entry points reject garbage without raising; audit raises only AssertionError')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    garbage = [None, 7, 'x', [1, 2], {0: (b'a',)}, {0: None}, {99: (b'a', b'b')}, {None: 3}, {'0': (b'a', b'b')}]
    s2 = PR.System.from_genesis(sysm.gen)
    for g in garbage:
        PR.decode_aggregate(s2, 1, g if isinstance(g, dict) else {})
        PR.judge(s2, {0: b'x'}, None, g, None, 0, g)
        v = PR.Replica(s2)
        for kind in ('register', 'beacon', 'keys', 'key-dispute', 'site', 'complaint', 'reveal', 'noise', 'certificate',
                     'sums', 'result', 'abort', None, 7):
            v.consume(dict(kind=kind, t=g, site=g, node=g, msg=g, posted=g, sigs=g, sums=g, seed=g, root=g, reveals=g))
            v.consume(dict(kind=kind, t=1, site=g, node=g, msg=g, posted=g, sigs=g, sums=g, seed=g, root=g, reveals=g))
            v.consume(g)
    ent = log.entries
    bad_logs = [ent + [None], ent[:-1] + [dict(kind='result', t=None)], [e if e['kind'] != 'site' else dict(e, site=99) for e in ent],
                [e if e['kind'] != 'sums' else dict(e, sums={'x': 1}) for e in ent], [e if e['kind'] != 'beacon' else dict(e, t=None) for e in ent],
                [e if e['kind'] != 'certificate' else dict(e, sigs=7) for e in ent], ent[:-2]]
    for bl in bad_logs:
        try:                                 # any exception other than AssertionError fails this test
            PR.audit(sysm.gen, copy.deepcopy(bl))
        except AssertionError:
            pass


@scenario('tampered round fields and injected junk entries are detected by the audit')
def _():
    sysm, sites, nodes, log, _ = build()
    for i in (3,):
        sites[i].offline = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert audit_ok(sysm, log, st)
    def tampered(fn):
        ent = copy.deepcopy(log.entries); fn(ent)
        try:
            s2, _o = PR.audit(sysm.gen, ent)
            return M.model_digest(sysm.cfg, s2) != M.model_digest(sysm.cfg, st)
        except AssertionError:
            return True
    for kind in ('certificate', 'sums', 'result'):
        assert tampered(lambda ent, k=kind: next(e for e in ent if e['kind'] == k).update(t=9)), kind
    assert tampered(lambda ent: next(e for e in ent if e['kind'] == 'result').update(valid=[0, 1]))
    k = next(i for i, e in enumerate(log.entries) if e['kind'] == 'site')
    assert tampered(lambda ent: ent.insert(k, dict(kind='site', t=1, site=3, msg=None)))     # junk for a silent site


@scenario('round numbers must increase; a skipped number burns nobody')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    st2, rep2 = PR.run_round(sysm, sites, nodes, log, st, 1)
    assert rep2['aborted'] == 'invalid round number' and not sysm.node_burned
    st3, rep3 = PR.run_round(sysm, sites, nodes, log, st, 3)
    assert rep3['aborted'] is None and not sysm.node_burned and audit_ok(sysm, log, st3)


@scenario('public record budget: a chunk is proved at most max_uses times, without strikes')
def _():
    sysm, sites, nodes, log, _ = build(chunks_per_site=2)
    sysm.max_uses = 1
    sysm.gen = sysm.genesis()
    st = M.init_model(sysm.cfg)
    for t in (1, 2, 3, 4):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
    assert all(u <= 1 for v in sysm.chunk_uses.values() for u in v)
    assert not any(sysm.site_strikes.values()) and audit_ok(sysm, log, st)


print('%d relay-attack protocol scenarios passed' % len(results))
