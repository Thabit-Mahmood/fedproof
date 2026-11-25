"""Honest parties act only on the log. Each scenario is an attack by the relay against the log-driven protocol and
checks that it fails. The relay is played by the test: it may call any party's public
method at any time, with any arguments, and may append, drop, duplicate or reorder bulletin entries."""
import sys, time, copy, inspect
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import model as M, protocol as PR, pq
from test_protocol import build

results = []
def scenario(name):
    def deco(fn):
        t0 = time.time(); fn(); results.append(name); print('  ok  %-78s %.1fs' % (name, time.time() - t0))
    return deco

def audit_ok(sysm, log, state):
    st2, out = PR.audit(sysm.gen, log.entries)
    return M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, state)

def honest_sum(sites, valid):
    return sum(sites[i].last['u'] for i in valid)

EMPTY = dict(round_keys=[], share_rounds=[])
st0 = None


def public_methods(cls):
    return {n for n, f in inspect.getmembers(cls, inspect.isfunction) if not n.startswith('_')}


@scenario('relay asks for next-round keys mid-round: refused, round-1 keys kept, nothing opened')
def _():
    sysm, sites, nodes, log, _ = build()
    assert all(nd.round_key(1) is None for nd in nodes)              # before the round-1 beacon: refused
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    assert all(nd.round_key(2) is None and nd.round_key(9) is None for nd in nodes)
    assert all(1 in nd.round_dk for nd in nodes)
    st, rep = PR.continue_round(sysm, sites, nodes, log, 1)
    assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4]
    assert not any(e['kind'] in ('complaint', 'reveal') for e in log.entries)
    assert all(nd._secrets_held() == EMPTY for nd in nodes) and audit_ok(sysm, log, st)


@scenario('no share can be handed to a node; a fake reveal entry changes nothing')
def _():
    assert 'accept_revealed' not in public_methods(PR.Node)
    for n in public_methods(PR.Node):
        assert not ({'row', 'share', 'shares', 'msg', 'message'} & set(inspect.signature(getattr(PR.Node, n)).parameters)), n
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    before = {m: nodes[m].shares[(1, 0)].copy() for m in range(sysm.K)}
    log.append(dict(kind='reveal', t=1, site=0, node=3, seed=b'\x01' * 32, sig=b'x'))          # no complaint, no signature
    log.append(dict(kind='reveal', t=1, site=0, node=3, seed=b'\x01' * 32,
                    sig=pq.sign(nodes[3].sk, PR._hb(b'reveal', 1, 0, 3, b'\x01' * 32))))     # signed by the wrong party
    assert all(np.array_equal(nodes[m].shares[(1, 0)], before[m]) for m in range(sysm.K))
    st, rep = PR.continue_round(sysm, sites, nodes, log, 1)
    assert rep['aborted'] is None and np.array_equal(rep['agg'], honest_sum(sites, rep['valid'])) and audit_ok(sysm, log, st)


@scenario('seeds only for a logged complaint, once, and erased at close; nodes sign no complaint on request')
def _():
    assert 'complaint_sig' not in public_methods(PR.Node) and 'reveal_seed' not in public_methods(PR.Site)
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    assert all(sites[i].reveal(1, m) is None for i in range(sysm.N) for m in range(sysm.K))      # no complaint logged
    assert all(nd.complaints(1) == [] for nd in nodes)                                          # nothing failed
    st, rep = PR.continue_round(sysm, sites, nodes, log, 1)
    assert all(s.last['seeds'] is None for s in sites)
    assert all(sites[i].reveal(1, m) is None for i in range(sysm.N) for m in range(sysm.K))
    # with a real complaint: one reveal, and asking again returns nothing
    sysm, sites, nodes, log, _ = build()
    sites[2].bad_share_to = 4
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    assert len([e for e in log.entries if e['kind'] == 'reveal']) == 1
    assert sites[2].reveal(1, 4) is None and sites[2].reveal(1, 0) is None
    st, rep = PR.continue_round(sysm, sites, nodes, log, 1)
    assert rep['valid'] == [0, 1, 3, 4] and sysm.site_strikes[2] == 1 and audit_ok(sysm, log, st)


@scenario('a second key request returns the same key; there is no relay-facing erase')
def _():
    assert 'erase' not in public_methods(PR.Node)
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    posted = [e for e in log.entries if e['kind'] == 'keys'][0]['posted']
    for nd in nodes:
        assert nd.round_key(1) is None or nd.round_key(1) == posted[nd.idx]     # keys already logged: no new pair
        assert nd.key_dispute(1) is None and len(nd.round_dk) == 1
    st, rep = PR.continue_round(sysm, sites, nodes, log, 1)
    assert rep['aborted'] is None and not sysm.node_burned and not any(e['kind'] in ('complaint', 'key-dispute') for e in log.entries)
    # a request before the keys are logged, made twice, gives the same signed key both times
    sysm, sites, nodes, log, _ = build()
    log.append(dict(kind='beacon', t=1, reveals={nd.idx: nd.beacon_reveal(1) for nd in nodes}))
    assert all(nd.round_key(1) == nd.round_key(1) for nd in nodes)


@scenario('a copy with a bad signature is never counted, even after the genuine one was verified')
def _():
    sysm, sites, nodes, log, _ = build()
    orig = log.append
    def relay(e, _o=orig):
        if e.get('kind') == 'site' and e['site'] == 0:
            v = PR.Replica(sysm)                                  # the relay verifies the genuine message first
            for x in log.entries:
                v.consume(x)
            assert v.check_site(1, 0, e['msg'])[0] == 'proven'
            bad = dict(e['msg']); bad['sig'] = bytes(len(e['msg']['sig']))
            _o(dict(e, msg=bad))
        _o(e)
    log.append = relay
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    log.append = orig
    assert rep['valid'] == [0, 1, 2, 3, 4] and not any(sysm.site_strikes.values()) and audit_ok(sysm, log, st)


@scenario('a duplicate after the certificate is ignored by everyone, the audit included')
def _():
    sysm, sites, nodes, log, _ = build()
    orig = log.append
    def relay(e, _o=orig):
        _o(e)
        if e.get('kind') == 'certificate':
            _o(copy.deepcopy([x for x in log.entries if x['kind'] == 'site' and x['site'] == 0][0]))
    log.append = relay
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    log.append = orig
    assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4] and audit_ok(sysm, log, st)


@scenario('a complaint logged after nodes signed: every party and the audit abort the round alike')
def _():
    sysm, sites, nodes, log, _ = build()
    sites[4].bad_share_to = 6
    nodes[6].complaints = lambda t: []                          # corrupt node 6 keeps quiet at first
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1, stop_before='certificate')
    valid = rep['valid']
    assert valid == [0, 1, 2, 3, 4]
    sigs = {}
    for nd in nodes:
        out = nd.sign_valid_set(1, valid)
        if out is not None:
            sigs[nd.idx] = out[1]
    log.append(dict(kind='complaint', t=1, site=4, node=6, sig=pq.sign(nodes[6].sk, PR._hb(b'complaint', 1, 4, 6))))
    log.append(dict(kind='certificate', t=1, valid=valid, sigs=sigs))
    views = [sysm.view] + [p.view for p in sites + nodes]
    assert all(v.round.closed == 'aborted' for v in views)
    assert all(nd.release_sum(1) is None and nd._secrets_held() == EMPTY for nd in nodes)
    st2, hist = PR.audit(sysm.gen, log.entries)
    assert hist[-1]['outcome'] == 'aborted' and M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, st0)
    sites[4].bad_share_to = None
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 2)
    assert rep['aborted'] is None and audit_ok(sysm, log, st)


@scenario('complaints from a node with no accepted key are ignored; no honest site is dropped or struck')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[5].withhold_key = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    for i in range(sysm.N):
        log.append(dict(kind='complaint', t=1, site=i, node=5, sig=pq.sign(nodes[5].sk, PR._hb(b'complaint', 1, i, 5))))
        assert sites[i].reveal(1, 5) is None
    st, rep = PR.continue_round(sysm, sites, nodes, log, 1)
    assert rep['valid'] == [0, 1, 2, 3, 4] and not any(sysm.site_strikes.values()) and audit_ok(sysm, log, st)


@scenario('nodes erase when the log closes the round, with no call from the relay; one crash affects no other')
def _():
    sysm, sites, nodes, log, _ = build()
    seen = {}
    orig = log.append
    def relay(e, _o=orig):
        _o(e)
        if e.get('kind') == 'sums':                              # before any further call or entry
            seen['held'] = [nd._secrets_held() for nd in nodes]
    log.append = relay
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    log.append = orig
    assert rep['aborted'] is None and all(h == EMPTY for h in seen['held'])
    # a relay that abandons a round after the certificate: the next beacon closes it and every node erases
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2, stop_before='certificate')
    assert all(2 in nd.round_dk for nd in nodes)
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 3)
    assert sysm.view.history[-2]['outcome'] == 'abandoned' and rep['aborted'] is None
    assert all(nd._secrets_held() == EMPTY for nd in nodes) and audit_ok(sysm, log, st)
    # a corrupted node whose erasure raises: the others still erase
    sysm, sites, nodes, log, _ = build()
    def boom(t):
        raise RuntimeError('corrupt node crashes in erase')
    nodes[6]._erase = boom
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None and all(nd._secrets_held() == EMPTY for nd in nodes[:6])


@scenario('the signature binds shapes: a reshaped copy is not the site\'s message, and the site is not struck')
def _():
    sysm, sites, nodes, log, _ = build()
    orig = log.append
    def relay(e, _o=orig):
        if e.get('kind') == 'site' and e['site'] == 1:
            m = dict(e['msg']); m['digests'] = np.ascontiguousarray(m['digests'].T)
            assert PR.site_msg_digest(1, 1, m) != PR.site_msg_digest(1, 1, e['msg'])
            e = dict(e, msg=m)
        _o(e)
    log.append = relay
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    log.append = orig
    assert rep['valid'] == [0, 2, 3, 4] and sysm.site_strikes[1] == 0 and audit_ok(sysm, log, st)
    a = np.zeros((2, 3), dtype=np.uint64)
    assert PR._hb(a) != PR._hb(a.reshape(3, 2)) and PR._hb(b'ab', b'c') != PR._hb(b'a', b'bc')


@scenario('a node raising inside a call: the round completes, pointers and audit stay consistent')
def _():
    sysm, sites, nodes, log, _ = build()
    def boom(*a):
        raise RuntimeError('corrupt node 6 crashes')
    nodes[6].sign_valid_set = boom
    nodes[5].release_sum = boom
    st = M.init_model(sysm.cfg)
    for t in (1, 2, 3):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
        assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4] and not any(sysm.site_strikes.values())
    assert audit_ok(sysm, log, st)


@scenario('every honest party\'s replica agrees with the driver and the audit after mixed deviations')
def _():
    sysm, sites, nodes, log, _ = build()
    sites[1].bad_share_to = 2; nodes[0].false_complaint = 3; nodes[4].wrong_sum = True; sites[3].offline = True
    st = M.init_model(sysm.cfg)
    for t in (1, 2):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
        sites[3].offline = False
    st2, hist = PR.audit(sysm.gen, log.entries)
    views = [p.view for p in sites + nodes]
    assert all(v.head == sysm.view.head == log.head for v in views)
    assert all(v.history == sysm.view.history for v in views)
    assert [dict(h, model=None) for h in hist] == [dict(h, model=None) for h in sysm.view.history]
    assert M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, st)


print('%d log-driven protocol scenarios passed' % len(results))
