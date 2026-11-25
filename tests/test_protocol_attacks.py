"""Deviating sites, nodes and coordinator against the complete protocol. Every scenario ends with an audit of the public log."""
import sys, time, copy, itertools
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, protocol as PR, pq
from test_protocol import build

def cheat(point, fn):
    def hook(name, cs, ids):
        if name == point:
            return fn(cs, ids)
    return hook

def crafted_U(cs, ids):          # replace the summed gradient of the first layer by a chosen vector
    nb = ids.shape[1]
    v = (F.to_signed(cs.get(ids)) << np.arange(nb)[None, :]).sum(1)
    v[:64 * 12] = (1 << (nb - 1)) + 12345
    cs.set(ids, ((v[:, None] >> np.arange(nb)[None, :]) & 1).astype(np.uint64))

def honest_sum(sites, valid):
    return sum(sites[i].last['u'] for i in valid)

def audit_ok(sysm, log, state):
    st2, out = PR.audit(sysm.gen, log.entries)
    return M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, state)

results = []
def scenario(name):
    def deco(fn):
        t0 = time.time(); fn(); results.append(name); print('  ok  %-62s %.1fs' % (name, time.time() - t0))
    return deco

@scenario('site submits a crafted update with a forged proof')
def _():
    sysm, sites, nodes, log, _ = build()
    sites[1].cheat_hook = cheat('U', crafted_U)
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None and rep['valid'] == [0, 2, 3, 4] and 'invalid' in rep['excluded'][1]
    assert np.array_equal(rep['agg'], honest_sum(sites, rep['valid'])) and sysm.site_strikes[1] == 1
    assert audit_ok(sysm, log, st)

@scenario('site stays silent after committing noise (selective abort): out this round, no strike')
def _():
    sysm, sites, nodes, log, _ = build()
    sites[3].offline = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    # silence could be censorship by the relay, so it costs the round and a fresh noise commitment, not a strike
    assert rep['valid'] == [0, 1, 2, 4] and rep['excluded'][3] == 'silent' and sysm.site_strikes[3] == 0
    assert sysm.noise_root[3] is None
    sites[3].offline = False                       # it comes back: must first commit new noise, joins one round later
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    assert rep['valid'] == [0, 1, 2, 4]
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 3)
    assert rep['valid'] == [0, 1, 2, 3, 4] and audit_ok(sysm, log, st)

@scenario('site sends a wrong share to one node (complaint upheld)')
def _():
    sysm, sites, nodes, log, _ = build()
    sites[2].bad_share_to = 4
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['valid'] == [0, 1, 3, 4] and rep['excluded'][2] == 'complaint upheld' and sysm.site_strikes[2] == 1
    assert np.array_equal(rep['agg'], honest_sum(sites, rep['valid'])) and 4 not in sysm.node_burned and audit_ok(sysm, log, st)

@scenario('site sends a wrong share and refuses to answer the complaint')
def _():
    sysm, sites, nodes, log, _ = build()
    sites[2].bad_share_to = 0; sites[2].refuse_reveal = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['valid'] == [0, 1, 3, 4] and rep['excluded'][2] == 'complaint upheld, no reveal'
    assert sysm.site_strikes[2] == 0 and audit_ok(sysm, log, st)      # out this round; a missing reveal could be censorship

@scenario('node files a false complaint against an honest site')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[5].false_complaint = 1
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['valid'] == [0, 1, 2, 3, 4] and 5 in sysm.node_burned and sysm.site_strikes[1] == 0
    assert np.array_equal(rep['agg'], honest_sum(sites, rep['valid'])) and audit_ok(sysm, log, st)
    comp = [e for e in log.entries if e['kind'] == 'complaint']
    rev = [e for e in log.entries if e['kind'] == 'reveal']
    assert len(comp) == 1 and len(rev) == 1 and (rev[0]['site'], rev[0]['node']) == (1, 5)   # one share row (node 5's own) public

@scenario('one node posts a wrong sum of shares')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[3].wrong_sum = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None and rep['liars'] == [3] and np.array_equal(rep['agg'], honest_sum(sites, rep['valid']))
    assert 3 in sysm.node_burned and audit_ok(sysm, log, st)

@scenario('two nodes post wrong sums (the tolerated maximum here)')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[0].wrong_sum = True; nodes[6].wrong_sum = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None and rep['liars'] == [0, 6] and np.array_equal(rep['agg'], honest_sum(sites, rep['valid']))
    assert audit_ok(sysm, log, st)

@scenario('three nodes post wrong sums (beyond the bound): round aborts, nothing wrong is accepted')
def _():
    sysm, sites, nodes, log, _ = build()
    for m in (0, 2, 6): nodes[m].wrong_sum = True
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1)
    assert rep['aborted'] == 'aggregate could not be decoded'
    assert M.model_digest(sysm.cfg, st) == M.model_digest(sysm.cfg, st0) and audit_ok(sysm, log, st)

@scenario('node offline from the start; another withholds its beacon value, then returns')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[1].offline = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    # unsigned reveals can be dropped or faked by the relay, so a missing one only sits the node out of the round
    assert rep['aborted'] is None and rep['burned'] == [] and 1 not in sysm.view.history[-1]['burned']
    msg = [e for e in log.entries if e['kind'] == 'site' and e['t'] == 1][0]['msg']
    assert msg['cts'][1] == b''
    nodes[4].withhold_beacon = True
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    assert rep['aborted'] is None and [e for e in log.entries if e['kind'] == 'site' and e['t'] == 2][0]['msg']['cts'][4] == b''
    nodes[4].withhold_beacon = False               # it returns: its round-3 reveal hashes forward to its round-1 value
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 3)
    assert rep['aborted'] is None and np.array_equal(rep['agg'], honest_sum(sites, rep['valid']))
    assert len([e for e in log.entries if e['kind'] == 'site' and e['t'] == 3][0]['msg']['cts'][4]) > 0
    assert not sysm.node_burned and audit_ok(sysm, log, st)

@scenario('too few valid sites: round aborts before any sum is released')
def _():
    sysm, sites, nodes, log, _ = build()
    for i in (0, 1, 2): sites[i].offline = True
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1)
    assert rep['aborted'] and rep['valid'] == [3, 4] and not any(e['kind'] in ('sums', 'certificate') for e in log.entries)
    assert M.model_digest(sysm.cfg, st) == M.model_digest(sysm.cfg, st0)
    for i in (0, 1, 2): sites[i].offline = False
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)          # the three must first re-commit noise
    assert rep['aborted'] and rep['valid'] == [3, 4]
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 3)
    assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4] and audit_ok(sysm, log, st)

@scenario('site replays its previous message; another proves against a stale model')
def _():
    sysm, sites, nodes, log, _ = build()
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1)
    old = [e for e in log.entries if e['kind'] == 'site' and e['site'] == 0][0]['msg']
    sites[0].replay = old; sites[1].stale_model = st0
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    assert rep['valid'] == [2, 3, 4] and 0 in rep['excluded'] and 1 in rep['excluded'] and audit_ok(sysm, log, st)

@scenario('repeated cheating leads to permanent exclusion')
def _():
    sysm, sites, nodes, log, _ = build()
    st = M.init_model(sysm.cfg)
    sites[4].cheat_hook = cheat('U', crafted_U)
    for t in (1, 2, 3):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
        assert rep['aborted'] is None and 4 not in rep['valid']
    assert sysm.excluded(4) and sysm.site_strikes[4] == 2 and audit_ok(sysm, log, st)

@scenario('coordinator shows different valid sets to different nodes (all 64 splits, one node signs anything)')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[6].sign_anything = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    A = rep['valid']; B = A[:-1]
    assert A == [0, 1, 2, 3, 4]
    worst = 0
    for split in itertools.product((0, 1), repeat=6):
        digs = {}
        for nd in nodes:
            shown = (A, B)[1 if nd.idx == 6 else split[nd.idx]]
            for sh in ((A, B) if nd.idx == 6 else (shown,)):
                out = nd.sign_valid_set(1, sh)
                if out is not None and pq.verify(sysm.node_pk[nd.idx], PR._hb(b'valid-set', 1, out[0]), out[1]):
                    digs.setdefault(out[0], set()).add(nd.idx)
        worst = max(worst, sum(len(v) >= sysm.quorum for v in digs.values()))
        assert len(digs) <= 2 and all(len(v) <= 1 for d, v in digs.items() if d != sysm.view.set_digest(1, A))
    assert worst == 1, 'two different valid sets could both be certified'

@scenario('equivocation inside a round: no quorum, no sums released')
def _():
    sysm, sites, nodes, log, _ = build()
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1, equivocate=lambda m, v: v if m < 4 else v[:-1])
    assert rep['aborted'] and 'no quorum' in rep['aborted'] and not any(e['kind'] == 'sums' for e in log.entries)
    assert audit_ok(sysm, log, st)

@scenario('tampered public log is detected by the audit')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    ent = copy.deepcopy(log.entries)
    k = next(i for i, e in enumerate(ent) if e['kind'] == 'result')
    ent[k]['agg'][0] += 1
    try:
        PR.audit(sysm.gen, ent); raise SystemExit('audit accepted a tampered aggregate')
    except AssertionError:
        pass
    ent = [e for e in copy.deepcopy(log.entries) if not (e['kind'] == 'site' and e['site'] == 2)]
    try:
        st2, _o = PR.audit(sysm.gen, ent); bad = M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, st)
    except (AssertionError, KeyError):
        bad = False
    assert not bad, 'audit accepted a log with a removed site message'

@scenario('honest rounds publish nothing about a single site beyond commitments, ciphertexts, digests and proof')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    kinds = sorted(set(e['kind'] for e in log.entries))
    assert kinds == ['beacon', 'certificate', 'keys', 'register', 'result', 'site', 'sums'], kinds      # keys: nodes' public round keys
    m0 = [e for e in log.entries if e['kind'] == 'site'][0]['msg']
    assert sorted(m0) == ['cts', 'ctx', 'digests', 'next_noise', 'proof', 'root_a', 'root_d1', 'root_m', 'sig']
    sizes = {len(c) for e in log.entries if e['kind'] == 'site' for c in e['msg']['cts']}
    def fixed_part(pf):          # everything except the Merkle paths, whose length depends only on the public query positions
        return (pf['q_prox'].shape, pf['q_lin'].shape, pf['q_quad'].shape, tuple(o['vals'].shape for o in pf['opens']))
    psz = {fixed_part(e['msg']['proof']) for e in log.entries if e['kind'] == 'site'}
    assert len(sizes) == 1 and len(psz) == 1                      # message sizes do not depend on the data

@scenario('coordinator forges a complaint in a node\'s name, or a noise commitment in a site\'s name')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[5].false_complaint = 1
    sites[3].offline = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert audit_ok(sysm, log, st)
    ent = copy.deepcopy(log.entries)
    k = next(i for i, e in enumerate(ent) if e['kind'] == 'complaint'); ent[k]['node'] = 2      # blame another node
    try:
        PR.audit(sysm.gen, ent); raise SystemExit('audit accepted a complaint attributed to the wrong node')
    except AssertionError:
        pass
    # a forgery: an unsigned complaint with no seed, inserted to strike an honest site.
    # Every party and the audit ignore it: no strike, and site 0 is not removed from the set.
    ent = [e for e in copy.deepcopy(log.entries) if e['kind'] in ('register', 'beacon', 'keys', 'site', 'noise')]
    ent.append(dict(kind='complaint', t=1, site=0, node=0, seed=None, verdict='site'))
    ent.append(dict(kind='certificate', t=1, valid=[1, 2, 4], sigs={})); ent.append(dict(kind='abort', t=1, reason='forged'))
    st2, hist = PR.audit(sysm.gen, ent)
    assert hist[-1]['outcome'] == 'aborted' and hist[-1]['struck'] == [] and hist[-1]['burned'] == []
    sites[3].offline = False
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    ent = copy.deepcopy(log.entries)
    k = next(i for i, e in enumerate(ent) if e['kind'] == 'noise'); ent[k]['root'] = b'\x07' * 48   # a root the site never signed
    try:
        st2, _o = PR.audit(sysm.gen, ent); same = M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, st)
    except (AssertionError, KeyError):
        same = True          # rejected outright
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 3)
    assert 3 in rep['valid'] and audit_ok(sysm, log, st)          # the real site is unaffected

@scenario('censored site never opens a data commitment beyond its zero-knowledge budget')
def _():
    sysm, sites, nodes, log, _ = build(chunks_per_site=1)
    st = M.init_model(sysm.cfg)
    s0 = sites[0]
    assert s0.max_opens == (sysm.prm.ell // 2 + sysm.prm.tprime) // sysm.prm.t and s0.max_opens >= 2
    # a coordinator that sees but never logs site 0's messages (it does log its fresh noise commitments)
    produced = []
    orig = log.append
    def censor(e, _o=orig):
        if e.get('kind') == 'site' and e.get('site') == 0:
            produced.append(e['t']); return
        _o(e)
    log.append = censor
    for t in range(1, 2 * s0.max_opens + 4):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
    log.append = orig
    assert len(produced) == s0.max_opens and s0.opens[0] == s0.max_opens, (produced, s0.max_opens, s0.opens)
    assert sysm.site_strikes[0] == 0 and audit_ok(sysm, log, st)
    # an honest, uncensored site is never limited: openings per commitment stay at most two
    sysm, sites, nodes, log, _ = build(chunks_per_site=1)
    st = M.init_model(sysm.cfg)
    for t in (1, 2, 3, 4):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
        assert rep['valid'] == [0, 1, 2, 3, 4] and all(max(s.opens) <= 2 for s in sites)

print('%d protocol scenarios passed' % len(results))
