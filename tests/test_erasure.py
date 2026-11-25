"""Fresh node keys every round and erasure of keys and shares at the end of each round (forward secrecy).

Threat: an attacker who seizes nodes after the fact, one or several at different times, and reads the public log.
Without erasure, the encrypted shares stay on the log forever and any long-lived key or stored share opens them, so
the privacy threshold would have to hold over the whole life of the system. With erasure it has to hold within a round.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, protocol as PR, pq, message as SP
from test_protocol import build


def audit_ok(sysm, log, state):
    st2, out = PR.audit(sysm.gen, log.entries)
    return M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, state)


def logged_ciphertexts(log):
    """(t, site, node, blob, aad) for every share ciphertext on the public log."""
    out, ctx_by_t = [], {}
    for e in log.entries:
        if e['kind'] == 'site' and e['msg'] is not None:
            for m, blob in enumerate(e['msg']['cts']):
                out.append((e['t'], e['site'], m, blob))
    return out


def seize(nodes):
    """Everything an attacker gets by taking over these nodes now: decapsulation keys and stored share rows."""
    dks, shares = [], {}
    for nd in nodes:
        dks += list(nd.round_dk.values())
        for k, v in nd.shares.items():
            shares.setdefault(k, {})[nd.idx] = v
    return dks, shares


results = []
def scenario(name):
    def deco(fn):
        t0 = time.time(); fn(); results.append(name); print('  ok  %-70s %.1fs' % (name, time.time() - t0))
    return deco


@scenario('after every round no node holds a round key or a share')
def _():
    sysm, sites, nodes, log, _ = build()
    st = M.init_model(sysm.cfg)
    for t in range(1, 4):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
        assert rep['aborted'] is None
        for nd in nodes:
            h = nd._secrets_held()
            assert h['round_keys'] == [] and h['share_rounds'] == [], (t, nd.idx, h)
    # each round used different keys
    ks = [e['posted'] for e in log.entries if e['kind'] == 'keys']
    for m in range(sysm.K):
        assert len({k[m][0] for k in ks}) == len(ks)
    assert audit_ok(sysm, log, st)


CAPTURED = []                       # the public context of every site message, in posting order (test instrument)
_orig_prove = SP.site_prove
def _capture(pub, *a, **k):
    CAPTURED.append(pub.ctx)
    return _orig_prove(pub, *a, **k)
PR.SP.site_prove = _capture


def site_entries(log):
    return [e for e in log.entries if e['kind'] == 'site' and e['msg'] is not None]


@scenario('seizing every node after the rounds yields no key and no share')
def _():
    sysm, sites, nodes, log, _ = build()
    st = M.init_model(sysm.cfg)
    for t in range(1, 4):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
    dks, shares = seize(nodes)
    assert dks == [] and shares == {}
    assert len(site_entries(log)) == 3 * sysm.N


@scenario('negative control: without erasure, seized nodes open the log and reveal one update')
def _():
    del CAPTURED[:]
    sysm, sites, nodes, log, _ = build()
    for nd in nodes:
        nd.no_erase = True
    st = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 1)
    u_site2 = sites[2].last['u'].copy()
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    # long after round 1 the attacker seizes w nodes and decrypts their round-1 ciphertexts from the public log
    shr = sysm.shr
    take = list(range(shr.w))
    ents = site_entries(log)
    k = [j for j, e in enumerate(ents) if e['t'] == 1 and e['site'] == 2][0]
    e = ents[k]; ctx = e['msg']['ctx']
    assert ctx in CAPTURED
    rows = []
    for m in take:
        dk = nodes[m].round_dk[1]
        pt = pq.open_(dk, e['msg']['cts'][m], SP.H(ctx) + m.to_bytes(4, 'little'))
        assert pt is not None
        rows.append(np.frombuffer(pt, dtype=np.uint64)[:shr.nblk])
    Cm = shr.secret_coeffs(take)
    mat = np.array(rows, dtype=np.uint64)
    rec = np.array([F.matvec(Cm, mat[:, B].copy()) for B in range(shr.nblk)]).reshape(-1)[:sysm.cfg.d]
    assert np.array_equal(F.to_signed(rec), u_site2), 'the leak this design closes did not reproduce'


@scenario('a node that withholds its round key sits the round out; the round completes')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[5].withhold_key = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None and 5 not in sysm.node_burned
    assert np.array_equal(rep['agg'], sum(sites[i].last['u'] for i in rep['valid']))
    msg = [e for e in log.entries if e['kind'] == 'site'][0]['msg']
    assert msg['cts'][5] == b''
    assert audit_ok(sysm, log, st)


@scenario('a node that posts an invalid key (fails the modulus check) receives nothing')
def _():
    sysm, sites, nodes, log, _ = build()
    nodes[3].bad_key = True
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None and all(e['msg']['cts'][3] == b'' for e in log.entries if e['kind'] == 'site')
    assert not pq.ek_valid(b'\xff' * pq.KEM_EK) and pq.ek_valid(pq.kem_keygen()[0])
    assert audit_ok(sysm, log, st)


@scenario('a key substituted in transit (no valid signature) is never used by any site')
def _():
    sysm, sites, nodes, log, _ = build()
    atk_ek, atk_dk = pq.kem_keygen()
    real = nodes[4].round_key
    def swapped(t, _r=real):
        ek, sg = _r(t)
        return atk_ek, sg                                  # the relay replaces the key, keeps the signature
    nodes[4].round_key = swapped
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None
    for e in log.entries:
        if e['kind'] == 'site' and e['msg'] is not None:
            assert pq.open_(atk_dk, e['msg']['cts'][4], b'x') is None and e['msg']['cts'][4] == b''
    assert audit_ok(sysm, log, st)


@scenario('a key from an earlier round, replayed with its signature, is rejected')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    old = [e for e in log.entries if e['kind'] == 'keys'][0]['posted'][6]
    nodes[6].round_key = lambda t, _o=old: _o
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    assert rep['aborted'] is None and all(e['msg']['cts'][6] == b'' for e in log.entries if e['kind'] == 'site' and e['t'] == 2)
    assert audit_ok(sysm, log, st)


@scenario('a site proof is bound to the round keys: changing one key invalidates it')
def _():
    sysm, sites, nodes, log, _ = build()
    st0 = M.init_model(sysm.cfg)
    st, rep = PR.run_round(sysm, sites, nodes, log, st0, 1)
    e_site = [e for e in log.entries if e['kind'] == 'site' and e['t'] == 1][0]
    e_keys = [e for e in log.entries if e['kind'] == 'keys' and e['t'] == 1][0]
    beacon_entry = [e for e in log.entries if e['kind'] == 'beacon' and e['t'] == 1][0]
    # rebuild the public state of round 1 from the log, then alter one key and re-verify the site message
    v = PR.Replica(PR.System.from_genesis(sysm.gen), cached=False)
    for e in log.entries[:log.entries.index(e_site)]:
        v.consume(e)
    assert v.check_site(1, e_site['site'], e_site['msg'])[0] == 'proven'
    r = v.open_round(1)
    other = dict(r.round_ek); other[0] = pq.kem_keygen()[0]
    r.round_ek, r.keys_digest = other, PR.keys_digest(1, other)
    assert v.check_site(1, e_site['site'], e_site['msg'])[0] == 'mismatch'
    pub = v.public_for(1, e_site['site'], e_site['msg']['next_noise'])          # and the proof itself fails there too
    c = PR.chunk_index(e_site['site'], 1, len(v.data_roots[e_site['site']]))
    assert not SP.site_verify(pub, v.data_roots[e_site['site']][c], v.noise_root[e_site['site']], e_site['msg'])[0]


@scenario('deviations combined with erasure: wrong share to a node, then a false complaint')
def _():
    sysm, sites, nodes, log, _ = build()
    sites[1].bad_share_to = 2
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert 1 not in rep['valid'] and 2 not in sysm.node_burned
    sites[1].bad_share_to = None
    nodes[0].false_complaint = 3
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    assert 0 in sysm.node_burned and 3 in rep['valid']
    for nd in nodes:
        assert nd._secrets_held() == dict(round_keys=[], share_rounds=[])
    assert audit_ok(sysm, log, st)


print('%d erasure scenarios passed' % len(results))
