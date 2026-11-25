"""The public surface of a party, key disputes, beacon reveals, late results, canonical heads and decoding with a
liar the random combination misses. Each scenario is an attack and checks that it fails."""
import sys, time, copy, inspect
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import field as F, model as M, protocol as PR, pq, wire as W
from fedproof.sharing import Sharing
from test_protocol import build

results = []
def scenario(name):
    def deco(fn):
        t0 = time.time(); fn(); results.append(name); print('  ok  %-80s %.1fs' % (name, time.time() - t0))
    return deco

def audit_ok(sysm, log, state):
    st2, out = PR.audit(sysm.gen, log.entries)
    return M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, state)

SITE_REQ = {'round_message', 'reveal', 'noise_only'}
NODE_REQ = {'beacon_reveal', 'round_key', 'key_dispute', 'complaints', 'sign_valid_set', 'release_sum'}


def public(cls):
    return {n for n, f in inspect.getmembers(cls, inspect.isfunction) if not n.startswith('_')}


@scenario('the public surface of a party is exactly its protocol requests; replica and data fixed at setup')
def _():
    assert public(PR.Site) == SITE_REQ, public(PR.Site) ^ SITE_REQ
    assert public(PR.Node) == NODE_REQ, public(PR.Node) ^ NODE_REQ
    sysm, sites, nodes, log, _ = build()
    for p in (sites[0], nodes[0]):
        try:
            p._attach(PR.Replica(sysm)); raise SystemExit('a second replica was attached')
        except AssertionError:
            pass
    try:
        sites[0]._register(); raise SystemExit('a site registered twice')
    except AssertionError:
        pass
    from experiments import harness as NH                                     # the networked party answers the same requests only
    assert NH.REQUESTS['Site'] == SITE_REQ and NH.REQUESTS['Node'] == NODE_REQ


@scenario('a site never proves against a data commitment the bulletin does not name')
def _():
    sysm, sites, nodes, log, _ = build()
    c = PR.chunk_index(1, 2, len(sites[1].chunks))
    real = sites[1].blocks[c]
    sites[1].blocks[c] = sites[1].blocks[1 - c]                  # whatever replaced it, it is not the logged root
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 2)
    assert 1 not in rep['valid'] and sysm.site_strikes[1] == 0      # silent, never a failing signed proof
    sites[1].blocks[c] = real


@scenario('a junk key posted in a node\'s name is not disputed and burns nobody')
def _():
    sysm, sites, nodes, log, _ = build()
    log.append(dict(kind='beacon', t=1, reveals={nd.idx: nd.beacon_reveal(1) for nd in nodes}))
    log.append(dict(kind='keys', t=1, posted={m: (b'junk', b'nosig') for m in range(sysm.K)}))
    sigs = {nd.idx: nd.key_dispute(1) for nd in nodes}
    assert all(v is None for v in sigs.values())
    forged = {m: pq.sign(nodes[m].sk, PR._hb(b'key-dispute', 1, m, b'junk')) for m in range(sysm.K)}   # even signed
    log.append(dict(kind='key-dispute', t=1, sigs=forged))
    log.append(dict(kind='abort', t=1, reason='relay'))
    assert not sysm.node_burned
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 2)    # sites re-commit noise first
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 3)
    assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4] and not sysm.node_burned and audit_ok(sysm, log, st)


@scenario('a key signed with a stolen signing key is disputed even if the node was never asked for one')
def _():
    sysm, sites, nodes, log, _ = build()
    stolen = nodes[4].sk
    atk_ek, _dk = pq.kem_keygen()
    nodes[4].round_key = lambda t: None                          # the relay never asks node 4 (modelled: no key)
    orig = log.append
    def relay(e, _o=orig):
        if e.get('kind') == 'keys':
            posted = dict(e['posted']); posted[4] = (atk_ek, pq.sign(stolen, PR._hb(b'round-key', e['t'], 4, atk_ek)))
            e = dict(e, posted=posted)
        return _o(e)
    log.append = relay
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    log.append = orig
    del nodes[4].round_key
    assert 4 in sysm.node_burned and 4 in sysm.view.history[-1]['burned'] and rep['aborted'] is None
    assert all(e['msg']['cts'][4] == b'' for e in log.entries if e['kind'] == 'site')
    assert audit_ok(sysm, log, st)


@scenario('beacon values only for the next few rounds, one round per gap; junk beacons open nothing')
def _():
    sysm, sites, nodes, log, _ = build()
    nd = nodes[0]
    assert nd.beacon_reveal(PR.MAX_STEP + 1) is None and nd.beacon_reveal(sysm.max_rounds) is None
    v = nd.beacon_reveal(2)
    v1 = nd.beacon_reveal(1)                                          # earlier rounds of the gap reveal nothing new
    assert v is not None and nd.beacon_reveal(2) == v and nd.beacon_reveal(3) is None
    assert v1 is not None and PR.H(b'beacon-chain' + v) == v1
    for t in (1, 3, sysm.max_rounds):
        log.append(dict(kind='beacon', t=t, reveals={}))          # no reveals: ignored by every replica
    assert sysm.view.round is None and sysm.view.last_t == 0
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 2)
    assert rep['aborted'] is None and audit_ok(sysm, log, st)


@scenario('a true result copied after the next beacon is not a contradiction')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    res = copy.deepcopy([e for e in log.entries if e['kind'] == 'result'][0])
    st, rep = PR.run_round(sysm, sites, nodes, log, st, 2)
    log.append(res)
    assert audit_ok(sysm, log, st)
    bad = copy.deepcopy(res); bad['agg'][0] += 1
    log.append(bad)
    try:
        PR.audit(sysm.gen, log.entries); raise SystemExit('a false result was accepted')
    except AssertionError:
        pass


@scenario('the head chains wire bytes: unencodable entries refused, list and tuple distinct, copies agree')
def _():
    sysm, sites, nodes, log, _ = build()
    assert log.append(dict(kind='junk', x=np.array([object()], dtype=object))) is False
    assert log.append(dict(kind='junk', x=np.bool_(True))) is False
    assert PR._hb([1, 2]) != PR._hb((1, 2)) and W.encode([1, 2]) != W.encode((1, 2))
    log.append(dict(kind='junk', x=2 ** 200, y=[b'a', (b'b',)]))
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    st2, _h = PR.audit(sysm.gen, copy.deepcopy(log.entries))
    assert M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, st)
    v = PR.Replica(PR.System.from_genesis(sysm.gen))
    for e in log.entries:
        v.consume(copy.deepcopy(e))
    assert v.head == log.head == sysm.view.head


@scenario('a liar missed by the random combination never gets honest nodes blamed or a wrong sum accepted')
def _():
    class Reg:
        pass
    reg = Reg(); reg.cfg = M.Config(arch=(64, 12, 10), b=8); reg.Tn = 1; reg.shr = Sharing(reg.cfg.d, 2, 1, 7)
    sh = reg.shr
    agg = np.random.default_rng(1).integers(-5000, 5000, reg.cfg.d)
    vals = np.zeros((sh.nblk, sh.w), dtype=np.uint64)
    flat = np.zeros(sh.nblk * sh.kp, dtype=np.int64); flat[:reg.cfg.d] = agg
    vals[:, :sh.kp] = F.from_int(flat.reshape(sh.nblk, sh.kp)); vals[:, sh.kp:] = F.rand_field((sh.nblk, sh.T))
    shares = sh.share_values(vals)
    sums = {m: shares[m].copy() for m in range(sh.K)}
    j = 5
    sums[0] = sums[0].copy(); sums[0][j] = (int(sums[0][j]) + 12345) % F.P
    real = F.rand_field
    for always in (True, False):                                 # the ground event on every try, or on the first only
        calls = [0]
        def ground(n, drbg):
            g = real(n, drbg).copy(); calls[0] += 1
            if always or calls[0] == 1:
                g[j] = 0
            return g
        F.rand_field = ground
        try:
            out = PR.decode_aggregate(reg, 1, sums)
        finally:
            F.rand_field = real
        if always:
            assert out is None                                     # abort, nobody blamed
        else:
            assert out is not None and out[1] == [0] and np.array_equal(out[0], agg)


@scenario('honest rounds after all fixes: no abort, strike or burn; every replica agrees with the audit')
def _():
    sysm, sites, nodes, log, _ = build()
    st = M.init_model(sysm.cfg)
    for t in (1, 2, 4):
        st, rep = PR.run_round(sysm, sites, nodes, log, st, t)
        assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4]
    assert not sysm.node_burned and not any(sysm.site_strikes.values())
    assert all(p.view.head == log.head and p.view.history == sysm.view.history for p in sites + nodes)
    assert audit_ok(sysm, log, st)


print('%d public-surface protocol scenarios passed' % len(results))
