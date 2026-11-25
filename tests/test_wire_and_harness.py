"""Strict wire decoding, ill-typed requests, beacon gaps, early results and the networked harness under malformed
or spoofed frames."""
import sys, time, copy, socket, threading, struct
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')
import numpy as np
from fedproof import model as M, protocol as PR, wire as W
from test_protocol import build

results = []
def scenario(name):
    def deco(fn):
        t0 = time.time(); fn(); results.append(name); print('  ok  %-80s %.1fs' % (name, time.time() - t0))
    return deco

def audit_ok(sysm, log, state):
    st2, out = PR.audit(sysm.gen, log.entries)
    return M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, state)

def rejects(b):
    try:
        W.decode(b); return False
    except Exception:
        return True


@scenario('only canonical bytes decode: padded integers, duplicate and colliding keys are refused')
def _():
    good = W.encode(dict(kind='junk', x=5))
    assert W.decode(good) == dict(kind='junk', x=5)
    assert rejects(b'I' + struct.pack('<H', 2) + (5).to_bytes(2, 'little'))       # 5 in two bytes
    assert rejects(b'I' + struct.pack('<H', 0))                                     # empty integer
    dup = b'M' + struct.pack('<I', 2) + W.encode('a') + W.encode(1) + W.encode('a') + W.encode(2)
    assert rejects(dup)
    col = b'M' + struct.pack('<I', 2) + W.encode(1) + W.encode('x') + W.encode(True) + W.encode('y')
    assert rejects(col)
    assert rejects(b'L' + struct.pack('<I', 1 << 23) + b'N' * (1 << 23))           # element budget
    for v in (None, True, -7, 2 ** 200, b'ab', 'z', [1, (2, b'x')], {3: np.arange(6, dtype=np.uint64).reshape(2, 3)}):
        b = W.encode(v)
        assert W.encode(W.decode(b)) == b


@scenario('ill-typed round numbers and indices are refused before any one-shot state is spent')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1, stop_before='certificate')
    for bad in (True, 1.0, '1', None):
        assert sites[0].round_message(bad) is None and sites[0].reveal(bad, 0) is None and sites[0].noise_only(bad) is None
        for nd in nodes:
            assert nd.sign_valid_set(bad, rep['valid']) is None and nd.release_sum(bad) is None
            assert nd.round_key(bad) is None and nd.complaints(bad) is None and nd.key_dispute(bad) is None
    assert all(nd.sign_valid_set(1, tuple(rep['valid'])) is None for nd in nodes)     # a tuple is not the wire list
    assert all(nd.sign_valid_set(1, [float(i) for i in rep['valid']]) is None for nd in nodes)
    st, rep = PR.continue_round(sysm, sites, nodes, log, 1)
    assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4] and audit_ok(sysm, log, st)


@scenario('asking each node for a different round in one gap cannot wedge the beacon')
def _():
    sysm, sites, nodes, log, _ = build()
    for nd in nodes:
        nd.beacon_reveal(1 + nd.idx % PR.MAX_STEP)
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    assert rep['aborted'] is None and rep['valid'] == [0, 1, 2, 3, 4]
    nd = nodes[0]
    v3 = nd.beacon_reveal(3)
    assert v3 is not None and nd.beacon_reveal(2) is not None and nd.beacon_reveal(4) is None   # never further ahead
    assert audit_ok(sysm, log, st)


@scenario('a true result logged before the sums is checked at close, not reported as tampering')
def _():
    sysm, sites, nodes, log, _ = build()
    st, rep = PR.run_round(sysm, sites, nodes, log, M.init_model(sysm.cfg), 1)
    ent = copy.deepcopy(log.entries)
    i = next(k for k, e in enumerate(ent) if e['kind'] == 'sums')
    res = next(e for e in ent if e['kind'] == 'result')
    moved = ent[:i] + [copy.deepcopy(res)] + ent[i:]
    st2, hist = PR.audit(sysm.gen, moved)
    assert M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, st)
    bad = copy.deepcopy(res); bad['agg'][0] += 1
    try:
        PR.audit(sysm.gen, ent[:i] + [bad] + ent[i:]); raise SystemExit('a false early result was accepted')
    except AssertionError:
        pass


@scenario('harness: a party survives malformed frames, and spoofed or unknown responses are ignored')
def _():
    from experiments import harness as NH
    a, b = socket.socketpair()
    net = NH.Net.__new__(NH.Net)
    net.sock, net.n, net.next_id, net.waiting, net.lock, net.target, net.trace = a, 0, 0, {}, threading.Lock(), {-1: 'router'}, []
    threading.Thread(target=net._read, daemon=True).start()
    ev, box = threading.Event(), []
    net.waiting[0] = (ev, box); net.target[0] = 'n0'
    def send(sender, body):
        nb = sender.encode()
        b.sendall(NH.frame(NH.R, bytes([len(nb)]) + nb + body))
    send('s3', W.encode(dict(id=0, r='spoof')))            # a corrupted site answers for n0
    send('n0', b'\xff garbage')                             # undecodable
    send('n0', W.encode(dict(id=99, r='unknown')))          # no such request
    send('n0', W.encode(dict(id=0, r='genuine')))
    assert ev.wait(5) and box[0][0]['r'] == 'genuine'
    a.close(); b.close()


print('%d wire and harness scenarios passed' % len(results))
