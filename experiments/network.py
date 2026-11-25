"""E6 traffic and round time on industrial links, E7 scaling, with the networked harness (experiments/harness.py).

Method, in three layers, each checked against the one below:
  1. live runs: every plant and node in its own process over TCP with the real protocol and real proofs;
  2. replay runs: the same message sizes and measured costs over real sockets, parties standing for dedicated machines,
     at any number of plants (scale_trace);
  3. the virtual-time emulator over the same traces, for the full sweep.
Links are shaped in user space by default; with --netns (root, experiments/netns_links.sh) the kernel shapes them,
with loss and jitter as well."""
import os
import sys
import json
import itertools
sys.path.insert(0, '.')
import numpy as np
from fedproof import protocol as PR
from experiments import common as C
from experiments.system import params
from experiments import harness as NH

# link profiles (per plant and per node; one-way delay). Values are assumptions to be justified in the paper's text.
PROFILES = {
    'industrial-ethernet': dict(down=100.0, up=100.0, delay_ms=2.0),
    'private-5g': dict(down=100.0, up=50.0, delay_ms=10.0),
    'constrained-wan': dict(down=10.0, up=2.0, delay_ms=60.0),
}
ROUTER = dict(egress=1000.0, ingress=1000.0)
SWEEP_BW = (2, 5, 10, 20, 50, 100)
SWEEP_DELAY = (1, 5, 10, 25, 50, 100)


class Args:
    def __init__(self, **kw):
        self.__dict__.update(dict(N=10, K=7, kp=2, Tn=1, A_min=7, rounds=2, down=None, up=None, delay_ms=0.0,
                                  egress=None, ingress=None, netns=False), **kw)


def system_for(ds, level='P128', rounds=2, K=7, kp=2, Tn=1, A_min=7, drop_site=None, drop_node=None):
    d = C.load(ds, 'iid', 0)
    cfg = C.make_cfg(d['meta'])
    ch = [c[:rounds + 1] for c in C.chunks(d['plants'], cfg.b)]
    reg, sites, nodes, log = PR.setup(cfg, params(level), ch, K, kp, Tn, A_min, max_rounds=rounds + 2)
    if drop_site is not None:
        sites[drop_site].offline = True
    if drop_node is not None:
        nodes[drop_node].offline = True
    return reg, sites, nodes, log


def _traffic(res):
    by = dict(proofs=0, ciphertexts=0, other=0)
    for ev in res['trace']:
        if ev[0] == 'append':
            by['proofs' if ev[2] == 'site' else 'other'] += ev[1]
    return by


def e6(quick=False, netns=False):
    st_ = C.Store('E6_network')
    rounds = 1 if quick else 2
    traces = {}
    for ds in C.DATASETS if not quick else C.DATASETS[:1]:
        if not C.available(ds, 'iid', 0):
            continue
        # unshaped live run: the trace every emulated setting replays
        key = '%s|live|unshaped' % ds
        tpath = os.path.join(C.RESULTS, 'trace_%s.json' % ds)
        if not st_.done(key) or not os.path.exists(tpath):
            res = NH.live_run(Args(rounds=rounds, **ROUTER), system_for(ds, rounds=rounds))
            json.dump(res, open(tpath, 'w'), default=str)
            st_.add(key, dict(dataset=ds, mode='live', profile='unshaped', round_s=[r['wall_s'] for r in res['rounds']],
                              audit_same=res['audit_same'], bytes_by_kind=_traffic(res),
                              bytes_down_per_party=_mean_bytes(res['bytes_down']), bytes_up_per_party=_mean_bytes(res['bytes_up'])))
        traces[ds] = _load_trace(tpath)
        if ds != C.DATASETS[0]:
            continue
        # live runs on the three profiles, with and without dropouts (one plant and one node offline)
        losses = (0.0, 1.0) if netns else (0.0,)
        for (pname, prof), drop, loss in itertools.product(PROFILES.items(), (False, True), losses):
            key = '%s|live|%s|drop=%s|loss=%g|%s' % (ds, pname, drop, loss, 'kernel' if netns else 'user')
            if st_.done(key) or quick and drop:
                continue
            if netns:                                  # kernel shaping for this profile (the harness runs as root)
                import subprocess
                subprocess.run(['bash', 'experiments/netns_links.sh', 'up', '17', str(prof['down']), str(prof['up']),
                                str(prof['delay_ms']), str(loss)], check=True)
            res = NH.live_run(Args(rounds=rounds, netns=netns, **prof, **ROUTER),
                              system_for(ds, rounds=rounds, drop_site=9 if drop else None, drop_node=6 if drop else None))
            em = NH.emulate(res, prof['down'], prof['up'], prof['delay_ms'] / 1e3, ROUTER['egress'], cores=os.cpu_count(), ingress=ROUTER['ingress'])
            em1 = NH.emulate(res, prof['down'], prof['up'], prof['delay_ms'] / 1e3, ROUTER['egress'], cores=None, ingress=ROUTER['ingress'])
            live = [r['wall_s'] for r in res['rounds']]
            st_.add(key, dict(dataset=ds, mode='live', profile=pname, dropout=drop, loss_pct=loss, shaping='kernel' if netns else 'user',
                              round_s=live, emulated_same_cores=em, emulated_one_machine_per_party=em1,
                              error_pct=[100 * (e - l) / l for e, l in zip(em, live)], audit_same=res['audit_same'],
                              valid=[r['valid'] for r in res['rounds']], bytes_by_kind=_traffic(res)))
            print('E6 %-55s live %s emulated %s' % (key, ['%.2f' % x for x in live], ['%.2f' % x for x in em]), flush=True)
    # emulated sweep: every trace under every bandwidth and delay, one machine per party
    for ds, tr in traces.items():
        for bw, dl in itertools.product(SWEEP_BW, SWEEP_DELAY):
            key = '%s|emulated|bw%g|delay%g' % (ds, bw, dl)
            if st_.done(key):
                continue
            em = NH.emulate(tr, bw, bw / 2.0, dl / 1e3, ROUTER['egress'], cores=None, ingress=ROUTER['ingress'])
            st_.add(key, dict(dataset=ds, mode='emulated', down=bw, up=bw / 2.0, delay_ms=dl, round_s=em))


def e7(quick=False):
    """Scaling: the recorded trace scaled to 10, 25, 50 and 100 plants; replayed live over sockets at 25 and 50 plants to
    check the emulator where parties stand for dedicated machines, emulated up to 100."""
    st_ = C.Store('E7_scaling')
    ds = C.DATASETS[0]
    tpath = os.path.join(C.RESULTS, 'trace_%s.json' % ds)
    if not os.path.exists(tpath):
        print('E7 needs the E6 trace first'); return
    tr = _load_trace(tpath)
    prof = PROFILES['private-5g']
    for n in ((10, 25) if quick else (10, 25, 50, 100)):
        sc = NH.scale_trace(tr, n)
        em = NH.emulate(sc, prof['down'], prof['up'], prof['delay_ms'] / 1e3, ROUTER['egress'], cores=None, ingress=ROUTER['ingress'])
        node_verify = sum(v for k, v in sc['consume']['n0'].items())
        row = dict(dataset=ds, plants=n, profile='private-5g', emulated_round_s=em, node_read_cpu_s_total=node_verify)
        if n in (25, 50) or (quick and n == 25):
            key = '%s|%d|replay' % (ds, n)
            if not st_.done(key):
                live = NH.replay_run(sc, prof['down'], prof['up'], prof['delay_ms'], ROUTER['egress'], ROUTER['ingress'])
                row.update(replay_round_s=live, error_pct=[100 * (e - l) / l for e, l in zip(em, live)])
                st_.add(key, row)
                print('E7 %d plants: replay %s emulated %s' % (n, ['%.2f' % x for x in live], ['%.2f' % x for x in em]), flush=True)
        key = '%s|%d|emulated' % (ds, n)
        if not st_.done(key):
            st_.add(key, row)


def _mean_bytes(d):
    vals = [sum(v.values()) for v in d.values()] if d else [0]
    return float(np.mean(vals))


def _load_trace(path):
    tr = json.load(open(path))
    tr['calls_cpu'] = {int(k): v for k, v in tr['calls_cpu'].items()}
    tr['consume'] = {nm: {int(k): v for k, v in d.items()} for nm, d in tr['consume'].items()}
    tr['trace'] = [tuple(ev) if ev[0] != 'gather' else ('gather', ev[1]) for ev in tr['trace']]
    return tr
