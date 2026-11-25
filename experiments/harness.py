"""Network harness for the protocol: a live run with every party in its own process over TCP, with shaped links, and a
virtual-time emulator that replays the recorded run under any network and machine model.

Live run. A router process plays the bulletin host and the relay's network: it keeps the log, streams every entry to
every party and forwards the driver's requests. Each site and node runs in its own process with its own replica and
answers requests in order with the entries it has read (one connection per party, so entries and requests share the
party's links, in order). Links are shaped in user space: each direction is a FIFO with a bandwidth and a one-way
delay (store and forward); the router also has a shared egress bandwidth. Every byte counted is a framed wire byte
(fedproof/wire.py). Each party records the CPU time of every entry it reads and every request it answers (thread CPU
time, so contention on the test machine does not inflate it).

Emulator. The recorded trace (entry sizes, request and response sizes, CPU times) is replayed as a dependency graph
in virtual time: links as FIFO queues, each party serial, all parties sharing C cores by processor sharing (C = the
test machine's cores to check the emulator against the live run; C = one per party to predict a deployment).

Usage: python3 experiments/harness.py [--rounds 3] [--down 50] [--up 20] [--delay-ms 20] [--egress 1000] [--out runs/net_live.json]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, socket, struct, threading, time, json, argparse, heapq, queue, collections
import multiprocessing as mp
sys.path.insert(0, '.')
from fedproof import wire as W, protocol as PR, model as M

E, Q, R, X = b'E', b'Q', b'R', b'X'          # frame types: entry, request, response, stop
ROUTER_BIND = os.environ.get('FEDPROOF_ROUTER_BIND', '127.0.0.1')   # 0.0.0.0 with kernel-shaped namespaces
REQUESTS = dict(Site={'round_message', 'reveal', 'noise_only'},
                Node={'beacon_reveal', 'round_key', 'key_dispute', 'complaints', 'sign_valid_set', 'release_sum'})


def frame(kind, payload):
    return kind + struct.pack('<I', len(payload)) + payload


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(min(1 << 20, n - len(buf)))
        if not b:
            raise EOFError
        buf += b
    return bytes(buf)


MAX_FRAME = 128 << 20


def recv_frame(sock):
    """One frame. A frame longer than MAX_FRAME is read and discarded; its payload comes back as None."""
    h = recv_exact(sock, 5)
    n, = struct.unpack('<I', h[1:])
    if n > MAX_FRAME:
        while n:
            n -= len(recv_exact(sock, min(n, 1 << 20)))
        return h[:1], None
    return h[:1], recv_exact(sock, n)


class Pipe:
    """A shaped direction: FIFO, bandwidth in Mbit/s (None = unshaped), one-way delay in seconds. A shared upstream
    pipe (the router's egress) can be chained in front: store and forward through both."""

    def __init__(self, sock, mbps=None, delay=0.0, upstream=None):
        self.sock, self.bps, self.delay, self.up = sock, (mbps * 1e6 if mbps else None), delay, upstream
        self.free = 0.0
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.bytes = collections.Counter()
        threading.Thread(target=self._run, daemon=True).start()

    def reserve(self, nbytes, now):
        with self.lock:
            start = max(now, self.free)
            self.free = start + (nbytes * 8 / self.bps if self.bps else 0.0)
            return self.free

    def send(self, kind, payload):
        b = frame(kind, payload)
        now = time.monotonic()
        t = self.up.reserve(len(b), now) if self.up is not None else now
        done = self.reserve(len(b), t)
        self.bytes[kind.decode()] += len(b)
        self.q.put((done + self.delay, b))

    def _run(self):
        while True:
            when, b = self.q.get()
            dt = when - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            try:
                self.sock.sendall(b)
            except OSError:
                return


class Egress:
    def __init__(self, mbps):
        self.bps, self.free, self.lock = (mbps * 1e6 if mbps else None), 0.0, threading.Lock()

    def reserve(self, nbytes, now):
        with self.lock:
            start = max(now, self.free)
            self.free = start + (nbytes * 8 / self.bps if self.bps else 0.0)
            return self.free


# ---------------------------------------------------------------- party process
def enter_netns(ns):
    """Join a network namespace made by experiments/netns_links.sh (kernel shaping with tc; needs root)."""
    fd = os.open('/var/run/netns/' + ns, os.O_RDONLY)
    os.setns(fd, os.CLONE_NEWNET)
    os.close(fd)


def party_main(party, name, n0, port, up_mbps, delay, netns=None, host='127.0.0.1'):
    if netns:
        enter_netns(netns)
    sock = socket.create_connection((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.sendall(frame(b'H', W.encode(name)))
    up = Pipe(sock, up_mbps, delay)
    view = party.view
    consumed = n0
    rec = dict(consume=[], calls=[])
    while True:
        try:
            kind, payload = recv_frame(sock)
        except EOFError:
            return
        if payload is None:
            continue                                      # oversized frame: dropped
        if kind == E:
            c0 = time.thread_time()
            try:
                e = W.decode(payload)                     # strict: only canonical bytes decode
            except Exception:
                consumed += 1                             # not an entry anyone can read: every reader skips it alike
                continue
            view.consume(e, payload)
            rec['consume'].append((consumed, time.thread_time() - c0))
            consumed += 1
        elif kind == Q:
            try:
                req = W.decode(payload)
                if not (isinstance(req, dict) and type(req.get('id')) is int and type(req.get('after')) is int):
                    continue
            except Exception:
                continue                                  # a malformed request is dropped; the party carries on
            if req.get('m') == '_stats':
                up.send(R, W.encode(dict(id=req['id'], r=rec, bytes_up=dict(up.bytes))))
                continue
            if req['after'] > consumed:                   # asks about entries this party has not seen: refuse
                up.send(R, W.encode(dict(id=req['id'], r=None)))
                continue
            c0 = time.thread_time()
            m = req.get('m')
            ok = m in REQUESTS[type(party).__name__] and isinstance(req.get('a'), list)
            res = PR._call(getattr(party, m), *req['a']) if ok else None        # only the protocol's requests
            dt = time.thread_time() - c0
            try:
                out = W.encode(dict(id=req['id'], r=res))
            except Exception:
                out = W.encode(dict(id=req['id'], r=None))
            rec['calls'].append((req['id'], dt, len(out) + 5))
            up.send(R, out)
        elif kind == X:
            return


# ---------------------------------------------------------------- router process
def router_main(port_q, names, down_mbps, delay, egress_mbps, ready, ingress_mbps=None):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((ROUTER_BIND, 0)); srv.listen(len(names) + 4)
    port_q.put(srv.getsockname()[1])
    eg = Egress(egress_mbps)
    pipes, socks = {}, {}
    drv, _ = srv.accept()
    drv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    to_drv = Pipe(drv, ingress_mbps, 0.0)        # the relay host's shared ingress: every response queues here
    for _ in names:
        c, _a = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        k, p = recv_frame(c)
        nm = W.decode(p)
        socks[nm] = c
        pipes[nm] = Pipe(c, down_mbps, delay, upstream=eg)

    def pump(nm, c):                       # responses from a party go to the driver (router and driver share a host)
        while True:
            try:
                k, p = recv_frame(c)
            except (EOFError, OSError):
                return
            if p is not None:
                nb = nm.encode()
                to_drv.send(k, bytes([len(nb)]) + nb + p)       # the sender's name, set by the router, not the party
    for nm, c in socks.items():
        threading.Thread(target=pump, args=(nm, c), daemon=True).start()
    ready.set()
    while True:
        k, p = recv_frame(drv)
        if k == E:
            for nm in names:
                pipes[nm].send(E, p)
        elif k == Q:
            nm_len = p[0]
            nm, body = p[1:1 + nm_len].decode(), p[1 + nm_len:]
            pipes[nm].send(Q, body)
        elif k == X:
            out = {nm: dict(pipes[nm].bytes) for nm in names}
            to_drv.send(R, b'\x06router' + W.encode(dict(id=-1, r=out)))
            for nm in names:
                pipes[nm].send(X, b'')
            time.sleep(0.5)
            return


# ---------------------------------------------------------------- driver side
class Net:
    def __init__(self, port, n0):
        self.sock = socket.create_connection(('127.0.0.1', port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.n, self.next_id = n0, 0
        self.waiting, self.lock = {}, threading.Lock()
        self.target = {-1: 'router'}
        self.trace = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        """A response counts only if it comes from the connection the request went to and answers a pending id."""
        while True:
            try:
                k, p = recv_frame(self.sock)
            except (EOFError, OSError):
                return
            try:
                ln = p[0]; sender = p[1:1 + ln].decode(); body = p[1 + ln:]
                msg = W.decode(body)
                rid = msg['id']
                w = self.waiting.get(rid)
                if w is None or self.target.get(rid) != sender:
                    continue                                  # spoofed, late or unknown: ignored
                self.waiting.pop(rid); self.target.pop(rid, None)
                ev, box = w
                box.append((msg, len(body) + 5, time.monotonic()))
                ev.set()
            except Exception:
                continue

    def call(self, name, method, args):
        with self.lock:
            rid = self.next_id; self.next_id += 1
        body = W.encode(dict(id=rid, m=method, a=list(args), after=self.n))
        ev, box = threading.Event(), []
        self.target[rid] = name
        self.waiting[rid] = (ev, box)
        nb = name.encode()
        t0 = time.monotonic()
        self.sock.sendall(frame(Q, bytes([len(nb)]) + nb + body))
        ev.wait()
        msg, rsize, t1 = box[0]
        return rid, msg, len(body) + 5, rsize, t0, t1

    def append_raw(self, payload):
        self.sock.sendall(frame(E, payload))
        self.n += 1


class LiveLog:
    """Driver end of the bulletin: sends each entry to the router and feeds the driver's own replica (same host)."""

    def __init__(self, net, view):
        self.net, self.view = net, view
        self.entries, self.consume = [], []

    def append(self, e):
        raw = W.encode(e)
        t0 = time.monotonic()
        self.net.append_raw(raw)
        e2 = W.decode(raw)                         # every reader, the driver included, sees the decoded wire form
        c0 = time.thread_time()
        self.view.consume(e2, raw)
        self.consume.append(time.thread_time() - c0)
        self.entries.append(e2)
        self.net.trace.append(('append', len(raw) + 5, e2.get('kind'), self.consume[-1], t0))


def build_system(N, K, kp, Tn, A_min, seed=0, chunks_per_site=2, max_rounds=16, cfg=None, prm=None):
    sys.path.insert(0, 'tests')
    from test_protocol import build
    return build(N=N, K=K, kp=kp, Tn=Tn, A_min=A_min, seed=seed, chunks_per_site=chunks_per_site, max_rounds=max_rounds,
                 cfg=cfg, prm=prm)


def live_run(args, system=None):
    """system: (registry, sites, nodes, bulletin) from protocol.setup, with any deviation flags already set (they travel
    into the party processes); default: the small test system. With args.netns each party runs in namespace fpp<i>
    and reaches the router at 10.201.<i>.1, links shaped by the kernel (user-space shaping then off)."""
    if system is None:
        sysm, sites, nodes, log, _ = build_system(args.N, args.K, args.kp, args.Tn, args.A_min, max_rounds=args.rounds + 2)
    else:
        sysm, sites, nodes, log = system
    n0 = len(log.entries)
    names = ['s%d' % s.idx for s in sites] + ['n%d' % nd.idx for nd in nodes]
    ctx = mp.get_context('fork')
    port_q, ready = ctx.Queue(), ctx.Event()
    kern = getattr(args, 'netns', False)
    rt = ctx.Process(target=router_main, args=(port_q, names, None if kern else args.down, 0.0 if kern else args.delay_ms / 1e3,
                                               args.egress, ready, args.ingress))
    rt.start()
    port = port_q.get()
    net = Net(port, n0)
    procs = []
    for j, (p, nm) in enumerate(zip(list(sites) + list(nodes), names)):
        ns = ('fpp%d' % j) if getattr(args, 'netns', False) else None
        host = ('10.201.%d.1' % j) if ns else '127.0.0.1'
        pr = ctx.Process(target=party_main, args=(p, nm, n0, port, None if ns else args.up, 0.0 if ns else args.delay_ms / 1e3, ns, host))
        pr.start(); procs.append(pr)
    ready.wait()
    view = sysm.view
    live = LiveLog(net, view)
    class Remote:                              # the driver's handle on a party: every method is a request
        pass

    def bind(name, idx):
        b = Remote(); b.idx = idx; b.name = name
        for m in ('beacon_reveal', 'round_key', 'key_dispute', 'round_message', 'complaints', 'reveal', 'noise_only',
                  'sign_valid_set', 'release_sum'):
            def mk(m=m):
                def f(*a):
                    return net.call(name, m, a)
                f.__name__ = m
                f.party = name
                return f
            setattr(b, m, mk())
        return b
    bs = [bind('s%d' % s.idx, s.idx) for s in sites]
    bn = [bind('n%d' % nd.idx, nd.idx) for nd in nodes]
    pool = __import__('concurrent.futures').futures.ThreadPoolExecutor(max_workers=64)

    def gather(calls):
        after = net.n
        futs = [pool.submit(fn, *a) for fn, a in calls]
        outs, recs = [], []
        for (fn, a), fu in zip(calls, futs):
            rid, msg, qs, rs, t0, t1 = fu.result()
            outs.append(msg['r'])
            recs.append(dict(id=rid, party=fn.party, m=fn.__name__, after=after, q=qs, r=rs, t0=t0, t1=t1))
        net.trace.append(('gather', recs))
        return outs

    st = M.init_model(sysm.cfg)
    rounds = []
    for t in range(1, args.rounds + 1):
        net.trace.append(('round', t))
        t0 = time.monotonic()
        st, rep = PR.run_round(sysm, bs, bn, live, st, t, gather=gather)
        rounds.append(dict(t=t, wall_s=time.monotonic() - t0, aborted=rep['aborted'], valid=rep['valid']))
        print('round %d: %.2fs aborted=%s valid=%s' % (t, rounds[-1]['wall_s'], rep['aborted'], rep['valid']), flush=True)
    stats = {}
    for nm in names:
        rid, msg, *_ = net.call(nm, '_stats', ())
        stats[nm] = msg
    ev, box = threading.Event(), []
    net.waiting[-1] = (ev, box)
    net.sock.sendall(frame(X, b''))
    ev.wait(30)
    down_bytes = box[0][0]['r'] if box else {}
    for pr in procs:
        pr.join(10)
    rt.join(10)
    gen = sysm.gen
    st2, hist = PR.audit(gen, log.entries[:n0] + live.entries)      # the registrations, then the live log
    audit_same = M.model_digest(sysm.cfg, st2) == M.model_digest(sysm.cfg, st)
    calls_cpu = {}
    for nm in names:
        for rid, dt, sz in stats[nm]['r']['calls']:
            calls_cpu[rid] = dt
    consume = {nm: dict(stats[nm]['r']['consume']) for nm in names}
    return dict(config=vars(args), rounds=rounds, audit_same=audit_same, trace=net.trace, calls_cpu=calls_cpu,
                consume=consume, driver_consume=live.consume, n0=n0, names=names,
                bytes_down={nm: down_bytes.get(nm, {}) for nm in names},
                bytes_up={nm: stats[nm]['bytes_up'] for nm in names})


# ---------------------------------------------------------------- replay: real sockets, recorded sizes and costs
def replay_party(name, consume, port, up_mbps, delay):
    """A party that answers like the recorded one: it 'reads' each entry for its recorded CPU time and answers each
    request after the recorded CPU time with a payload of the recorded size. Sleeping stands for a dedicated machine."""
    sock = socket.create_connection(('127.0.0.1', port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.sendall(frame(b'H', W.encode(name)))
    up = Pipe(sock, up_mbps, delay)
    while True:
        try:
            kind, payload = recv_frame(sock)
        except EOFError:
            return
        if kind == E:
            idx = struct.unpack('<I', payload[:4])[0]
            time.sleep(consume.get(idx, 0.0))
        elif kind == Q:
            rid, cpu, rsize = struct.unpack('<IdI', payload[:16])
            time.sleep(cpu)
            up.send(R, W.encode(dict(id=rid, r=None, pad=bytes(max(0, rsize - 40)))))
        elif kind == X:
            return


def replay_run(res, down, up, delay_ms, egress=None, ingress=None):
    """Run a recorded (or scaled) trace live over sockets with shaped links. Returns per-round wall times."""
    names = res['names']; n0 = res['n0']
    ctx = mp.get_context('fork')
    port_q, ready = ctx.Queue(), ctx.Event()
    rt = ctx.Process(target=router_main, args=(port_q, names, down, delay_ms / 1e3, egress, ready, ingress))
    rt.start(); port = port_q.get()
    net = Net(port, n0)
    procs = [ctx.Process(target=replay_party, args=(nm, res['consume'].get(nm, {}), port, up, delay_ms / 1e3)) for nm in names]
    for p in procs:
        p.start()
    ready.wait()
    walls, t0, idx, k = [], None, n0, 0
    for ev in res['trace']:
        if ev[0] == 'round':
            now = time.monotonic()
            if t0 is not None:
                walls.append(now - t0)
            t0 = now
        elif ev[0] == 'append':
            size, dcpu = ev[1], ev[3]
            net.sock.sendall(frame(E, struct.pack('<I', idx) + bytes(max(0, size - 9))))
            net.n += 1; idx += 1
            time.sleep(dcpu)
        else:
            import concurrent.futures as cf
            def one(r):
                ev2, box = threading.Event(), []
                with net.lock:
                    rid = net.next_id; net.next_id += 1
                net.target[rid] = r['party']
                net.waiting[rid] = (ev2, box)
                nb = r['party'].encode()
                body = struct.pack('<IdI', rid, res['calls_cpu'].get(r['id'], 0.0), r['r']) + bytes(max(0, r['q'] - 21))
                net.sock.sendall(frame(Q, bytes([len(nb)]) + nb + body))
                ev2.wait()
            with cf.ThreadPoolExecutor(max_workers=max(1, len(ev[1]))) as pool:
                list(pool.map(one, ev[1]))
    walls.append(time.monotonic() - t0)
    ev2, box = threading.Event(), []
    net.waiting[-1] = (ev2, box)
    net.sock.sendall(frame(X, b''))
    ev2.wait(30)
    for p in procs:
        p.join(10)
    rt.join(10)
    return walls


def scale_trace(res, n_sites):
    """The recorded trace with n_sites plants instead of the recorded number: every plant request and plant entry of a
    round is repeated for the extra plants (sizes and costs copied from recorded plants in turn), and every party,
    the new plants included, reads every entry. Node and site costs per entry are copied from the recorded ones."""
    sites0 = [nm for nm in res['names'] if nm.startswith('s')]
    nodes0 = [nm for nm in res['names'] if nm.startswith('n')]
    new_sites = ['s%d' % i for i in range(n_sites)]
    src = {nm: sites0[i % len(sites0)] for i, nm in enumerate(new_sites)}
    src.update({nm: nm for nm in nodes0})
    names = new_sites + nodes0
    out = dict(names=names, n0=res['n0'], trace=[], calls_cpu={}, consume={nm: {} for nm in names})
    idx_old, idx_new, rid = res['n0'], res['n0'], 0
    pending_site = []
    def flush_sites():
        nonlocal idx_new
        if not pending_site:
            return
        recorded = list(pending_site)
        for j in range(n_sites):
            e_old, ev = recorded[j % len(recorded)]
            out['trace'].append(ev)
            for nm in names:
                out['consume'][nm][idx_new] = res['consume'].get(src[nm], {}).get(e_old, 0.0)
            idx_new += 1
        pending_site.clear()
    for ev in res['trace']:
        if ev[0] == 'append' and ev[2] in ('site', 'noise'):
            pending_site.append((idx_old, ev)); idx_old += 1
            continue
        flush_sites()
        if ev[0] == 'append':
            out['trace'].append(ev)
            for nm in names:
                out['consume'][nm][idx_new] = res['consume'].get(src[nm], {}).get(idx_old, 0.0)
            idx_old += 1; idx_new += 1
        elif ev[0] == 'gather':
            recs = ev[1]
            srecs = [r for r in recs if r['party'].startswith('s')]
            nrecs = [r for r in recs if not r['party'].startswith('s')]
            new = []
            if srecs:
                for j in range(n_sites):
                    r = dict(srecs[j % len(srecs)]); old = r['id']; r['party'] = new_sites[j]; r['id'] = rid
                    out['calls_cpu'][rid] = res['calls_cpu'].get(old, 0.0); rid += 1; new.append(r)
            for r in nrecs:
                r = dict(r); old = r['id']; r['id'] = rid
                out['calls_cpu'][rid] = res['calls_cpu'].get(old, 0.0); rid += 1; new.append(r)
            out['trace'].append(('gather', new))
        else:
            out['trace'].append(ev)
    flush_sites()
    return out


# ---------------------------------------------------------------- emulator
class DAG:
    def __init__(self):
        self.nodes, self.children = [], collections.defaultdict(list)

    def add(self, kind, deps=(), work=0.0, link=None, size=0, owner=None, tag=None):
        i = len(self.nodes)
        deps = [d for d in deps if d is not None]
        self.nodes.append(dict(kind=kind, deps=deps, work=work, link=link, size=size, owner=owner, tag=tag,
                               left=len(deps), rel=0.0, done=None))
        for d in deps:
            self.children[d].append(i)
        return i


def simulate(dag, links, cores=None):
    """links: name -> (Mbit/s or None, one-way delay s). cores: None = unlimited (each party on its own machine)."""
    nodes = dag.nodes
    heap, seq = [], 0
    now = 0.0
    lfree = collections.defaultdict(float)
    active = {}
    oq = collections.defaultdict(collections.deque)
    obusy = {}

    def push(t, typ, i):
        nonlocal seq
        heapq.heappush(heap, (t, seq, typ, i)); seq += 1

    def finish(i, t):
        nodes[i]['done'] = t
        o = nodes[i]['owner']
        if nodes[i]['kind'] == 'cpu' and o is not None:
            obusy[o] = False
            if oq[o]:
                start_cpu(oq[o].popleft(), t)
        for c in dag.children[i]:
            n = nodes[c]
            n['left'] -= 1
            n['rel'] = max(n['rel'], t)
            if n['left'] == 0:
                push(n['rel'], 'rel', c)

    def start_cpu(i, t):
        o = nodes[i]['owner']
        if o is not None:
            obusy[o] = True
        if nodes[i]['work'] <= 0:
            push(t, 'done', i)
        else:
            active[i] = nodes[i]['work']

    for i, n in enumerate(nodes):
        if n['left'] == 0:
            push(0.0, 'rel', i)
    while heap or active:
        rate = 1.0 if (cores is None or len(active) <= cores) else cores / len(active)
        tf = now + min(active.values()) / rate if active else float('inf')
        th = heap[0][0] if heap else float('inf')
        t = min(tf, th)
        for i in active:
            active[i] -= (t - now) * rate
        now = t
        if tf <= th:
            for i in [i for i, w in active.items() if w <= 1e-12]:
                del active[i]
                finish(i, now)
            continue
        _t, _s, typ, i = heapq.heappop(heap)
        n = nodes[i]
        if typ == 'done':
            finish(i, now)
        elif n['kind'] == 'join':
            finish(i, now)
        elif n['kind'] == 'link':
            bw, dl = links.get(n['link'], (None, 0.0))
            start = max(now, lfree[n['link']])
            lfree[n['link']] = start + (n['size'] * 8 / (bw * 1e6) if bw else 0.0)
            push(lfree[n['link']] + dl, 'done', i)
        elif n['kind'] == 'cpu':
            o = n['owner']
            if o is not None and obusy.get(o):
                oq[o].append(i)
            else:
                start_cpu(i, now)
    return nodes


def trace_to_dag(res):
    """The recorded live run as a dependency graph: what each step waited for, how big it was, how long it computed."""
    dag = DAG()
    names = res['names']
    last_consume = {nm: None for nm in names}
    consumed_node = {nm: {} for nm in names}
    drv = None
    idx = res['n0']
    k_app = 0
    round_start = {}
    for ev in res['trace']:
        if ev[0] == 'round':
            drv = dag.add('join', [drv], tag=('round', ev[1]))
            round_start[ev[1]] = drv
            continue
        if ev[0] == 'append':
            _a, size, kind, dcpu, _t0 = ev
            dc = dag.add('cpu', [drv], work=dcpu, owner='driver', tag=('consume', 'driver', idx))
            for nm in names:
                eg = dag.add('link', [drv], link='egress', size=size)
                dn = dag.add('link', [eg], link='down:' + nm, size=size)
                w = res['consume'][nm].get(idx, 0.0)
                c = dag.add('cpu', [dn, last_consume[nm]], work=w, owner=nm, tag=('consume', nm, idx))
                last_consume[nm] = c
                consumed_node[nm][idx + 1] = c
            drv = dc
            idx += 1
            k_app += 1
            continue
        recs = ev[1]
        ends = []
        for r in recs:
            nm = r['party']
            eg = dag.add('link', [drv], link='egress', size=r['q'])
            dn = dag.add('link', [eg], link='down:' + nm, size=r['q'])
            w = res['calls_cpu'].get(r['id'], 0.0)
            c = dag.add('cpu', [dn, last_consume[nm]], work=w, owner=nm, tag=('call', nm, r['m']))
            last_consume[nm] = c
            up = dag.add('link', [c], link='up:' + nm, size=r['r'])
            ends.append(dag.add('link', [up], link='ingress', size=r['r']))
        drv = dag.add('join', [drv] + ends)
    end = dag.add('join', [drv], tag=('end',))
    return dag, round_start, end


def emulate(res, down, up, delay, egress, cores=None, ingress=None):
    names = res['names']
    links = {'egress': (egress, 0.0), 'ingress': (ingress, 0.0)}
    for nm in names:
        links['down:' + nm] = (down, delay)
        links['up:' + nm] = (up, delay)
    dag, round_start, end = trace_to_dag(res)
    nodes = simulate(dag, links, cores)
    ts = sorted(round_start)
    starts = [nodes[round_start[t]]['done'] for t in ts] + [nodes[end]['done']]
    return [starts[i + 1] - starts[i] for i in range(len(ts))]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=5); ap.add_argument('--K', type=int, default=7)
    ap.add_argument('--kp', type=int, default=2); ap.add_argument('--Tn', type=int, default=1)
    ap.add_argument('--A_min', type=int, default=3); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--down', type=float, default=None, help='party downlink Mbit/s (default unshaped)')
    ap.add_argument('--up', type=float, default=None, help='party uplink Mbit/s')
    ap.add_argument('--delay-ms', type=float, default=0.0, dest='delay_ms')
    ap.add_argument('--egress', type=float, default=None, help='router egress Mbit/s')
    ap.add_argument('--ingress', type=float, default=None, help='router ingress Mbit/s (all responses share it)')
    ap.add_argument('--out', default='runs/net_live.json')
    a = ap.parse_args()
    t0 = time.time()
    res = live_run(a)
    cores = os.cpu_count()
    em = emulate(res, a.down, a.up, a.delay_ms / 1e3, a.egress, cores=cores, ingress=a.ingress)
    em_inf = emulate(res, a.down, a.up, a.delay_ms / 1e3, a.egress, cores=None, ingress=a.ingress)
    out = dict(config=vars(a), machine=dict(cores=cores), audit_same=res['audit_same'],
               live_round_s=[r['wall_s'] for r in res['rounds']], emulated_round_s_same_cores=em,
               emulated_round_s_one_machine_per_party=em_inf,
               error_pct=[100 * (e - l['wall_s']) / l['wall_s'] for e, l in zip(em, res['rounds'])],
               bytes_down=res['bytes_down'], bytes_up=res['bytes_up'],
               tcpip_overhead_note='framed payload bytes; add about 52 bytes of IPv4+TCP headers per 1448-byte segment (3.6%) and any TLS record overhead',
               entries=[(ev[2], ev[1]) for ev in res['trace'] if ev[0] == 'append'],
               total_s=time.time() - t0)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=str)
    json.dump(dict(res, config=vars(a)), open(a.out.replace('.json', '_trace.json'), 'w'), default=str)
    print(json.dumps({k: out[k] for k in ('audit_same', 'live_round_s', 'emulated_round_s_same_cores', 'error_pct',
                                         'emulated_round_s_one_machine_per_party')}, indent=1))
