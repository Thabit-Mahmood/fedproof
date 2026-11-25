"""The complete protocol in one process: registry, bulletin, sites, committee nodes, rounds, complaints, audit.

Design rule: every party acts only on what the public bulletin shows. Each honest site and node keeps
its own Replica, a deterministic interpreter of the bulletin that every party, the round driver and the auditor run in
exactly the same way. A request from the relay (the round driver) is honoured only when the party's own replica says the
bulletin justifies it; otherwise it is ignored. Erasure happens when the party's replica sees the round close.

Only public messages travel between parties (through the Bulletin). Private state stays inside Site and Node objects.
Flags on Site and Node let tests script deviations; honest parties never set them.
"""
import hashlib
import os
import numpy as np
from . import wire as W
from . import field as F
from . import ligero as L
from . import model as M
from . import pq
from . import step as ST
from . import message as SP
from .hashing import H
from .sharing import Sharing, rs_decode, _lagrange_row


# ---------------------------------------------------------------- canonical hashing (types, lengths and shapes bound)
def _canon(v):
    """Canonical, unambiguous byte encoding of the values that appear in messages and log entries."""
    if v is None:
        return b'N'
    if isinstance(v, bool):
        return b'B' + (b'1' if v else b'0')
    if isinstance(v, (int, np.integer)):
        b = int(v).to_bytes(16, 'little', signed=True)
        return b'I' + b
    if isinstance(v, (bytes, bytearray)):
        return b'Y' + len(v).to_bytes(8, 'little') + bytes(v)
    if isinstance(v, str):
        e = v.encode()
        return b'S' + len(e).to_bytes(8, 'little') + e
    if isinstance(v, np.ndarray):
        if v.dtype.kind not in 'iub':
            raise TypeError('array dtype %s is not canonical' % v.dtype)
        a = np.ascontiguousarray(v)
        meta = ('%s|%s' % (a.dtype.str, a.shape)).encode()
        return b'A' + len(meta).to_bytes(8, 'little') + meta + H(a.tobytes())
    if isinstance(v, (list, tuple)):
        parts = [_canon(x) for x in v]
        return (b'L' if isinstance(v, list) else b'U') + len(parts).to_bytes(8, 'little') + b''.join(len(p).to_bytes(8, 'little') + p for p in parts)
    if isinstance(v, dict):
        items = sorted(((_canon(k), _canon(x)) for k, x in v.items()))
        return b'D' + len(items).to_bytes(8, 'little') + b''.join(len(k).to_bytes(8, 'little') + k + len(x).to_bytes(8, 'little') + x for k, x in items)
    if isinstance(v, float):
        return b'F' + repr(v).encode()
    raise TypeError('cannot encode %r' % type(v))


def _chain(head, raw):
    """Bulletin head after one more entry: a hash chain over the exact wire bytes of every entry."""
    return hashlib.sha3_384(b'sfl-head' + head + len(raw).to_bytes(8, 'little') + raw).digest()


def _hb(*parts):
    h = hashlib.sha3_384()
    for p in parts:
        c = _canon(p)
        h.update(len(c).to_bytes(8, 'little')); h.update(c)
    return h.digest()


# ---------------------------------------------------------------- registry
class System:
    """Public parameters and the registry fixed at setup. Derived public state (burned nodes, strikes, roots, model)
    lives in Replica objects, which compute it from the bulletin; `view` is the round driver's replica."""

    def __init__(self, cfg, prm, N, K, kp, Tn, A_min, f_max=0, max_rounds=64, quorum=None):
        self.cfg, self.prm, self.N, self.K = cfg, prm, N, K
        self.shr = Sharing(cfg.d, kp, Tn, K)
        self.Tn = Tn
        self.A_min = A_min
        self.max_rounds = max_rounds
        self.quorum = quorum or (K + Tn) // 2 + 1      # two quorums intersect in more than Tn nodes -> in an honest one
        self.site_pk, self.node_pk, self.beacon_commit = {}, {}, {}
        self.init_data_roots, self.init_noise_root = {}, {}
        self.max_strikes = 1
        self.max_uses = None
        self.rows_d, self.rows_n = ST.block_rows(cfg, prm.ell)
        self._ident = None
        self.args = (cfg, prm, N, K, kp, Tn, A_min, f_max, max_rounds, quorum)
        self.view = None

    # convenience accessors for tests and experiments: the driver's view of the derived state
    node_burned = property(lambda s: s.view.burned)
    site_strikes = property(lambda s: s.view.strikes)
    round_ek = property(lambda s: s.view.round_ek)
    chunk_uses = property(lambda s: s.view.chunk_uses)
    data_roots = property(lambda s: s.view.data_roots)
    noise_root = property(lambda s: s.view.noise_root)
    keys_digest = property(lambda s: s.view.keys_digest)

    def excluded(self, i):
        return self.view.excluded(i)

    def genesis(self):
        return dict(args=self.args, site_pk=dict(self.site_pk), node_pk=dict(self.node_pk),
                    beacon_commit=dict(self.beacon_commit), data_roots={i: list(v) for i, v in self.init_data_roots.items()},
                    noise_root=dict(self.init_noise_root), max_strikes=self.max_strikes, max_uses=self.max_uses)

    @staticmethod
    def from_genesis(g):
        sysm = System(*g['args'])
        sysm.site_pk, sysm.node_pk = dict(g['site_pk']), dict(g['node_pk'])
        sysm.beacon_commit = dict(g['beacon_commit'])
        sysm.init_data_roots = {i: list(v) for i, v in g['data_roots'].items()}
        sysm.init_noise_root = dict(g['noise_root'])
        sysm.max_strikes = g['max_strikes']
        sysm.max_uses = g['max_uses']
        return sysm

    def ident(self):
        if self._ident is None:
            self._ident = _hb(repr(self.cfg), self.prm.ident(), self.shr.ident(), self.N, self.K, self.A_min, self.quorum,
                              [self.site_pk[i] for i in sorted(self.site_pk)],
                              [(self.node_pk[m], self.beacon_commit[m]) for m in sorted(self.node_pk)])
        return self._ident


def beacon_bits(cfg, beacon, site):
    nbits = cfg.d * cfg.J * cfg.r
    raw = hashlib.shake_256(b'sfl-beta' + beacon + site.to_bytes(4, 'little')).digest(-(-nbits // 8))
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits].reshape(cfg.d, cfg.J, cfg.r).astype(np.int64)


def chunk_index(site, t, nchunks):
    return (t + site) % nchunks


def keys_digest(t, keys):
    return _hb(b'round-keys', t, [(m, keys[m]) for m in sorted(keys)])


def round_ctx(reg, t, site, mdigest, beacon, chunk, next_noise_root, kdig):
    return _hb(b'sfl-round-ctx', reg.ident(), t, site, mdigest, beacon, chunk, next_noise_root, kdig)


def site_msg_digest(t, site, msg):
    pf = msg['proof']
    return _hb(b'site-msg', t, site, msg['ctx'], msg['root_d1'], msg['root_m'], msg['root_a'], msg['next_noise'],
               list(msg['cts']), msg['digests'], pf['q_prox'], pf['q_lin'], pf['q_quad'],
               [(op['vals'], list(op['salts']), list(op['path'])) for op in pf['opens']])


# ---------------------------------------------------------------- message checks (pure functions; never raise)
MAX_MSG_BYTES = 64 << 20
MAX_STEP = 4                    # a round number may skip at most MAX_STEP - 1 numbers
MSG_KEYS = {'ctx', 'root_d1', 'root_m', 'root_a', 'next_noise', 'cts', 'digests', 'proof', 'sig'}


def well_formed(reg, msg):
    try:
        if not isinstance(msg, dict) or set(msg) != MSG_KEYS:
            return False
        if not all(SP.is_digest(msg[k]) for k in ('ctx', 'root_d1', 'root_m', 'root_a', 'next_noise')):
            return False
        if not isinstance(msg['sig'], (bytes, bytearray)) or len(msg['sig']) > 8192:
            return False
        if not isinstance(msg['cts'], list) or len(msg['cts']) != reg.K or not all(type(c) in (bytes, bytearray) for c in msg['cts']):
            return False
        dg, pf = msg['digests'], msg['proof']
        if type(dg) is not np.ndarray or dg.dtype != np.uint64 or dg.shape != (reg.K, reg.prm.sigma):
            return False
        if not isinstance(pf, dict) or set(pf) != {'q_prox', 'q_lin', 'q_quad', 'opens'} or not isinstance(pf['opens'], list):
            return False
        size = dg.nbytes + sum(len(c) for c in msg['cts'])
        for k in ('q_prox', 'q_lin', 'q_quad'):
            if type(pf[k]) is not np.ndarray or pf[k].dtype != np.uint64 or pf[k].ndim != 2:
                return False
            size += pf[k].nbytes
        if len(pf['opens']) > 16:
            return False
        for op in pf['opens']:
            if not isinstance(op, dict) or set(op) != {'vals', 'salts', 'path'}:
                return False
            if type(op['vals']) is not np.ndarray or op['vals'].dtype != np.uint64 or op['vals'].ndim != 2:
                return False
            if not isinstance(op['salts'], list) or not isinstance(op['path'], list):
                return False
            if not all(type(x) in (bytes, bytearray) and len(x) == L.SALT for x in op['salts']):
                return False
            if not all(type(x) in (bytes, bytearray) and len(x) == 48 for x in op['path']):
                return False
            size += op['vals'].nbytes + 32 * len(op['salts']) + 48 * len(op['path'])
        return size <= MAX_MSG_BYTES
    except Exception:
        return False


def valid_sum(reg, v):
    return (type(v) is np.ndarray and v.dtype == np.uint64 and v.shape == (reg.shr.nblk,) and bool((v < np.uint64(F.P)).all()))


def sanitize_keys(posted):
    out = {}
    if not isinstance(posted, dict):
        return out
    for m, v in posted.items():
        if type(m) is int and isinstance(v, tuple) and len(v) == 2 and all(type(x) in (bytes, bytearray) for x in v):
            if len(v[0]) <= 4 * pq.KEM_EK and len(v[1]) <= 8192:
                out[m] = (bytes(v[0]), bytes(v[1]))
    return out


def sanitize_sigs(d):
    out = {}
    if isinstance(d, dict):
        for m, s in d.items():
            if type(m) is int and type(s) in (bytes, bytearray) and len(s) <= 8192:
                out[m] = bytes(s)
    return out


def sanitize_reveals(d):
    out = {}
    if isinstance(d, dict):
        for m, r in d.items():
            if type(m) is int and SP.is_digest(r):
                out[m] = bytes(r)
    return out


_VCACHE = {}


def _verify_proof(reg, pub, root_d0, root_nz, msg, cached=True):
    """site_verify, memoised on the full canonical content of everything it reads (a pure function)."""
    if not cached:
        return SP.site_verify(pub, root_d0, root_nz, msg)
    try:
        key = _hb(b'vcache', reg.ident(), pub.ctx, root_d0, root_nz, [bool(x) for x in pub.key_present],
                  [(np.asarray(w), np.asarray(b)) for w, b in pub.weights], np.asarray(pub.beta),
                  site_msg_digest(0, 0, msg), msg['sig'])
    except Exception:
        return False, None
    if key not in _VCACHE:
        if len(_VCACHE) > 4096:
            _VCACHE.clear()
        _VCACHE[key] = SP.site_verify(pub, root_d0, root_nz, msg)
    return _VCACHE[key]


def judge(reg, round_ek, pub, msg, gamma, node, seed):
    """Public resolution of node's complaint against a site, given the seed the site revealed (or None).
    Returns 'site' (the share was bad, or no valid seed), 'node' (the share was good: the node lied) or 'void'
    (the node had no accepted key this round: its complaint does not count). Never raises."""
    try:
        ek = round_ek.get(node)
        if ek is None:
            return 'void'
        if not isinstance(seed, (bytes, bytearray)) or len(seed) != 32:
            return 'site'
        pt = pq.open_with_seed(ek, msg['cts'][node], SP._aad(pub, node), bytes(seed))
        if pt is None or len(pt) != 8 * (reg.shr.nblk + pub.G):
            return 'site'
        row = np.frombuffer(pt, dtype=np.uint64)
        if (row >= np.uint64(F.P)).any() or not ST.node_check(reg.shr, pub.G, gamma, node, row, msg['digests'][node]):
            return 'site'
        return 'node'
    except Exception:
        return 'site'


def decode_aggregate(reg, t, sums):
    """sums: node -> array (nblk,). Locate lying nodes, reconstruct every block. Returns (vector, liars) or None.
    Refuses unless at least w + 2*Tn sums are available. Never raises."""
    try:
        shr = reg.shr
        sums = {m: v for m, v in sums.items() if type(m) is int and 0 <= m < shr.K and valid_sum(reg, v)}
        avail = sorted(sums)
        if len(avail) < shr.w + 2 * reg.Tn:
            return None
        seed = _hb(b'combine', t, [sums[m] for m in avail])
        cap = (len(avail) - shr.w) // 2                   # errors the code corrects; at least Tn by the check above
        liars = set()
        for attempt in range(shr.K):
            use = [m for m in avail if m not in liars]
            emax = (len(use) - shr.w) // 2
            gam = F.rand_field(shr.nblk, F.Drbg(seed + bytes([attempt])))
            comb = [int(F.fdot(gam, sums[m])) for m in use]
            res = rs_decode([shr.alphas[m] for m in use], comb, shr.w, emax)
            if res is None:
                return None
            bad = [use[i] for i in res[1]]
            liars.update(bad)
            good = [m for m in use if m not in bad]
            if len(good) < shr.w:
                return None
            if len(liars) > cap:
                return None
            base = good[:shr.w]
            Cm = shr.secret_coeffs(base)
            mat = np.array([sums[m] for m in base], dtype=np.uint64)
            secrets = np.array([F.matvec(mat.T.copy(), Cm[i]) for i in range(shr.kp)], dtype=np.uint64).T
            extra = [m for m in good if m not in base]
            # every other node's full vector is checked against the base. If the base held a liar (missed by the random
            # combination, probability about 1/|F|), more than cap nodes disagree with it (at least
            # len(avail) - w - liars + 1 honest ones): then nobody is blamed and a fresh combination is tried. With an
            # honest base, the nodes that disagree are liars.
            pts = [shr.alphas[m] for m in base]
            newbad = [m for m in extra
                      if not np.array_equal(F.matvec(mat.T.copy(), np.array(_lagrange_row(pts, shr.alphas[m]), dtype=np.uint64)), sums[m])]
            if len(newbad) + len(liars) > cap:
                continue
            if newbad:
                liars.update(newbad)
                continue
            return F.to_signed(secrets.reshape(-1)[:reg.cfg.d]), sorted(liars)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------- the replica: the bulletin's one interpretation
class Round:
    def __init__(self, t, beacon):
        self.t, self.beacon = t, beacon
        self.keys_posted = None         # sanitized dict from the first keys entry
        self.disputes = {}
        self.keys_final = False
        self.round_ek, self.keys_digest = {}, b''
        self.expected = []              # sites expected to post this round
        self.firsts = {}                # site -> its first counted message (valid signature and matching context)
        self.pubs, self.gammas, self.proven, self.invalid = {}, {}, [], []
        self.complaints = {}            # (site, node) -> first valid complaint entry
        self.reveals = {}               # (site, node) -> seed from the first valid reveal
        self.noise_only = {}            # site -> root from its first valid noise-only entry
        self.cert = None                # (valid list, digest) once certified
        self.burn_now = set()           # nodes found lying in complaints (applied at certification)
        self.mismatched = set()         # sites with a signed message for another context (set aside, no strike)
        self.revealed = set()           # nodes with a good beacon reveal this round
        self.dispute_burned = set()     # nodes removed because a key they did not make was signed in their name
        self.early_results = []         # result entries logged before the round closed
        self.closed = None


class Replica:
    """Deterministic interpreter of the bulletin. Every honest party, the driver and the auditor run this same code on the
    same entries, so they agree on every outcome. Entries that the rules do not justify are ignored (but hashed)."""

    def __init__(self, reg, cached=True):
        self.reg, self.cached = reg, cached
        cfg = reg.cfg
        self.head = _hb(b'bulletin')
        self.burned = set()
        self.strikes = {i: 0 for i in range(reg.N)}
        self.data_roots = {i: list(v) for i, v in reg.init_data_roots.items()}
        self.noise_root = dict(reg.init_noise_root)
        self.chunk_uses = {i: [0] * len(v) for i, v in reg.init_data_roots.items()}
        self.last_reveal = {}
        self.last_t = 0
        self.state = M.init_model(cfg)
        self.round = None
        self.history = []               # per-round outcomes
        self.contradictions = []        # logged claims that contradict the computed outcome
        self.aggs = {}                  # t -> decoded aggregate of every round that produced a result
        self.ignored = 0
        self.listeners = []             # callbacks(event, round) for the owning party

    # -- public derived state
    def excluded(self, i):
        return self.strikes[i] > self.reg.max_strikes

    @property
    def round_ek(self):
        return self.round.round_ek if self.round is not None and self.round.closed is None else {}

    @property
    def keys_digest(self):
        return self.round.keys_digest if self.round is not None and self.round.closed is None else b''

    def open_round(self, t):
        r = self.round
        return r if (r is not None and r.t == t and r.closed is None) else None

    def chunk_exhausted(self, i, t):
        if self.reg.max_uses is None:
            return False
        c = chunk_index(i, t, len(self.data_roots[i]))
        return self.chunk_uses[i][c] >= self.reg.max_uses

    def expected_sites(self, t):
        return [i for i in range(self.reg.N) if not self.excluded(i) and self.noise_root.get(i) is not None
                and not self.chunk_exhausted(i, t)]

    def ctx_for(self, t, site, next_root):
        r = self.open_round(t)
        c = chunk_index(site, t, len(self.data_roots[site]))
        return round_ctx(self.reg, t, site, M.model_digest(self.reg.cfg, self.state), r.beacon, c, next_root, r.keys_digest)

    def public_for(self, t, site, next_root):
        cfg = self.reg.cfg
        r = self.open_round(t)
        return SP.Public(cfg, self.reg.prm, self.reg.shr, M.proof_weights(cfg, self.state), beacon_bits(cfg, r.beacon, site),
                         self.ctx_for(t, site, next_root), key_present=[m in r.round_ek for m in range(self.reg.K)])

    # -- feeding
    def consume(self, e, raw=None):
        """Read one bulletin entry. An entry that has no wire encoding cannot be on the bulletin: it is ignored and not
        hashed. Otherwise the head chains the exact wire bytes, so every reader of the same log has the same head."""
        if raw is None:
            try:
                raw = W.encode(e)
            except Exception:
                self.ignored += 1
                return
        try:
            self._consume(e)
        except Exception:
            self.ignored += 1
        self.head = _chain(self.head, raw)

    def _emit(self, ev):
        for fn in list(self.listeners):
            try:
                fn(ev, self.round)
            except Exception:
                pass

    def _consume(self, e):
        if not isinstance(e, dict):
            self.ignored += 1; return
        kind, t = e.get('kind'), e.get('t')
        if kind == 'register':
            return
        if kind == 'beacon':
            return self._on_beacon(t, e)
        r = self.open_round(t) if type(t) is int else None
        if r is None:
            if kind == 'result' and type(t) is int:
                self._check_result(t, e)
            else:
                self.ignored += 1
            return
        handler = getattr(self, '_on_' + str(kind).replace('-', '_'), None)
        if handler is None:
            self.ignored += 1; return
        handler(r, e)

    def _on_beacon(self, t, e):
        reg = self.reg
        if type(t) is not int or t <= self.last_t or t > reg.max_rounds or t - self.last_t > MAX_STEP:
            self.ignored += 1; return
        reveals = sanitize_reveals(e.get('reveals'))
        if self._count_good(t, reveals) < reg.Tn + 1:      # at least one honest node contributes, or no round opens
            self.ignored += 1; return
        if self.round is not None and self.round.closed is None:
            self._close(self.round, 'abandoned')
        good = {}
        for m in range(reg.K):                            # reveals are unsigned, so a missing or wrong one (which the
            if m in self.burned:                          # relay could fake) only makes the node sit this round out
                continue
            r_ = reveals.get(m)
            if not self._good_reveal(m, t, r_):
                continue
            good[m] = r_
            self.last_reveal[m] = (r_, t)
        self.last_t = t
        beacon = _hb(b'beacon', t, self.head, [good[m] for m in sorted(good)])
        self.round = Round(t, beacon)
        self.round.revealed = set(good)
        self.round.expected = self.expected_sites(t)
        self._emit('beacon')

    def _good_reveal(self, m, t, r_):
        prev, tprev = self.last_reveal.get(m, (self.reg.beacon_commit[m], 0))
        if r_ is None:
            return False
        x = r_
        for _ in range(t - tprev):
            x = H(b'beacon-chain' + x)
        return x == prev

    def _count_good(self, t, reveals):
        out = set(self.burned)
        r = self.round
        if r is not None and r.closed is None:            # nodes the close of the open round will remove
            out |= set(r.burn_now) | set(r.disputes)
            try:
                out |= self.judged(r)[2]
            except Exception:
                pass
        return sum(1 for m in range(self.reg.K) if m not in out and self._good_reveal(m, t, reveals.get(m)))

    def _on_keys(self, r, e):
        if r.keys_posted is not None or r.keys_final:
            self.ignored += 1; return
        r.keys_posted = sanitize_keys(e.get('posted'))
        self._finalize_keys(r, provisional=True)
        self._emit('keys')

    def _on_key_dispute(self, r, e):
        if r.keys_posted is None or r.keys_final:
            self.ignored += 1; return
        for m, sg in sanitize_sigs(e.get('sigs')).items():
            v = r.keys_posted.get(m)
            if (v is not None and m not in r.disputes and self._key_signed(r.t, m, v)
                    and pq.verify(self.reg.node_pk.get(m, b''), _hb(b'key-dispute', r.t, m, v[0]), sg)):
                r.disputes[m] = sg                         # a key signed with m's key that m did not make
        self._finalize_keys(r, provisional=True)
        self._emit('keys')

    def _key_signed(self, t, m, v):
        return m in self.reg.node_pk and pq.verify(self.reg.node_pk[m], _hb(b'round-key', t, m, v[0]), v[1])

    def _finalize_keys(self, r, provisional=False):
        if r.keys_final:
            return
        keys = {}
        for m in range(self.reg.K):
            if m in self.burned or m in r.disputes or m not in r.revealed:
                continue
            v = (r.keys_posted or {}).get(m)
            if v is None or not pq.ek_valid(v[0]) or not pq.verify(self.reg.node_pk[m], _hb(b'round-key', r.t, m, v[0]), v[1]):
                continue
            keys[m] = v[0]
        r.round_ek, r.keys_digest = keys, keys_digest(r.t, keys)
        if not provisional:
            r.keys_final = True
            r.dispute_burned = set(r.disputes) - self.burned
            self.burned |= set(r.disputes)           # a disputed key proves the signing key is in other hands
                                                     # (a node with no valid key only sits this round out)

    def check_site(self, t, i, msg):
        """What a site entry would mean in the open round t, without changing anything:
        ('ignored', None, None), ('mismatch', None, None), ('invalid', pub, None) or ('proven', pub, gamma)."""
        reg = self.reg
        r = self.open_round(t)
        if r is None or r.keys_posted is None or r.cert is not None:
            return 'ignored', None, None
        if type(i) is not int or i not in r.expected or i in r.firsts or not well_formed(reg, msg):
            return 'ignored', None, None
        if not pq.verify(reg.site_pk[i], site_msg_digest(t, i, msg), msg['sig']):
            return 'ignored', None, None                   # not the site's message: never counted, never blamed on it
        pub = self.public_for(t, i, msg['next_noise'])
        if msg['ctx'] != pub.ctx:
            return 'mismatch', None, None                  # made for another context (stale, replayed or reordered)
        c = chunk_index(i, t, len(self.data_roots[i]))
        ok, gamma = _verify_proof(reg, pub, self.data_roots[i][c], self.noise_root[i], msg, self.cached)
        return ('proven', pub, gamma) if ok else ('invalid', pub, None)

    def _on_site(self, r, e):
        if r.keys_posted is None or r.cert is not None:
            self.ignored += 1; return
        if not r.keys_final:
            self._finalize_keys(r)                         # the first site entry fixes the round keys
        i, msg = e.get('site'), e.get('msg')
        verdict, pub, gamma = self.check_site(r.t, i, msg)
        if verdict == 'ignored':
            self.ignored += 1; return
        if verdict == 'mismatch':
            r.mismatched.add(i)                            # it never counts, and the site is not blamed for it
            self.ignored += 1; return
        r.firsts[i] = msg
        ok = verdict == 'proven'
        if ok:
            r.pubs[i], r.gammas[i] = pub, gamma
            r.proven.append(i)
        else:
            r.invalid.append(i)
        self._emit(('site', i))

    def _on_complaint(self, r, e):
        if r.cert is not None:
            self.ignored += 1; return
        i, m, sg = e.get('site'), e.get('node'), e.get('sig')
        if i not in r.pubs or type(m) is not int or m not in r.round_ek or m in self.burned or (i, m) in r.complaints:
            self.ignored += 1; return
        if not pq.verify(self.reg.node_pk[m], _hb(b'complaint', r.t, i, m), sg):
            self.ignored += 1; return
        r.complaints[(i, m)] = sg
        self._emit(('complaint', i, m))

    def _on_reveal(self, r, e):
        if r.cert is not None:
            self.ignored += 1; return
        i, m, seed, sg = e.get('site'), e.get('node'), e.get('seed'), e.get('sig')
        if (i, m) not in r.complaints or (i, m) in r.reveals or not isinstance(seed, (bytes, bytearray)):
            self.ignored += 1; return
        if not pq.verify(self.reg.site_pk[i], _hb(b'reveal', r.t, i, m, bytes(seed)), sg):
            self.ignored += 1; return
        r.reveals[(i, m)] = bytes(seed)

    def _on_noise(self, r, e):
        i, root, sg = e.get('site'), e.get('root'), e.get('sig')
        if type(i) is not int or not (0 <= i < self.reg.N) or i in r.noise_only or i in r.proven or self.excluded(i):
            self.ignored += 1; return
        if not SP.is_digest(root) or not pq.verify(self.reg.site_pk[i], _hb(b'noise-only', r.t, i, root), sg):
            self.ignored += 1; return
        r.noise_only[i] = bytes(root)

    def judged(self, r, proven_only=False):
        """(valid list, sites with upheld complaints, lying nodes) from the round so far. With proven_only, only the
        sites whose revealed seed opened a bad share (a fault anyone can check); a missing reveal also upholds the
        complaint, but it could be censorship, so it costs the site this round and nothing more."""
        upheld, liars = set(), set()
        for (i, m) in sorted(r.complaints):
            seed = r.reveals.get((i, m))
            v = judge(self.reg, r.round_ek, r.pubs[i], r.firsts[i], r.gammas[i], m, seed)
            if v == 'site' and (seed is not None or not proven_only):
                upheld.add(i)
            elif v == 'node':
                liars.add(m)
        valid = sorted(i for i in r.proven if i not in upheld)
        return valid, upheld, liars

    def set_digest(self, t, valid):
        return _hb(b'vset', t, list(valid), self.head)

    def _on_certificate(self, r, e):
        if r.cert is not None:
            self.ignored += 1; return
        if not r.keys_final:
            self._finalize_keys(r)
        valid, upheld, liars = self.judged(r)
        r.burn_now = liars
        if len(valid) < self.reg.A_min or e.get('valid') != valid:
            return self._close(r, 'aborted')
        dig = self.set_digest(r.t, valid)
        sigs = sanitize_sigs(e.get('sigs'))
        n = sum(1 for m, sg in sigs.items() if m in self.reg.node_pk and m not in self.burned and m not in liars
                and pq.verify(self.reg.node_pk[m], _hb(b'valid-set', r.t, dig), sg))
        if n < self.reg.quorum:
            return self._close(r, 'aborted')
        r.cert = (valid, dig)
        self._emit('certified')

    def _on_sums(self, r, e):
        if r.cert is None:
            self.ignored += 1; return
        reg = self.reg
        raw, rs = e.get('sums'), e.get('sigs')
        if not isinstance(raw, dict) or not isinstance(rs, dict):
            return self._close(r, 'aborted')
        sums = {m: v for m, v in raw.items()
                if type(m) is int and m in reg.node_pk and m not in self.burned and m not in r.burn_now and valid_sum(reg, v)
                and pq.verify(reg.node_pk[m], _hb(b'sum', r.t, v), rs.get(m))}
        dec = decode_aggregate(reg, r.t, sums)
        valid = r.cert[0]
        if dec is None or (np.abs(dec[0]) > len(valid) * reg.cfg.ranges()['u_absmax']).any():
            return self._close(r, 'aborted')
        agg, liars = dec
        r.burn_now = set(r.burn_now) | set(liars)
        self.state = M.apply_update(reg.cfg, self.state, agg, len(valid))
        r.agg, r.liars = agg, liars
        self.aggs[r.t] = agg
        self._close(r, 'result')

    def _on_result(self, r, e):
        r.early_results.append(e)                         # checked against the round's record when it closes

    def _check_result(self, t, e):
        h = next((h for h in self.history if h['t'] == t), None)
        if h is None or h['outcome'] != 'result':
            self.contradictions.append((t, 'result logged for a round that produced none'))
            return
        try:
            same = (e.get('valid') == h['valid'] and e.get('model') == h['model']
                    and np.array_equal(np.asarray(e.get('agg')), self.aggs[t]))
        except Exception:
            same = False
        if not same:
            self.contradictions.append((t, 'logged result differs from the computed one'))

    def _on_abort(self, r, e):
        self._close(r, 'aborted')                         # a relay may always abort a round

    def _close(self, r, how):
        if r.closed is not None:
            return
        reg = self.reg
        if not r.keys_final:
            self._finalize_keys(r)
        valid, _u, liars = self.judged(r)             # complaints and reveals are frozen once a set is certified
        if r.cert is not None:
            valid = r.cert[0]
        proven_bad = self.judged(r, proven_only=True)[1]
        r.burn_now = set(r.burn_now) | liars
        self.burned |= r.burn_now
        r.struck = sorted(set(r.invalid) | proven_bad)   # strikes only for faults anyone can check from the log:
        for i in r.struck:                                # a signed proof that fails, or a revealed bad share
            self.strikes[i] += 1
        for i in range(reg.N):                            # pointers: a proven message moves data and noise forward
            if i in r.proven:
                msg = r.firsts[i]
                c = chunk_index(i, r.t, len(self.data_roots[i]))
                self.data_roots[i][c] = msg['root_d1']
                self.chunk_uses[i][c] += 1
                self.noise_root[i] = msg['next_noise']
            elif i in r.noise_only:
                self.noise_root[i] = r.noise_only[i]
            elif i in r.expected or (self.noise_root.get(i) is not None and i in r.firsts):
                self.noise_root[i] = None                 # its noise block may have been opened: commit fresh noise
        r.closed = how
        r.final_valid = valid if how == 'result' else []
        self.history.append(dict(t=r.t, outcome=how, valid=list(r.final_valid), struck=list(r.struck),
                                 liars=list(getattr(r, 'liars', [])), burned=sorted(set(r.burn_now) | r.dispute_burned),
                                 model=M.model_digest(reg.cfg, self.state)))
        for e in r.early_results:
            self._check_result(r.t, e)
        self._emit('closed')


# ---------------------------------------------------------------- the bulletin
class Bulletin:
    """Append-only public log. Every subscribed replica reads every entry, in order; the relay cannot skip a reader."""

    def __init__(self, reg):
        self.entries, self.raw = [], []
        self.head = _hb(b'bulletin')
        self.reg = reg
        self.readers = []

    def subscribe(self, replica):
        for e, raw in zip(self.entries, self.raw):
            replica.consume(e, raw)
        self.readers.append(replica)

    def append(self, entry):
        """The bulletin stores bytes: an entry is encoded once, and every reader gets the decoded wire form of exactly
        those bytes. Anything that does not encode is refused (returns False)."""
        try:
            raw = W.encode(entry)
            e = W.decode(raw)
        except Exception:
            return False
        self.entries.append(e)
        self.raw.append(raw)
        self.head = _chain(self.head, raw)
        for rd in self.readers:
            rd.consume(e, raw)
        return True


# ---------------------------------------------------------------- parties
class Site:
    def __init__(self, idx, reg, chunks, rng=None):
        self.idx, self.reg, self.rng = idx, reg, rng
        self.pk, self.sk = pq.sig_keygen()
        self.chunks = chunks
        self.blocks = []
        self.noise = None                                 # (rho, Committed) the site's current registered noise block
        self.pending_noise = {}                           # root -> (rho, Committed) committed but not yet registered
        self.last = None
        self.view = None
        self._msg = {}                                    # t -> message (one per round, never a second proof)
        self._noise_only = {}                             # t -> (root, sig)
        self._revealed = set()
        self.spent_noise = set()
        # test-only deviation flags
        self.cheat_hook = None; self.offline = False; self.bad_share_to = None; self.refuse_reveal = False
        self.replay = None; self.stale_model = None

    def _rand_bits(self, shape):
        n = int(np.prod(shape))
        raw = os.urandom(-(-n // 8)) if self.rng is None else self.rng.read(-(-n // 8))
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n].reshape(shape).astype(np.int64)

    def _register(self):
        """Commit the data chunks and the first noise block; once, at setup."""
        assert not self.blocks, 'a site registers once'
        cfg, prm = self.reg.cfg, self.reg.prm
        self.blocks = [L.commit_block(ST.data_block_matrix(cfg, prm.ell, X, y, self.rng), prm, self.rng) for X, y in self.chunks]
        self.opens = [0] * len(self.chunks)
        self.max_opens = (prm.ell // 2 + prm.tprime) // prm.t
        root = self._commit_noise()
        self.noise = self.pending_noise.pop(root)
        return dict(pk=self.pk, data_roots=[b.root for b in self.blocks], noise_root=root)

    def _commit_noise(self):
        cfg, prm = self.reg.cfg, self.reg.prm
        rho = self._rand_bits((cfg.d, cfg.J, cfg.r))
        blk = L.commit_block(ST.noise_block_matrix(cfg, prm.ell, rho), prm, self.rng)
        self.pending_noise[blk.root] = (rho, blk)
        return blk.root

    def _attach(self, view):
        assert self.view is None, 'a party has one replica, fixed at setup'
        self.view = view
        view.listeners.append(self._on_event)

    def _on_event(self, ev, r):
        if ev != 'closed':
            return
        v = self.view
        root = v.noise_root.get(self.idx)                 # follow the public pointers exactly
        if self.idx in r.proven and self.last is not None and self.last['t'] == r.t and self._mine(r):
            self.blocks[self.last['chunk']] = self.last['d1']
            self.opens[self.last['chunk']] = 1
        if root is None:
            self.noise = None
        elif self.noise is None or self.noise[1].root != root:
            self.noise = self.pending_noise.pop(root, None)
        self.pending_noise = {}
        if self.last is not None:                         # forget the encapsulation seeds of the closed round
            self.last['seeds'] = None
        self._msg = {k: m for k, m in self._msg.items() if k > r.t}

    def _mine(self, r):
        """True if the message the bulletin counted for this site in round r is the one this site made."""
        m = self._msg.get(r.t)
        f = r.firsts.get(self.idx)
        return m is not None and f is not None and site_msg_digest(r.t, self.idx, f) == site_msg_digest(r.t, self.idx, m)

    def round_message(self, t, state=None, beacon=None):
        """The site's signed message for round t, or None. Honoured only while round t is open on the site's own view,
        its keys are known and the site is expected; asked twice, the site returns the same message."""
        if type(t) is not int:
            return None
        if self.offline:
            return None
        if self.replay is not None:
            return self.replay
        v = self.view
        r = v.open_round(t)
        if r is None or r.keys_posted is None or r.cert is not None or self.idx not in r.expected:
            return None
        if t in self._msg:
            return self._msg[t]
        if self.noise is None or self.noise[1].root != v.noise_root.get(self.idx):
            return None
        if self.noise[1].root in self.spent_noise and self.cheat_hook is None:
            return None
        cfg, reg = self.reg.cfg, self.reg
        c = chunk_index(self.idx, t, len(self.chunks))
        if self.blocks[c].root != v.data_roots[self.idx][c]:
            return None                                   # never prove against a commitment the bulletin does not name
        if self.opens[c] + 1 > self.max_opens and self.cheat_hook is None:
            return None
        if not r.keys_final:
            keys = dict(r.round_ek)
        else:
            keys = r.round_ek
        self.opens[c] += 1
        X, y = self.chunks[c]
        rho, nzblk = self.noise
        self.spent_noise.add(nzblk.root)
        next_root = self._commit_noise()
        if self.stale_model is not None:
            st_w, st_d = M.proof_weights(cfg, self.stale_model), M.model_digest(cfg, self.stale_model)
        else:
            st_w, st_d = M.proof_weights(cfg, v.state), M.model_digest(cfg, v.state)
        kdig = keys_digest(t, keys)
        ctx = round_ctx(reg, t, self.idx, st_d, r.beacon, c, next_root, kdig)
        pub = SP.Public(cfg, reg.prm, reg.shr, st_w, beacon_bits(cfg, r.beacon, self.idx), ctx,
                        key_present=[m in keys for m in range(reg.K)])
        hook = self.cheat_hook
        if self.bad_share_to is not None:
            tgt = self.bad_share_to

            def hook(name, cs, ids, _h=self.cheat_hook):
                if name == 'shares':
                    sh = ids.copy()
                    sh[tgt, 0] = (int(sh[tgt, 0]) + 1) % F.P
                    return sh
                return _h(name, cs, ids) if _h else None
        msg, priv = SP.site_prove(pub, dict(X=X, y=y, rho=rho), self.blocks[c], nzblk,
                                  [keys.get(m) for m in range(reg.K)], self.rng, hook=hook,
                                  strict=self.cheat_hook is None)
        msg['ctx'] = ctx
        msg['next_noise'] = next_root
        msg['sig'] = pq.sign(self.sk, site_msg_digest(t, self.idx, msg))
        self.last = dict(t=t, chunk=c, d1=priv['d1'], seeds=priv['seeds'], u=priv['u'])
        self._msg[t] = msg
        return msg

    def reveal(self, t, node):
        """Reveal the encapsulation seed for node's share, only if the bulletin holds a valid complaint by that node
        against this site in the open round t; once per complaint."""
        if type(t) is not int or type(node) is not int:
            return None
        if self.refuse_reveal or self.offline:
            return None
        r = self.view.open_round(t)
        if r is None or (self.idx, node) not in r.complaints or (self.idx, node) in self._revealed:
            return None
        if self.last is None or self.last['t'] != t or self.last['seeds'] is None or not self._mine(r):
            return None
        seed = self.last['seeds'][node]
        if seed is None:
            return None
        self._revealed.add((self.idx, node))
        return bytes(seed), pq.sign(self.sk, _hb(b'reveal', t, self.idx, node, bytes(seed)))

    def noise_only(self, t):
        """A signed fresh noise commitment, when the site has no proven message in the open round t."""
        if type(t) is not int:
            return None
        if self.offline:
            return None
        r = self.view.open_round(t)
        if r is None or self.idx in r.proven or self.view.excluded(self.idx) or r.cert is not None:
            return None
        if t not in self._noise_only:
            root = self._commit_noise()
            self._noise_only[t] = (root, pq.sign(self.sk, _hb(b'noise-only', t, self.idx, root)))
        return self._noise_only[t]


class Node:
    """A committee node, driven only by its own view of the bulletin."""

    def __init__(self, idx, reg, rng=None):
        self.idx, self.reg = idx, reg
        self.pk, self.sk = pq.sig_keygen()
        self.round_dk, self.round_ek_mine = {}, {}
        seed = os.urandom(48) if rng is None else rng.read(48)
        chain = [seed]
        for _ in range(reg.max_rounds):
            chain.append(H(b'beacon-chain' + chain[-1]))
        self.chain = chain
        self.shares = {}                                  # (t, site) -> share row
        self.failed = {}                                  # (t, site) -> True when this node's own check failed
        self.complained = set()
        self.signed = {}                                  # t -> (digest, valid list)
        self._revealed_for = {}                           # last logged beacon -> the furthest round revealed after it
        self.released = set()
        self.view = None
        # test-only deviation flags
        self.offline = False; self.withhold_beacon = False; self.false_complaint = None; self.wrong_sum = False
        self.sign_anything = False; self.withhold_key = False; self.bad_key = False; self.no_erase = False

    def _register(self):
        return dict(pk=self.pk, beacon_commit=self.chain[-1])

    def _attach(self, view):
        assert self.view is None, 'a party has one replica, fixed at setup'
        self.view = view
        view.listeners.append(self._on_event)

    def _on_event(self, ev, r):
        if ev == 'closed':
            self._erase(r.t)
        elif isinstance(ev, tuple) and ev[0] == 'site':
            self._check_share(r, ev[1])

    def _erase(self, t):
        """Forget the round key and every share of rounds up to t. Called only by this node's own replica when the
        bulletin closes round t; nothing outside the node can trigger it."""
        if self.no_erase:
            return
        for k in [k for k in self.round_dk if k <= t]:
            del self.round_dk[k]
        for k in [k for k in self.shares if k[0] <= t]:
            del self.shares[k]

    def _secrets_held(self):
        return dict(round_keys=sorted(self.round_dk), share_rounds=sorted({k[0] for k in self.shares}))

    def beacon_reveal(self, t):
        """The chain value for round t: only for a round within MAX_STEP of the last logged beacon, and for one round
        number per gap (a value for t also gives the values of every earlier round)."""
        last = self.view.last_t
        if self.offline or self.withhold_beacon or type(t) is not int or not (last < t <= min(last + MAX_STEP, self.reg.max_rounds)):
            return None
        hi = self._revealed_for.get(last)
        if hi is not None and t > hi:
            return None                                   # never reveal further ahead than already revealed in this gap
        self._revealed_for[last] = t if hi is None else hi
        return self.chain[self.reg.max_rounds - t]

    def round_key(self, t):
        """This node's key for round t: generated once, only when the bulletin shows round t open and no keys yet."""
        if type(t) is not int:
            return None
        if self.offline or self.withhold_key:
            return None
        r = self.view.open_round(t)
        if r is None or self.idx in self.view.burned:
            return None
        if t not in self.round_ek_mine:
            if r.keys_posted is not None:
                return None
            ek, dk = pq.kem_keygen()
            if self.bad_key:
                ek = b'\xff' * len(ek)
            self.round_dk[t] = dk
            self.round_ek_mine[t] = (ek, pq.sign(self.sk, _hb(b'round-key', t, self.idx, ek)))
        return self.round_ek_mine[t]

    def key_dispute(self, t):
        """Dispute a key logged under this node's name that carries a valid signature by this node's signing key but
        that this node did not make: proof that the signing key is in other hands. Junk keys are not disputed."""
        if type(t) is not int:
            return None
        r = self.view.open_round(t)
        if self.offline or r is None or r.keys_posted is None or r.keys_final:
            return None
        v = r.keys_posted.get(self.idx)
        mine = self.round_ek_mine.get(t)
        if v is not None and self.view._key_signed(t, self.idx, v) and (mine is None or v[0] != mine[0]):
            return pq.sign(self.sk, _hb(b'key-dispute', t, self.idx, v[0]))
        return None

    def _check_share(self, r, i):
        if self.offline or i not in r.pubs:
            return
        dk = self.round_dk.get(r.t)
        if dk is None or self.idx not in r.round_ek:
            return                                         # without its own key a node never complains
        pub, gamma, msg = r.pubs[i], r.gammas[i], r.firsts[i]
        pt = pq.open_(dk, msg['cts'][self.idx], SP._aad(pub, self.idx))
        ok = False
        if pt is not None and len(pt) == 8 * (self.reg.shr.nblk + pub.G):
            row = np.frombuffer(pt, dtype=np.uint64)
            if (row < np.uint64(F.P)).all() and ST.node_check(self.reg.shr, pub.G, gamma, self.idx, row, msg['digests'][self.idx]):
                self.shares[(r.t, i)] = row[:self.reg.shr.nblk].copy()
                ok = True
        if not ok or self.false_complaint == i:
            self.failed[(r.t, i)] = True

    def complaints(self, t):
        """Signed complaints for the sites whose share failed this node's own check in the open round t."""
        if type(t) is not int:
            return None
        r = self.view.open_round(t)
        if self.offline or r is None or r.cert is not None:
            return []
        out = []
        for (rt, i) in sorted(self.failed):
            if rt == t and (t, i) not in self.complained:
                self.complained.add((t, i))
                out.append((i, pq.sign(self.sk, _hb(b'complaint', t, i, self.idx))))
        return out

    def sign_valid_set(self, t, valid):
        """Sign the proposed set only if it equals the set this node's own view computes (at least A_min sites);
        at most one signature per round."""
        if type(t) is not int or not isinstance(valid, list) or not all(type(x) is int for x in valid):
            return None
        if self.offline:
            return None
        v = self.view
        r = v.open_round(t)
        if r is None or r.cert is not None:
            return None
        try:
            vl = [int(i) for i in valid]
        except Exception:
            return None
        mine, _u, _l = v.judged(r)
        if not self.sign_anything and (vl != mine or len(vl) < self.reg.A_min):
            return None
        dig = v.set_digest(t, vl)
        if t in self.signed and self.signed[t][0] != dig and not self.sign_anything:
            return None
        self.signed.setdefault(t, (dig, vl))
        return dig, pq.sign(self.sk, _hb(b'valid-set', t, dig))

    def release_sum(self, t):
        """Release, once, the sum of shares over the set this node signed, only after the bulletin certified that set."""
        if type(t) is not int:
            return None
        if self.offline or t in self.released or t not in self.signed:
            return None
        r = self.view.open_round(t)
        dig, vl = self.signed[t]
        if r is None or r.cert is None or r.cert[1] != dig or r.cert[0] != vl:
            return None
        if any((t, i) not in self.shares for i in vl):
            return None
        self.released.add(t)
        acc = np.zeros(self.reg.shr.nblk, dtype=np.uint64)
        for i in vl:
            acc = F.fadd(acc, self.shares[(t, i)])
        if self.wrong_sum:
            acc = F.fadd(acc, F.U64(1))
        return acc, pq.sign(self.sk, _hb(b'sum', t, acc))


# ---------------------------------------------------------------- one-time setup
def setup(cfg, prm, site_chunks, K, kp, Tn, A_min, max_rounds=64, rng=None, max_uses=None):
    N = len(site_chunks)
    reg = System(cfg, prm, N, K, kp, Tn, A_min, max_rounds=max_rounds)
    reg.max_uses = max_uses
    sites = [Site(i, reg, ch, rng) for i, ch in enumerate(site_chunks)]
    nodes = [Node(m, reg, rng) for m in range(K)]
    for nd in nodes:
        rg = nd._register()
        reg.node_pk[nd.idx], reg.beacon_commit[nd.idx] = rg['pk'], rg['beacon_commit']
    for s in sites:
        reg.site_pk[s.idx] = s.pk
    regs = []
    for s in sites:                                   # data and first-round noise are committed before training starts
        rg = s._register()
        reg.init_data_roots[s.idx] = list(rg['data_roots'])
        reg.init_noise_root[s.idx] = rg['noise_root']
        regs.append(rg)
    log = Bulletin(reg)
    for s, rg in zip(sites, regs):
        log.append(dict(kind='register', site=s.idx, data_roots=list(rg['data_roots']), noise_root=rg['noise_root']))
    reg.view = Replica(reg)
    log.subscribe(reg.view)
    for p in sites + nodes:
        v = Replica(reg)
        log.subscribe(v)
        p._attach(v)
    reg.gen = reg.genesis()
    return reg, sites, nodes, log


# ---------------------------------------------------------------- round driver (the untrusted relay, played honestly)
def _call(fn, *a):
    try:
        return fn(*a)
    except Exception:
        return None


def _seq(calls):
    return [_call(fn, *a) for fn, a in calls]


def _pair(x):
    return x if isinstance(x, tuple) and len(x) == 2 else None


def run_round(reg, sites, nodes, log, state=None, t=None, equivocate=None, stop_before=None, gather=None):
    """Runs round t: asks parties for their messages and appends them to the bulletin. Outcomes are whatever the
    replicas compute; the driver reads them from its own replica. `state` is accepted for compatibility and unused."""
    if t is None:
        t, state = state, None
    G = gather or _seq                       # how the requests of one phase are issued (in turn, or concurrently)
    D = reg.view
    rep = dict(t=t, aborted=None, valid=[], excluded={}, liars=[], burned=[])
    burned0, strikes0 = set(D.burned), dict(D.strikes)
    reveals = dict(zip([nd.idx for nd in nodes], G([(nd.beacon_reveal, (t,)) for nd in nodes])))
    log.append(dict(kind='beacon', t=t, reveals=sanitize_reveals(reveals)))
    r = D.open_round(t)
    if r is None:
        rep['aborted'] = 'invalid round number'
        return D.state, rep
    posted = dict(zip([nd.idx for nd in nodes], G([(nd.round_key, (t,)) for nd in nodes])))
    log.append(dict(kind='keys', t=t, posted=sanitize_keys(posted)))
    disputes = sanitize_sigs(dict(zip([nd.idx for nd in nodes], G([(nd.key_dispute, (t,)) for nd in nodes]))))
    if disputes:
        log.append(dict(kind='key-dispute', t=t, sigs=disputes))
    for s, msg in zip(sites, G([(s.round_message, (t,)) for s in sites])):
        if msg is not None:
            log.append(dict(kind='site', t=t, site=s.idx, msg=msg))
    for nd, cs in zip(nodes, G([(nd.complaints, (t,)) for nd in nodes])):
        for c in (cs if isinstance(cs, list) else []):
            c = _pair(c)
            if c is not None:
                log.append(dict(kind='complaint', t=t, site=c[0], node=nd.idx, sig=c[1]))
    pend = sorted(r.complaints)
    for (i, m), out in zip(pend, G([(sites[i].reveal, (t, m)) for (i, m) in pend])):
        out = _pair(out)
        if out is not None:
            log.append(dict(kind='reveal', t=t, site=i, node=m, seed=out[0], sig=out[1]))
    for s, out in zip(sites, G([(s.noise_only, (t,)) for s in sites])):
        out = _pair(out)
        if out is not None:
            log.append(dict(kind='noise', t=t, site=s.idx, root=out[0], sig=out[1]))
    rep['_base'] = (burned0, strikes0)
    if stop_before == 'certificate':
        rep['valid'] = D.judged(r)[0]
        return D.state, rep
    return continue_round(reg, sites, nodes, log, t, equivocate, rep, gather)


def continue_round(reg, sites, nodes, log, t, equivocate=None, rep=None, gather=None):
    """The second half of a round (valid set, certificate, sums, result), for a round left open by
    run_round(..., stop_before='certificate'); tests insert relay behaviour in between."""
    G = gather or _seq
    D = reg.view
    r = D.open_round(t)
    if rep is None or '_base' not in rep:
        rep = dict(t=t, aborted=None, valid=[], excluded={}, liars=[], burned=[], _base=(set(D.burned), dict(D.strikes)))
    burned0, strikes0 = rep['_base']
    if r is None:
        rep['aborted'] = 'round %s is not open' % t
        return _finish(D, rep, burned0, strikes0, t)
    valid, upheld, liars = D.judged(r)
    rep['valid'] = valid
    if len(valid) < reg.A_min:
        log.append(dict(kind='abort', t=t, reason='fewer than %d valid sites' % reg.A_min))
        rep['aborted'] = 'fewer than %d valid sites' % reg.A_min
        return _finish(D, rep, burned0, strikes0, t)
    sigs = {}
    shown = [valid if equivocate is None else equivocate(nd.idx, valid) for nd in nodes]
    for nd, out in zip(nodes, G([(nd.sign_valid_set, (t, sh)) for nd, sh in zip(nodes, shown)])):
        out = _pair(out)
        if out is not None and out[0] == D.set_digest(t, valid):
            sigs[nd.idx] = out[1]
    rep['cert_counts'] = [len(sigs)]
    log.append(dict(kind='certificate', t=t, valid=valid, sigs=sigs))
    if r.cert is None:
        rep['aborted'] = 'no quorum on the valid set (%d of %d signatures)' % (len(sigs), reg.quorum)
        log.append(dict(kind='abort', t=t, reason=rep['aborted']))
        return _finish(D, rep, burned0, strikes0, t)
    sums, sumsigs = {}, {}
    for nd, out in zip(nodes, G([(nd.release_sum, (t,)) for nd in nodes])):
        out = _pair(out)
        if out is not None:
            sums[nd.idx], sumsigs[nd.idx] = out
    log.append(dict(kind='sums', t=t, sums=sums, sigs=sanitize_sigs(sumsigs)))
    if r.closed != 'result':
        rep['aborted'] = 'aggregate could not be decoded'
        log.append(dict(kind='abort', t=t, reason=rep['aborted']))
        return _finish(D, rep, burned0, strikes0, t)
    rep['agg'], rep['liars'] = r.agg, r.liars
    log.append(dict(kind='result', t=t, valid=valid, agg=r.agg, model=M.model_digest(reg.cfg, D.state)))
    return _finish(D, rep, burned0, strikes0, t)


def _finish(D, rep, burned0, strikes0, t):
    rep.pop('_base', None)
    rep['burned'] = sorted(D.burned - burned0)
    r = D.round
    if r is not None and r.cert:
        rep['valid'] = list(r.cert[0])
    if r is not None and r.t == t:
        for i in r.expected:
            if i in r.invalid:
                rep['excluded'][i] = 'invalid proof'
            elif i in getattr(r, 'struck', []):
                rep['excluded'][i] = 'complaint upheld'
            elif i not in r.firsts:
                rep['excluded'][i] = 'silent'
            elif i in r.proven and i not in rep['valid']:
                rep['excluded'][i] = 'complaint upheld, no reveal'
        rep['struck'] = list(getattr(r, 'struck', []))
    return D.state, rep


# ---------------------------------------------------------------- independent audit from the public log alone
def audit(genesis, entries, strict=True):
    """Recompute the whole training from the genesis registry and the bulletin entries, verifying every proof afresh.
    Returns (final_state, per-round outcomes). With strict=True, raises AssertionError if a logged result contradicts
    the computed one. Entries the rules do not justify are ignored, exactly as every honest party ignores them."""
    try:
        reg = System.from_genesis(genesis)
        v = Replica(reg, cached=False)
        for e in entries:
            v.consume(e)
    except Exception as ex:
        raise AssertionError('malformed log: %s: %s' % (type(ex).__name__, ex))
    if strict and v.contradictions:
        raise AssertionError('log contradicts its own computation: %r' % v.contradictions[:3])
    return v.state, v.history
