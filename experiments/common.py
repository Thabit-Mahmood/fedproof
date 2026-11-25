"""Shared pieces of the paper's experiment suite: data, model settings, metrics, baselines, result files.

Every result row is appended to runs/<experiment>.jsonl with its key, the code version and the machine, so an
interrupted run resumes where it stopped and every number in the paper can be traced to the code that produced it.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # before numpy: the harness forks
    os.environ.setdefault(_v, '1')

import os, sys, json, time, hashlib, platform, glob, math, functools
sys.path.insert(0, '.')
import numpy as np
from fedproof import model as M, dp

DATA = os.environ.get('FEDPROOF_DATA', 'data/prepared')
RESULTS = os.environ.get('FEDPROOF_RESULTS', 'runs')
DATASETS = tuple(os.environ.get('FEDPROOF_DATASETS', 'edge-iiotset,wustl-iiot-2021,x-iiotid').split(','))
SPLITS = ('iid', 'dir1.0', 'dir0.3', 'natural')       # natural: only where the dataset has a device or scenario column
PLANTS = 10
PER_PLANT = 32768                                     # records per plant (10 plants)
B = 128                                               # records per chunk: one chunk per plant per round
A_MIN = 7                                             # valid-set floor: no sum is opened over fewer plants
F_MAX = 3                                             # corrupted plants tolerated (fewer than a third of 10)
H_MIN = A_MIN - F_MAX                                 # honest plants in every opened sum, enforced by the floor
# consortium-size variant of the privacy experiment: 50 plants, same rule (f < N/3, H = A_min - f)
SCALE = dict(tag='iid50', plants=50, per_plant=8192, A_min=35, f=15)
SCALE['H'] = SCALE['A_min'] - SCALE['f']


def code_version():
    h = hashlib.sha256()
    for f in sorted(glob.glob('fedproof/*.py') + glob.glob('experiments/*.py')):
        h.update(f.encode()); h.update(open(f, 'rb').read())
    return h.hexdigest()[:12]


def machine():
    return dict(host=platform.node(), system=platform.system(), release=platform.release(), cpu=platform.processor(),
                cores=os.cpu_count(), python=platform.python_version(), numpy=np.__version__)


class Store:
    """Append-only JSON-lines result file, keyed; rows already present are skipped on a rerun."""

    def __init__(self, name):
        os.makedirs(RESULTS, exist_ok=True)
        self.path = os.path.join(RESULTS, name + '.jsonl')
        self.rows = {}
        if os.path.exists(self.path):
            for line in open(self.path):
                try:
                    r = json.loads(line)
                    self.rows[r['key']] = r
                except Exception:
                    pass
        self.ver, self.mach = code_version(), machine()

    def done(self, key):
        return key in self.rows

    def add(self, key, row):
        row = dict(row, key=key, code=self.ver, machine=self.mach['host'], cores=self.mach['cores'], at=time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(self.path, 'a') as f:
            f.write(json.dumps(row, default=_jsonable) + '\n')
        self.rows[key] = row
        return row


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------- data
def data_path(dataset, split, seed=0):
    return os.path.join(DATA, '%s_%s_seed%d' % (dataset, split.replace(':', '-'), seed))


def available(dataset, split, seed=0):
    return os.path.exists(data_path(dataset, split, seed) + '.npz')


@functools.lru_cache(maxsize=6)
def load(dataset, split, seed=0):
    base = data_path(dataset, split, seed)
    z = np.load(base + '.npz')
    meta = json.load(open(base + '.json'))
    n = meta['plants']
    plants = [(z['X_plant%d' % i].astype(np.int64), z['y_plant%d' % i].astype(np.int64)) for i in range(n)]
    return dict(meta=meta, plants=plants, X_pub=z['X_pub'].astype(np.int64), y_pub=z['y_pub'].astype(np.int64),
                X_test=z['X_test'].astype(np.int64), y_test=z['y_test'].astype(np.int64))


def normal_class(meta):
    for i, c in enumerate(meta['classes']):
        if str(c).strip().lower() in ('normal', 'benign', '0') or 'normal' in str(c).lower():
            return i
    return int(np.argmax([meta['class_counts'][c] for c in meta['classes']]))


def tuning_path(dataset):
    return os.path.join(RESULTS, 'tuning_%s.json' % dataset)


R_DEFAULT = 14                                        # noise bits outside E2: about epsilon 2 at two uses and H_MIN


def make_cfg(meta, noisy=True, **kw):
    """The model for a dataset: two hidden layers of 64 and 32, 8-bit centred inputs, chunks of B, R_DEFAULT noise bits;
    clipping threshold, learning rate and truncation from the public-data tuning when it exists (the learning rate for
    a run with noise is the one tuned under that noise; noisy=False takes the one tuned without noise)."""
    base = dict(arch=(meta['n_features'], 64, 32, len(meta['classes'])), Bx=8, b=B, r=R_DEFAULT, center=True)
    tp = tuning_path(meta['dataset'])
    if os.path.exists(tp):
        t = json.load(open(tp))
        base.update(t['chosen'])
        r = kw.get('r', base['r'])
        if noisy and str(r) in t.get('lr_noisy', {}):         # learning rate chosen on public data under this noise
            base['lr_shift'] = t['lr_noisy'][str(r)]
    base.update(kw)
    if 'arch' in base:
        base['arch'] = tuple(base['arch'])
    cfg = M.Config(**base)
    cfg.ranges()
    return cfg


def chunks(plants, b):
    out = []
    for X, y in plants:
        n = len(y) // b
        out.append([(X[i * b:(i + 1) * b], y[i * b:(i + 1) * b]) for i in range(n)])
    return out


# ---------------------------------------------------------------- metrics
def evaluate(pred, y, ncls, normal=None):
    from sklearn.metrics import f1_score, recall_score, accuracy_score
    labels = list(range(ncls))
    out = dict(acc=float(accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
               per_class_recall=[float(x) for x in recall_score(y, pred, labels=labels, average=None, zero_division=0)])
    if normal is not None:
        att = y != normal
        out['attack_detection'] = float((pred[att] != normal).mean()) if att.any() else None   # attack flagged as attack
        out['false_alarm'] = float((pred[~att] != normal).mean()) if (~att).any() else None
    return out


def eval_int(cfg, state, X, y, normal=None):
    return evaluate(M.predict(cfg, M.proof_weights(cfg, state), X), y, cfg.arch[-1], normal)


# ---------------------------------------------------------------- floating-point references
def float_central(d, seed, hidden=(64, 32), max_iter=200):
    from sklearn.neural_network import MLPClassifier
    X = np.concatenate([p[0] for p in d['plants']]) / 255.0
    y = np.concatenate([p[1] for p in d['plants']])
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=seed, early_stopping=False).fit(X, y)
    return clf.predict(d['X_test'] / 255.0)


class FloatMLP:
    """Floating-point MLP with softmax cross-entropy, for federated averaging of gradients (one chunk per plant per
    round, like the integer system)."""

    def __init__(self, arch, seed):
        g = np.random.default_rng(seed)
        self.W = [g.normal(0, math.sqrt(2.0 / a), (b, a)) for a, b in zip(arch[:-1], arch[1:])]
        self.b = [np.zeros(b) for b in arch[1:]]
        self.vW = [np.zeros_like(w) for w in self.W]; self.vb = [np.zeros_like(b) for b in self.b]

    def forward(self, X):
        hs = [X]
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W.T + b
            hs.append(np.maximum(z, 0) if l < len(self.W) - 1 else z)
        return hs

    def grad(self, X, y, clip=None):
        hs = self.forward(X)
        z = hs[-1]; z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        d = p.copy(); d[np.arange(len(y)), y] -= 1
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for l in range(len(self.W) - 1, -1, -1):
            gW[l] = d.T @ hs[l]; gb[l] = d.sum(0)
            if l:
                d = (d @ self.W[l]) * (hs[l] > 0)
        return gW, gb

    def step(self, gW, gb, n, lr=0.05, mom=0.9):
        for l in range(len(self.W)):
            self.vW[l] = mom * self.vW[l] - lr * gW[l] / n; self.W[l] += self.vW[l]
            self.vb[l] = mom * self.vb[l] - lr * gb[l] / n; self.b[l] += self.vb[l]

    def predict(self, X):
        return self.forward(X)[-1].argmax(1)


def float_fedavg(d, arch, seed, rounds, b=B, lr=0.05):
    net = FloatMLP(arch, seed)
    sites = chunks(d['plants'], b)
    for t in range(rounds):
        GW = [np.zeros_like(w) for w in net.W]; Gb = [np.zeros_like(x) for x in net.b]; n = 0
        for ch in sites:
            X, y = ch[t % len(ch)]
            gW, gb = net.grad(X / 255.0, y)
            for l in range(len(GW)):
                GW[l] += gW[l]; Gb[l] += gb[l]
            n += len(y)
        net.step(GW, Gb, n, lr=lr)
    return net.predict(d['X_test'] / 255.0)


# ---------------------------------------------------------------- privacy
_EPS_CACHE = os.path.join(RESULTS, 'eps_cache.json')


def epsilon(cfg, n_uses, H_min=H_MIN, delta=1e-5):
    """Rigorous epsilon of one record used n_uses times with H_min honest plants adding noise (cached on disk)."""
    os.makedirs(RESULTS, exist_ok=True)
    key = '%s|%d|%d|%g' % (repr(cfg), n_uses, H_min, delta)
    cache = json.load(open(_EPS_CACHE)) if os.path.exists(_EPS_CACHE) else {}
    if key not in cache:
        a = dp.account(cfg, H_min=H_min, n_uses=n_uses, delta=delta)
        cache[key] = dict(eps=a['eps'], eps_gauss=a['eps_gauss'], alpha=a['alpha'], z=a['z'], sigma=a['sigma'], Delta=a['Delta'])
        json.dump(cache, open(_EPS_CACHE, 'w'), indent=0)
    return cache[key]


def gaussian_sigma_for(eps, Delta, n_uses, delta=1e-5):
    """Smallest sigma of a Gaussian mechanism (standard RDP accountant) reaching (eps, delta) over n_uses releases."""
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 256))

    def eps_of(sig):
        return min(dp.rdp_to_dp(n_uses * a * Delta * Delta / (2 * sig * sig), a, delta) for a in alphas)
    lo, hi = 1e-3, 1e9
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if eps_of(mid) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def trusted_noise(sigma, d):
    """Rounded Gaussian noise added once by a trusted aggregator (the reference that our system avoids needing)."""
    return lambda rng: np.rint(rng.normal(0, sigma, d)).astype(np.int64)


# ---------------------------------------------------------------- running many independent runs on all cores
def workers():
    return max(1, int(os.environ.get('FEDPROOF_WORKERS', os.cpu_count() or 1)))


def run_jobs(store, jobs, fn, name):
    """Runs fn(job) -> (key, row) for every job whose key (job[0]) is not yet in store, on all cores (FEDPROOF_WORKERS to
    change). Rows are written by this process as they finish, so an interrupted run resumes where it stopped."""
    todo = [j for j in jobs if not store.done(j[0])]
    if not todo:
        return
    pool = None
    if workers() > 1 and len(todo) > 1:
        import multiprocessing as mp
        pool = mp.get_context('fork').Pool(min(workers(), len(todo)))
        it = pool.imap_unordered(fn, todo)
    else:
        it = map(fn, todo)
    try:
        for n, (key, row) in enumerate(it, 1):
            if row is None:
                continue
            store.add(key, row)
            extra = ' '.join('%s %.4g' % (k, row[k]) for k in ('eps', 'macro_f1', 'attack_success') if isinstance(row.get(k), (int, float)))
            print('%s [%d/%d] %-58s %s' % (name, n, len(todo), key, extra), flush=True)
    finally:
        if pool is not None:
            pool.close(); pool.join()


def epsilons(items):
    """epsilon() for many (cfg, n_uses, H_min) at once, the missing ones computed on all cores, then cached."""
    os.makedirs(RESULTS, exist_ok=True)
    cache = json.load(open(_EPS_CACHE)) if os.path.exists(_EPS_CACHE) else {}
    keyf = lambda cfg, u, H: '%s|%d|%d|%g' % (repr(cfg), u, H, 1e-5)
    miss = list({keyf(*it): it for it in items if keyf(*it) not in cache}.values())
    if miss:
        if workers() > 1 and len(miss) > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(min(workers(), len(miss))) as pool:
                res = pool.map(_eps_one, miss)
        else:
            res = list(map(_eps_one, miss))
        for it, a in zip(miss, res):
            cache[keyf(*it)] = a
        json.dump(cache, open(_EPS_CACHE, 'w'), indent=0)
    return [cache[keyf(*it)] for it in items]


def _eps_one(it):
    cfg, u, H = it
    a = dp.account(cfg, H_min=H, n_uses=u, delta=1e-5)
    return dict(eps=a['eps'], eps_gauss=a['eps_gauss'], alpha=a['alpha'], z=a['z'], sigma=a['sigma'], Delta=a['Delta'])
