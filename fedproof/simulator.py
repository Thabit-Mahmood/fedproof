"""Plain (proof-free) simulation of the federated protocol with the exact integer semantics.

Used for utility, privacy-utility and robustness experiments. The proof-carrying pipeline computes bit-identical
vectors (tested separately), so results here are results for the full system.
"""
import numpy as np
from . import model as M


def load_digits_int(seed=0, test=297):
    from sklearn.datasets import load_digits
    d = load_digits()
    X = np.minimum(d.data.astype(np.int64), 15)
    y = d.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    te, tr = idx[:test], idx[test:]
    return X[tr], y[tr], X[te], y[te]


def partition(y, N, rng, alpha=None):
    """IID (alpha None) or Dirichlet label-skew partition into N index lists."""
    if alpha is None:
        idx = rng.permutation(len(y))
        return [np.sort(part) for part in np.array_split(idx, N)]
    C = int(y.max()) + 1
    out = [[] for _ in range(N)]
    for c in range(C):
        ids = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet(alpha * np.ones(N))
        cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
        for k, part in enumerate(np.split(ids, cuts)):
            out[k].extend(part.tolist())
    return [np.array(sorted(a), dtype=np.int64) for a in out]


def make_chunks(X, y, parts, b, rng):
    """Each site's data cut into chunks of exactly b records (fixed before training; remainder dropped)."""
    sites = []
    for p in parts:
        p = rng.permutation(p)
        nchunk = len(p) // b
        sites.append([(X[p[i * b:(i + 1) * b]], y[p[i * b:(i + 1) * b]]) for i in range(nchunk)])
    return sites


def site_noise(cfg, rng):
    return rng.integers(0, 1 << cfg.r, size=(cfg.J, cfg.d), dtype=np.int64).sum(0)


def run_plain(cfg, sites, rounds, Xte, yte, seed=0, noise=True, clip=True, eval_every=10, transform=None,
              craft=None, corrupt=(), log=None, metric=None, central_noise=None, start_state=None, on_round=None,
              aggregate=None):
    """Plain simulation of training.

    transform(site, round, X, y, weights) -> (X, y): a corrupted site alters its data (the only attack the full system allows).
    craft(site, round, weights, honest_tq_list) -> vector: a corrupted site submits an arbitrary vector instead (possible
        only in a system that checks a norm bound but not the training; used as a baseline, impossible here).
    metric(weights) -> float: extra evaluation (for example backdoor success rate).
    central_noise(rng) -> vector: a trusted aggregator adds this once to the sum instead of every site adding its own
        noise (the trusted-aggregator reference; not part of the system).
    on_round(t, state): called after every round (for example to keep models for later attacks).
    aggregate(list of vectors) -> vector: a server that sees every noised update and combines them by a robust rule
        (Krum, median, trimmed mean); returns a vector on the scale of the sum. A baseline only: it needs the
        individual updates, which our system never reveals.
    """
    rng = np.random.default_rng(seed)
    state = M.init_model(cfg) if start_state is None else start_state
    hist = []
    mean_noise = cfg.J * ((1 << cfg.r) - 1) // 2
    for t in range(rounds):
        W = M.proof_weights(cfg, state)
        tqs = {}
        for i, chunks in enumerate(sites):
            if not chunks:
                continue
            Xc, yc = chunks[t % len(chunks)]
            if transform is not None and i in corrupt:
                Xc, yc = transform(i, t, Xc, yc, W)
            tqs[i] = M.plain_step(cfg, W, Xc, yc, clip=clip)['tq']
        if craft is not None:
            hon = [tqs[i] for i in tqs if i not in corrupt]
            for i in corrupt:
                tqs[i] = craft(i, t, W, hon)
        agg = np.zeros(cfg.d, dtype=np.int64)
        if aggregate is not None:
            agg = aggregate([tqs[i] + (site_noise(cfg, rng) if noise else mean_noise) for i in sorted(tqs)])
        else:
            for i in tqs:
                agg += tqs[i] + (site_noise(cfg, rng) if (noise and central_noise is None) else mean_noise)
        if central_noise is not None:
            agg += central_noise(rng)
        state = M.apply_update(cfg, state, agg, len(tqs))
        if on_round is not None:
            on_round(t + 1, state)
        if (t + 1) % eval_every == 0 or t == rounds - 1:
            Wn = M.proof_weights(cfg, state)
            acc = float((M.predict(cfg, Wn, Xte) == yte).mean())
            hist.append((t + 1, acc) if metric is None else (t + 1, acc, metric(Wn)))
            if log:
                log('round %d acc %.4f' % (t + 1, acc))
    return state, hist


# ---------------------------------------------------------------- robust aggregation rules (baselines)
def agg_median(vecs, f=0):
    V = np.stack(vecs).astype(np.float64)
    return np.rint(np.median(V, 0) * len(vecs)).astype(np.int64)


def agg_trimmed_mean(vecs, f):
    V = np.sort(np.stack(vecs).astype(np.float64), 0)
    return np.rint(V[f:len(vecs) - f].mean(0) * len(vecs)).astype(np.int64)


def agg_krum(vecs, f):
    """Krum: the update with the smallest sum of squared distances to its n - f - 2 nearest others, scaled to n."""
    V = np.stack(vecs).astype(np.float64)
    n = len(V); k = max(1, n - f - 2)
    D = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
    scores = [np.sort(np.delete(D[i], i))[:k].sum() for i in range(n)]
    return np.rint(V[int(np.argmin(scores))] * n).astype(np.int64)
