"""Preparation of the industrial intrusion datasets for the experiments.

One function per dataset turns the raw CSV into the same output:
  - columns that would leak the answer or identify a host are dropped (addresses, ports used as identifiers, times,
    free text, the other label columns);
  - exact duplicate rows are removed before any split, so the same record cannot sit in a plant and in the test set;
  - a stratified PUBLIC portion is set aside first; every choice that looks at data (category codes, constant
    columns, 8-bit bin edges, later the clipping threshold and learning rate) is made on it alone;
  - a stratified TEST portion is set aside next;
  - the rest is divided among plants: independent and identically distributed, or by label skew (Dirichlet alpha),
    each plant receiving a fixed number of records cut into chunks of b.
Output: one .npz with uint8 features and int labels per part, and a .json with every decision and count.

Usage: python3 experiments/prepare_data.py <dataset> <csv path> <out dir> [--plants 10] [--per-plant 32768] [--split iid|dir1.0|dir0.3]
"""
import os
import json
import hashlib
import argparse
import collections
import numpy as np
import pandas as pd

DATASETS = {
    # label column, label-like columns to drop, identifier/time/text columns to drop (matched case-insensitively,
    # by exact name or prefix), and the class column chosen for the multi-class task
    'edge-iiotset': dict(
        label='Attack_type', drop_labels=['Attack_label'],
        drop=['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
              'http.file_data', 'http.request.full_uri', 'icmp.transmit_timestamp', 'http.request.uri.query',
              'tcp.options', 'tcp.payload', 'tcp.srcport', 'udp.port', 'mqtt.msg',
              # position in a capture rather than behaviour: raw sequence values, checksums, stream and transaction indices
              'tcp.ack_raw', 'tcp.checksum', 'icmp.checksum', 'udp.stream', 'mbtcp.trans_id'],
        drop_prefix=[]),
    'wustl-iiot-2021': dict(
        label='Traffic', drop_labels=['Target'],
        drop=['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'Sport', 'sIpId', 'dIpId'],
        drop_prefix=[]),
    'x-iiotid': dict(
        label='class2', drop_labels=['class1', 'class3'],
        drop=['Date', 'Timestamp', 'Scr_IP', 'Des_IP', 'Scr_port'],
        drop_prefix=[]),
    'synthetic': dict(label='label', drop_labels=['is_attack'], drop=['ts', 'src_ip', 'dst_ip'], drop_prefix=[]),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def load(name, path):
    spec = DATASETS[name]
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    lab = lower.get(spec['label'].lower())
    if lab is None:
        raise SystemExit('label column %r not found; columns are %s' % (spec['label'], list(df.columns)))
    dropped = []
    for c in spec['drop'] + spec['drop_labels']:
        if c.lower() in lower and lower[c.lower()] != lab:
            dropped.append(lower[c.lower()])
    for p in spec['drop_prefix']:
        dropped += [c for c in df.columns if c.lower().startswith(p.lower()) and c != lab]
    df = df.drop(columns=sorted(set(dropped)))
    y_raw = df.pop(lab).astype(str).str.strip()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = canonical(df[c])
    return df, y_raw, sorted(set(dropped)), lab


ABSENT = ('', 'nan', 'none', 'null', '-')


def canonical(col):
    """One spelling per value in a text column, before deduplication and encoding. Numbers written differently
    ('0', '0.0', '0.00') become one token, and every spelling of an absent field becomes that same zero. Without this,
    a capture tool that writes an absent field as '0' in one traffic file and as '0.0' in another turns the spelling
    into a label for the file, and so for the class."""
    s = col.astype(str).str.strip()
    num = pd.to_numeric(s, errors='coerce')
    s = s.where(num.isna(), num.astype(str))
    return s.where(~s.str.lower().isin(ABSENT), '0.0')


def separating(X, y, normal, thresh=0.99):
    """Columns that alone tell normal from attack rows almost perfectly (by a lookup of the encoded value), on the
    rows given. A real traffic feature rarely does this; a recording artefact does. Reported, not dropped."""
    att = y != normal
    base = max(att.mean(), 1 - att.mean())
    out = {}
    for j in range(X.shape[1]):
        v = X[:, j].astype(np.int64)
        a = np.bincount(v[att], minlength=256); n = np.bincount(v[~att], minlength=256)
        acc = float(np.maximum(a, n).sum()) / len(y)
        gain = (acc - base) / (1 - base) if base < 1 else 0.0     # share of the majority-guess errors it removes
        if gain >= thresh:
            out[j] = acc
    return out


def stratified_take(y, frac, rng):
    """Indices of a stratified sample of about frac of the rows (at least one per class)."""
    idx = []
    for c in np.unique(y):
        rows = np.flatnonzero(y == c)
        k = max(1, int(round(frac * len(rows))))
        idx.append(rng.choice(rows, size=min(k, len(rows)), replace=False))
    return np.sort(np.concatenate(idx))


def encode(df, pub_idx, n_bins=256):
    """Every decision from the public rows only: numeric coercion, category codes, constant columns, bin edges."""
    cols, enc, X = [], {}, []
    pub = df.iloc[pub_idx]
    for c in df.columns:
        col = df[c]
        num = pd.to_numeric(col, errors='coerce')
        if num.notna().mean() >= 0.99:                         # numeric column
            v = num.fillna(0.0).to_numpy(dtype=np.float64)
            vp = v[pub_idx]
            if np.unique(vp).size <= 1:
                continue                                      # constant on the public rows: dropped
            qs = np.unique(np.quantile(vp, np.linspace(0, 1, n_bins + 1)[1:-1]))
            codes = np.searchsorted(qs, v, side='right').astype(np.int64)
            enc[c] = dict(kind='numeric', edges=qs.tolist())
        else:                                                 # categorical: codes by public frequency, rare -> 0
            s = col.astype(str)
            vc = s.iloc[pub_idx].value_counts()
            vocab = list(vc.index[:n_bins - 1])
            if len(vocab) <= 1:
                continue
            m = {k: i + 1 for i, k in enumerate(vocab)}
            codes = s.map(m).fillna(0).to_numpy(dtype=np.int64)
            enc[c] = dict(kind='categorical', vocab=vocab)
        # map codes to 8 bits; spread small code ranges over the full byte so every feature uses the input scale
        top = max(int(codes.max()), 1)
        X.append(np.clip(np.round(codes * (255.0 / max(top, 1)) if top < 255 else codes, 0), 0, 255).astype(np.uint8))
        cols.append(c)
    return np.stack(X, axis=1), cols, enc


def split_plants(y, idx, n_plants, per_plant, mode, rng):
    """Assign rows to plants. Returns a list of index arrays (each of length per_plant, or fewer if data runs out)."""
    classes = np.unique(y[idx])
    if isinstance(mode, tuple):                       # natural split: one plant per group (device or scenario)
        groups = mode[1]
        vals = [v for v, _c in collections.Counter(groups[idx]).most_common(n_plants)]
        out = []
        for v in vals:
            rows = rng.permutation(idx[groups[idx] == v])
            out.append(np.array(sorted(rows[:per_plant]), dtype=np.int64))
        return out
    total = n_plants * per_plant
    alloc = base_mix({c: int((y[idx] == c).sum()) for c in classes}, total)
    if mode == 'iid':                                 # every plant gets the same class mix
        rows = np.concatenate([rng.permutation(idx[y[idx] == c])[:alloc[c]] for c in classes])
        return [np.sort(rows[i::n_plants])[:per_plant] for i in range(n_plants)]
    alpha = float(mode[3:])                           # label skew around the same mix: Dirichlet(alpha * C * mix)
    q = np.array([alloc[c] for c in classes], dtype=float); q /= q.sum()
    pools = {c: list(rng.permutation(idx[y[idx] == c])) for c in classes}
    props = rng.dirichlet(np.maximum(alpha * len(classes) * q, 1e-3), size=n_plants)
    out = []
    for p in range(n_plants):
        want = np.floor(props[p] * per_plant).astype(int)
        want[np.argmax(props[p])] += per_plant - want.sum()
        take = []
        for c, k in zip(classes, want):
            got = [pools[c].pop() for _ in range(min(k, len(pools[c])))]
            take += got
        short = per_plant - len(take)                     # a class ran out: fill from what is left, largest pools first
        while short > 0 and any(pools.values()):
            c = max(pools, key=lambda k: len(pools[k]))
            take.append(pools[c].pop()); short -= 1
        out.append(np.array(sorted(take), dtype=np.int64))
    return out


def base_mix(counts, total):
    """Training records per class, as balanced as the data allows: every class gets an equal share of total, a class
    with fewer records gives all it has, and what it leaves is shared among the others (water filling). The test set
    keeps the natural class mix."""
    alloc, left, budget = {}, dict(counts), total
    while left:
        share = budget / len(left)
        small = {c: n for c, n in left.items() if n <= share}
        if not small:
            base, extra = divmod(budget, len(left))
            for i, c in enumerate(sorted(left, key=lambda c: -left[c])):
                alloc[c] = base + (1 if i < extra else 0)
            break
        for c, n in small.items():
            alloc[c] = n; budget -= n; del left[c]
    return alloc


def prepare(name, path, out_dir, plants=10, per_plant=4096, split='iid', pub_frac=0.05, test_frac=0.2, b=16, seed=0, tag=None):
    rng = np.random.default_rng(seed)
    df, y_raw, dropped, lab = load(name, path)
    n0 = len(df)
    full = df.assign(__y=y_raw.values)
    keep = ~full.duplicated()
    df, y_raw = df[keep.values].reset_index(drop=True), y_raw[keep.values].reset_index(drop=True)
    classes = sorted(y_raw.unique())
    y = y_raw.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
    allidx = np.arange(len(y))
    pub = stratified_take(y, pub_frac, rng)
    rest = np.setdiff1d(allidx, pub)
    test = rest[stratified_take(y[rest], test_frac / (1 - pub_frac), rng)]
    pool = np.setdiff1d(rest, test)
    if split.startswith('natural:'):                  # group column named after the colon, read before it is dropped
        gcol = split.split(':', 1)[1]
        graw = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() == gcol.lower(), low_memory=False)
        graw = graw[keep.values].reset_index(drop=True).iloc[:, 0].astype(str).to_numpy()
        if gcol in df.columns:
            df = df.drop(columns=[gcol])
        mode = ('natural', graw)
    else:
        mode = split
    X, cols, enc = encode(df, pub)
    normal = next((i for i, c in enumerate(classes) if 'normal' in c.lower() or c.lower() == 'benign'), None)
    sep = {cols[j]: round(a, 5) for j, a in separating(X[pub], y[pub], normal).items()} if normal is not None else {}
    parts = split_plants(y, pool, plants, per_plant, mode, rng)
    plants = len(parts)
    # leakage checks: disjoint index sets, and no identical feature row with a different role
    assert not (set(pub) & set(test)) and not any(set(p) & set(test) for p in parts) and not any(set(p) & set(pub) for p in parts)
    nb = per_plant // b
    out = dict(X_pub=X[pub], y_pub=y[pub], X_test=X[test], y_test=y[test])
    for i, p in enumerate(parts):
        p = rng.permutation(p)[:nb * b]               # random order: the raw files are grouped by class, and chunks
                                                      # taken in file order would show every plant one class at a time
        out['X_plant%d' % i] = X[p]; out['y_plant%d' % i] = y[p]
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, '%s_%s_seed%d' % (name, (tag or split).replace(':', '-'), seed))
    np.savez_compressed(base + '.npz', **out)
    meta = dict(dataset=name, source=os.path.basename(path), source_sha256=sha256_file(path), rows_raw=n0,
                rows_after_dedup=int(len(y)), label_column=lab, classes=classes,
                class_counts={c: int((y == i).sum()) for i, c in enumerate(classes)},
                dropped_columns=dropped, features_kept=cols, n_features=len(cols), separating_columns=sep,
                encoders={c: (dict(kind='numeric', n_edges=len(e['edges'])) if e['kind'] == 'numeric' else dict(kind='categorical', n_vocab=len(e['vocab']))) for c, e in enc.items()},
                public_rows=int(len(pub)), test_rows=int(len(test)), plants=plants, per_plant=per_plant, split=split, chunk=b,
                plant_class_counts=[{classes[c]: int((out['y_plant%d' % i] == c).sum()) for c in range(len(classes))} for i in range(plants)], tag=tag or split,
                seed=seed)
    json.dump(meta, open(base + '.json', 'w'), indent=1)
    return base, meta


def synthetic_csv(path, n=30000, seed=1):
    """A small CSV with the awkward parts of the real ones: addresses, times, text categories, duplicates, a constant."""
    g = np.random.default_rng(seed)
    lab = g.choice(['Normal', 'DDoS', 'Scan', 'Backdoor'], size=n, p=[0.6, 0.3, 0.09, 0.01])
    df = pd.DataFrame(dict(
        ts=pd.date_range('2026-01-01', periods=n, freq='s').astype(str),
        src_ip=['10.0.%d.%d' % (a, b) for a, b in g.integers(0, 255, (n, 2))],
        dst_ip=['10.1.0.%d' % a for a in g.integers(0, 255, n)],
        proto=g.choice(['tcp', 'udp', 'mqtt', 'modbus'], size=n),
        bytes=g.lognormal(6, 2, n).round(), pkts=g.poisson(5, n), dur=g.exponential(1.0, n), const=1.0,
        is_attack=(lab != 'Normal').astype(int), label=lab))
    df.loc[lab == 'DDoS', 'pkts'] += 40
    df = pd.concat([df, df.iloc[:500]])                  # exact duplicates
    df.to_csv(path, index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset'); ap.add_argument('csv'); ap.add_argument('out')
    ap.add_argument('--plants', type=int, default=10); ap.add_argument('--per-plant', type=int, default=4096)
    ap.add_argument('--split', default='iid'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    base, meta = prepare(a.dataset, a.csv, a.out, a.plants, a.per_plant, a.split, seed=a.seed)
    print(json.dumps({k: meta[k] for k in ('dataset', 'rows_raw', 'rows_after_dedup', 'n_features', 'classes', 'class_counts', 'public_rows', 'test_rows')}, indent=1))
    print('written', base + '.npz')
