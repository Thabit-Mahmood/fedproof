"""Hashing: SHA3-384 Merkle trees with salted leaves and batched openings; SHAKE256 Fiat-Shamir transcript."""
import hashlib
import numpy as np
from . import field as F

DIGEST = 48


def H(b):
    return hashlib.sha3_384(b).digest()


def leaf_hash(salt, col_bytes):
    return H(b'\x00' + salt + col_bytes)


def node_hash(left, right):
    return H(b'\x01' + left + right)


class MerkleTree:
    def __init__(self, leaves):
        n = len(leaves)
        assert n & (n - 1) == 0 and n >= 2
        self.levels = [list(leaves)]
        cur = self.levels[0]
        while len(cur) > 1:
            cur = [node_hash(cur[i], cur[i + 1]) for i in range(0, len(cur), 2)]
            self.levels.append(cur)
        self.root = cur[0]
        self.n = n

    def open(self, idxs):
        """Batched opening for sorted distinct leaf indices: sibling hashes in a fixed order."""
        known = sorted(set(int(i) for i in idxs))
        proof = []
        for lvl in range(len(self.levels) - 1):
            ks = set(known)
            nxt = []
            for i in known:
                if (i ^ 1) not in ks:
                    proof.append(self.levels[lvl][i ^ 1])
                if not nxt or nxt[-1] != i >> 1:
                    nxt.append(i >> 1)
            known = nxt
        return proof


def merkle_verify(root, n, idxs, leaves, proof):
    """Check that leaves (digests) sit at the given distinct indices under root."""
    if n < 2 or n & (n - 1) or len(idxs) != len(leaves) or len(set(int(i) for i in idxs)) != len(idxs):
        return False
    cur = {int(i): l for i, l in zip(idxs, leaves)}
    if any(i < 0 or i >= n for i in cur):
        return False
    it = iter(proof)
    size = n
    try:
        while size > 1:
            nxt = {}
            for i in sorted(cur):
                if i >> 1 in nxt:
                    continue
                sib = cur[i ^ 1] if (i ^ 1) in cur else next(it)
                if not isinstance(sib, (bytes, bytearray)) or len(sib) != DIGEST:
                    return False
                left, right = (cur[i], sib) if i % 2 == 0 else (sib, cur[i])
                nxt[i >> 1] = node_hash(left, right)
            cur = nxt
            size //= 2
    except StopIteration:
        return False
    if next(it, None) is not None:
        return False
    return cur.get(0) == root


class Transcript:
    """Fiat-Shamir transcript. Every absorbed item is length-prefixed and labelled; challenges ratchet the state."""

    def __init__(self, domain):
        self.state = H(b'sfl-transcript-v1' + domain)

    def absorb(self, label, data):
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data).tobytes()
        if isinstance(data, int):
            data = data.to_bytes(16, 'little', signed=True)
        if isinstance(data, str):
            data = data.encode()
        lab = label.encode()
        self.state = H(b'A' + self.state + len(lab).to_bytes(4, 'little') + lab + len(data).to_bytes(8, 'little') + H(data))

    def _squeeze(self, label, nbytes):
        lab = label.encode()
        out = hashlib.shake_256(b'C' + self.state + lab).digest(nbytes)
        self.state = H(b'R' + self.state + len(lab).to_bytes(4, 'little') + lab)
        return out

    def chal_bytes(self, label, nbytes):
        return self._squeeze(label, nbytes)

    def chal_field(self, label, count):
        """count uniform field elements (rejection sampling)."""
        seed = self._squeeze(label, 64)
        ctr = 0
        out = np.zeros(0, dtype=np.uint64)
        while len(out) < count:
            need = count - len(out)
            buf = hashlib.shake_256(seed + ctr.to_bytes(8, 'little')).digest(8 * (need + 64))
            v = np.frombuffer(buf, dtype='<u8')
            out = np.concatenate([out, v[v < np.uint64(F.P)]])
            ctr += 1
        return out[:count].copy()

    def chal_indices(self, label, t, n):
        """t distinct indices in [0, n), n a power of two."""
        assert n & (n - 1) == 0 and t <= n
        seed = self._squeeze(label, 64)
        ctr = 0
        seen = []
        seen_set = set()
        while len(seen) < t:
            buf = hashlib.shake_256(seed + ctr.to_bytes(8, 'little')).digest(4 * 4 * t)
            for v in np.frombuffer(buf, dtype='<u4'):
                i = int(v) & (n - 1)
                if i not in seen_set:
                    seen_set.add(i); seen.append(i)
                    if len(seen) == t:
                        break
            ctr += 1
        return np.array(sorted(seen), dtype=np.int64)

    def fork(self):
        t = Transcript.__new__(Transcript)
        t.state = self.state
        return t
