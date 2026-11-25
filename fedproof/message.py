"""A site's proof-carrying round message, and its public verification."""
import numpy as np
from . import field as F
from . import ligero as L
from . import pq
from . import step as ST
from .hashing import Transcript, H


class Public:
    """Everything public that a site's round statement depends on."""

    def __init__(self, cfg, prm, shr, weights, beta, ctx_bytes, key_present=None):
        self.cfg, self.prm, self.shr, self.weights, self.beta, self.ctx = cfg, prm, shr, weights, beta, ctx_bytes
        self.G = prm.sigma
        # which nodes hold a valid key this round; a node without one must receive an empty ciphertext
        self.key_present = [True] * shr.K if key_present is None else list(key_present)


def _aad(pub, node):
    return H(pub.ctx) + node.to_bytes(4, 'little')


def _transcript(pub, roots, ct_hash):
    T = Transcript(b'sfl-site-proof')
    T.absorb('ctx', pub.ctx)
    T.absorb('cfg', repr(pub.cfg))
    T.absorb('sharing', pub.shr.ident())
    for name, r in roots:
        T.absorb('root-' + name, r)
    T.absorb('ciphertexts', ct_hash)
    return T


def site_prove(pub, wit, d0, nz, node_eks, rng=None, hook=None, strict=True):
    """Returns (public message, private state). d0, nz: the Committed data and noise blocks."""
    cfg, prm, shr, G = pub.cfg, pub.prm, pub.shr, pub.G
    cs = ST.new_cs(prm.ell, True, rng)
    cs.lenient = hook is not None
    ctx = ST.build_phase1(cs, cfg, shr, G, pub.weights, pub.beta, wit, hook)
    d1 = L.commit_block(cs.message_matrix('D1'), prm, rng)
    mb = L.commit_block(cs.message_matrix('M'), prm, rng)
    cts, seeds = [], []
    for m in range(shr.K):
        if node_eks[m] is None:                    # node without a valid key this round: removed, receives nothing
            cts.append(b''); seeds.append(None)
            continue
        blob, seed = pq.seal(node_eks[m], np.ascontiguousarray(ctx['shares'][m]).tobytes(), _aad(pub, m),
                             seed=None if rng is None else rng.read(32))
        cts.append(blob); seeds.append(seed)
    T = _transcript(pub, [('d0', d0.root), ('nz', nz.root), ('d1', d1.root), ('m', mb.root)], H(b''.join(cts)))
    ch = ST.derive_challenges(T, cfg, shr, G)
    digests = ST.build_phase2(cs, cfg, shr, G, ctx, ch)
    ab = L.commit_block(cs.message_matrix('A'), prm, rng, with_masks=True)
    T.absorb('root-a', ab.root)
    T.absorb('digests', digests)
    sysd = cs.system()
    proof = L.prove(prm, [d0, nz, d1, mb, ab], sysd, T, strict=strict)
    msg = dict(root_d1=d1.root, root_m=mb.root, root_a=ab.root, cts=cts, digests=digests, proof=proof)
    priv = dict(d1=d1, seeds=seeds, shares=ctx['shares'], u=F.to_signed(cs.get(ctx['u'])), cs=cs, sysd=sysd, gamma=ch['gamma'])
    return msg, priv


def is_digest(x):
    return isinstance(x, (bytes, bytearray)) and len(x) == 48


def site_verify(pub, root_d0, root_nz, msg):
    """Public verification of a site's message. Returns (ok, gamma) where gamma is what nodes need to check shares."""
    try:
        cfg, prm, shr, G = pub.cfg, pub.prm, pub.shr, pub.G
        if not isinstance(msg, dict) or not all(is_digest(r) for r in (root_d0, root_nz, msg['root_d1'], msg['root_m'], msg['root_a'])):
            return False, None
        dg = msg['digests']
        if not (isinstance(dg, np.ndarray) and dg.dtype == np.uint64 and dg.shape == (shr.K, G) and (dg < np.uint64(F.P)).all()):
            return False, None
        if len(msg['cts']) != shr.K or any(not isinstance(c, (bytes, bytearray)) for c in msg['cts']):
            return False, None
        want = pq.KEM_CT + 8 * (shr.nblk + G) + 16
        if len(pub.key_present) != shr.K or any(len(c) != (want if kp_ else 0) for c, kp_ in zip(msg['cts'], pub.key_present)):
            return False, None
        cs = ST.new_cs(prm.ell, False)
        ctx = ST.build_phase1(cs, cfg, shr, G, pub.weights, pub.beta)
        T = _transcript(pub, [('d0', root_d0), ('nz', root_nz), ('d1', msg['root_d1']), ('m', msg['root_m'])],
                        H(b''.join(msg['cts'])))
        ch = ST.derive_challenges(T, cfg, shr, G)
        ST.build_phase2(cs, cfg, shr, G, ctx, ch, dg)
        T.absorb('root-a', msg['root_a'])
        T.absorb('digests', dg)
        sysd = cs.system()
        rows = sysd['block_rows']
        metas = [(root_d0, rows[0]), (root_nz, rows[1]), (msg['root_d1'], rows[2]), (msg['root_m'], rows[3]),
                 (msg['root_a'], rows[4] + L.MASKS_PER_REP * prm.sigma)]
        return bool(L.verify(prm, metas, sysd, T, msg['proof'])), ch['gamma']
    except Exception:              # any malformed input is a rejection, never a crash
        return False, None


def message_size(msg):
    return 3 * 48 + sum(len(c) for c in msg['cts']) + 8 * msg['digests'].size + L.proof_size(msg['proof'])
