"""Post-quantum transport and signatures: ML-KEM-768 + AES-256-GCM (KEM-DEM), ML-DSA-65."""
import hashlib
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from kyber_py.ml_kem import ML_KEM_768
from dilithium_py.ml_dsa import ML_DSA_65

KEM_CT = 1088
KEM_EK = 1184
SIG_LEN = 3309
SIG_PK = 1952


def kem_keygen():
    return ML_KEM_768.keygen()          # (ek, dk)


def ek_valid(ek):
    """FIPS 203 input check on an ML-KEM-768 encapsulation key: type, length, and the modulus check
    (every 12-bit coefficient of the encoded vector t_hat is below q = 3329). Public and deterministic."""
    if type(ek) not in (bytes, bytearray) or len(ek) != KEM_EK:
        return False
    b = ek[:384 * 3]
    for i in range(0, len(b), 3):
        x0, x1, x2 = b[i], b[i + 1], b[i + 2]
        if (x0 | ((x1 & 0x0F) << 8)) >= 3329 or ((x1 >> 4) | (x2 << 4)) >= 3329:
            return False
    return True


def _dem_key(K, aad):
    return hashlib.sha3_256(b'sfl-dem' + K + hashlib.sha3_256(aad).digest()).digest()


def seal(ek, plaintext, aad, seed=None):
    """Encrypt to ek. Returns (ciphertext bytes, encapsulation seed). The seed re-derives the whole ciphertext."""
    seed = os.urandom(32) if seed is None else seed
    K, c = ML_KEM_768._encaps_internal(ek, seed)
    ct = AESGCM(_dem_key(K, aad)).encrypt(b'\x00' * 12, plaintext, aad)      # one-time key, fixed nonce
    return c + ct, seed


def open_(dk, blob, aad):
    """Decrypt. Returns plaintext or None when the ciphertext is malformed or fails authentication."""
    if len(blob) < KEM_CT + 16:
        return None
    try:
        K = ML_KEM_768.decaps(dk, blob[:KEM_CT])
        return AESGCM(_dem_key(K, aad)).decrypt(b'\x00' * 12, blob[KEM_CT:], aad)
    except (InvalidTag, ValueError):
        return None


def open_with_seed(ek, blob, aad, seed):
    """Public re-derivation used in complaints: returns plaintext if seed reproduces the KEM ciphertext, else None."""
    if len(blob) < KEM_CT + 16 or len(seed) != 32:
        return None
    try:
        K, c = ML_KEM_768._encaps_internal(ek, seed)
    except Exception:
        return None
    if c != blob[:KEM_CT]:
        return None
    try:
        return AESGCM(_dem_key(K, aad)).decrypt(b'\x00' * 12, blob[KEM_CT:], aad)
    except (InvalidTag, ValueError):
        return None


def sig_keygen():
    return ML_DSA_65.keygen()           # (pk, sk)


def sign(sk, msg):
    return ML_DSA_65.sign(sk, msg)


def verify(pk, msg, sig):
    try:
        return bool(ML_DSA_65.verify(pk, msg, sig))
    except Exception:
        return False


# ---------------------------------------------------------------- optional C backend (same standards, same bytes)
class _SigAPI:
    """One interface over both pqcrypto APIs: 0.x (generate_keypair, verify returns a bool) and 1.x (keygen, verify
    returns None or raises)."""

    def __init__(self, m):
        self._m = m
        self.generate_keypair = getattr(m, 'generate_keypair', None) or m.keygen
        self.sign = m.sign

    def verify(self, pk, msg, sig):
        try:
            r = self._m.verify(pk, msg, sig)
        except Exception:
            return False
        return True if r is None else bool(r)


class _KemAPI:
    def __init__(self, m):
        self.generate_keypair = getattr(m, 'generate_keypair', None) or m.keygen
        self.decrypt = getattr(m, 'decrypt', None) or m.decaps


def _fast_backend():
    """If the pqcrypto package (PQClean C code) is installed and FEDPROOF_FAST_PQ is not '0', use it for key generation,
    signing, verification and decapsulation. Encapsulation from a seed stays in kyber_py, because complaints re-derive
    ciphertexts from revealed seeds. The backend is used only if it interoperates byte for byte with the reference
    implementation in a self-test run at import (keys, signatures and shared secrets cross-checked both ways)."""
    if os.environ.get('FEDPROOF_FAST_PQ', '1') == '0':
        return None
    try:
        from pqcrypto.sign import ml_dsa_65 as _s
        from pqcrypto.kem import ml_kem_768 as _k
    except Exception:
        return None
    fsig, fkem = _SigAPI(_s), _KemAPI(_k)
    try:
        pk, sk = fsig.generate_keypair()
        m = b'sfl-interop'
        s1 = fsig.sign(sk, m)
        ok = ML_DSA_65.verify(pk, m, s1)
        pk2, sk2 = ML_DSA_65.keygen()
        s2 = ML_DSA_65.sign(sk2, m)
        ok = ok and fsig.verify(pk2, m, s2) and not fsig.verify(pk2, m + b'x', s2)
        ek, dk = fkem.generate_keypair()
        K, c = ML_KEM_768._encaps_internal(ek, b'\x07' * 32)
        ok = ok and fkem.decrypt(dk, c) == K
        ek2, dk2 = ML_KEM_768.keygen()
        K2, c2 = ML_KEM_768._encaps_internal(ek2, b'\x09' * 32)
        ok = ok and fkem.decrypt(dk2, c2) == K2 and ek_valid(ek)
        return (fsig, fkem) if ok else None
    except Exception:
        return None


_FAST = _fast_backend()
BACKEND = 'pqcrypto (PQClean C) + kyber_py seeded encapsulation' if _FAST else 'kyber_py + dilithium_py (pure Python)'
if _FAST:
    _fsig, _fkem = _FAST

    def kem_keygen():                   # noqa: F811  (same (ek, dk) formats as FIPS 203)
        return _fkem.generate_keypair()

    def sig_keygen():                   # noqa: F811
        return _fsig.generate_keypair()

    def sign(sk, msg):                  # noqa: F811
        return _fsig.sign(sk, msg)

    def verify(pk, msg, sig):           # noqa: F811
        try:
            return bool(_fsig.verify(pk, msg, sig))
        except Exception:
            return False

    def open_(dk, blob, aad):           # noqa: F811
        if len(blob) < KEM_CT + 16:
            return None
        try:
            K = _fkem.decrypt(dk, blob[:KEM_CT])
            return AESGCM(_dem_key(K, aad)).decrypt(b'\x00' * 12, blob[KEM_CT:], aad)
        except Exception:
            return None
