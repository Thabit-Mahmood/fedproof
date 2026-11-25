"""Wire format for bulletin entries and requests: a typed, length-prefixed, invertible encoding.

Used by the network harness so that every byte counted is a byte that would cross a link. Arrays carry dtype and
shape; decoding refuses unknown tags, oversize lengths and trailing bytes (it never executes anything, unlike pickle).
"""
import struct
import numpy as np

MAX_LEN = 1 << 30
_DT = {b'u8': np.uint64, b'i8': np.int64, b'u1': np.uint8, b'i1': np.int8, b'u4': np.uint32, b'i4': np.int32}
_DTR = {np.dtype(v).str[1:].encode(): k for k, v in _DT.items()}


def encode(v):
    out = []
    _enc(v, out)
    return b''.join(out)


def _enc(v, out):
    if v is None:
        out.append(b'N')
    elif isinstance(v, bool):
        out.append(b'T' if v else b'F')
    elif isinstance(v, (int, np.integer)):
        v = int(v)
        n = (v.bit_length() + 8) // 8
        out.append(b'I' + struct.pack('<H', n) + v.to_bytes(n, 'little', signed=True))
    elif isinstance(v, float):
        out.append(b'D' + struct.pack('<d', v))
    elif isinstance(v, (bytes, bytearray)):
        out.append(b'Y' + struct.pack('<I', len(v)) + bytes(v))
    elif isinstance(v, str):
        e = v.encode()
        out.append(b'S' + struct.pack('<I', len(e)) + e)
    elif isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        code = _DTR.get(a.dtype.str[1:].encode())
        if code is None:
            raise TypeError('array dtype %s not on the wire' % a.dtype)
        a = a.astype(a.dtype.newbyteorder('<'), copy=False)
        out.append(b'A' + code + struct.pack('<B', a.ndim) + b''.join(struct.pack('<I', s) for s in a.shape)
                   + struct.pack('<I', a.nbytes) + a.tobytes())
    elif isinstance(v, (list, tuple)):
        out.append((b'L' if isinstance(v, list) else b'U') + struct.pack('<I', len(v)))
        for x in v:
            _enc(x, out)
    elif isinstance(v, dict):
        out.append(b'M' + struct.pack('<I', len(v)))
        for k, x in v.items():
            _enc(k, out); _enc(x, out)
    else:
        raise TypeError('type %r not on the wire' % type(v))


MAX_ELEMENTS = 1 << 22            # decoded values per message (memory amplification bound)


def decode(b):
    """Strict decoding: the input must be exactly the canonical encoding of the value it decodes to (minimal integer
    lengths, no duplicate or colliding dictionary keys), so equal values always come from equal bytes."""
    budget = [MAX_ELEMENTS]
    v, i = _dec(memoryview(b), 0, 0, budget)
    if i != len(b):
        raise ValueError('trailing bytes')
    if encode(v) != bytes(b):
        raise ValueError('not the canonical encoding')
    return v


def _dec(b, i, depth, budget):
    if depth > 32:
        raise ValueError('nesting too deep')
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError('too many elements')
    tag = bytes(b[i:i + 1]); i += 1
    if tag == b'N':
        return None, i
    if tag in (b'T', b'F'):
        return tag == b'T', i
    if tag == b'I':
        n, = struct.unpack_from('<H', b, i); i += 2
        return int.from_bytes(b[i:i + n], 'little', signed=True), i + n
    if tag == b'D':
        return struct.unpack_from('<d', b, i)[0], i + 8
    if tag in (b'Y', b'S'):
        n, = struct.unpack_from('<I', b, i); i += 4
        if n > MAX_LEN or i + n > len(b):
            raise ValueError('bad length')
        raw = bytes(b[i:i + n])
        return (raw if tag == b'Y' else raw.decode()), i + n
    if tag == b'A':
        code = bytes(b[i:i + 2]); i += 2
        if code not in _DT:
            raise ValueError('bad dtype')
        nd, = struct.unpack_from('<B', b, i); i += 1
        shape = struct.unpack_from('<%dI' % nd, b, i); i += 4 * nd
        n, = struct.unpack_from('<I', b, i); i += 4
        dt = np.dtype(_DT[code]).newbyteorder('<')
        if n != int(np.prod(shape, dtype=np.int64)) * dt.itemsize or i + n > len(b):
            raise ValueError('bad array length')
        a = np.frombuffer(bytes(b[i:i + n]), dtype=dt).reshape(shape).astype(_DT[code])
        return a, i + n
    if tag in (b'L', b'U'):
        n, = struct.unpack_from('<I', b, i); i += 4
        if n > MAX_LEN:
            raise ValueError('bad count')
        xs = []
        for _ in range(n):
            x, i = _dec(b, i, depth + 1, budget); xs.append(x)
        return (xs if tag == b'L' else tuple(xs)), i
    if tag == b'M':
        n, = struct.unpack_from('<I', b, i); i += 4
        if n > MAX_LEN:
            raise ValueError('bad count')
        d = {}
        for _ in range(n):
            k, i = _dec(b, i, depth + 1, budget)
            x, i = _dec(b, i, depth + 1, budget)
            d[k] = x
        return d, i
    raise ValueError('unknown tag %r' % tag)
