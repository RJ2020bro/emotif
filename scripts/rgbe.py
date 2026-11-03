"""Minimal Radiance .hdr (RGBE) reader: header, new-style run-length encoded scanlines. Returns float32 (H, W, 3)."""
import numpy as np


def read_hdr(path):
    with open(path, "rb") as fh:
        data = fh.read()
    i = 0
    while True:                               # header lines until an empty line
        j = data.index(b"\n", i)
        line = data[i:j]
        i = j + 1
        if line == b"":
            break
        if line.startswith(b"FORMAT=") and b"32-bit_rle_rgbe" not in line:
            raise ValueError("only 32-bit_rle_rgbe is supported")
    j = data.index(b"\n", i)
    dims = data[i:j].split()
    i = j + 1
    if dims[0] != b"-Y" or dims[2] != b"+X":
        raise ValueError(f"unsupported orientation {dims}")
    H, W = int(dims[1]), int(dims[3])
    out = np.zeros((H, W, 4), np.uint8)
    buf = memoryview(data)
    for y in range(H):
        if buf[i] != 2 or buf[i + 1] != 2 or (buf[i + 2] << 8 | buf[i + 3]) != W:
            raise ValueError("expected new-style RLE scanlines")
        i += 4
        for c in range(4):
            x = 0
            while x < W:
                n = buf[i]
                i += 1
                if n > 128:
                    n -= 128
                    out[y, x:x + n, c] = buf[i]
                    i += 1
                else:
                    out[y, x:x + n, c] = np.frombuffer(buf[i:i + n], np.uint8)
                    i += n
                x += n
    e = out[..., 3].astype(np.float32)
    scale = np.where(out[..., 3] > 0, np.ldexp(1.0, (e - 136).astype(np.int32)), 0.0)
    return out[..., :3].astype(np.float32) * scale[..., None]
