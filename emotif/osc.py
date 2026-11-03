"""Open Sound Control (OSC 1.0) over UDP: sends the decoded state to an external renderer such as Unreal Engine 5
(its built-in OSC plugin), TouchDesigner or Max. Messages, one datagram each:

  /emotif/state        f f f f   valence, arousal (each -1..1), confidence (0..1, -1 if unknown), signal quality (0..1)
  /emotif/hold         i         1 while the brain-only decoder holds its value (facial activity out of range)
  /emotif/<head>       f         each extra head of the model, e.g. /emotif/stress (-1..1)

Only float32, int32 and string arguments are needed, so the encoder is written out here (no dependency).
"""
import math
import re
import socket
import struct


def _pad(b):
    return b + b"\0" * (4 - len(b) % 4)


def message(address, *args):
    """One OSC message. Arguments: float -> 'f', bool/int -> 'i', str -> 's'."""
    if not address.startswith("/"):
        raise ValueError("an OSC address starts with '/'")
    tags, data = ",", b""
    for a in args:
        if isinstance(a, bool) or isinstance(a, int):
            tags += "i"
            data += struct.pack(">i", int(a))
        elif isinstance(a, float):
            tags += "f"
            data += struct.pack(">f", a)
        elif isinstance(a, str):
            tags += "s"
            data += _pad(a.encode("utf-8"))
        else:
            raise TypeError(f"unsupported OSC argument {a!r}")
    return _pad(address.encode("ascii")) + _pad(tags.encode("ascii")) + data


def parse(datagram):
    """(address, [args]) for a message made by `message` (used by the tests and the receiver example)."""
    def cstr(b, i):
        j = b.index(b"\0", i)
        s = b[i:j].decode("utf-8")
        return s, (j // 4 + 1) * 4
    addr, i = cstr(datagram, 0)
    tags, i = cstr(datagram, i)
    out = []
    for t in tags[1:]:
        if t == "f":
            out.append(struct.unpack(">f", datagram[i:i + 4])[0])
            i += 4
        elif t == "i":
            out.append(struct.unpack(">i", datagram[i:i + 4])[0])
            i += 4
        elif t == "s":
            s, i = cstr(datagram, i)
            out.append(s)
        else:
            raise ValueError(f"unsupported OSC type tag {t!r}")
    return addr, out


def parse_target(text):
    """'127.0.0.1:8001' or ':8001' or '8001' -> ('127.0.0.1', 8001)."""
    host, _, port = text.rpartition(":")
    port = int(port)
    if not 0 < port < 65536:
        raise ValueError(f"bad OSC port {port}")
    return (host or "127.0.0.1"), port


class OscSender:
    def __init__(self, host, port):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_state(self, v, a, conf=None, quality=1.0, hold=False, extra=None):
        clip = lambda x, lo, hi: max(lo, min(hi, float(x)))  # noqa: E731
        c = -1.0 if conf is None else clip(conf, 0, 1)
        self.sock.sendto(message("/emotif/state", clip(v, -1, 1), clip(a, -1, 1), c, clip(quality, 0, 1)), self.addr)
        self.sock.sendto(message("/emotif/hold", 1 if hold else 0), self.addr)
        for name, x in (extra or {}).items():
            if re.fullmatch(r"[a-z][a-z_]{0,23}", str(name)) and name not in ("state", "hold") and math.isfinite(float(x)):
                self.sock.sendto(message(f"/emotif/{name}", clip(x, -1, 1)), self.addr)

    def close(self):
        self.sock.close()
