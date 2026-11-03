"""Waveshare RoArm-M2-S driver with a safety governor and a simulated arm.

Protocol (firmware: github.com/waveshareteam/roarm_m2, json_cmd.h):
  serial  115200 baud, one JSON object per line ('\\n')
  HTTP    GET http://<ip>/js?json=<url-encoded JSON>   (AP mode default 192.168.4.1)
  {"T":1041,"x":..,"y":..,"z":..,"t":..}   move end effector, no interpolation (we stream small steps)
  {"T":105}                                 feedback -> {"T":1051,"x","y","z","b","s","e","t",...}
  {"T":0} emergency stop, {"T":999} clear it, {"T":100} move to init pose

Kinematics follow RoArmM2_computePosbyJointRad (gripper mode): x forward, y left, z up, millimetres,
origin at the base axis on the shoulder-joint plane.
"""
import json
import math
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict

L2A, L2B = 236.82, 30.00
L3A, L3B = 280.15, 1.73
L2 = math.hypot(L2A, L2B)
L3 = math.hypot(L3A, L3B)
T2 = math.atan2(L2B, L2A)
INIT = {"x": L3A + L2B, "y": 0.0, "z": L2A - L3B, "t": math.pi}   # firmware initX/initZ/initT
GRIP_OPEN, GRIP_CLOSED = 1.08, math.pi                          # EoAT clamp range (rad)


def forward(b, s, e):
    """Joint angles (rad) -> (x, y, z) mm. Same formula as the firmware (EEMode 0)."""
    a1 = math.pi / 2 - (s + T2)
    a2 = math.pi / 2 - (e + s)
    r = L2 * math.cos(a1) + L3 * math.cos(a2)
    z = L2 * math.sin(a1) + L3 * math.sin(a2)
    return r * math.cos(b), r * math.sin(b), z


def inverse(x, y, z):
    """(x, y, z) mm -> joint angles (b, s, e) rad, elbow-up branch, or None if unreachable."""
    b = math.atan2(y, x)
    r = math.hypot(x, y)
    d2 = r * r + z * z
    c = (d2 - L2 * L2 - L3 * L3) / (2 * L2 * L3)
    if c < -1 or c > 1:
        return None
    gamma = math.acos(c)                   # angle between link 2 and link 3 directions
    phi = math.atan2(z, r)
    a1 = phi + math.atan2(L3 * math.sin(gamma), L2 + L3 * math.cos(gamma))
    a2 = a1 - gamma
    s = math.pi / 2 - a1 - T2
    e = math.pi / 2 - a2 - s
    if not (-math.pi / 2 - 1e-6 <= s <= math.pi / 2 + 1e-6 and -1e-6 <= e <= math.pi + 1e-6):
        return None
    return b, s, e


@dataclass
class Limits:
    r_min: float = 150.0      # keep the gripper away from the base column
    r_max: float = 430.0
    z_min: float = -80.0      # shoulder plane is z = 0; table is roughly 120 mm below it
    z_max: float = 380.0
    base_max_deg: float = 150.0
    speed_mm_s: float = 120.0
    grip_speed_rad_s: float = 2.0


@dataclass
class ArmState:
    connected: bool = False
    transport: str = "none"
    enabled: bool = False
    estop: bool = False
    x: float = INIT["x"]
    y: float = INIT["y"]
    z: float = INIT["z"]
    t: float = INIT["t"]
    target: dict = field(default_factory=lambda: dict(INIT))
    last_cmd_age: float = 1e9
    clamped: bool = False
    feedback: dict = field(default_factory=dict)
    error: str = ""


def clamp_target(x, y, z, t, lim):
    """Project a requested pose into the safe workspace. Returns (x, y, z, t, clamped)."""
    orig = (x, y, z, t)
    r = math.hypot(x, y)
    ang = math.atan2(y, x) if r > 1e-9 else 0.0
    amax = math.radians(lim.base_max_deg)
    ang = max(-amax, min(amax, ang))
    r = max(lim.r_min, min(lim.r_max, r))
    z = max(lim.z_min, min(lim.z_max, z))
    # pull inward until reachable by the 2-link arm
    for _ in range(60):
        if inverse(r * math.cos(ang), r * math.sin(ang), z) is not None:
            break
        r = max(lim.r_min, r - 5.0)
        if r == lim.r_min:
            z = z - 5.0 if z > 0 else z + 5.0
    x, y = r * math.cos(ang), r * math.sin(ang)
    t = max(GRIP_OPEN, min(GRIP_CLOSED, t))
    clamped = any(abs(a - b) > 1e-6 for a, b in zip(orig, (x, y, z, t)))
    return x, y, z, t, clamped


# ---------------------------------------------------------------- transports
class SimTransport:
    """Simulated arm: tracks T:1041 targets at a finite speed and answers T:105."""
    name = "sim"

    def __init__(self, speed_mm_s=400.0):
        self.pos = dict(INIT)
        self.goal = dict(INIT)
        self.estop = False
        self.speed = speed_mm_s
        self._t = time.monotonic()
        self.log = []

    def _advance(self):
        now = time.monotonic()
        dt, self._t = now - self._t, now
        if self.estop:
            return
        dx, dy, dz = (self.goal[k] - self.pos[k] for k in "xyz")
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        step = self.speed * dt
        f = 1.0 if d <= step or d == 0 else step / d
        for k, dk in zip("xyz", (dx, dy, dz)):
            self.pos[k] += dk * f
        self.pos["t"] += (self.goal["t"] - self.pos["t"]) * min(1.0, dt * 8)

    def send(self, cmd):
        self._advance()
        self.log.append(cmd)
        T = cmd.get("T")
        if T == 0:
            self.estop = True
        elif T == 999:
            self.estop = False
        elif T == 100 and not self.estop:
            self.goal = dict(INIT)
        elif T in (1041, 104) and not self.estop:
            self.goal = {k: float(cmd[k]) for k in ("x", "y", "z", "t")}
        elif T == 105:
            j = inverse(self.pos["x"], self.pos["y"], self.pos["z"]) or (0.0, 0.0, math.pi / 2)
            return {"T": 1051, **{k: round(v, 2) for k, v in self.pos.items()},
                    "b": j[0], "s": j[1], "e": j[2], "v": 1200}
        return None

    def close(self):
        pass


class SerialTransport:
    name = "serial"

    def __init__(self, port, baud=115200):
        import serial
        self.ser = serial.serial_for_url(port, baudrate=baud, timeout=0.05)
        self.lock = threading.Lock()
        self.feedback = None
        self.buf = b""
        self.running = True
        self.reader = threading.Thread(target=self._read_loop, daemon=True)
        self.reader.start()

    def _read_loop(self):
        while self.running:
            try:
                chunk = self.ser.read(256)
            except Exception:
                break
            if not chunk:
                continue
            self.buf += chunk
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                try:
                    msg = json.loads(line.decode("utf-8", "replace").strip())
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("T") == 1051:
                    self.feedback = msg

    def send(self, cmd):
        data = (json.dumps(cmd, separators=(",", ":")) + "\n").encode()
        with self.lock:
            self.ser.write(data)
        if cmd.get("T") == 105:
            t0 = time.monotonic()
            self.feedback = None
            while time.monotonic() - t0 < 0.3:
                if self.feedback is not None:
                    return self.feedback
                time.sleep(0.01)
        return None

    def close(self):
        self.running = False
        self.reader.join(timeout=1)
        self.ser.close()


class HttpTransport:
    name = "http"

    def __init__(self, host, timeout=0.6):
        self.base = host if host.startswith("http") else f"http://{host}"
        self.timeout = timeout

    def send(self, cmd):
        url = f"{self.base}/js?json={urllib.parse.quote(json.dumps(cmd, separators=(',', ':')))}"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            body = r.read().decode("utf-8", "replace").strip()
        if cmd.get("T") == 105 and body:
            try:
                return json.loads(body)
            except ValueError:
                return None
        return None

    def close(self):
        pass


# ---------------------------------------------------------------- governor
class Governor:
    """All motion goes through here. Streams T:1041 at `rate_hz` toward the latest client target,
    limited to Limits.speed_mm_s, inside the safe workspace. Holds position when disabled, when the
    client target is older than `watchdog_s`, or after an emergency stop."""

    def __init__(self, transport, limits=None, rate_hz=20.0, watchdog_s=1.0):
        self.tx = transport
        self.lim = limits or Limits()
        self.rate = rate_hz
        self.watchdog = watchdog_s
        self.state = ArmState(connected=True, transport=transport.name)
        self._target_time = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._sync_from_feedback()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _sync_from_feedback(self):
        try:
            fb = self.tx.send({"T": 105})
        except Exception as e:  # noqa: BLE001 - surface any transport error to the UI
            self.state.error = f"feedback failed: {e}"
            fb = None
        if fb and all(k in fb for k in ("x", "y", "z")):
            self.state.feedback = fb
            self.state.x, self.state.y, self.state.z = float(fb["x"]), float(fb["y"]), float(fb["z"])
            if "t" in fb:
                self.state.t = float(fb["t"])
        self.state.target = {"x": self.state.x, "y": self.state.y, "z": self.state.z, "t": self.state.t}

    def set_target(self, x, y, z, t):
        x, y, z, t, clamped = clamp_target(float(x), float(y), float(z), float(t), self.lim)
        with self._lock:
            self.state.target = {"x": x, "y": y, "z": z, "t": t}
            self.state.clamped = clamped
            self._target_time = time.monotonic()

    def enable(self):
        with self._lock:
            if self.state.estop:
                raise RuntimeError("clear the emergency stop first")
        self._sync_from_feedback()
        with self._lock:
            self.state.enabled = True
            self._target_time = time.monotonic()

    def disable(self):
        with self._lock:
            self.state.enabled = False

    def estop(self):
        with self._lock:
            self.state.enabled = False
            self.state.estop = True
        self.tx.send({"T": 0})

    def clear_estop(self):
        self.tx.send({"T": 999})
        self._sync_from_feedback()
        with self._lock:
            self.state.estop = False

    def home(self):
        self.set_target(INIT["x"], INIT["y"], INIT["z"], INIT["t"])

    def snapshot(self):
        with self._lock:
            d = asdict(self.state)
        d["last_cmd_age"] = time.monotonic() - self._target_time if self._target_time else 1e9
        d["limits"] = asdict(self.lim)
        return d

    def _loop(self):
        period = 1.0 / self.rate
        n = 0
        while self._running:
            t0 = time.monotonic()
            with self._lock:
                live = (self.state.enabled and not self.state.estop
                        and t0 - self._target_time <= self.watchdog)
                tgt = dict(self.state.target)
                cur = {"x": self.state.x, "y": self.state.y, "z": self.state.z, "t": self.state.t}
            if live:
                dx, dy, dz = (tgt[k] - cur[k] for k in "xyz")
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                step = self.lim.speed_mm_s * period
                f = 1.0 if d <= step or d == 0 else step / d
                nxt = {k: cur[k] + (tgt[k] - cur[k]) * f for k in "xyz"}
                gstep = self.lim.grip_speed_rad_s * period
                nxt["t"] = cur["t"] + max(-gstep, min(gstep, tgt["t"] - cur["t"]))
                if d > 0.05 or abs(tgt["t"] - cur["t"]) > 1e-3:
                    try:
                        self.tx.send({"T": 1041, **{k: round(v, 2) for k, v in nxt.items()}})
                        with self._lock:
                            self.state.x, self.state.y, self.state.z, self.state.t = nxt["x"], nxt["y"], nxt["z"], nxt["t"]
                            self.state.error = ""
                    except Exception as e:  # noqa: BLE001
                        with self._lock:
                            self.state.error = f"send failed: {e}"
                            self.state.enabled = False
            n += 1
            if n % int(self.rate) == 0 and not live:
                try:
                    fb = self.tx.send({"T": 105})
                    if fb:
                        with self._lock:
                            self.state.feedback = fb
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.tx.close()


def connect(kind, address=""):
    if kind == "sim":
        return Governor(SimTransport())
    if kind == "serial":
        if not address:
            raise ValueError("serial port required, e.g. /dev/ttyUSB0 or COM5")
        return Governor(SerialTransport(address))
    if kind == "http":
        return Governor(HttpTransport(address or "192.168.4.1"))
    raise ValueError(f"unknown arm transport {kind!r}")
