"""Backend: device streaming, session recording, browser-sourced streams, and the arm governor over
simulated, HTTP and serial (pseudo-terminal) transports."""
import asyncio
import http.server
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from emotif import datasets, osc, roarm, server  # noqa: E402


async def recv_until(ws, kind, timeout=8.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        m = await ws.receive(timeout=timeout)
        d = json.loads(m.data)
        if d.get("type") == kind:
            return d
    raise AssertionError(f"no {kind} message")


async def test_streaming_and_recording(tmp):
    hub = server.Hub(sessions_dir=os.path.join(tmp, "sessions"))
    async with TestClient(TestServer(server.build_app(hub))) as c:
        assert (await (await c.get("/api/health")).json())["ok"]
        boards = (await (await c.get("/api/boards")).json())["boards"]
        assert any(b["id"] == "MUSE_2_BOARD" and b["available"] for b in boards)
        ws = await c.ws_connect("/ws")
        r = await c.post("/api/device/connect", json={"kind": "brainflow", "board": "SYNTHETIC_BOARD"})
        assert r.status == 200, await r.text()
        meta = await recv_until(ws, "meta")
        assert meta["fs"] == 250 and len(meta["channels"]) == 16
        assert len((await recv_until(ws, "data"))["samples"][0]) == 16

        r = await c.post("/api/session/start", json={"participant": "P01"})
        assert r.status == 400 and "consent" in (await r.json())["error"]
        r = await c.post("/api/session/start", json={"participant": "P 01!", "consent": True, "protocol": "test"})
        sid = (await r.json())["id"]
        assert sid.endswith("_P01")
        await ws.send_str(json.dumps({"type": "label", "valence": 1, "arousal": -1}))
        await ws.send_str(json.dumps({"type": "marker", "name": "stimulus", "value": {"clip": "calm"}}))
        await ws.send_str(json.dumps({"type": "affect", "v": 0.3, "a": -0.2, "conf": 0.7, "label": "Calm", "quality": 0.9}))
        await asyncio.sleep(3.0)
        st = await (await c.get("/api/state")).json()
        assert st["recording"]["seconds"] > 1.5, st
        await c.post("/api/session/stop")
        sessions = (await (await c.get("/api/sessions")).json())["sessions"]
        assert sessions[0]["id"] == sid and sessions[0]["n_samples"] > 250
        r = await c.get(f"/api/sessions/{sid}/events.csv")
        assert r.status == 200 and "stimulus" in await r.text()
        assert (await c.get(f"/api/sessions/{sid}/../../etc")).status == 404
        tr = datasets.load_csv_file(os.path.join(hub.sessions_dir, sid, "eeg.csv"))
        assert tr and tr[0].labels == {"valence": 1, "arousal": -1} and abs(tr[0].fs - 250) < 1e-6
        affect = open(os.path.join(hub.sessions_dir, sid, "affect.csv")).read()
        assert "Calm" in affect
        await c.post("/api/device/disconnect")

        # browser-owned stream (simulator / Web Bluetooth) recorded by the server
        await ws.send_str(json.dumps({"type": "stream_meta", "fs": 256, "channels": ["TP9", "AF7", "AF8", "TP10"], "label": "Muse BLE"}))
        await asyncio.sleep(0.1)
        sid2 = (await (await c.post("/api/session/start", json={"participant": "P02", "consent": True})).json())["id"]
        for k in range(4):
            await ws.send_str(json.dumps({"type": "samples", "samples": [[k, k + 1, k + 2, k + 3]] * 256}))
        await asyncio.sleep(0.2)
        await c.post("/api/session/stop")
        lines = open(os.path.join(hub.sessions_dir, sid2, "eeg.csv")).read().strip().splitlines()
        assert lines[0].startswith("time_s,TP9,AF7,AF8,TP10") and len(lines) == 1 + 4 * 256
        await ws.close()
    print("server: device stream, sessions, client stream OK")


async def test_arm_api():
    async with TestClient(TestServer(server.build_app(server.Hub(sessions_dir=tempfile.mkdtemp())))) as c:
        assert (await c.post("/api/arm/enable")).status == 409
        r = await c.post("/api/arm/connect", json={"kind": "sim"})
        assert r.status == 200
        ws = await c.ws_connect("/ws")
        await c.post("/api/arm/enable")
        tgt = {"x": 250.0, "y": 120.0, "z": 100.0, "t": 2.0}
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            await ws.send_str(json.dumps({"type": "arm_target", **tgt}))
            await asyncio.sleep(0.05)
        assert (await recv_until(ws, "arm"))["connected"]
        s = (await (await c.get("/api/state")).json())["arm"]
        assert math.dist((s["x"], s["y"], s["z"]), (250, 120, 100)) < 1.0, s
        # watchdog: stop streaming targets -> arm holds
        await ws.send_str(json.dumps({"type": "arm_target", "x": 300, "y": -150, "z": 200, "t": 3.0}))
        await asyncio.sleep(2.5)
        s1 = (await (await c.get("/api/state")).json())["arm"]
        await asyncio.sleep(1.0)
        s2 = (await (await c.get("/api/state")).json())["arm"]
        assert (s1["x"], s1["y"], s1["z"]) == (s2["x"], s2["y"], s2["z"]), "watchdog should hold"
        await c.post("/api/arm/estop")
        r = await c.post("/api/arm/enable")
        assert r.status == 400 and "emergency" in (await r.json())["error"]
        await c.post("/api/arm/clear_estop")
        assert (await c.post("/api/arm/enable")).status == 200
        # out-of-workspace target is clamped
        await ws.send_str(json.dumps({"type": "arm_target", "x": 900, "y": 0, "z": 900, "t": 9}))
        await asyncio.sleep(0.2)
        s = (await (await c.get("/api/state")).json())["arm"]
        assert s["clamped"] and s["target"]["z"] <= 380 and s["target"]["t"] <= math.pi + 1e-9
        await c.post("/api/arm/disconnect")
        await ws.close()
    print("server: arm API, watchdog, e-stop, clamping OK")


def test_governor_speed_limit():
    g = roarm.Governor(roarm.SimTransport(), roarm.Limits(speed_mm_s=100.0), rate_hz=50)
    g.enable()
    start = (g.state.x, g.state.y, g.state.z)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        g.set_target(start[0], start[1], start[2] - 200, math.pi)
        time.sleep(0.02)
    moved = math.dist(start, (g.state.x, g.state.y, g.state.z))
    g.close()
    assert 80 <= moved <= 105, moved
    print(f"governor: moved {moved:.1f} mm in 1 s at 100 mm/s limit OK")


def test_http_transport():
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.urlparse(self.path)
            assert q.path == "/js"
            cmd = json.loads(urllib.parse.parse_qs(q.query)["json"][0])
            got.append(cmd)
            body = json.dumps({"T": 1051, "x": 300, "y": 0, "z": 200, "t": 3.1}) if cmd["T"] == 105 else ""
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    g = roarm.connect("http", f"127.0.0.1:{srv.server_address[1]}")
    assert (g.state.x, g.state.z) == (300.0, 200.0)
    g.enable()
    for _ in range(10):
        g.set_target(300, 0, 180, 3.1)
        time.sleep(0.05)
    g.estop()
    g.close()
    srv.shutdown()
    kinds = [c["T"] for c in got]
    assert 1041 in kinds and kinds[-1] == 0, kinds
    print("roarm http transport OK")


def test_serial_transport():
    import pty
    master, slave = pty.openpty()
    name = os.ttyname(slave)
    got, stop = [], threading.Event()

    def firmware():
        buf = b""
        while not stop.is_set():
            try:
                chunk = os.read(master, 256)
            except OSError:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                cmd = json.loads(line)
                got.append(cmd)
                if cmd["T"] == 105:
                    os.write(master, b'noise\r\n{"T":1051,"x":310.2,"y":0,"z":236.8,"t":3.14}\r\n')

    th = threading.Thread(target=firmware, daemon=True)
    th.start()
    g = roarm.connect("serial", name)
    assert abs(g.state.x - 310.2) < 1e-6
    g.enable()
    for _ in range(10):
        g.set_target(300, 20, 230, 3.0)
        time.sleep(0.05)
    g.close()
    stop.set()
    os.close(slave)
    kinds = [c["T"] for c in got]
    assert kinds.count(105) >= 2 and 1041 in kinds, kinds
    print("roarm serial transport OK")


def test_lsl_source():
    """An LSL EEG stream (what Emotiv, g.tec and OpenViBE publish) is found by name and read with labels."""
    from pylsl import StreamInfo, StreamOutlet
    from emotif.devices import LSLSource
    info = StreamInfo("EmotifTest", "EEG", 4, 256, "float32", "emotif-test")
    ch = info.desc().append_child("channels")
    for n in ["TP9", "AF7", "AF8", "TP10"]:
        ch.append_child("channel").append_child_value("label", n)
    out = StreamOutlet(info)
    stop = threading.Event()

    def push():
        k = 0
        while not stop.is_set():
            for _ in range(16):
                out.push_sample([math.sin(k / 10) * 10, 1, 2, 3])
                k += 1
            time.sleep(16 / 256)

    threading.Thread(target=push, daemon=True).start()
    src = LSLSource("EmotifTest", 5)
    assert src.fs == 256 and src.channels == ["TP9", "AF7", "AF8", "TP10"]
    time.sleep(0.8)
    rows, t0 = [], time.time()
    while time.time() - t0 < 1.0:
        rows += src.read()
        time.sleep(0.05)
    src.close()
    stop.set()
    assert 150 < len(rows) < 400 and len(rows[0]) == 4, len(rows)
    print(f"lsl source OK: {len(rows)} samples/s with channel labels")


async def test_osc_bridge():
    """The decoded state the app sends reaches a UDP listener (the Unreal Engine side) as OSC."""
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(3)
    port = rx.getsockname()[1]
    assert osc.parse_target(f":{port}") == ("127.0.0.1", port) and osc.parse_target("10.0.0.2:9000") == ("10.0.0.2", 9000)
    assert osc.parse(osc.message("/a/b", 1.5, 7, "hi")) == ("/a/b", [1.5, 7, "hi"])
    assert len(osc.message("/emotif/state", 0.0, 0.0, 0.0, 0.0)) % 4 == 0
    hub = server.Hub(sessions_dir=tempfile.mkdtemp(), osc_target=("127.0.0.1", port))
    async with TestClient(TestServer(server.build_app(hub))) as c:
        ws = await c.ws_connect("/ws")
        await ws.send_str(json.dumps({"type": "state", "v": 0.25, "a": -2.0, "conf": None, "quality": 0.5, "hold": True,
                                      "extra": {"stress": 0.5, "Bad/Name": 1.0}}))
        got = {}
        loop = asyncio.get_running_loop()
        while len(got) < 3:
            data = await loop.run_in_executor(None, rx.recv, 512)
            addr, args = osc.parse(data)
            got[addr] = args
        await ws.close()
    rx.close()
    hub.osc.close()
    assert got["/emotif/state"] == [0.25, -1.0, -1.0, 0.5], got      # arousal clipped to -1, unknown confidence = -1
    assert got["/emotif/hold"] == [1] and got["/emotif/stress"] == [0.5]
    assert not any("Bad" in k for k in got), got
    print("osc bridge OK: /emotif/state and /emotif/hold reach a UDP listener")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(test_streaming_and_recording(tmp))
    asyncio.run(test_arm_api())
    asyncio.run(test_osc_bridge())
    test_governor_speed_limit()
    test_http_transport()
    test_serial_transport()
    test_lsl_source()
    print("all server tests passed")


if __name__ == "__main__":
    main()
