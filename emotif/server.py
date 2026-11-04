"""Emotif backend: serves the app, streams EEG devices, records sessions, and drives the RoArm-M2.

  python -m emotif.server                     # http://localhost:8000
  python -m emotif.server --host 0.0.0.0      # reachable from a tablet on the same network

Everything binds to localhost by default: EEG is health data and the arm is a physical device.
"""
import argparse
import asyncio
import importlib.util
import json
import os
import threading
import time

from aiohttp import WSMsgType, web

from . import devices, osc, recorder, roarm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
MODELS = os.path.join(ROOT, "models")
SESSIONS = os.path.join(ROOT, "data", "sessions")


class Hub:
    def __init__(self, sessions_dir=SESSIONS, models_dir=MODELS, osc_target=None):
        self.osc = osc.OscSender(*osc_target) if osc_target else None   # decoded state out to Unreal Engine etc.
        self.sessions_dir = sessions_dir
        self.models_dir = models_dir
        self.clients = set()
        self.source = None            # devices.* when the server owns acquisition
        self.stream = None            # {"fs", "channels", "origin": "device"|"client", "label"}
        self.rec = None
        self.arm = None
        self.reader = None
        self.reader_stop = threading.Event()
        self.loop = None
        self.queue = None
        self.last_error = ""

    # ---------------------------------------------------------------- streaming
    async def broadcast(self, msg):
        if not self.clients:
            return
        data = json.dumps(msg, separators=(",", ":"))
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except (ConnectionResetError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def _read_loop(self, src):
        while not self.reader_stop.is_set():
            try:
                rows = src.read()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"device read failed: {e}"
                break
            if rows:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, rows)
            time.sleep(0.04)

    async def pump(self):
        while True:
            rows = await self.queue.get()
            if self.stream is None or self.stream["origin"] != "device":
                continue
            if self.rec:
                self.rec.samples(rows)
            await self.broadcast({"type": "data", "samples": rows})

    async def connect_device(self, cfg):
        await self.disconnect_device()
        src = await asyncio.get_running_loop().run_in_executor(None, devices.open_source, cfg)
        self.source = src
        self.stream = {"fs": src.fs, "channels": src.channels, "origin": "device", "label": src.label}
        self.reader_stop.clear()
        self.reader = threading.Thread(target=self._read_loop, args=(src,), daemon=True)
        self.reader.start()
        await self.broadcast({"type": "meta", **self.stream})

    async def disconnect_device(self):
        if self.rec and self.stream and self.stream["origin"] == "device":
            self.stop_recording()
        if self.source:
            self.reader_stop.set()
            if self.reader:
                self.reader.join(timeout=2)
            try:
                self.source.close()
            except Exception:  # noqa: BLE001
                pass
            self.source = None
            self.stream = None
            await self.broadcast({"type": "device_closed"})

    # ---------------------------------------------------------------- recording
    def start_recording(self, body):
        if self.rec:
            raise ValueError("a session is already recording")
        if not self.stream:
            raise ValueError("connect a device or start a stream before recording")
        if not body.get("consent"):
            raise ValueError("confirm the participant's consent before recording")
        self.rec = recorder.Recorder(self.sessions_dir, body.get("participant", ""), self.stream["fs"],
                                     self.stream["channels"], self.stream.get("label", ""),
                                     body.get("protocol", ""), body.get("notes", ""),
                                     {"origin": self.stream["origin"]})
        return self.rec.id

    def stop_recording(self):
        if not self.rec:
            return None
        sid = self.rec.id
        self.rec.close()
        self.rec = None
        return sid

    # ---------------------------------------------------------------- state
    def state(self):
        return {
            "stream": self.stream,
            "recording": None if not self.rec else {"id": self.rec.id, "seconds": round(self.rec.t, 1)},
            "arm": self.arm.snapshot() if self.arm else None,
            "error": self.last_error,
        }

    async def arm_loop(self):
        while True:
            await asyncio.sleep(0.1)
            if self.arm:
                await self.broadcast({"type": "arm", **self.arm.snapshot()})

    # ---------------------------------------------------------------- websocket
    async def handle_ws_message(self, ws, msg):
        t = msg.get("type")
        if t == "stream_meta":           # browser-owned source (simulator, Web Bluetooth Muse)
            if self.stream and self.stream["origin"] == "device":
                return {"type": "error", "message": "a hardware device is already streaming"}
            if self.rec:
                self.stop_recording()
            self.stream = {"fs": float(msg["fs"]), "channels": list(msg["channels"]), "origin": "client",
                           "label": str(msg.get("label", "browser"))}
        elif t == "samples":
            if self.stream and self.stream["origin"] == "client" and self.rec:
                self.rec.samples(msg.get("samples", []))
        elif t == "stream_end":
            if self.stream and self.stream["origin"] == "client":
                self.stop_recording()
                self.stream = None
        elif t == "marker" and self.rec:
            self.rec.event(str(msg.get("name", "marker")), json.dumps(msg.get("value", "")))
        elif t == "label" and self.rec:
            self.rec.set_labels(msg.get("valence"), msg.get("arousal"))
        elif t == "state":               # decoded state, every window, whether or not a session is recording
            if self.osc:
                self.osc.send_state(float(msg["v"]), float(msg["a"]), msg.get("conf"), float(msg.get("quality", 1.0)),
                                    bool(msg.get("hold", False)), msg.get("extra") or {})
        elif t == "affect" and self.rec:
            self.rec.affect(float(msg["v"]), float(msg["a"]), msg.get("conf"), str(msg.get("label", "")),
                            float(msg.get("quality", 1.0)))
        elif t == "arm_target":
            if self.arm:
                self.arm.set_target(msg["x"], msg["y"], msg["z"], msg.get("t", roarm.INIT["t"]))
                if self.rec and msg.get("log"):
                    self.rec.event("arm_target", json.dumps({k: msg[k] for k in ("x", "y", "z", "t") if k in msg}))
        elif t == "ping":
            return {"type": "pong", "t": msg.get("t")}
        return None

    async def ws(self, request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        self.clients.add(ws)
        if self.stream and self.stream["origin"] == "device":
            await ws.send_str(json.dumps({"type": "meta", **self.stream}))
        try:
            async for m in ws:
                if m.type != WSMsgType.TEXT:
                    continue
                try:
                    reply = await self.handle_ws_message(ws, json.loads(m.data))
                except (KeyError, ValueError, TypeError) as e:
                    reply = {"type": "error", "message": f"bad message: {e}"}
                if reply:
                    await ws.send_str(json.dumps(reply))
        finally:
            self.clients.discard(ws)
        return ws


_QA = None


def _session_qa():
    """research/session_qa.py, loaded by path so the server doesn't need research/ on sys.path."""
    global _QA
    if _QA is None:
        spec = importlib.util.spec_from_file_location("emotif_session_qa", os.path.join(ROOT, "research", "session_qa.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _QA = mod
    return _QA


def json_error(msg, status=400):
    return web.json_response({"ok": False, "error": msg}, status=status)


def build_app(hub=None):
    hub = hub or Hub()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["hub"] = hub
    r = web.RouteTableDef()

    async def body(request):
        try:
            return await request.json() if request.can_read_body else {}
        except ValueError:
            raise web.HTTPBadRequest(text="body must be JSON")

    @r.get("/")
    async def index(_):
        # mark pages served by this backend so the app knows a server exists without probing for one
        with open(os.path.join(WEB, "index.html"), encoding="utf-8") as fh:
            html = fh.read()
        html = '<meta name="emotif-server" content="1">\n' + html
        return web.Response(text=html, content_type="text/html", charset="utf-8", headers={"Cache-Control": "no-cache"})

    @r.get("/api/health")
    async def health(_):
        return web.json_response({"ok": True, "app": "emotif", "version": recorder.VERSION})

    @r.get("/api/state")
    async def state(_):
        return web.json_response(hub.state())

    @r.get("/api/boards")
    async def boards(_):
        return web.json_response({"boards": devices.list_boards()})

    @r.post("/api/device/connect")
    async def dev_connect(request):
        try:
            await hub.connect_device(await body(request))
        except Exception as e:  # noqa: BLE001 - report device errors to the user verbatim
            return json_error(f"Couldn't open the device: {e}")
        return web.json_response({"ok": True, "stream": hub.stream})

    @r.post("/api/device/disconnect")
    async def dev_disconnect(_):
        await hub.disconnect_device()
        return web.json_response({"ok": True})

    @r.post("/api/session/start")
    async def s_start(request):
        try:
            sid = hub.start_recording(await body(request))
        except (ValueError, OSError) as e:
            return json_error(str(e))
        return web.json_response({"ok": True, "id": sid})

    @r.post("/api/session/stop")
    async def s_stop(_):
        return web.json_response({"ok": True, "id": hub.stop_recording()})

    @r.get("/api/sessions")
    async def s_list(_):
        return web.json_response({"sessions": recorder.list_sessions(hub.sessions_dir)})

    @r.get("/api/sessions/{sid}/qa")
    async def s_qa(request):
        """Blink/jaw/contact check of a finished session (research/session_qa.py), for the app's guided session."""
        sid = request.match_info["sid"]
        if "/" in sid or "\\" in sid or ".." in sid:
            raise web.HTTPNotFound()
        folder = os.path.join(hub.sessions_dir, sid)
        if not os.path.isfile(os.path.join(folder, "meta.json")):
            raise web.HTTPNotFound()
        if hub.rec and hub.rec.id == sid:
            return json_error("this session is still recording")
        try:
            res = await asyncio.get_running_loop().run_in_executor(None, _session_qa().qa, folder)
        except Exception as e:  # unreadable or too-short recording
            return json_error(f"check failed: {e}")
        return web.json_response(res)

    @r.get("/api/sessions/{sid}/{name}")
    async def s_file(request):
        sid, name = request.match_info["sid"], request.match_info["name"]
        if name not in ("eeg.csv", "events.csv", "affect.csv", "meta.json") or "/" in sid or ".." in sid:
            raise web.HTTPNotFound()
        p = os.path.join(hub.sessions_dir, sid, name)
        if not os.path.isfile(p):
            raise web.HTTPNotFound()
        return web.FileResponse(p, headers={"Content-Disposition": f'attachment; filename="{sid}_{name}"'})

    @r.get("/api/models")
    async def m_list(_):
        out = []
        if os.path.isdir(hub.models_dir):
            for f in sorted(os.listdir(hub.models_dir)):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(hub.models_dir, f)) as fh:
                            m = json.load(fh)
                        out.append({"file": f, "name": m.get("name", f), "heads": list(m.get("heads", {})),
                                    "datasets": m.get("datasets", []), "model": m.get("model", "")})
                    except ValueError:
                        continue
        return web.json_response({"models": out})

    @r.post("/api/models/save")
    async def m_save(request):
        b = await body(request)
        m = b.get("model")
        if not isinstance(m, dict) or m.get("format") != "emotif-model-v2" or not m.get("heads"):
            return json_error("not an Emotif v2 model")
        os.makedirs(hub.models_dir, exist_ok=True)
        base = "personal_" + recorder.safe_code(str(b.get("name", "model")).replace(" ", "_").replace("·", ""))
        name, k = f"{base}.json", 1
        while os.path.exists(os.path.join(hub.models_dir, name)):
            k += 1
            name = f"{base}_{k}.json"
        with open(os.path.join(hub.models_dir, name), "w") as fh:
            json.dump(m, fh)
        return web.json_response({"ok": True, "file": name})

    @r.get("/models/{name}")
    async def m_get(request):
        name = request.match_info["name"]
        p = os.path.join(hub.models_dir, name)
        if "/" in name or ".." in name or not name.endswith(".json") or not os.path.isfile(p):
            raise web.HTTPNotFound()
        return web.FileResponse(p)

    # -------------------------------------------------- robot arm
    def arm_required():
        if not hub.arm:
            raise web.HTTPConflict(text=json.dumps({"ok": False, "error": "connect the arm first"}),
                                   content_type="application/json")

    @r.post("/api/arm/connect")
    async def a_connect(request):
        b = await body(request)
        if hub.arm:
            hub.arm.close()
            hub.arm = None
        try:
            hub.arm = await asyncio.get_running_loop().run_in_executor(
                None, roarm.connect, b.get("kind", "sim"), b.get("address", ""))
        except Exception as e:  # noqa: BLE001
            return json_error(f"Couldn't reach the arm: {e}")
        return web.json_response({"ok": True, "arm": hub.arm.snapshot()})

    @r.post("/api/arm/disconnect")
    async def a_disconnect(_):
        if hub.arm:
            hub.arm.disable()
            hub.arm.close()
            hub.arm = None
        return web.json_response({"ok": True})

    @r.post("/api/arm/{action}")
    async def a_action(request):
        arm_required()
        action = request.match_info["action"]
        loop = asyncio.get_running_loop()
        try:
            if action == "enable":
                await loop.run_in_executor(None, hub.arm.enable)
            elif action == "disable":
                hub.arm.disable()
            elif action == "estop":
                await loop.run_in_executor(None, hub.arm.estop)
            elif action == "clear_estop":
                await loop.run_in_executor(None, hub.arm.clear_estop)
            elif action == "home":
                hub.arm.home()
            elif action == "limits":
                b = await body(request)
                for k, v in b.items():
                    if hasattr(hub.arm.lim, k):
                        setattr(hub.arm.lim, k, float(v))
            else:
                raise web.HTTPNotFound()
            if hub.rec:
                hub.rec.event("arm", action)
        except RuntimeError as e:
            return json_error(str(e))
        return web.json_response({"ok": True, "arm": hub.arm.snapshot()})

    r.get("/ws")(hub.ws)
    app.add_routes(r)

    async def on_start(app_):
        hub.loop = asyncio.get_running_loop()
        hub.queue = asyncio.Queue()
        app_["tasks"] = [asyncio.create_task(hub.pump()), asyncio.create_task(hub.arm_loop())]

    async def on_cleanup(app_):
        for t in app_["tasks"]:
            t.cancel()
        hub.stop_recording()
        await hub.disconnect_device()
        if hub.arm:
            hub.arm.disable()
            hub.arm.close()

    app.on_startup.append(on_start)
    app.on_cleanup.append(on_cleanup)
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--sessions", default=SESSIONS)
    p.add_argument("--osc", default=None, metavar="HOST:PORT",
                   help="also send the decoded state as OSC over UDP (e.g. 127.0.0.1:8001 for the Unreal Engine 5 showcase)")
    a = p.parse_args(argv)
    osc_target = osc.parse_target(a.osc) if a.osc else None
    os.makedirs(a.sessions, exist_ok=True)
    if a.host not in ("127.0.0.1", "localhost"):
        print("warning: listening beyond this computer. Anyone on the network can open the app and move the arm.")
    print(f"Emotif {recorder.VERSION}: open http://localhost:{a.port}")
    if osc_target:
        print(f"sending the decoded state as OSC to {osc_target[0]}:{osc_target[1]} (/emotif/state, /emotif/hold)")
    web.run_app(build_app(Hub(sessions_dir=a.sessions, osc_target=osc_target)), host=a.host, port=a.port, print=None, shutdown_timeout=2)


if __name__ == "__main__":
    main()
