"""Full app in headless Chromium against the real backend:
simulator decoding, headset switching, built-in model, a recorded protocol with SAM ratings,
personalization, sessions/replay, the arm twin (embody, draw, brain-intent control) and the
server-governed arm, plus a browser-only run with no server (as when published as an artifact).

  python tests/test_browser.py [screenshot_dir]
"""
import functools
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emotif import datasets  # noqa: E402

SHOTS = sys.argv[1] if len(sys.argv) > 1 else None


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait(page, expr, timeout=30, what=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if page.evaluate(expr):
            return
        time.sleep(0.2)
    raise AssertionError(f"timeout: {what or expr}")


def shot(page, name):
    if SHOTS:
        page.screenshot(path=os.path.join(SHOTS, f"{name}.png"), timeout=120000)


# Headless Chromium renders WebGL in software, which can starve the main thread; protocol buttons are
# therefore clicked with dispatch_event (same handlers, no pointer-actionability wait).
def new_page(browser, errors, **kw):
    page = browser.new_page(**kw)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.route("**/fonts.googleapis.com/**", lambda r: r.abort())
    page.route("**/fonts.gstatic.com/**", lambda r: r.abort())
    page.on("framenavigated", lambda f: print("  navigated:", f.url) if os.environ.get("EMOTIF_DEBUG") else None)
    return page


def get_json(port, path):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}").read())


def run_with_server(browser, errors):
    port = free_port()
    sessions = tempfile.mkdtemp()
    srv = subprocess.Popen([sys.executable, "-m", "emotif.server", "--port", str(port), "--sessions", sessions],
                           cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5)
                break
            except OSError:
                time.sleep(0.2)
        page = new_page(browser, errors, viewport={"width": 1280, "height": 800})
        page.goto(f"http://127.0.0.1:{port}/")
        wait(page, "window.emotif && window.emotif.link.available", 10, "server detected")
        assert page.evaluate("window.emotif.world.ok"), "3-D world failed"

        # ---- simulator: decoder recovers the steered state
        page.locator("nav.nav button[data-view=setup]").dispatch_event("click")
        page.fill("#calibSec", "25"); page.press("#calibSec", "Tab")
        page.select_option("#simMode", "manual")
        assert page.evaluate("window.emotif.modelId") == "builtin:brain", "brain-only decoder is the default"
        page.evaluate("window.emotif.selectModel('markers')")   # the simulator's states are written in marker terms
        page.evaluate("window.emotif.useSource('sim')")
        wait(page, "window.emotif.engine.calibrated", 60, "calibration")
        page.locator("nav.nav button[data-view=live]").dispatch_event("click")
        hits = 0
        for v, a in [(0.8, 0.7), (0.8, -0.7), (-0.8, -0.7), (-0.8, 0.7)]:
            page.evaluate(f"window.emotif.sim.target={{v:{v},a:{a}}}")
            time.sleep(13)
            acc = []
            for _ in range(4):
                time.sleep(1.5)
                acc.append(page.evaluate("[window.emotif.engine.last.v, window.emotif.engine.last.a]"))
            mv = sum(x[0] for x in acc) / 4
            ma = sum(x[1] for x in acc) / 4
            ok = mv * v > 0.1 and ma * a > 0.1
            hits += ok
            print(f"  steer ({v:+.1f},{a:+.1f}) -> ({mv:+.2f},{ma:+.2f}) {page.inner_text('#word')} {'ok' if ok else 'MISS'}")
            if v > 0 and a > 0:
                shot(page, "live_happy")
        assert hits == 4, f"{hits}/4 quadrants"
        assert page.evaluate("window.emotif.engine.asrActive && window.emotif.engine.detectorActive"), "cleaning should be active after the baseline"
        assert page.evaluate("document.querySelectorAll('#cleanRow .chip').length") == 8, "per-channel detector chips"
        print("cleaning live: ASR + detector active;", page.inner_text("#stClean"))
        page.evaluate("document.documentElement.dataset.theme='dark'")
        time.sleep(1.5)
        shot(page, "live_dark")
        page.evaluate("document.documentElement.dataset.theme='light'")

        # ---- switch to a Muse layout and the built-in model trained on real Muse data
        page.locator("nav.nav button[data-view=setup]").dispatch_event("click")
        page.select_option("#simMontage", "muse")
        wait(page, "window.emotif.engine.C === 4", 5, "muse layout")
        shot(page, "setup")
        page.evaluate("document.getElementById('fitTopo').scrollIntoView()")
        wait(page, "document.querySelectorAll('#fitList li').length === 4", 10, "fit list for Muse layout")
        shot(page, "setup_fit")

        # ---- method view: live pipeline numbers and embedded evidence figures
        page.locator("nav.nav button[data-view=method]").dispatch_event("click")
        wait(page, "document.querySelectorAll('#pipeline .stage-box').length === 7", 5, "pipeline")
        assert page.evaluate("document.querySelectorAll('#evidence img').length") == 7, "7 result figures embedded"
        time.sleep(1)
        shot(page, "method")
        page.locator("nav.nav button[data-view=models]").dispatch_event("click")
        page.locator("#modelList .card-opt:nth-child(1)").dispatch_event("click")
        wait(page, "window.emotif.modelId === 'builtin:brain' && window.emotif.engine.model !== null", 5, "brain-only model loaded")
        wait(page, "window.emotif.engine.last && window.emotif.engine.last.source === 'model' && window.emotif.engine.last.cmp", 60, "model decoding with comparison")
        assert page.locator("#modelDetail table.data").count() >= 1
        time.sleep(0.5)
        shot(page, "models")
        check_brain_vs_face(page)
        page.evaluate("window.emotif.selectModel('markers')")

        # ---- a short recorded protocol with SAM ratings
        page.locator("nav.nav button[data-view=study]").dispatch_event("click")
        page.fill("#pcode", "T01")
        page.evaluate("document.getElementById('consent').checked = true")
        page.evaluate("""window.emotif.startProtocol({id:'test', name:'Test', minutes:1, summary:'', build: () => ({order:['a','b'], steps:[
          {kind:'instruction', title:'Ready', body:'Test run'},
          {kind:'baseline', seconds:12},
          {kind:'task', id:'calm1', title:'Slow breathing', seconds:8, task:'breathing', design:{v:1,a:-1}, condition:'mirror'},
          {kind:'sam', block:'calm1'},
          {kind:'task', id:'stress1', title:'Count backward', seconds:8, task:'arithmetic', design:{v:-1,a:1}, condition:'mirror'},
          {kind:'sam', block:'stress1'},
          {kind:'task', id:'watch1', title:'Watch', seconds:6, task:'watch', design:{v:1,a:-1}, condition:'guide', target:'calm'},
          {kind:'sam', block:'watch1'}]})})""")
        wait(page, "!!window.emotif.recording", 5, "recording started")
        page.locator(".runner button.btn.primary").dispatch_event("click")
        wait(page, "document.querySelector('.runner h2')?.textContent === 'Slow breathing'", 20, "breathing step")
        page.evaluate("window.emotif.sim.target={v:0.7,a:-0.7}")
        shot(page, "study_breathing")
        ratings = {"calm1": (8, 2), "stress1": (2, 8), "watch1": (7, 3)}
        for blk, (sv, sa) in ratings.items():
            wait(page, "document.querySelector('.runner h2')?.textContent === 'How did you feel?'", 25, f"SAM {blk}")
            page.locator(f".sam:nth-of-type(1) .sam-pt:nth-child({sv})").dispatch_event("click")
            page.locator(f".sam:nth-of-type(2) .sam-pt:nth-child({sa})").dispatch_event("click")
            if blk == "calm1":
                shot(page, "study_sam")
            page.locator(".runner .btn.primary.large").dispatch_event("click")
            if blk == "calm1":
                page.evaluate("window.emotif.sim.target={v:-0.7,a:0.7}")
            if blk == "stress1":
                page.evaluate("window.emotif.sim.target={v:0.7,a:-0.7}")
                wait(page, "document.body.classList.contains('immersive')", 10, "immersive watch")
                assert page.evaluate("window.emotif.director.mode") == "guide"
                time.sleep(2)
                shot(page, "study_immersive")
        wait(page, "!window.emotif.protocolActive", 10, "protocol end")
        wait(page, "!window.emotif.recording", 10, "recording stopped")
        assert page.locator("#runSummary tbody tr").count() == 3
        sess = get_json(port, "/api/sessions")["sessions"]
        assert sess and sess[0]["participant"] == "T01", sess
        sid = sess[0]["id"]
        ev = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sessions/{sid}/events.csv").read().decode()
        for k in ("protocol_start", "block_start", "sam", "condition", "protocol_end"):
            assert k in ev, k
        trials = datasets.load_sessions(os.path.join(sessions, sid))
        labels = {t.trial_id.split(":")[-1]: t.labels for t in trials}
        assert labels == {"calm1": {"valence": 1, "arousal": -1}, "stress1": {"valence": -1, "arousal": 1},
                          "watch1": {"valence": 1, "arousal": -1}}, labels   # SAM 3 -> low (1-3 low, 4-6 mid, 7-9 high)
        print(f"protocol: 3 rated blocks recorded to {sid}, reloaded for training with SAM labels")

        # ---- personalization from the ratings
        page.locator("nav.nav button[data-view=models]").dispatch_event("click")
        page.locator("#personal .btn.primary").dispatch_event("click")
        wait(page, "!!window.emotif.personal", 30, "personal model")
        page.locator("#personal .btn:has-text('Use personal model')").dispatch_event("click")
        wait(page, "window.emotif.modelId === 'personal'", 5)
        page.locator("#personal .btn:has-text('Save to server')").dispatch_event("click")
        wait(page, "document.querySelector('#personal').textContent.includes('Saved as models/')", 10, "saved")
        rep = page.evaluate("window.emotif.personal.report")
        assert "two rated blocks" in rep, rep        # 3 blocks can't support leave-one-block-out per class
        print("personalization:", rep[:160])
        for f in os.listdir(os.path.join(ROOT, "models")):
            if f.startswith("personal_"):
                os.remove(os.path.join(ROOT, "models", f))

        # ---- sessions list and replay
        page.locator("nav.nav button[data-view=sessions]").dispatch_event("click")
        wait(page, "document.querySelectorAll('#sessionList tbody tr').length >= 1", 10, "sessions table")
        page.locator("#sessionList tbody tr:first-child button:has-text('Report')").dispatch_event("click")
        wait(page, "document.querySelectorAll('#report tbody tr').length === 3", 10, "session report with 3 blocks")
        time.sleep(1)
        shot(page, "sessions")
        page.locator("#sessionList tbody tr:first-child button:has-text('Replay')").dispatch_event("click")
        wait(page, "window.emotif.sourceKind === 'csv' && window.emotif.engine.C === 4", 10, "replay")

        # ---- arm: local twin with simulated brain intent
        page.evaluate("window.emotif.useSource('sim')")
        page.locator("nav.nav button[data-view=arm]").dispatch_event("click")
        page.evaluate("const c = document.getElementById('armClear'); c.checked = true; c.dispatchEvent(new Event('change'))")
        page.locator("#armEnable").dispatch_event("click")
        assert page.evaluate("window.emotif.arm.enabled")
        page.locator("#armMode button[data-v=intent]").dispatch_event("click")
        wait(page, "!!window.emotif.intent && window.emotif.engine.total > 512", 8)
        i0 = page.evaluate("window.emotif.arm.intentIndex")
        page.locator("#simBlink").dispatch_event("click")
        wait(page, f"window.emotif.arm.intentIndex === {(i0 + 1) % 8}", 6, "double blink advances the action")
        p0 = page.evaluate("[window.emotif.arm.pose.x, window.emotif.arm.pose.y, window.emotif.arm.pose.z]")
        page.locator("#simClench").dispatch_event("click")
        wait(page, "window.emotif.arm.moving", 4, "clench starts motion")
        time.sleep(1.2)
        p1 = page.evaluate("[window.emotif.arm.pose.x, window.emotif.arm.pose.y, window.emotif.arm.pose.z]")
        moved = sum((a - b) ** 2 for a, b in zip(p0, p1)) ** 0.5
        assert moved > 10, (p0, p1)
        wait(page, "!window.emotif.arm.moving", 6, "clench release stops motion")
        print(f"intent control: double blink -> '{page.evaluate('window.emotif.arm.action')}', clench moved the arm {moved:.0f} mm")
        shot(page, "arm_intent")
        page.locator("#armMode button[data-v=draw]").dispatch_event("click")
        page.locator("#penBtn").dispatch_event("click")
        time.sleep(4)
        assert page.evaluate("window.emotif.arm.trail.length") > 5
        shot(page, "arm_draw")

        # ---- arm through the server governor (simulated hardware transport)
        page.evaluate("window.emotif.link.api('/api/arm/connect', {kind:'sim'}).then(() => { window.emotif.arm.useServer = true; })")
        wait(page, "window.emotif.arm.connectedToReal", 5, "server arm")
        page.locator("#armMode button[data-v=embody]").dispatch_event("click")
        if page.evaluate("window.emotif.arm.enabled"):
            page.locator("#armEnable").dispatch_event("click")
        page.locator("#armEnable").dispatch_event("click")
        wait(page, "window.emotif.arm.server && window.emotif.arm.server.enabled", 5, "server arm enabled")
        s0 = page.evaluate("[window.emotif.arm.server.x, window.emotif.arm.server.z]")
        time.sleep(3)
        s1 = page.evaluate("[window.emotif.arm.server.x, window.emotif.arm.server.z]")
        assert s0 != s1, "server arm should follow the embody targets"
        page.keyboard.press("Escape")
        wait(page, "!window.emotif.arm.server.enabled", 3, "Esc stops the arm")
        print("server arm: follows targets; Esc emergency stop disables it")

        # ---- phone layout
        mob = new_page(browser, errors, viewport={"width": 390, "height": 844}, device_scale_factor=2)
        mob.goto(f"http://127.0.0.1:{port}/")
        wait(mob, "window.emotif && window.emotif.engine.windows > 10", 20)
        mob.locator(".tabbar button[data-view=live]").dispatch_event("click")
        time.sleep(1)
        assert mob.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "no horizontal scroll"
        shot(mob, "phone_live")
        mob.close()
        page.close()
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()


def check_brain_vs_face(page):
    """Brain vs face panel: both decoders report, and a simulated jaw clench shows up in the face strip.
    (Whether the brain-only decoder moves less is tested on real data and known-answer data, not here: the simulator's
    clench is largely removed by ASR and the detector before it reaches either decoder.)"""
    page.locator("nav.nav button[data-view=live]").dispatch_event("click")
    time.sleep(3)
    assert page.evaluate("(() => { const o = window.emotif.engine.last; return !!o && !!o.cmp && typeof o.faceHold === 'boolean'; })()")
    page.evaluate("window.emotif.sim.clench(5)")
    wait(page, "window.emotif.faceHist.slice(-12).some(p => p.jaw > 1.0)", 10, "jaw activity in the face strip")
    assert page.inner_text("#cmpBrain") != "—" and page.inner_text("#cmpStd") != "—"
    shot(page, "brain_vs_face")
    print(f"brain vs face panel OK: both decoders live, clench raised the jaw index to "
          f"{page.evaluate('Math.max(...window.emotif.faceHist.slice(-12).map(p => p.jaw))'):.1f} (log units over baseline)")


AXE = os.path.join(ROOT, "app", "node_modules", "axe-core", "axe.min.js")


def check_accessibility(page):
    """axe-core (WCAG 2.1 A/AA rules) on every view in light and dark themes: no serious or critical violations.
    Then keyboard: the first Tab lands on the skip link."""
    page.add_script_tag(path=AXE)
    found = []
    for theme in ("light", "dark"):
        page.evaluate(f"document.documentElement.dataset.theme='{theme}'")
        for view in ("live", "study", "arm", "models", "sessions", "method", "setup"):
            page.evaluate(f"window.emotif.go('{view}')")
            time.sleep(0.4)
            res = page.evaluate("""async () => { const r = await axe.run(document, {runOnly: {type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']}});
                return r.violations.filter(v => v.impact === 'serious' || v.impact === 'critical')
                  .map(v => ({id: v.id, n: v.nodes.length, ex: v.nodes.slice(0, 3).map(n => n.target.join(' ') + ' :: ' + (n.failureSummary || '').slice(0, 160))})); }""")
            found += [dict(r, view=view, theme=theme) for r in res]
    assert not found, json.dumps(found, indent=1)
    page.evaluate("document.activeElement && document.activeElement.blur(); window.scrollTo(0,0)")
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.classList.contains('skip-link')"), "first Tab should reach the skip link"
    page.evaluate("delete document.documentElement.dataset.theme; window.emotif.go('live')")
    print("accessibility OK: axe-core WCAG 2.1 AA, 7 views x 2 themes, no serious or critical violations; skip link first in tab order")


def check_edf_playback(page):
    """An EDF exported by any headset plays through the full pipeline."""
    import numpy as np
    from emotif import edf
    rng = np.random.default_rng(3)
    fs = 256
    t = np.arange(fs * 70) / fs
    x = np.stack([20 * np.sin(2 * np.pi * 10 * t + k) + rng.normal(0, 8, t.size) for k in range(4)])
    path = os.path.join(tempfile.mkdtemp(), "muse_export.edf")
    edf.write_edf(path, x, fs, ["TP9", "AF7", "AF8", "TP10"], extra=[("Accel X", "g", rng.normal(0, 1, 70 * 52), 52)])
    page.evaluate("window.emotif.engine.settings.calibSec = 20")
    page.set_input_files("#csvFile", path)
    wait(page, "window.emotif.sourceKind === 'csv' && window.emotif.engine.calibrated && window.emotif.engine.windows > 90", 60, "EDF playback decoded")
    assert page.evaluate("window.emotif.engine.channels.join()") == "TP9,AF7,AF8,TP10"
    wait(page, "/^(State: |Baseline done)/.test(document.getElementById('announcer').textContent)", 30, "screen-reader announcement")
    page.evaluate("window.emotif.go('setup')")
    page.locator("#sonify").dispatch_event("click")
    wait(page, "window.emotif.sonifier.running", 5, "sound display starts")
    page.locator("#sonify").dispatch_event("click")
    wait(page, "!window.emotif.sonifier.running", 5, "sound display stops")
    page.evaluate("window.emotif.go('live')")
    print("EDF playback OK: 4-channel EDF with an accelerometer channel decoded end to end; "
          f"announced '{page.evaluate('document.getElementById(\"announcer\").textContent')}'; sound display toggles")


def check_personal_states(page):
    """Calibration protocol listed; a calibrated personal model (made by the app's own states module on synthetic
    windows, two states marked as passing) loads for the participant and only those states appear live."""
    import subprocess
    code = ("const S=require(process.argv[1]);const {N_FEATURES:NF}=require(process.argv[2]);let a=5;"
            "const r=()=>{a=(a*1103515245+12345)%2147483648;return a/2147483648;};const d=[];let b=0;"
            "for(let rep=0;rep<3;rep++)for(let k=0;k<6;k++){b++;for(let w=0;w<12;w++){const x=new Float64Array(NF),f=new Float64Array(51);"
            "for(let i=0;i<50;i++)x[i*2]=r()+(i%6===k?1.5:0);for(let i=0;i<51;i++)f[i]=r()-0.5;d.push({x,f,y:k,block:b});}}"
            "const m=S.calibrateStates(d,S.STATES,{perms:19});S.STATES.forEach((n,i)=>{m.checks[n].on=i===1||i===3;});console.log(JSON.stringify(m));")
    dist = os.path.join(ROOT, "app", "dist-test")
    model = subprocess.run(["node", "-e", code, os.path.join(dist, "states.cjs"), os.path.join(dist, "montage.cjs")],
                           capture_output=True, text=True, check=True).stdout
    page.evaluate("window.emotif.go('study')")
    names = page.inner_text("#protoList")
    assert all(n in names for n in ("Personal state calibration", "Daily check-in", "Headset check", "Blink and jaw check")), names
    page.fill("#pcode", "T99")
    page.dispatch_event("#pcode", "change")
    assert "No calibration for T99" in page.inner_text("#statesReport")
    page.evaluate(f"localStorage.setItem('emotif.states:T99', {json.dumps(model)}); window.emotif.loadPersonalStates()")
    rep = page.inner_text("#statesReport")
    assert "Stress" in rep and "shown" in rep and "Face check" in rep, rep
    page.evaluate("window.emotif.go('live')")
    wait(page, "!document.getElementById('statesText').hidden && /Stress/.test(document.getElementById('statesText').textContent)", 30, "personal states live")
    txt = page.inner_text("#statesText")
    assert "Anger" in txt and "Calm" not in txt, txt
    print(f"personal states OK: calibration protocol listed, model loads per participant, live shows only passing states ({txt})")


def check_mind_control(page):
    """Natural world: crystals appear near the path; a focus command lifts the nearest, a relax command sets it down
    and it blooms (the decoder's command is stubbed so the test doesn't depend on simulated alpha)."""
    page.evaluate("window.emotif.go('live')")
    page.select_option("#worldStyle", "natural")
    page.evaluate("window.emotif.world.interactive = true; window.emotif.world.explore = true")
    wait(page, "window.emotif.world.aiming", 90, "a crystal within reach")
    page.evaluate("window.emotif.mind.update = () => ({ level: 1.2, command: 'focus', holdFor: 2, held: false })")
    wait(page, "window.emotif.world.carrying", 20, "focus lifts the crystal")
    assert page.inner_text("#mindLabel") == "Focus"
    page.evaluate("window.emotif.mind.update = () => ({ level: -1.2, command: 'relax', holdFor: 2, held: false })")
    wait(page, "!window.emotif.world.carrying && window.emotif.world.placedCount >= 1", 20, "relaxing sets it down")
    time.sleep(2.5)
    shot(page, "world_mind_control")
    page.evaluate("delete window.emotif.mind.update")
    page.select_option("#worldStyle", "auto")
    print("mind control OK: crystal targeted, lifted on focus, placed and blooming on relax")


def check_inner_world(page):
    """Prompt -> world in every art style, explorer movement, arm scene switch, and persistence across reloads."""
    import io
    import numpy as np
    from PIL import Image

    def viewport_spread():
        box = page.locator("#liveViewport").bounding_box()
        img = np.asarray(Image.open(io.BytesIO(page.screenshot(clip=box, timeout=120000))).convert("RGB"), dtype=float)
        return float(img.std())

    page.evaluate("window.emotif.go('live')")
    page.evaluate("window.emotif.world.radius = 2")   # software WebGL in CI: draw a 5 x 5 chunk world
    page.fill("#worldPrompt", "Minecraft forest with a lake")
    page.locator("#worldGo").dispatch_event("click")
    wait(page, "window.emotif.world.spec.name === 'Minecraft forest with a lake'", 10, "world from theme")
    assert page.evaluate("window.emotif.world.spec.style") == "voxel"
    for style in ("natural", "voxel", "lowpoly", "dream", "neon"):
        page.select_option("#worldStyle", style)
        assert page.evaluate("window.emotif.world.spec.style") == style
        page.evaluate("window.emotif.world.buildAll()")
        n, f = page.evaluate("[window.emotif.world.chunkCount, window.emotif.world.featureCount]")
        r = page.evaluate("window.emotif.world.viewRadius")
        assert (2 * r + 1) ** 2 <= n <= (2 * r + 3) ** 2 and f > 0, (style, r, n, f)
        time.sleep(0.6)
        spread = viewport_spread()
        assert spread > 8, f"{style}: the world looks blank (pixel std {spread:.1f})"
        shot(page, f"world_{style}")
    page.select_option("#worldStyle", "auto")
    x0 = page.evaluate("[window.emotif.world.pos.x, window.emotif.world.pos.z]")
    wait(page, f"Math.hypot(window.emotif.world.pos.x - {x0[0]}, window.emotif.world.pos.z - {x0[1]}) > 0.5", 30, "explorer wanders")
    page.locator("#worldExplore").dispatch_event("click")
    assert page.evaluate("window.emotif.world.explore") is False
    page.locator("#worldExplore").dispatch_event("click")
    page.fill("#worldPrompt", "neon synthwave mountains")
    page.press("#worldPrompt", "Enter")
    wait(page, "window.emotif.world.spec.style === 'neon'", 10, "theme on Enter")
    page.evaluate("window.emotif.go('arm')")
    assert page.evaluate("window.emotif.world.mode") == "arm"
    shot(page, "arm_scene")
    page.evaluate("window.emotif.go('live')")
    assert page.evaluate("window.emotif.world.mode") == "world"
    name = page.evaluate("window.emotif.world.spec.name")
    page.reload()
    wait(page, "window.emotif && window.emotif.world.ok", 20, "reload")
    assert page.evaluate("window.emotif.world.spec.name") == name, "world should persist across reloads"
    print(f"inner world OK: theme -> world on device, 5 art styles (natural, voxel, low-poly, dream, neon) render streamed chunks with features, explorer wanders, arm scene separate, '{name}' persists")


def run_without_server(browser, errors):
    """As a published artifact: one static file, no backend."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.join(ROOT, "web"))
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    page = new_page(browser, errors, viewport={"width": 1280, "height": 800})
    page.goto(f"http://127.0.0.1:{httpd.server_address[1]}/index.html")
    wait(page, "window.emotif && window.emotif.engine.windows > 10", 20, "standalone decoding")
    assert not page.evaluate("window.emotif.link.available")
    assert page.is_disabled("#recBtn")
    check_accessibility(page)
    check_edf_playback(page)
    page.evaluate("window.emotif.useSource('sim')")
    page.locator("nav.nav button[data-view=sessions]").dispatch_event("click")
    assert "server" in page.inner_text("#sessionList")
    # booth demo: self-running tour with captions, then restores the previous state
    page.evaluate("window.emotif.toggleDemo()")
    wait(page, "!document.getElementById('demoCaption').hidden && document.getElementById('demoText').textContent.length > 20", 10, "demo caption")
    wait(page, "window.emotif.engine.calibrated", 60, "demo baseline")
    wait(page, "document.getElementById('demoStep').textContent.startsWith('2')", 20, "demo step 2")
    page.evaluate("window.emotif.toggleDemo()")
    assert page.evaluate("document.getElementById('demoCaption').hidden && window.emotif.demo === null")
    check_personal_states(page)
    check_mind_control(page)
    check_inner_world(page)
    page.close()
    httpd.shutdown()
    print("browser-only mode OK (no server), booth demo OK")


def main():
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        run_without_server(b, errors)
        run_with_server(b, errors)
        b.close()
    bad = [e for e in errors if "net::ERR" not in e]
    assert not bad, bad
    print("browser e2e OK, no console errors")


if __name__ == "__main__":
    main()
