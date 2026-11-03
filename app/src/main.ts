import { Engine, emotionWord, fitCheck, type WindowEvent, type Normalization, type FitStatus, type Cleaning } from "./engine";
import { World, affectCss } from "./world";
import { STYLES, STYLE_LABELS, interpretPrompt, validateSpec, type Style, type WorldSpec } from "./worldspec";
import { Director, TARGETS, type DirectorMode, type VA } from "./director";
import { ArmController, INTENT_ACTIONS, INIT, type ArmMode } from "./arm";
import { ServerLink, ServerDeviceSource, type SessionMeta, type ServerState } from "./server";
import { Sonifier, Announcer } from "./sonify";
import { CortexSource } from "./cortex";
import { STATES, STATE_LABEL, calibrateStates, validateStates, type StateWindow } from "./states";
import { CheckinAccumulator, MIN_DAYS, trends, type CheckinEntry } from "./checkin";
import { AlphaControl } from "./interact";
import { Simulator, CsvPlayback, MuseBluetooth, OpenBciSerial, SIM_MONTAGES, type Source, type SourceSink, type SimMode } from "./sources";
import { IntentDetector, median } from "./intent";
import { ProtocolRunner, PROTOCOLS, samClass, type RunResult, type Protocol } from "./protocol";
import { validateModel, contributions, personalizationCv, fitLogistic, headInput, inputDim, isBrainOnly, modelInput, type ModelFile, type LabeledWindow, type Head } from "./model";
import { REGIONS, REGION_PAIRS, N_FEATURES, coverage, canonical, knownNames, DEVICES } from "./montage";
import { BANDS } from "./dsp";
import { drawCircumplex, drawCompare, drawTopomap, drawTraces, drawTimeline, type TimelineEvent } from "./charts";
import { ICONS } from "./icons";
import builtinMuse from "../../models/muse_mental_logreg.json";
import builtinGameemo from "../../models/gameemo_logreg.json";
import builtinBrain from "../../models/gameemo_brain.json";
import builtinInsight from "../../models/insight_brain.json";
import artifactModel from "../../models/artifact_detector.json";

// ------------------------------------------------------------------ helpers
type Attrs = Record<string, string | number | boolean | ((e: Event) => void) | undefined>;
function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...kids: (Node | string | null | undefined | false)[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v as EventListener);
    else if (k === "class") el.className = String(v);
    else if (k === "html") el.innerHTML = String(v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
  for (const c of kids) if (c !== null && c !== undefined && c !== false) el.append(c);
  return el;
}
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const store = {
  get<T>(k: string, d: T): T { try { const v = localStorage.getItem("emotif." + k); return v === null ? d : JSON.parse(v) as T; } catch { return d; } },
  set(k: string, v: unknown) { try { localStorage.setItem("emotif." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};
const fmtSigned = (x: number) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(2);
const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function seg(id: string, options: [string, string][], value: string, onChange: (v: string) => void): HTMLDivElement {
  const el = h("div", { class: "seg", id, role: "group" });
  for (const [v, label] of options) {
    const b = h("button", { type: "button", "aria-pressed": String(v === value), "data-v": v }, label);
    b.addEventListener("click", () => { el.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); onChange(v); });
    el.append(b);
  }
  return el;
}
function setSeg(id: string, v: string) { $(id)?.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === v))); }
function codeSeedLocal(code: string): number {
  let x = 2166136261;
  for (const ch of code) { x ^= ch.charCodeAt(0); x = Math.imul(x, 16777619); }
  return x >>> 0;
}

function download(name: string, text: string, type = "application/json") {
  const a = h("a", { href: URL.createObjectURL(new Blob([text], { type })), download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

const WORLD_LOCKED = "The world is locked during a study session, so every condition shows the same world.";

type ModelEntry = { id: string; name: string; desc: string; model: ModelFile | null; origin: string; file?: string };
type View = "live" | "study" | "arm" | "models" | "sessions" | "method" | "setup";
const VIEWS: [View, string, keyof typeof ICONS][] = [
  ["live", "Live", "live"], ["study", "Study", "study"], ["arm", "Robot arm", "arm"],
  ["models", "Models", "models"], ["sessions", "Sessions", "sessions"], ["method", "Method", "method"], ["setup", "Setup", "setup"],
];

// ------------------------------------------------------------------ app
class App {
  link = new ServerLink();
  engine = new Engine();
  world = new World();
  director = new Director();
  arm = new ArmController(this.link);
  sim = new Simulator();
  csv = new CsvPlayback();
  muse = new MuseBluetooth();
  checkinAcc: CheckinAccumulator | null = null;
  mind = new AlphaControl();
  brainSeeded = false;
  checkinDone: CheckinAccumulator | null = null;
  cyton = new OpenBciSerial();
  sonifier = new Sonifier();
  announcer: Announcer | null = null;
  private spoken = { word: "", since: 0, calibrated: false };
  source: Source | null = null;
  sourceKind = "sim";
  origin: "browser" | "server" = "browser";
  streamLabel = "";
  rawChannels: string[] = [];
  intent: IntentDetector | null = null;
  runner!: ProtocolRunner;
  view: View = "live";
  trail: VA[] = [];
  events: TimelineEvent[] = [];
  blocks: { t0: number; t1: number; label: string }[] = [];
  runs: RunResult[] = [];
  models: ModelEntry[] = [];
  modelId = "markers";
  uiDirector: { mode: DirectorMode; target: string } = { mode: "mirror", target: "calm" };
  protocolActive = false;
  recording: { id: string; seconds: number } | null = null;
  serverState: ServerState | null = null;
  pendingSamples: number[][] = [];
  lastStrata = 0;
  lastTrailLen = -1;
  lastDrawMode = false;
  bandView = 2;
  personal: { model: ModelFile; report: string } | null = null;
  armCal: { running: boolean; text: string } = { running: false, text: "" };
  traceMode: "clean" | "filtered" = "clean";
  demo: { on: boolean; t0: number; prevSource: string; prevMode: SimMode; prevCalib: number; prevModel: string } | null = null;
  lastFit: { status: FitStatus; ptp: number; mainsRatio: number }[] = [];
  worldStyle: Style | "auto" = "auto";
  cmpDim: "a" | "v" = "a";
  faceHist: { t: number; eye: number; jaw: number }[] = [];
  figures: Record<string, string> = (window as unknown as { __EMOTIF_FIGS?: Record<string, string> }).__EMOTIF_FIGS ?? {};

  async init() {
    document.documentElement.lang = "en";
    document.documentElement.dataset.theme = store.get<string>("theme", "") || "";
    if (!document.documentElement.dataset.theme) delete document.documentElement.dataset.theme;
    const saved = store.get<Partial<Engine["settings"]>>("settings", {});
    this.engine.settings = { ...this.engine.settings, ...saved, cleaning: { ...this.engine.settings.cleaning, ...(saved.cleaning ?? {}) } };
    try { this.engine.setArtifactModel(artifactModel); } catch { this.engine.setArtifactModel(null); }
    this.buildShell();
    this.world.reducedMotion = store.get("reducedMotion", window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
    if (!this.world.init()) {
      document.querySelectorAll(".viewport").forEach(v => v.append(h("div", { class: "nogl" }, "This browser can't run WebGL, so the 3-D world is off. Decoding, charts and arm control still work.")));
      $("worldStatus").textContent = "The inner world needs WebGL, which this browser doesn't have.";
      ($("worldGo") as HTMLButtonElement).disabled = true;
    } else {
      const saved = store.get<WorldSpec | null>("worldSpec", null);
      if (saved) this.world.setSpec(validateSpec(saved));
      this.world.explore = store.get("explore", true);
      this.world.interactive = store.get("worldMind", true);
      this.world.onInteract = (type, info) => {
        this.marker(`mind_${type}`, info);
        this.announcer?.say(`mind:${type}`, type === "lift" ? "Crystal lifted" : `Crystal placed. ${info.placed} placed so far`, 2);
      };
      ($("worldExplore") as HTMLInputElement).checked = this.world.explore;
      this.paintWorldStatus();
    }
    this.runner = new ProtocolRunner(document.body, {
      now: () => this.engine.t,
      event: (type, value) => this.marker(type, value),
      label: va => this.link.send({ type: "label", valence: va ? va.v : "", arousal: va ? va.a : "" }),
      condition: (mode, target) => {
        this.director.mode = mode;
        if (target && TARGETS[target]) this.director.target = { ...TARGETS[target].va };
        this.marker("condition", { mode, target });
      },
      recalibrate: secs => { this.engine.settings.calibSec = secs; this.engine.resetCalibration(); this.world.clearStrata(); },
      immersive: on => { document.body.classList.toggle("immersive", on); if (on) this.go("live"); },
      covered: on => { this.world.paused = on; },
      finish: r => this.onRunFinished(r),
    });
    this.engine.listeners.add(ev => this.onWindow(ev));
    this.loadModels();
    this.loadPersonalStates();
    this.go(store.get<View>("view", "live"));
    const serverUp = await this.link.detect();
    this.onServer(serverUp);
    this.useSource("sim");
    this.loop();
    window.addEventListener("keydown", e => {
      if (e.key === "Escape" && this.arm.enabled) this.estop();
    });
  }

  // ---------------------------------------------------------------- shell
  buildShell() {
    const nav = h("nav", { class: "nav", "aria-label": "Sections" });
    const tab = h("nav", { class: "tabbar", "aria-label": "Sections" });
    for (const [v, label, icon] of VIEWS) {
      nav.append(h("button", { "data-view": v, onclick: () => this.go(v), html: `${ICONS[icon]}<span>${label}</span>` }));
      tab.append(h("button", { "data-view": v, onclick: () => this.go(v), html: `${ICONS[icon]}<span>${label.split(" ")[0]}</span>` }));
    }
    const app = h("div", { class: "app" },
      h("aside", { class: "sidebar" },
        h("div", { class: "brand", html: `${ICONS.logo}<span>emotif</span><small>2.0</small>` }),
        nav,
        h("div", { class: "sidebar-foot" },
          h("div", { id: "serverLine" }, h("span", { class: "dot" }), "Checking for server…"),
          h("div", {}, "EEG stays on this computer."))),
      h("main", { class: "main", id: "main", tabindex: "-1" },
        h("header", { class: "topbar" },
          h("div", { class: "status" },
            h("span", { class: "pill", id: "stSource" }, "No signal"),
            h("span", { class: "pill", id: "stQuality", title: "Share of clean 2-second windows over the last 10 s" },
              h("span", { class: "qbar", html: "<i></i><i></i><i></i><i></i>" }), h("span", {}, "—")),
            h("span", { class: "pill warn", id: "stCalib" }, "Waiting"),
            h("span", { class: "pill", id: "stClean", hidden: true, title: "Artifact repair (ASR) and AI artifact detector" }, "Cleaning"),
            h("span", { class: "pill rec", id: "stRec", hidden: true }, "REC")),
          h("div", { class: "top-actions" },
            h("button", { class: "btn small", id: "demoBtn", onclick: () => this.toggleDemo(), title: "Self-running demo for an exhibit booth" }, "Booth demo"),
            h("button", { class: "estop", id: "estopTop", hidden: true, onclick: () => this.estop(), title: "Emergency stop (Esc)" }, "STOP ARM"),
            h("button", { class: "icon-btn", id: "themeBtn", "aria-label": "Switch theme", onclick: () => this.toggleTheme() }))),
        this.viewLive(), this.viewStudy(), this.viewArm(), this.viewModels(), this.viewSessions(), this.viewMethod(), this.viewSetup()),
      tab);
    const live = h("div", { id: "announcer", class: "sr-only", role: "status", "aria-live": "polite" });
    document.body.append(h("a", { class: "skip-link", href: "#main" }, "Skip to content"), app, live);
    this.announcer = new Announcer(live);
    this.paintThemeIcon();
  }

  go(v: View) {
    if (!VIEWS.some(x => x[0] === v)) v = "live";
    this.view = v;
    store.set("view", v);
    document.querySelectorAll<HTMLElement>(".view").forEach(s => { s.hidden = s.dataset.view !== v; });
    document.querySelectorAll<HTMLElement>(".nav button, .tabbar button").forEach(b => {
      if (b.dataset.view === v) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
    });
    if (this.world.ok) {
      if (v === "arm") { this.world.attach($("armViewport")); this.world.focusArm(true); }
      else { this.world.attach($("liveViewport")); this.world.focusArm(false); }
    }
    if (v === "sessions") this.refreshSessions();
    if (v === "models") this.renderModelDetail();
    if (v === "setup") this.renderMontage();
    if (v === "method") this.renderMethod();
  }

  setSound(on: boolean) {
    try { if (on) this.sonifier.start(); else this.sonifier.stop(); }
    catch (e) { this.showErr("srcErr", (e as Error).message); ($("sonify") as HTMLInputElement).checked = false; }
  }

  toggleTheme() {
    const r = document.documentElement;
    const dark = r.dataset.theme ? r.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    r.dataset.theme = dark ? "light" : "dark";
    store.set("theme", r.dataset.theme);
    this.paintThemeIcon();
  }
  paintThemeIcon() {
    const r = document.documentElement;
    const dark = r.dataset.theme ? r.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    $("themeBtn").innerHTML = dark ? ICONS.sun : ICONS.moon;
  }

  // ---------------------------------------------------------------- live view
  viewLive(): HTMLElement {
    const bands = h("div", { class: "bands" });
    BANDS.forEach(([name], i) => bands.append(h("div", { class: "band" }, h("span", {}, name), h("div", { class: "bar" }, h("i", { id: `bb${i}` })), h("b", { id: `bv${i}` }, "—"))));
    const bandSel = h("select", { id: "topoBand", "aria-label": "Band shown on the scalp map", onchange: e => { this.bandView = Number((e.target as HTMLSelectElement).value); } });
    BANDS.forEach(([name, lo, hi], i) => bandSel.append(h("option", { value: i, selected: i === 2 }, `${name} ${lo}–${hi} Hz`)));
    const target = h("select", { id: "dirTarget", "aria-label": "Target state", onchange: e => {
      this.uiDirector.target = (e.target as HTMLSelectElement).value;
      if (!this.protocolActive) this.director.target = { ...TARGETS[this.uiDirector.target].va };
    } });
    Object.entries(TARGETS).forEach(([k, t]) => target.append(h("option", { value: k }, t.label)));
    return h("section", { class: "view", tabindex: "0", "data-view": "live" },
      h("div", { class: "live" },
        h("div", { class: "stage" },
          h("div", { class: "viewport", id: "liveViewport" }),
          h("div", { class: "readout" },
            h("span", { class: "eyebrow" }, "Decoded state"),
            h("span", { class: "word", id: "word" }, "—"),
            h("span", { class: "va", id: "vaText" }, "V +0.00 · A +0.00"),
            h("span", { class: "va", id: "extraText", hidden: true }, ""),
            h("span", { class: "va", id: "statesText", hidden: true }, ""),
            h("span", { class: "sub", id: "subText" }, "")),
          h("div", { class: "mind-hud", id: "mindHud", hidden: true, role: "status" },
            h("span", { class: "mind-label", id: "mindLabel" }, "Relax"),
            h("div", { class: "mind-bar" }, h("i", { id: "mindNeg" }), h("i", { id: "mindPos" })),
            h("span", { class: "mind-hint", id: "mindHint" }, "Focus to lift · relax to place")),
          h("div", { class: "stage-tools" },
            h("button", { class: "btn small", onclick: () => { this.world.focusArm(false); this.world.resumeAutoRotate(); } }, "Recentre"),
            h("button", { class: "btn small", onclick: () => this.world.clearStrata(), title: "Remove the beacons the explorer has left" }, "Clear trail")),
          h("div", { class: "base-card", id: "baseCard", hidden: true },
            h("div", { class: "base-top" }, h("b", {}, "Resting baseline"), h("span", { class: "num", id: "baseTime" }, "")),
            h("div", { class: "meter" }, h("i", { id: "baseBar" })),
            h("p", { id: "baseText" }, "Sit still with your eyes open and your jaw relaxed. Emotif is learning what your signal looks like at rest."),
            h("div", { class: "fit-chips", id: "baseFit" }),
            h("p", { class: "err", id: "baseErr", hidden: true })),
          h("div", { class: "demo-caption", id: "demoCaption", hidden: true },
            h("span", { class: "demo-step", id: "demoStep" }, ""), h("p", { id: "demoText" }, ""),
            h("button", { class: "btn small", onclick: () => this.toggleDemo() }, "End demo"))),
        h("div", { class: "timeline" },
          h("div", { class: "legend" },
            h("span", {}, h("i", { style: "background:var(--accent)" }), "Valence"),
            h("span", {}, h("i", { style: "background:var(--ink-2)" }), "Arousal"),
            h("span", { class: "grow" }),
            h("span", { id: "timeText", class: "num" }, "0:00")),
          h("canvas", { class: "plot", id: "timeline" })),
        h("aside", { class: "live-side" },
          h("section", { class: "panel" }, h("h3", {}, "Affect map", h("span", { class: "chip", id: "srcChip" }, "markers")),
            h("canvas", { class: "plot", id: "circ" }),
            h("p", { class: "note", id: "circNote" }, "Dot: decoded state. Diamond: what the world is showing.")),
          h("section", { class: "panel", id: "cmpPanel" }, h("div", { class: "panel-h" }, "Brain vs face",
              seg("cmpDim", [["a", "Arousal"], ["v", "Valence"]], "a", v => { this.cmpDim = v as "a" | "v"; })),
            h("canvas", { class: "plot cmp-plot", id: "cmpPlot", role: "img", "aria-label": "Brain-only and standard decoder outputs over the last 30 seconds, with eye and jaw activity" }),
            h("div", { class: "cmp-legend" },
              h("span", {}, h("i", { class: "sw solid" }), "Brain-only ", h("b", { class: "num", id: "cmpBrain" }, "—")),
              h("span", {}, h("i", { class: "sw dash" }), "Standard ", h("b", { class: "num", id: "cmpStd" }, "—"))),
            h("p", { class: "note", id: "cmpNote" }, "")),
          h("section", { class: "panel" }, h("div", { class: "panel-h" }, "Environment"),
            seg("dirMode", [["mirror", "Mirror"], ["guide", "Guide"], ["sham", "Sham"]], "mirror", v => {
              this.uiDirector.mode = v as DirectorMode;
              if (!this.protocolActive) this.director.mode = this.uiDirector.mode;
              this.marker("director_mode", v);
            }),
            h("div", { class: "row" }, h("label", { class: "field" }, "Guide toward", target)),
            h("p", { class: "note", id: "dirNote" }, "Mirror: the world shows your decoded state.")),
          this.worldPanel(),
          h("section", { class: "panel" }, h("div", { class: "panel-h" }, "Scalp", bandSel),
            h("canvas", { class: "plot", id: "topo" }),
            h("p", { class: "note" }, "Band power relative to your baseline (red higher, blue lower). Ring colour shows contact quality.")),
          h("section", { class: "panel" }, h("div", { class: "panel-h" }, "Signals",
              seg("traceMode", [["clean", "Cleaned"], ["filtered", "Before cleaning"]], "clean", v => { this.traceMode = v as "clean" | "filtered"; })),
            h("canvas", { class: "plot", id: "traces" }),
            h("div", { class: "clean-row", id: "cleanRow" }), bands),
          h("section", { class: "panel" }, h("h3", {}, "Why this state"), h("ul", { class: "why", id: "why" })))));
  }

  worldPanel(): HTMLElement {
    this.worldStyle = store.get<Style | "auto">("worldStyle", "auto");
    const style = h("select", { id: "worldStyle", onchange: e => this.setWorldStyle((e.target as HTMLSelectElement).value as Style | "auto") });
    style.append(h("option", { value: "auto", selected: this.worldStyle === "auto" }, "From the theme"));
    for (const st of STYLES) style.append(h("option", { value: st, selected: this.worldStyle === st }, STYLE_LABELS[st]));
    const theme = h("input", { type: "text", id: "worldPrompt", maxlength: 120, autocomplete: "off", placeholder: "optional, e.g. forest with a lake", onkeydown: e => {
      if ((e as KeyboardEvent).key === "Enter") { e.preventDefault(); this.generateWorld(theme.value); }
    } });
    return h("section", { class: "panel" }, h("h3", {}, "World"),
      h("p", { class: "note" }, "Your decoded state drives the light, colour, mist, wind, weather and pace. The theme and style only set the scenery."),
      h("label", { class: "field" }, "Art style", style),
      h("div", { class: "row theme-row" }, h("label", { class: "field" }, "Theme", theme),
        h("button", { class: "btn", id: "worldGo", onclick: () => this.generateWorld(theme.value) }, "Apply")),
      h("label", { class: "check" }, h("input", { type: "checkbox", id: "worldExplore", checked: true, onchange: e => {
        this.world.explore = (e.target as HTMLInputElement).checked; store.set("explore", this.world.explore);
      } }), h("span", {}, "Explorer wanders on its own (click the world, then arrow keys or W A S D steer)")),
      h("label", { class: "check" }, h("input", { type: "checkbox", id: "worldMind", checked: store.get("worldMind", true), onchange: e => {
        this.world.interactive = (e.target as HTMLInputElement).checked; store.set("worldMind", this.world.interactive);
      } }), h("span", {}, "Move objects with your brain: focus to lift the glowing crystal, relax to set it down (alpha rhythm, face ignored)")),
      h("p", { class: "note" }, "Each world is grown from your resting brain signal: when the baseline finishes, its spectrum seeds the land, so no two sessions are the same. Photographs from an open image library give the ground and sky their texture (credits in Method)."),
      h("p", { class: "note", id: "worldStatus", role: "status" }, ""));
  }

  setWorldStyle(st: Style | "auto") {
    this.worldStyle = st;
    store.set("worldStyle", st);
    if (this.protocolActive) { $("worldStatus").textContent = WORLD_LOCKED + " The new style applies after the session."; return; }
    if (st !== "auto" && this.world.ok && this.world.spec.style !== st) {
      this.world.setSpec({ ...this.world.spec, style: st });
      store.set("worldSpec", this.world.spec);
      this.paintWorldStatus();
    }
  }

  paintWorldStatus() {
    const sp = this.world.spec;
    $("worldStatus").textContent = `${sp.name} · ${STYLE_LABELS[sp.style]}`;
  }

  /** Build the scenery from a short theme, on this computer (no network, no AI service). */
  generateWorld(text: string) {
    if (!this.world.ok) return;
    if (this.protocolActive) { $("worldStatus").textContent = WORLD_LOCKED; return; }
    const theme = text.trim().slice(0, 120);
    const spec = interpretPrompt(theme, this.worldStyle === "auto" ? undefined : this.worldStyle);
    this.world.setSpec(spec);
    store.set("worldSpec", this.world.spec);
    this.paintWorldStatus();
    this.marker("world", { theme, style: spec.style });
  }

  // ---------------------------------------------------------------- study view
  viewStudy(): HTMLElement {
    const list = h("div", { class: "cards", id: "protoList" });
    const sel = store.get("protocol", "quick");
    for (const p of PROTOCOLS) {
      list.append(h("button", { class: "card-opt", "aria-pressed": String(p.id === sel), "data-id": p.id, onclick: e => {
        list.querySelectorAll(".card-opt").forEach(x => x.setAttribute("aria-pressed", String(x === e.currentTarget)));
        store.set("protocol", p.id);
      } }, h("b", {}, p.name), h("span", {}, p.summary), h("span", { class: "meta" }, `about ${p.minutes} min`)));
    }
    return h("section", { class: "view", tabindex: "0", "data-view": "study", hidden: true },
      h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "Run a study session"),
        h("p", {}, "Guided tasks with ratings after each block. Ratings label the EEG for personal models and for analysis."))),
      h("div", { class: "cols-2" },
        h("div", { class: "cards" },
          h("section", { class: "panel" }, h("h3", {}, "Protocol"), list),
          h("section", { class: "panel", id: "runSummary" }, h("h3", {}, "Results"), h("p", { class: "note" }, "Finished sessions appear here."))),
        h("div", { class: "cards" },
          h("section", { class: "panel" }, h("h3", {}, "Participant"),
            h("label", { class: "field" }, "Participant code (no names)", h("input", { type: "text", id: "pcode", value: store.get("pcode", "P01"), maxlength: 32, autocomplete: "off", spellcheck: "false", onchange: () => { store.set("pcode", this.code); this.loadPersonalStates(); } })),
            h("label", { class: "check" }, h("input", { type: "checkbox", id: "consent" }),
              h("span", {}, "Signed consent is on file (and parent or guardian permission if the participant is under 18), as approved by the IRB.")),
            h("label", { class: "check" }, h("input", { type: "checkbox", id: "recordRun", checked: true }), h("span", { id: "recordLabel" }, "Record EEG, events and ratings to this computer")),
            h("div", { class: "row" },
              h("button", { class: "btn primary", id: "startRun", onclick: () => this.startProtocol() }, "Start session"),
              h("button", { class: "btn", id: "recBtn", onclick: () => this.toggleRecording() }, "Record without protocol")),
            h("p", { class: "err", id: "studyErr", hidden: true }),
            h("p", { class: "note", id: "studyNote" }, "Calibrate on a calm, still baseline first; the protocol does this for you.")),
          h("section", { class: "panel" }, h("h3", {}, "Your states"), h("div", { id: "statesReport" })),
          h("section", { class: "panel" }, h("h3", {}, "Daily check-ins"), h("div", { id: "checkinReport" })),
          h("section", { class: "panel" }, h("h3", {}, "During a session"),
            h("p", { class: "note" }, "Keep the participant still and the jaw relaxed during tasks. In the closed-loop protocol the participant doesn't see which condition is running; the world simply changes. Press Stop session at any time; data recorded so far is kept.")))));
  }

  // ---------------------------------------------------------------- arm view
  viewArm(): HTMLElement {
    const lim = this.arm.local.lim;
    const wheel = h("div", { class: "action-wheel", id: "actionWheel" });
    INTENT_ACTIONS.forEach((a, i) => wheel.append(h("span", { "data-i": i }, a)));
    const num = (id: string, label: string, value: number, step: number, on: (v: number) => void) =>
      h("label", { class: "field" }, label, h("input", { type: "number", id, value, step, onchange: e => on(Number((e.target as HTMLInputElement).value)) }));
    const jogBtn = (label: string, dx: number, dy: number, dz: number, dt = 0) => h("button", { class: "btn", type: "button", onclick: () => {
      this.arm.manual = { x: this.arm.manual.x + dx, y: this.arm.manual.y + dy, z: this.arm.manual.z + dz, t: this.arm.manual.t + dt };
    } }, label);
    return h("section", { class: "view", tabindex: "0", "data-view": "arm", hidden: true },
      h("div", { class: "armview" },
        h("div", { class: "stage" },
          h("div", { class: "viewport", id: "armViewport" }),
          h("div", { class: "readout" },
            h("span", { class: "eyebrow", id: "armEyebrow" }, "Digital twin · simulated"),
            h("span", { class: "word", id: "armWord" }, "Embody"),
            h("span", { class: "va", id: "armPose" }, ""),
            h("span", { class: "sub", id: "armSub" }, ""))),
        h("aside", { class: "arm-side" },
          h("section", { class: "panel" }, h("h3", {}, "RoArm-M2-S"),
            seg("armLink", [["sim", "Simulated"], ["serial", "USB"], ["http", "Wi-Fi"]], store.get<string>("armLink", "sim"), v => { store.set("armLink", v); this.paintArmLink(); }),
            h("label", { class: "field", id: "armAddrWrap" }, h("span", { id: "armAddrLabel" }, "Address"),
              h("input", { type: "text", id: "armAddr", value: store.get("armAddr", ""), spellcheck: "false", onchange: e => store.set("armAddr", (e.target as HTMLInputElement).value) })),
            h("div", { class: "row" }, h("button", { class: "btn", id: "armConnect", onclick: () => this.armConnect() }, "Connect"), h("span", { class: "chip", id: "armState" }, "twin only")),
            h("p", { class: "note", id: "armLinkNote" }),
            h("p", { class: "err", id: "armErr", hidden: true })),
          h("section", { class: "panel" }, h("h3", {}, "Safety"),
            h("label", { class: "check" }, h("input", { type: "checkbox", id: "armClear", onchange: () => this.paintArmButtons() }),
              h("span", {}, "The space around the arm is clear and an adult is supervising.")),
            h("div", { class: "row" },
              h("button", { class: "btn primary", id: "armEnable", onclick: () => this.armEnable(), disabled: true }, "Enable motion"),
              h("button", { class: "btn", onclick: () => this.armHome() }, "Home"),
              h("button", { class: "btn danger", onclick: () => this.estop() }, "Stop (Esc)")),
            h("label", { class: "field" }, h("span", { id: "speedLabel" }, `Speed limit ${lim.speed_mm_s} mm/s`),
              h("input", { type: "range", id: "armSpeed", min: 20, max: 200, step: 10, value: lim.speed_mm_s, oninput: e => this.setArmSpeed(Number((e.target as HTMLInputElement).value)) })),
            h("p", { class: "note" }, "Every command passes a workspace limit, a speed limit and a 1-second watchdog. If the browser stops sending, the arm holds still.")),
          h("section", { class: "panel" }, h("h3", {}, "Control"),
            seg("armMode", [["embody", "Embody"], ["draw", "Draw"], ["intent", "Intent"], ["manual", "Manual"]], "embody", v => this.setArmMode(v as ArmMode)),
            h("div", { id: "modeEmbody" }, h("p", { class: "note" }, "The arm's posture follows the decoded state: lifted and open when pleasant, low and closed when unpleasant; it sways faster with higher arousal.")),
            h("div", { id: "modeDraw", hidden: true, class: "cards" },
              h("p", { class: "note" }, "Put a pen in the gripper. The arm plots the session's affect on paper: higher arousal further away, more pleasant to the right. With the pen down, lower the pen-down height a few millimetres at a time until the tip just touches the paper."),
              h("div", { class: "row" },
                num("drawCx", "Centre x (mm)", this.arm.draw.cx, 5, v => { this.arm.draw.cx = v; }),
                num("drawSize", "Square (mm)", this.arm.draw.size, 10, v => { this.arm.draw.size = v; }),
                num("drawZ", "Pen-down height z (mm)", this.arm.draw.penZ, 1, v => { this.arm.draw.penZ = v; })),
              h("div", { class: "row" },
                h("button", { class: "btn", id: "penBtn", onclick: () => { this.arm.draw.lifted = !this.arm.draw.lifted; this.marker("pen", this.arm.draw.lifted ? "up" : "down"); } }, "Pen down"),
                h("button", { class: "btn ghost", onclick: () => { this.arm.trail = []; } }, "Clear drawing"))),
            h("div", { id: "modeIntent", hidden: true, class: "cards" },
              h("p", { class: "note" }, "Double-blink to step to the next action. Clench your jaw to run it; relax to stop. These use eye and jaw muscle signals picked up by the headset."),
              h("div", { class: "big-action", id: "intentAction" }, INTENT_ACTIONS[0]),
              wheel,
              h("div", { class: "field" }, "Jaw activity", h("div", { class: "meter", id: "clenchMeter" }, h("i"))),
              h("div", { class: "field" }, "Eye channel", h("div", { class: "meter", id: "blinkMeter" }, h("i"))),
              h("div", { class: "row" },
                h("button", { class: "btn", id: "calBtn", onclick: () => this.calibrateIntent() }, "Calibrate (40 s)"),
                h("button", { class: "btn ghost", id: "simBlink", onclick: () => this.sim.doubleBlink() }, "Simulate double blink"),
                h("button", { class: "btn ghost", id: "simClench", onclick: () => this.sim.clench(1.5) }, "Simulate clench")),
              h("p", { class: "note", id: "calText" }, ""),
              h("label", { class: "field" }, "Speed while clenching (mm/s)", h("input", { type: "range", min: 10, max: 80, step: 5, value: this.arm.speedMmS, oninput: e => { this.arm.speedMmS = Number((e.target as HTMLInputElement).value); } }))),
            h("div", { id: "modeManual", hidden: true, class: "cards" },
              h("div", { class: "jog" },
                jogBtn("x +10", 10, 0, 0), jogBtn("y +10", 0, 10, 0), jogBtn("z +10", 0, 0, 10),
                jogBtn("x −10", -10, 0, 0), jogBtn("y −10", 0, -10, 0), jogBtn("z −10", 0, 0, -10),
                jogBtn("Open", 0, 0, 0, -0.3), jogBtn("Close", 0, 0, 0, 0.3), h("button", { class: "btn", onclick: () => { this.arm.manual = { ...INIT }; } }, "Init pose")))),
          h("section", { class: "panel" }, h("h3", {}, "Readout"),
            h("dl", { class: "kv", id: "armKv" })))));
  }

  // ---------------------------------------------------------------- models view
  viewModels(): HTMLElement {
    return h("section", { class: "view", tabindex: "0", "data-view": "models", hidden: true },
      h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "Models"),
        h("p", {}, "Every model reads the same 17 scalp regions, so one model runs on any headset. Missing regions are masked."))),
      h("div", { class: "cols-2" },
        h("div", { class: "cards" },
          h("section", { class: "panel" }, h("h3", {}, "Available"), h("div", { class: "cards", id: "modelList" }),
            h("div", { class: "row" }, h("label", { class: "btn", for: "modelFile" }, "Load model file…"),
              h("input", { type: "file", id: "modelFile", accept: ".json,application/json", hidden: true, onchange: e => this.uploadModel(e) })),
            h("p", { class: "err", id: "modelErr", hidden: true })),
          h("section", { class: "panel" }, h("h3", {}, "Personalize"), h("div", { id: "personal" }))),
        h("section", { class: "panel", id: "modelDetail" })));
  }

  // ---------------------------------------------------------------- sessions view
  viewSessions(): HTMLElement {
    return h("section", { class: "view", tabindex: "0", "data-view": "sessions", hidden: true },
      h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "Sessions"),
        h("p", {}, "Recordings saved by the Emotif server. Each one loads straight into train.py.")),
        h("button", { class: "btn", onclick: () => this.refreshSessions() }, "Refresh")),
      h("section", { class: "panel" }, h("div", { id: "sessionList", class: "table-wrap" })),
      h("section", { class: "panel report", id: "report", hidden: true }));
  }

  // ---------------------------------------------------------------- setup view
  viewSetup(): HTMLElement {
    const montageSel = h("select", { id: "simMontage", onchange: e => { this.sim.montage = (e.target as HTMLSelectElement).value; store.set("simMontage", this.sim.montage); if (this.sourceKind === "sim") this.useSource("sim"); } });
    Object.entries(SIM_MONTAGES).forEach(([k, m]) => montageSel.append(h("option", { value: k, selected: k === store.get("simMontage", "lab8") }, m.label)));
    this.sim.montage = store.get("simMontage", "lab8");
    const simMode = h("select", { id: "simMode", onchange: e => { this.sim.mode = (e.target as HTMLSelectElement).value as SimMode; } },
      h("option", { value: "tour" }, "Guided tour: rest, then the four states"),
      h("option", { value: "manual" }, "Manual: drag on the affect map"),
      h("option", { value: "participant" }, "Virtual participant: responds to the world"));
    const boardSel = h("select", { id: "boardSel" });
    const card = (kind: string, title: string, body: Node[]) =>
      h("div", { class: "card-opt", "data-kind": kind, "aria-current": "false", role: "group", "aria-label": title }, h("b", {}, title), ...body,
        h("div", { class: "row" }, h("button", { class: "btn small", id: `use-${kind}`, onclick: () => this.useSource(kind) }, "Use this source")));
    const datalist = h("datalist", { id: "electrodeNames" });
    knownNames().filter(n => /^[A-Z]+[0-9Z]+$/.test(n)).sort().forEach(n => datalist.append(h("option", { value: n.replace(/Z$/, "z").replace(/^FP/, "Fp") })));
    return h("section", { class: "view", tabindex: "0", "data-view": "setup", hidden: true },
      h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "Setup"), h("p", {}, "Choose a signal source, name its electrodes, and tune processing."))),
      h("div", { class: "cols" },
        h("section", { class: "panel" }, h("h3", {}, "Signal source"),
          h("div", { class: "cards", id: "sourceCards" },
            card("sim", "Simulator", [h("span", {}, "Synthetic EEG for demos and testing, including blinks and jaw clenches."),
              h("label", { class: "field" }, "Electrode layout", montageSel), h("label", { class: "field" }, "Behaviour", simMode)]),
            card("muse", "Muse 2 / Muse S over Bluetooth", [h("span", {}, "Direct from the browser in Chrome or Edge. No other software needed.")]),
            card("emotiv", "Emotiv Insight / EPOC X", [h("span", {}, "Through the EMOTIV Launcher on this computer (Cortex). Needs a free Cortex app ID and an Emotiv licence that includes raw EEG."),
              h("div", { class: "row" },
                h("label", { class: "field" }, "Client ID", h("input", { type: "text", id: "cxId", autocomplete: "off", value: store.get("cxId", "") })),
                h("label", { class: "field" }, "Client secret", h("input", { type: "password", id: "cxSecret", autocomplete: "off", value: store.get("cxSecret", "") }))),
              h("label", { class: "field" }, "Headset ID (blank = first Insight, else first headset)", h("input", { type: "text", id: "cxHeadset", placeholder: "INSIGHT-XXXXXXXX", value: store.get("cxHeadset", "") })),
              h("p", { class: "note" }, "The ID and secret are kept in this browser only.")]),
            card("cyton", "OpenBCI Cyton over USB", [h("span", {}, "Direct from the browser (Web Serial) in Chrome or Edge: plug in the dongle, no other software. 8 channels at 250 Hz; rename electrodes on the right.")]),
            card("server", "Headset through the Emotif server", [h("span", { id: "serverCardNote" }, "40 boards via BrainFlow: OpenBCI, Muse, Neurosity, g.tec Unicorn, ANT Neuro, Mentalab, BrainBit, FreeEEG and more."),
              h("label", { class: "field" }, "Board", boardSel),
              h("div", { class: "row" },
                h("label", { class: "field" }, "Serial port", h("input", { type: "text", id: "bfSerial", placeholder: "/dev/ttyUSB0 or COM3", value: store.get("bfSerial", "") })),
                h("label", { class: "field" }, "MAC address", h("input", { type: "text", id: "bfMac", placeholder: "optional", value: store.get("bfMac", "") }))),
              h("label", { class: "field" }, "Electrode names, comma separated (optional)", h("input", { type: "text", id: "bfNames", placeholder: "Fp1, Fp2, C3, C4, P7, P8, O1, O2", value: store.get("bfNames", "") }))]),
            card("lsl", "Lab Streaming Layer", [h("span", {}, "Any app that publishes an LSL EEG stream (Emotiv, g.tec, OpenViBE)."),
              h("label", { class: "field" }, "Stream name (blank = first EEG stream)", h("input", { type: "text", id: "lslName", value: store.get("lslName", "") }))]),
            card("csv", "Recording file", [h("span", {}, "Replay EDF, EDF+ or BDF (exported by most headsets) or CSV (a time column plus one column per channel in µV)."),
              h("div", { class: "row" }, h("input", { type: "file", id: "csvFile", accept: ".csv,.edf,.bdf,text/csv", "aria-label": "Recording file", onchange: e => this.loadCsv(e) }),
                h("label", { class: "field", style: "max-width:140px" }, "Hz if no time column", h("input", { type: "number", id: "csvFs", value: 256, min: 100 })))])),
          h("p", { class: "err", id: "srcErr", hidden: true })),
        h("section", { class: "panel" }, h("h3", {}, "Electrodes"),
          h("p", { class: "note" }, "Name each channel with its 10-20 position so the decoder knows where it sits. Unknown names are ignored."),
          h("div", { id: "montageTable", class: "table-wrap" }), datalist,
          h("canvas", { class: "plot topo-lg", id: "montageTopo" }),
          h("p", { class: "note", id: "montageCov" })),
        h("section", { class: "panel" }, h("h3", {}, "Processing"),
          h("div", { class: "row" },
            h("label", { class: "field" }, "Baseline (s)", h("input", { type: "number", id: "calibSec", min: 10, max: 300, step: 5, value: this.engine.settings.calibSec, onchange: e => this.setSetting("calibSec", Number((e.target as HTMLInputElement).value)) })),
            h("label", { class: "field" }, "Reject above (µV p-p)", h("input", { type: "number", id: "rejectUv", min: 0, step: 10, value: this.engine.settings.rejectUv, onchange: e => this.setSetting("rejectUv", Number((e.target as HTMLInputElement).value)) }))),
          h("label", { class: "field" }, "Normalization",
            h("select", { id: "normSel", onchange: e => this.setSetting("normalization", (e.target as HTMLSelectElement).value as Normalization) },
              h("option", { value: "auto", selected: this.engine.settings.normalization === "auto" }, "Automatic (baseline for markers, whole session for models)"),
              h("option", { value: "baseline", selected: this.engine.settings.normalization === "baseline" }, "Resting baseline (frozen)"),
              h("option", { value: "session", selected: this.engine.settings.normalization === "session" }, "Whole session (running)"))),
          h("label", { class: "check" }, h("input", { type: "checkbox", id: "cleanAsr", checked: this.engine.settings.cleaning.asr, onchange: () => this.setCleaning() }),
            h("span", {}, "Repair artifacts (Artifact Subspace Reconstruction)")),
          h("label", { class: "field" }, "Repair threshold k (lower repairs more; 20 tested best)",
            h("input", { type: "number", id: "cleanK", min: 3, max: 50, step: 1, value: this.engine.settings.cleaning.asrK, onchange: () => this.setCleaning() })),
          h("label", { class: "check" }, h("input", { type: "checkbox", id: "cleanDet", checked: this.engine.settings.cleaning.detector, onchange: () => this.setCleaning() }),
            h("span", {}, "AI artifact detector: leave out channels with eye or muscle activity")),
          h("button", { class: "btn", onclick: () => { this.engine.resetCalibration(); this.world.clearStrata(); this.marker("recalibrate", ""); } }, "Recalibrate now"),
          h("p", { class: "note" }, "Pipeline: 0.5 Hz high-pass and 50/60 Hz notch, then ASR repairs eye, muscle and movement bursts using statistics learned from your baseline, then the AI detector checks every channel of every 2-second window. Changing these starts a new baseline.")),
        h("section", { class: "panel" }, h("h3", {}, "Electrode fit"),
          h("p", { class: "note" }, "Live contact check. Fix anything that isn't green before recording."),
          h("canvas", { class: "plot topo-lg", id: "fitTopo" }),
          h("ul", { class: "fit-list", id: "fitList" })),
        h("section", { class: "panel" }, h("h3", {}, "Display"),
          h("label", { class: "check" }, h("input", { type: "checkbox", id: "reduceMotion", checked: this.world.reducedMotion, onchange: e => { this.world.reducedMotion = (e.target as HTMLInputElement).checked; store.set("reducedMotion", this.world.reducedMotion); } }),
            h("span", {}, "Reduce motion in the 3-D world")),
          h("label", { class: "check" }, h("input", { type: "checkbox", id: "sonify", onchange: e => this.setSound((e.target as HTMLInputElement).checked) }),
            h("span", {}, "Play the decoded state as sound (pitch and chord follow valence, pulse follows arousal)")),
          h("p", { class: "note" }, "Screen readers hear the state when it changes and holds for 5 seconds, at most every 10 seconds. The world's brightness can't change fast enough to flash (fewer than 3 flashes per second, WCAG 2.3.1), even if the decoder jumps."),
          h("p", { class: "note" }, "Theme follows your system; switch it with the button at the top right.")),
        h("section", { class: "panel" }, h("h3", {}, "Privacy and safety"),
          h("p", { class: "note" }, "EEG never leaves this computer. Recordings use participant codes, not names. Emotion estimates from EEG are uncertain: treat them as research measurements, never as a diagnosis."),
          h("p", { class: "note" }, "Research with people needs approval from your school's review board (and for ISEF, the SRC/IRB forms) before you collect any data."))));
  }

  // ---------------------------------------------------------------- sources
  sink(origin: "browser" | "server"): SourceSink {
    return {
      onMeta: (fs, channels, label) => this.onMeta(fs, channels, label, origin),
      onRows: rows => this.onRows(rows),
      onStatus: (text, kind) => this.setSourceStatus(text, kind ?? "ok"),
      onEnd: () => { this.setSourceStatus("Signal ended", "warn"); },
    };
  }

  useSource(kind: string) {
    if (this.protocolActive) { this.showErr("srcErr", "Stop the running session before switching sources."); return; }
    this.source?.stop();
    if (this.origin === "browser" && this.link.available) this.link.send({ type: "stream_end" });
    this.showErr("srcErr", "");
    this.sourceKind = kind;
    store.set("sourceKind", kind);
    document.querySelectorAll<HTMLElement>("#sourceCards .card-opt").forEach(c => c.setAttribute("aria-current", String(c.dataset.kind === kind)));
    let src: Source;
    if (kind === "muse") src = this.muse;
    else if (kind === "cyton") src = this.cyton;
    else if (kind === "emotiv") {
      const cfg = { clientId: $<HTMLInputElement>("cxId").value.trim(), clientSecret: $<HTMLInputElement>("cxSecret").value.trim(),
        headsetId: $<HTMLInputElement>("cxHeadset").value.trim() || undefined };
      store.set("cxId", cfg.clientId); store.set("cxSecret", cfg.clientSecret); store.set("cxHeadset", cfg.headsetId ?? "");
      src = new CortexSource(cfg);
    }
    else if (kind === "csv") src = this.csv;
    else if (kind === "server" || kind === "lsl") {
      if (!this.link.available) { this.showErr("srcErr", "Start the Emotif server on this computer (python -m emotif.server) and open http://localhost:8000."); return; }
      const cfg: Record<string, unknown> = kind === "lsl" ? { kind: "lsl", lsl_name: $<HTMLInputElement>("lslName").value.trim() } : {
        kind: "brainflow", board: $<HTMLSelectElement>("boardSel").value, serial_port: $<HTMLInputElement>("bfSerial").value.trim(),
        mac_address: $<HTMLInputElement>("bfMac").value.trim(),
        channels: $<HTMLInputElement>("bfNames").value.split(",").map(s => s.trim()).filter(Boolean),
      };
      store.set("bfSerial", $<HTMLInputElement>("bfSerial").value.trim());
      store.set("bfMac", $<HTMLInputElement>("bfMac").value.trim());
      store.set("bfNames", $<HTMLInputElement>("bfNames").value.trim());
      store.set("lslName", $<HTMLInputElement>("lslName").value.trim());
      if (!(cfg.channels as string[] | undefined)?.length) delete cfg.channels;
      src = new ServerDeviceSource(this.link, cfg);
    } else src = this.sim;
    this.source = src;
    this.origin = kind === "server" || kind === "lsl" ? "server" : "browser";
    this.setSourceStatus("Starting…", "warn");
    Promise.resolve(src.start(this.sink(this.origin))).catch(e => this.setSourceStatus((e as Error).message, "error"));
  }

  loadCsv(e: Event) {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f) return;
    const binary = /\.(edf|bdf)$/i.test(f.name);
    const r = new FileReader();
    r.onload = () => {
      try {
        if (binary) {
          const skipped = this.csv.loadEdf(r.result as ArrayBuffer, f.name);
          this.useSource("csv");
          if (skipped.length) this.setSourceStatus(`Playing ${f.name}; left out ${skipped.join(", ")} (not EEG voltage at the main rate)`, "ok");
        } else { this.csv.load(String(r.result), Number($<HTMLInputElement>("csvFs").value) || 256, f.name); this.useSource("csv"); }
      } catch (err) { this.showErr("srcErr", (err as Error).message); }
    };
    if (binary) r.readAsArrayBuffer(f); else r.readAsText(f);
  }

  montageKey(label: string, channels: string[]) { return `montage:${label.split(" · ")[0]}:${channels.join(",")}`; }

  onMeta(fs: number, channels: string[], label: string, origin: "browser" | "server") {
    this.rawChannels = channels.slice();
    const saved = store.get<string[] | null>(this.montageKey(label, channels), null);
    const names = saved && saved.length === channels.length ? saved : channels;
    try { this.engine.setStream(fs, names); } catch (e) { this.setSourceStatus((e as Error).message, "error"); this.source?.stop(); return; }
    this.streamLabel = label;
    this.sim.calibSec = this.engine.settings.calibSec;
    this.intent = new IntentDetector(fs, names);
    const cal = store.get<IntentDetector["cal"] | null>(`intentCal:${label.split(" · ")[0]}`, null);
    if (cal) this.intent.cal = cal;
    this.intent.listeners.add(ev => this.onIntent(ev.kind));
    this.engine.rowTaps.clear();
    this.engine.rowTaps.add(row => this.intent?.process(row));
    this.trail = []; this.events = []; this.blocks = [];
    this.world.clearStrata();
    this.lastStrata = 0;
    if (origin === "browser" && this.link.available) this.link.send({ type: "stream_meta", fs, channels, label });
    $("stSource").textContent = label;
    this.renderMontage();
    this.paintArmButtons();
  }

  onRows(rows: number[][]) {
    this.engine.push(rows);
    if (this.origin === "browser" && this.recording) {
      for (const r of rows) this.pendingSamples.push(r);
      if (this.pendingSamples.length >= 64) { this.link.send({ type: "samples", samples: this.pendingSamples }); this.pendingSamples = []; }
    }
  }

  setSourceStatus(text: string, kind: "ok" | "warn" | "error") {
    const el = $("stSource");
    el.className = `pill ${kind === "error" ? "bad" : kind === "warn" ? "warn" : ""}`;
    if (kind !== "ok" || !this.streamLabel) el.textContent = text;
    else el.textContent = this.streamLabel;
    if (kind === "error") this.showErr("srcErr", text);
  }

  showErr(id: string, msg: string) { const el = $(id); if (!el) return; el.textContent = msg; el.hidden = !msg; }

  setSetting<K extends keyof Engine["settings"]>(k: K, v: Engine["settings"][K]) {
    if (k === "calibSec") v = Math.max(10, Math.min(300, Number(v) || 30)) as Engine["settings"][K];
    if (k === "rejectUv") v = Math.max(0, Number(v) || 0) as Engine["settings"][K];
    this.engine.settings[k] = v;
    this.sim.calibSec = this.engine.settings.calibSec;
    store.set("settings", this.engine.settings);
    if (k !== "rejectUv") this.engine.resetCalibration();
  }

  // ---------------------------------------------------------------- server
  onServer(up: boolean) {
    const line = $("serverLine");
    line.replaceChildren(h("span", { class: `dot ${up ? "ok" : ""}` }), up ? "Server connected" : "Browser-only mode");
    $<HTMLInputElement>("recordRun").disabled = !up;
    $("recordLabel").textContent = up ? "Record EEG, events and ratings to this computer" : "Recording needs the Emotif server (python -m emotif.server)";
    $<HTMLButtonElement>("recBtn").disabled = !up;
    if (!up) $<HTMLInputElement>("recordRun").checked = false;
    $("serverCardNote").textContent = up ? "40 boards via BrainFlow: OpenBCI, Muse, Neurosity, g.tec Unicorn, ANT Neuro, Mentalab, BrainBit, FreeEEG and more." : "Needs the Emotif server: run python -m emotif.server, then open http://localhost:8000.";
    this.paintArmLink();
    if (!up) return;
    this.link.api<{ boards: { id: string; label: string; available: boolean }[] }>("/api/boards").then(r => {
      const sel = $<HTMLSelectElement>("boardSel");
      sel.replaceChildren();
      for (const b of r.boards.filter(x => x.available)) sel.append(h("option", { value: b.id, selected: b.id === store.get("board", "MUSE_2_BOARD") }, b.label));
      sel.onchange = () => store.set("board", sel.value);
    }).catch(() => undefined);
    this.link.api<{ models: { file: string; name: string; heads: string[]; datasets: string[]; model: string }[] }>("/api/models").then(r => {
      for (const m of r.models) {
        if (this.models.some(x => x.id === `server:${m.file}` || x.name === m.name)) continue;   // bundled copy already listed
        this.models.push({ id: `server:${m.file}`, name: m.name, desc: `${m.model} · ${m.heads.join(" + ")} · ${m.datasets.join(", ")}`, model: null, origin: "server", file: m.file });
      }
      this.renderModelList();
    }).catch(() => undefined);
    const poll = () => this.link.api<ServerState>("/api/state").then(s => { this.serverState = s; this.recording = s.recording; this.paintRec(); }).catch(() => undefined);
    poll();
    window.setInterval(poll, 1000);
    this.link.on("close", () => { this.recording = null; this.paintRec(); });
  }

  paintRec() {
    const el = $("stRec");
    el.hidden = !this.recording;
    if (this.recording) el.textContent = `REC ${mmss(this.recording.seconds)}`;
    $("recBtn").textContent = this.recording ? "Stop recording" : "Record without protocol";
  }

  async toggleRecording() {
    if (this.recording) { await this.stopRecording(); return; }
    await this.startRecording("free");
  }
  async startRecording(protocol: string): Promise<boolean> {
    this.showErr("studyErr", "");
    const code = $<HTMLInputElement>("pcode").value.trim();
    store.set("pcode", code);
    try {
      const r = await this.link.api<{ ok: boolean; id: string }>("/api/session/start", {
        participant: code, consent: $<HTMLInputElement>("consent").checked, protocol,
        notes: `model=${this.modelId}; montage=${this.engine.channels.join(" ")}`,
      });
      this.recording = { id: r.id, seconds: 0 };
      this.pendingSamples = [];
      this.paintRec();
      this.marker("model", this.modelId);
      return true;
    } catch (e) { this.showErr("studyErr", (e as Error).message); return false; }
  }
  async stopRecording() {
    if (this.pendingSamples.length) { this.link.send({ type: "samples", samples: this.pendingSamples }); this.pendingSamples = []; }
    try { await this.link.api("/api/session/stop", {}); } catch { /* server gone */ }
    this.recording = null;
    this.paintRec();
  }

  marker(type: string, value: unknown) {
    this.link.send({ type: "marker", name: type, value });
    if (type === "block_start" && (value as { block: string }).block === "checkin") this.checkinAcc = new CheckinAccumulator(this.engine.channels.slice());
    if (type === "block_end" && this.checkinAcc && (value as { block: string }).block === "checkin") { this.checkinDone = this.checkinAcc; this.checkinAcc = null; }
    if (["block_start", "baseline_start", "recalibrate", "director_mode", "sam"].includes(type)) this.events.push({ t: this.engine.t, label: type });
    if (type === "block_start") { const v = value as { block: string }; this.blocks.push({ t0: this.engine.t, t1: 0, label: v.block }); }
    if (type === "block_end" && this.blocks.length) this.blocks[this.blocks.length - 1].t1 = this.engine.t;
  }

  // ---------------------------------------------------------------- protocol
  async startProtocol(custom?: Protocol) {
    this.showErr("studyErr", "");
    if (!this.engine.C) { this.showErr("studyErr", "Connect a signal source first (Setup)."); return; }
    const pid = store.get("protocol", "quick");
    const p = custom ?? PROTOCOLS.find(x => x.id === pid) ?? PROTOCOLS[0];
    const code = $<HTMLInputElement>("pcode").value.trim() || "P01";
    if (!/^[A-Za-z0-9_-]+$/.test(code)) { this.showErr("studyErr", "Use letters, numbers, - or _ in the participant code."); return; }
    if (!$<HTMLInputElement>("consent").checked) { this.showErr("studyErr", "Confirm consent before starting a session."); return; }
    if ($<HTMLInputElement>("recordRun").checked && !this.recording) { if (!(await this.startRecording(p.id))) return; }
    this.protocolActive = true;
    if (this.sourceKind === "sim" && this.sim.mode === "tour") { this.sim.mode = "participant"; $<HTMLSelectElement>("simMode").value = "participant"; }
    this.director.setSham(Date.now() & 0xffff);
    this.marker("sham_seed", this.director.shamSeed);
    this.runner.start(p, code);
  }

  async onRunFinished(r: RunResult) {
    this.protocolActive = false;
    this.director.mode = this.uiDirector.mode;
    this.director.target = { ...TARGETS[this.uiDirector.target].va };
    this.runs.push(r);
    if (this.recording && $<HTMLInputElement>("recordRun").checked) await this.stopRecording();
    this.renderRunSummary(r);
    this.renderPersonal();
    if (r.protocol === "states" && !r.aborted) await this.calibratePersonalStates(r);
    if (r.protocol === "checkin" && !r.aborted) this.saveCheckin(r);
    this.go("study");
  }

  // ---------------------------------------------------------------- personal states and daily check-ins
  get code(): string { return ($<HTMLInputElement>("pcode")?.value.trim() || "P01"); }

  /** Windows from the calibration blocks, one per second, skipping each block's first 5 s (the task settling in). */
  stateWindows(r: RunResult): StateWindow[] {
    const out: StateWindow[] = [];
    r.blocks.forEach((b, i) => {
      const k = b.state ? (STATES as readonly string[]).indexOf(b.state) : -1;
      if (k < 0) return;
      let last = -1e9;
      for (const w of this.engine.featureLog) {
        if (w.t < b.t0 + 5 || w.t > b.t1 || w.t - last < 1) continue;
        last = w.t;
        out.push({ x: Float64Array.from(w.x), f: Float64Array.from(w.f), y: k, block: i });
      }
    });
    return out;
  }

  async calibratePersonalStates(r: RunResult) {
    const box = $("statesReport");
    box.replaceChildren(h("p", { class: "note" }, "Checking each state on rounds the model didn't see (199 permutations)…"));
    await new Promise(res => setTimeout(res, 30));
    try {
      const m = calibrateStates(this.stateWindows(r), STATES, { perms: 199, seed: codeSeedLocal(this.code) });
      // manipulation check: the participant's own 0-10 ratings of the target state, per state
      store.set(`states:${this.code}`, m);
      this.engine.personalStates = m;
      this.link.send({ type: "marker", name: "personal_states", value: { checks: m.checks, faceOnly: m.faceOnly, bacc: m.bacc, baccP: m.baccP } });
      this.renderStatesReport(r);
    } catch (e) { box.replaceChildren(h("p", { class: "err" }, (e as Error).message)); }
  }

  loadPersonalStates() {
    const raw = store.get<unknown>(`states:${this.code}`, null);
    try { this.engine.personalStates = raw ? validateStates(raw) : null; } catch { this.engine.personalStates = null; }
    this.renderStatesReport(null);
    this.renderCheckins();
  }

  renderStatesReport(r: RunResult | null) {
    const box = $("statesReport");
    if (!box) return;
    const m = this.engine.personalStates;
    if (!m) { box.replaceChildren(h("p", { class: "note" }, `No calibration for ${this.code} yet. Run "Personal state calibration" (about 28 minutes).`)); return; }
    const rated = (st: string) => {
      const v = (r?.blocks ?? []).filter(b => b.state === st && b.intensity !== undefined).map(b => b.intensity!);
      return v.length ? (v.reduce((a, x) => a + x, 0) / v.length).toFixed(1) : "—";
    };
    const tb = h("tbody");
    for (const st of m.states) {
      const c = m.checks[st];
      tb.append(h("tr", {}, h("td", {}, STATE_LABEL[st as keyof typeof STATE_LABEL] ?? st),
        h("td", { class: "n" }, Number.isFinite(c.auc) ? c.auc.toFixed(2) : "—"), h("td", { class: "n" }, c.p.toFixed(3)),
        h("td", { class: "n" }, rated(st)), h("td", {}, c.on ? "shown" : "not shown")));
    }
    box.replaceChildren(
      h("p", { class: "note" }, `${this.code}: ${m.blocks} blocks, ${m.windows} windows (${m.created.slice(0, 10)}). Held-out accuracy across all states ${(m.bacc * 100).toFixed(0)} % (chance ${(100 / m.states.length).toFixed(0)} %, permutation p = ${m.baccP.toFixed(3)}).`),
      h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", {}, "State"), h("th", { class: "n" }, "AUC on unseen rounds"), h("th", { class: "n" }, "p"), h("th", { class: "n" }, "Your rating (0-10)"), h("th", {}, "Live"))), tb)),
      h("p", { class: "note" }, `Face check: eye and jaw activity alone separate this person's states at ${(m.faceOnly.bacc * 100).toFixed(0)} % (p = ${m.faceOnly.p.toFixed(3)}). The live decoder removes what the face explains before it learns, so a state is shown only if the brain signal itself separates it.`),
      h("p", { class: "note" }, "A state is shown live only when its AUC on unseen rounds beats 199 shuffles of the state labels (p ≤ 0.05). These are personal research measurements, not a diagnosis."));
  }

  saveCheckin(r: RunResult) {
    const blk = r.blocks.find(b => b.id === "checkin");
    const acc = this.checkinDone;
    this.checkinDone = null;
    const sum = acc?.summarize();
    if (!blk?.checkin || !sum || !acc) { $("checkinReport").replaceChildren(h("p", { class: "err" }, "Not enough clean signal in the check-in (needs 20 usable windows). Try again.")); return; }
    const entries = store.get<CheckinEntry[]>(`checkins:${this.code}`, []);
    const date = new Date().toISOString().slice(0, 10);
    const e: CheckinEntry = { date, answers: blk.checkin, markers: sum.markers, windows: sum.windows, channels: acc.channels };
    const kept = entries.filter(x => x.date !== date);     // one check-in per day; a repeat replaces it
    kept.push(e);
    store.set(`checkins:${this.code}`, kept);
    this.link.send({ type: "marker", name: "checkin_summary", value: e });
    this.renderCheckins();
  }

  renderCheckins() {
    const box = $("checkinReport");
    if (!box) return;
    const entries = store.get<CheckinEntry[]>(`checkins:${this.code}`, []);
    if (!entries.length) { box.replaceChildren(h("p", { class: "note" }, `No check-ins for ${this.code} yet.`)); return; }
    const rows = trends(entries);
    const kids: Node[] = [h("p", { class: "note" }, `${entries.length} day${entries.length === 1 ? "" : "s"} recorded (latest ${entries[entries.length - 1].date}).`)];
    if (!rows.length) kids.push(h("p", { class: "note" }, `Trends appear after ${MIN_DAYS} days.`));
    else {
      const tb = h("tbody");
      const lab = { mood: "mood", sleepHours: "hours slept", sleepQuality: "sleep quality" };
      rows.slice().sort((a, b) => a.p - b.p).slice(0, 6).forEach(t => tb.append(h("tr", {}, h("td", {}, t.marker), h("td", {}, lab[t.answer]),
        h("td", { class: "n" }, t.rho.toFixed(2)), h("td", { class: "n" }, t.p.toFixed(3)), h("td", { class: "n" }, t.pHolm.toFixed(3)))));
      kids.push(h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", {}, "Resting marker"), h("th", {}, "vs"), h("th", { class: "n" }, "Spearman ρ"), h("th", { class: "n" }, "p"), h("th", { class: "n" }, "Holm p"))), tb)),
        h("p", { class: "note" }, `Across ${rows[0].n} days, within this person; the six strongest of ${rows.length} tests. Exploratory: a Holm p above 0.05 means no reliable link yet.`));
    }
    box.replaceChildren(...kids);
  }

  renderRunSummary(r: RunResult) {
    const box = $("runSummary");
    const meanIn = (t0: number, t1: number) => {
      const s = this.engine.history.filter(p => p.t >= t0 + 2 && p.t <= t1);
      if (!s.length) return null;
      return { v: s.reduce((a, p) => a + p.v, 0) / s.length, a: s.reduce((a, p) => a + p.a, 0) / s.length };
    };
    const tbody = h("tbody");
    for (const b of r.blocks.filter(x => x.task !== "rest")) {
      const m = meanIn(b.t0, b.t1);
      tbody.append(h("tr", {},
        h("td", {}, b.id), h("td", {}, b.condition),
        h("td", { class: "n" }, b.sam ? String(b.sam.valence) : "—"), h("td", { class: "n" }, b.sam ? String(b.sam.arousal) : "—"),
        h("td", { class: "n" }, m ? fmtSigned(m.v) : "—"), h("td", { class: "n" }, m ? fmtSigned(m.a) : "—")));
    }
    box.replaceChildren(h("h3", {}, "Results"),
      h("p", { class: "note" }, `${PROTOCOLS.find(p => p.id === r.protocol)?.name ?? r.protocol} · ${r.code}${r.aborted ? " · stopped early" : ""}. Block order: ${r.order.join(", ")}.`),
      h("div", { class: "table-wrap" }, h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", {}, "Block"), h("th", {}, "World"), h("th", { class: "n" }, "SAM val."), h("th", { class: "n" }, "SAM aro."), h("th", { class: "n" }, "Decoded V"), h("th", { class: "n" }, "Decoded A"))),
        tbody)),
      h("p", { class: "note" }, "SAM is 1–9. Decoded values are means over each block, −1 to +1. Open Models → Personalize to fit a model to these ratings."));
  }

  // ---------------------------------------------------------------- per-window updates
  /** Seed the world from the first calibrated window's spectrum (absolute band power per channel) and the clock. */
  seedWorldFromBrain(ev: WindowEvent) {
    if (!ev.bandPower || !this.world.ok) return;
    let h = 2166136261 ^ (Date.now() & 0xffffffff);
    for (const r of ev.bandPower) for (const v of r) { h ^= Math.round(Math.log10(v + 1e-9) * 1000) & 0xffff; h = Math.imul(h, 16777619); }
    const spec = this.world.spec;
    this.world.setSpec({ ...spec, seed: ((spec.seed ^ (h >>> 0)) >>> 0) % 1000000 });
    this.marker("world_seed", { seed: this.world.spec.seed });
    this.paintWorldStatus();
  }

  onWindow(ev: WindowEvent) {
    if (!ev.calibrated) this.brainSeeded = false;
    else if (!this.brainSeeded) { this.brainSeeded = true; this.seedWorldFromBrain(ev); }
    if (ev.calibrated) {
      const st = this.mind.update(ev.features, ev.reject || !!ev.out?.faceHold, 0.25);
      this.world.setControl(st.command, st.level);
      const hud = $("mindHud");
      hud.hidden = !this.world.interactive;
      if (this.world.interactive) {
        const L = Math.max(-1.5, Math.min(1.5, st.level)) / 1.5;
        $("mindPos").style.width = `${Math.max(0, L) * 50}%`;
        $("mindNeg").style.width = `${Math.max(0, -L) * 50}%`;
        $("mindLabel").textContent = st.held ? "Face activity · holding" : st.command === "focus" ? "Focus" : st.command === "relax" ? "Relax" : "Steady";
        $("mindHint").textContent = this.world.carrying ? "Relax to set it down" : "Focus to lift the crystal";
      }
    }
    if (this.checkinAcc && ev.bandPower && !ev.reject) this.checkinAcc.add(ev.bandPower, ev.masked);
    const sc = $("stCalib");
    if (!ev.calibrated) { sc.textContent = `Baseline ${Math.round(ev.calibProgress * this.engine.settings.calibSec)}/${this.engine.settings.calibSec} s`; sc.className = "pill warn"; }
    else if (ev.reject) { sc.textContent = "Artifact · holding"; sc.className = "pill bad"; }
    else if (ev.out?.faceHold) { sc.textContent = "Face activity · holding"; sc.className = "pill warn"; }
    else { sc.textContent = "Decoding"; sc.className = "pill ok"; }
    this.paintCleaning(ev);
    const ro = document.querySelector<HTMLElement>("#liveViewport ~ .readout");
    if (ro) ro.hidden = !ev.calibrated;
    const q = $("stQuality"), pct = Math.round(ev.quality * 100);
    q.className = `pill ${ev.quality > 0.8 ? "ok" : ev.quality > 0.5 ? "warn" : "bad"}`;
    q.querySelectorAll(".qbar i").forEach((b, i) => b.classList.toggle("on", ev.quality > i / 4));
    (q.lastElementChild as HTMLElement).textContent = `${pct}%`;
    if (ev.bandPower) {
      const tot = [0, 0, 0, 0, 0];
      ev.bandPower.forEach((r, c) => { if (ev.good[c]) for (let b = 0; b < 5; b++) tot[b] += r[b]; });
      const sum = tot.reduce((a, b) => a + b, 0) || 1;
      tot.forEach((p, b) => { const pctB = 100 * p / sum; $(`bb${b}`).style.width = `${pctB.toFixed(1)}%`; $(`bv${b}`).textContent = `${pctB.toFixed(0)}%`; });
    }
    if (ev.face) {
      let eye = 0, jaw = 0, n = 0;
      ev.face.forEach((r, c) => { if (ev.good[c]) { eye += r[0]; jaw += (r[1] + r[2]) / 2; n++; } });
      if (n) { this.faceHist.push({ t: ev.t, eye: eye / n, jaw: jaw / n }); if (this.faceHist.length > 4 * 120) this.faceHist.shift(); }
    }
    const o = ev.out;
    if (ev.calibrated !== this.spoken.calibrated) {
      this.spoken.calibrated = ev.calibrated;
      if (ev.calibrated) this.announcer?.say("calib", "Baseline done. Decoding.", 0);
    }
    if (o) {
      this.sonifier.update(o.v, o.a);
      const w = emotionWord(o.v, o.a);
      if (w !== this.spoken.word) { this.spoken.word = w; this.spoken.since = ev.t; }
      else if (ev.calibrated && ev.t - this.spoken.since >= 5) this.announcer?.say(`state:${w}`, `State: ${w}`, 10);
      this.trail.push({ v: o.v, a: o.a });
      if (this.trail.length > 80) this.trail.shift();
      $("word").textContent = emotionWord(o.v, o.a);
      $("word").style.color = affectCss(o.v, o.a);
      $("vaText").textContent = `V ${fmtSigned(o.v)} · A ${fmtSigned(o.a)}${o.conf !== null ? ` · p ${o.conf.toFixed(2)}` : ""}`;
      const extras = Object.entries(o.extra ?? {});
      $("extraText").hidden = !extras.length;
      // extra heads report the expected class value (-1..1); shown as 0-100 % toward the head's positive class
      $("extraText").textContent = extras.map(([k, x]) => `${k[0].toUpperCase()}${k.slice(1).replace(/_/g, " ")} ${Math.round((x + 1) * 50)} %`).join(" · ") + (extras.length ? " (experimental)" : "");
      $("subText").textContent = `${o.source === "model" ? this.models.find(m => m.id === this.modelId)?.name ?? "model" : "neuro-markers"} · world: ${this.director.mode}`;
      $("srcChip").textContent = o.source === "model" ? "model" : "markers";
      if (this.recording) this.link.send({ type: "affect", v: o.v, a: o.a, conf: o.conf, label: emotionWord(o.v, o.a), quality: ev.quality });
      const sts = Object.entries(o.states ?? {});
      $("statesText").hidden = !sts.length;
      $("statesText").textContent = sts.length ? "You: " + sts.sort((a, b) => b[1] - a[1]).map(([k, p]) => `${STATE_LABEL[k as keyof typeof STATE_LABEL] ?? k} ${Math.round(p * 100)} %`).join(" · ") : "";
      const top = sts.length ? sts.reduce((a, b) => (b[1] > a[1] ? b : a)) : null;
      this.world.setBrainState(top && top[1] > 0.45 ? top[0] : null);
      if (this.link.available) this.link.send({ type: "state", v: o.v, a: o.a, conf: o.conf, quality: ev.quality, hold: !!o.faceHold, extra: { ...(o.extra ?? {}), ...(o.states ?? {}) } });
      this.renderWhy(ev);
    }
    $("timeText").textContent = mmss(ev.t);
  }

  paintCompare() {
    const d = this.cmpDim, o = this.engine.last, cmpOn = !!this.engine.compareModel;
    const hist = this.engine.history.filter(p => p.t >= this.engine.t - 30)
      .map(p => ({ t: p.t, brain: d === "a" ? p.a : p.v, std: cmpOn ? ((d === "a" ? p.ca : p.cv) ?? null) : null }));
    drawCompare($("cmpPlot") as HTMLCanvasElement, hist, this.faceHist, 30, this.engine.t);
    $("cmpBrain").textContent = o ? fmtSigned(d === "a" ? o.a : o.v) : "—";
    $("cmpStd").textContent = o?.cmp ? fmtSigned(d === "a" ? o.cmp.a : o.cmp.v) : "—";
    $("cmpNote").textContent = cmpOn
      ? `Both decoders read the same signal; the brain-only one first removes what eye and jaw activity explain, and holds its value when that activity is beyond its training range${o?.faceHold ? " (holding now)" : ""}. The strip shows eye and jaw activity above your baseline.`
      : "Select the brain-only model (Models) to see it next to the standard decoder. The strip shows eye and jaw activity above your baseline.";
  }

  renderWhy(ev: WindowEvent) {
    const ul = $("why");
    if (!ev.features) return;
    const x = ev.features;
    const items: { label: string; val: number }[] = [];
    const entry = this.models.find(m => m.id === this.modelId);
    const heads = entry?.model?.heads;
    const model = entry?.model ?? null;
    const lin = heads && model ? (Object.entries(heads) as [string, Head][]).map(([k, hd]) => [k, hd, contributions(hd, headInput(model, hd, x, ev.faceRegion))] as const).filter(([, , c]) => c) : [];
    if (lin.length && model) {
      const nb = inputDim(model);
      for (const [k, hd, c] of lin) {
        const idx = Array.from(c!.keys()).filter(i => i < nb).sort((i, j) => Math.abs(c![j]) - Math.abs(c![i])).slice(0, 3);
        for (const i of idx) items.push({ label: `${k}: ${featureLabel(i)}`, val: c![i] });
        if (hd.face) { let fsum = 0; for (let i = nb; i < c!.length; i++) fsum += c![i]; items.push({ label: `${k}: face removed`, val: fsum }); }
      }
    } else {
      const R = REGIONS.length, off = R * 5, maskOff = off + REGION_PAIRS.length * 5;
      REGION_PAIRS.slice(0, 3).forEach(([l, r], p) => {
        if (x[maskOff + REGIONS.indexOf(l)] && x[maskOff + REGIONS.indexOf(r)]) items.push({ label: `valence: alpha ${r} vs ${l}`, val: -x[off + p * 5 + 2] });
      });
      let ba = 0, n = 0;
      for (let r = 0; r < R; r++) if (x[maskOff + r]) { ba += x[r * 5 + 3] - x[r * 5 + 2]; n++; }
      if (n) items.push({ label: "arousal: beta over alpha", val: ba / n });
    }
    const mx = Math.max(1e-6, ...items.map(i => Math.abs(i.val)));
    ul.replaceChildren(...items.map(i => h("li", {}, h("span", {}, i.label), h("span", { class: "num" }, fmtSigned(i.val)),
      h("span", { class: "barx" }, h("i", { class: i.val < 0 ? "neg" : "", style: `left:${i.val < 0 ? 50 - 50 * Math.abs(i.val) / mx : 50}%;width:${50 * Math.abs(i.val) / mx}%` })))));
    if (!items.length) ul.replaceChildren(h("li", {}, h("span", { class: "note" }, "No left/right electrode pairs in this layout.")));
  }

  onIntent(kind: string) {
    if (kind === "double_blink") { this.arm.nextIntent(); this.marker("intent", "double_blink"); }
    else if (kind === "clench_start") { this.arm.moving = true; this.marker("intent", "clench_start"); }
    else if (kind === "clench_end") { this.arm.moving = false; this.marker("intent", "clench_end"); }
  }

  // ---------------------------------------------------------------- arm
  paintArmLink() {
    const kind = store.get<string>("armLink", "sim");
    $("armAddrWrap").hidden = kind === "sim";
    $("armAddrLabel").textContent = kind === "serial" ? "Serial port" : "Arm IP address";
    $<HTMLInputElement>("armAddr").placeholder = kind === "serial" ? "/dev/ttyUSB0 or COM5" : "192.168.4.1";
    $("armLinkNote").textContent = kind === "sim" ? "The twin moves on its own; nothing physical moves."
      : this.link.available ? (kind === "serial" ? "Plug the arm's USB-C port into this computer." : "Join the arm's Wi-Fi (RoArm-M2, password 12345678) or put both on one network.")
      : "Driving the real arm needs the Emotif server on this computer: python -m emotif.server";
    $<HTMLButtonElement>("armConnect").disabled = kind !== "sim" && !this.link.available;
    this.paintArmButtons();
  }

  async armConnect() {
    const kind = store.get<string>("armLink", "sim");
    this.showErr("armErr", "");
    if (kind === "sim") {
      if (this.arm.useServer && this.link.available) await this.link.api("/api/arm/disconnect", {}).catch(() => undefined);
      this.arm.useServer = false;
      this.arm.local.enabled = false;
      $("armState").textContent = "twin only";
      this.paintArmButtons();
      return;
    }
    const address = $<HTMLInputElement>("armAddr").value.trim();
    store.set("armAddr", address);
    try {
      await this.link.api("/api/arm/connect", { kind, address });
      this.arm.useServer = true;
      $("armState").textContent = kind === "serial" ? `USB ${address}` : `Wi-Fi ${address || "192.168.4.1"}`;
      this.marker("arm_connect", { kind, address });
    } catch (e) { this.showErr("armErr", (e as Error).message); }
    this.paintArmButtons();
  }

  async armEnable() {
    if (this.arm.enabled) {
      if (this.arm.connectedToReal) await this.link.api("/api/arm/disable", {}).catch(e => this.showErr("armErr", (e as Error).message));
      this.arm.local.enabled = false;
      this.paintArmButtons();
      return;
    }
    if (!$<HTMLInputElement>("armClear").checked) return;
    this.arm.syncManual();
    if (this.arm.connectedToReal) {
      try { await this.link.api("/api/arm/enable", {}); } catch (e) { this.showErr("armErr", (e as Error).message); return; }
    }
    this.arm.local.estop = false;
    this.arm.local.enabled = true;
    this.marker("arm", "enable");
    this.paintArmButtons();
  }

  armHome() {
    this.arm.manual = { ...INIT };
    if (this.arm.mode !== "manual" && this.arm.mode !== "intent") this.setArmMode("manual");
    if (this.arm.connectedToReal) this.link.api("/api/arm/home", {}).catch(() => undefined);
  }

  estop() {
    this.arm.local.enabled = false;
    this.arm.local.estop = true;
    if (this.arm.useServer && this.link.available) this.link.api("/api/arm/estop", {}).catch(() => undefined);
    this.marker("arm", "estop");
    this.paintArmButtons();
    window.setTimeout(() => {
      if (this.arm.useServer && this.link.available) this.link.api("/api/arm/clear_estop", {}).catch(() => undefined);
      this.arm.local.estop = false;
      this.paintArmButtons();
    }, 1500);
  }

  setArmSpeed(v: number) {
    this.arm.local.lim.speed_mm_s = v;
    $("speedLabel").textContent = `Speed limit ${v} mm/s`;
    if (this.arm.connectedToReal) this.link.api("/api/arm/limits", { speed_mm_s: v }).catch(() => undefined);
  }

  setArmMode(m: ArmMode) {
    this.arm.mode = m;
    this.arm.syncManual();
    setSeg("armMode", m);
    $("modeEmbody").hidden = m !== "embody"; $("modeDraw").hidden = m !== "draw"; $("modeIntent").hidden = m !== "intent"; $("modeManual").hidden = m !== "manual";
    $("armWord").textContent = { embody: "Embody", draw: "Draw", intent: "Intent", manual: "Manual" }[m];
    this.marker("arm_mode", m);
  }

  paintArmButtons() {
    const en = this.arm.enabled;
    const b = $<HTMLButtonElement>("armEnable");
    if (!b) return;
    b.textContent = en ? "Disable motion" : "Enable motion";
    b.disabled = !en && !$<HTMLInputElement>("armClear").checked;
    b.className = en ? "btn" : "btn primary";
    $("estopTop").hidden = !(en || this.arm.connectedToReal);
    $("armEyebrow").textContent = `Digital twin · ${this.arm.connectedToReal ? "mirroring the real arm" : "simulated"}${en ? " · enabled" : ""}`;
    const simOnly = this.sourceKind !== "sim";
    $("simBlink").hidden = simOnly; $("simClench").hidden = simOnly;
  }

  async calibrateIntent() {
    if (!this.intent || this.armCal.running) return;
    const it = this.intent, sim = this.sourceKind === "sim";
    this.armCal.running = true;
    $<HTMLButtonElement>("calBtn").disabled = true;
    const say = (s: string) => { $("calText").textContent = s; };
    const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));
    const sample = async (ms: number, pick: () => number) => { const out: number[] = []; const t0 = performance.now(); while (performance.now() - t0 < ms) { out.push(pick()); await sleep(30); } return out; };
    try {
      say("Relax your jaw and keep your eyes still… (6 s)");
      const rest = await sample(6000, () => it.levels.clenchRms);
      const clench: number[] = [];
      for (let k = 1; k <= 3; k++) {
        say(`Clench your jaw now (${k} of 3)…`);
        if (sim) this.sim.clench(2);
        await sleep(400);
        clench.push(...await sample(1500, () => it.levels.clenchRms));
        say("Relax…"); await sleep(2500);
      }
      const peaks: number[] = [];
      if (it.canBlink) {
        for (let k = 1; k <= 4; k++) {
          say(`Blink firmly twice now (${k} of 4)…`);
          if (sim) this.sim.doubleBlink();
          const s = await sample(1400, () => it.levels.frontal);
          peaks.push(Math.max(...s));
          say("Wait…"); await sleep(2200);
        }
      }
      it.calibrate(rest, clench, peaks);
      store.set(`intentCal:${this.streamLabel.split(" · ")[0]}`, it.cal);
      say(`Calibrated: blink threshold ${it.cal.blinkUv.toFixed(0)} µV, clench at ${it.cal.clenchRatio.toFixed(1)}× rest (rest ${median(rest).toFixed(1)}). Try it: double-blink, then clench.`);
      this.marker("intent_calibration", it.cal);
    } finally {
      this.armCal.running = false;
      $<HTMLButtonElement>("calBtn").disabled = false;
    }
  }

  // ---------------------------------------------------------------- models
  loadModels() {
    this.models = [
      { id: "markers", name: "Neuro-markers", desc: "No training. Valence from frontal alpha asymmetry, arousal from beta over alpha, both relative to your baseline.", model: null, origin: "built-in" },
    ];
    try {
      const m = validateModel(builtinMuse);
      this.models.push({ id: "builtin:muse_mental", name: m.name, desc: "Arousal from real Muse recordings (relaxed / neutral / concentrating). Valence stays on markers.", model: m, origin: "built-in" });
    } catch { /* bundled model invalid: skip */ }
    try {
      const m = validateModel(builtinBrain);
      this.models.unshift({ id: "builtin:brain", name: m.name, desc: "Recommended. Valence and arousal from brain activity only: every window's eye and jaw-muscle activity is measured and everything it explains is removed before decoding. Trained on 28 people (GAMEEMO, 14-channel Emotiv; labels are the games' intended emotions). Runs on any headset.", model: m, origin: "built-in" });
    } catch { /* bundled model invalid: skip */ }
    try {
      const m = validateModel(builtinGameemo);
      this.models.push({ id: "builtin:gameemo", name: m.name, desc: "Standard decoder on the same data, without the face removal; shown next to the brain-only decoder for comparison.", model: m, origin: "built-in" });
    } catch { /* bundled model invalid: skip */ }
    this.modelId = "markers";
    try {
      const m = validateModel(builtinInsight);
      this.models.push({ id: "builtin:insight", name: m.name, desc: "Brain-only valence and arousal (as above) plus an experimental stress head trained on 17 people recorded with an Emotiv Insight (horror video vs relaxing music). Stress was higher for the horror recording in 13 of 17 new people; treat it as a research measurement.", model: m, origin: "built-in" });
    } catch { /* bundled model invalid: leave it out */ }
    if (this.models.some(m => m.id === "builtin:brain")) { void this.selectModel("builtin:brain"); return; }
    this.renderModelList();
    this.renderPersonal();
  }

  renderModelList() {
    const list = $("modelList");
    if (!list) return;
    list.replaceChildren(...this.models.map(m => h("button", { class: "card-opt", "aria-pressed": String(m.id === this.modelId), onclick: () => this.selectModel(m.id) },
      h("b", {}, m.name), h("span", {}, m.desc), h("span", { class: "meta" }, m.origin))));
  }

  async selectModel(id: string) {
    const m = this.models.find(x => x.id === id);
    if (!m) return;
    this.showErr("modelErr", "");
    try {
      if (m.origin === "server" && !m.model && m.file) m.model = validateModel(await (await fetch(`/models/${encodeURIComponent(m.file)}`)).json());
      this.engine.setModel(m.model);
      // brain-only model: run the standard decoder trained on the same data alongside it for the comparison panel
      const std = isBrainOnly(m.model) ? this.models.find(x => x.model && !isBrainOnly(x.model) && x.model.datasets.join() === m.model!.datasets.join()) : undefined;
      this.engine.setCompareModel(std?.model ?? null);
      this.modelId = id;
      this.marker("model", id);
    } catch (e) { this.showErr("modelErr", (e as Error).message); }
    this.renderModelList();
    this.renderModelDetail();
    this.renderPersonal();
  }

  uploadModel(e: Event) {
    const input = e.target as HTMLInputElement, f = input.files?.[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = () => {
      try {
        const m = validateModel(JSON.parse(String(r.result)));
        const id = `upload:${f.name}`;
        this.models = this.models.filter(x => x.id !== id);
        this.models.push({ id, name: m.name || f.name, desc: `${m.model} · ${Object.keys(m.heads).join(" + ")} · ${m.datasets.join(", ")}`, model: m, origin: "your file" });
        this.selectModel(id);
      } catch (err) { this.showErr("modelErr", (err as Error).message); }
      input.value = "";
    };
    r.readAsText(f);
  }

  renderModelDetail() {
    const box = $("modelDetail");
    if (!box) return;
    const entry = this.models.find(m => m.id === this.modelId);
    if (!entry) return;
    const m = entry.model;
    if (!m) {
      box.replaceChildren(h("h3", {}, "Neuro-markers"),
        h("p", { class: "note" }, "Uses two well-studied EEG patterns, measured against your own resting baseline:"),
        h("dl", { class: "kv" },
          h("dt", {}, "Valence"), h("dd", {}, "alpha(right) − alpha(left), frontal"),
          h("dt", {}, "Arousal"), h("dd", {}, "beta − alpha, all regions")),
        h("p", { class: "note" }, "More right-hemisphere alpha means relatively more left-frontal activity, which is linked to approach and positive affect (Davidson's frontal asymmetry). Beta rises and alpha falls with alertness. Train a model for better accuracy on your own data."));
      return;
    }
    const kv = h("dl", { class: "kv" },
      h("dt", {}, "Outputs"), h("dd", {}, Object.keys(m.heads).join(", ")),
      h("dt", {}, "Classifier"), h("dd", {}, m.model),
      h("dt", {}, "Trained on"), h("dd", {}, m.datasets.join(", ")),
      h("dt", {}, "Subjects"), h("dd", {}, String(m.n_subjects ?? "—")),
      h("dt", {}, "Windows"), h("dd", {}, String(m.n_windows ?? "—")),
      h("dt", {}, "Created"), h("dd", {}, m.created ?? "—"));
    const tables: HTMLElement[] = [];
    for (const [target, cv] of Object.entries(m.cv ?? {})) {
      const lays = Array.from(new Set([...Object.keys(cv.within ?? {}), ...Object.keys(cv.loso ?? {})]));
      if (!lays.length) continue;          // trained without cross-validation (results are in docs/RESULTS.md)
      const tb = h("tbody");
      for (const l of lays) {
        tb.append(h("tr", {}, h("td", {}, l === "full" ? "Full layout" : DEVICES[l]?.label ?? l),
          h("td", { class: "n" }, cv.within?.[l] ? cv.within[l].bacc.toFixed(3) : "—"),
          h("td", { class: "n" }, cv.loso?.[l] ? cv.loso[l].bacc.toFixed(3) : "—")));
      }
      tables.push(h("p", { class: "note" }, `${target}: balanced accuracy. Chance = ${(cv.chance_bacc ?? NaN).toFixed(3)}.`),
        h("div", { class: "table-wrap" }, h("table", { class: "data" },
          h("thead", {}, h("tr", {}, h("th", {}, "Electrode layout"), h("th", { class: "n" }, "Within-subject"), h("th", { class: "n" }, "New subject (LOSO)"))), tb)));
    }
    const headSel = h("select", { id: "wHead", "aria-label": "Output shown on the weight map" });
    Object.keys(m.heads).forEach(k => headSel.append(h("option", { value: k }, k)));
    const bandSel = h("select", { id: "wBand", "aria-label": "Band shown on the weight map" });
    BANDS.forEach(([n], i) => bandSel.append(h("option", { value: i, selected: i === 2 }, n)));
    const cv = h("canvas", { class: "plot topo-lg", id: "weightTopo" });
    const paint = () => {
      const head = m.heads[headSel.value as "valence" | "arousal"];
      if (!head) return;
      if (head.layers.length !== 1) { const g = cv.getContext("2d"); if (g) g.clearRect(0, 0, cv.width, cv.height); return; }
      const W = head.layers[0].W, hi = head.classes.length - 1, b = Number(bandSel.value);
      const vals = REGIONS.map((_, r) => W[r * 5 + b][hi] - W[r * 5 + b][0]);
      const mx = Math.max(1e-9, ...vals.map(Math.abs));
      drawTopomap(cv, REGIONS, vals.map(v => v / mx * 2), REGIONS.map(() => "plain"), { range: 2 });
    };
    headSel.onchange = paint; bandSel.onchange = paint;
    box.replaceChildren(h("h3", {}, m.name), kv, ...(m.notes ? [h("p", { class: "note" }, m.notes)] : []), ...tables,
      h("div", { class: "panel-h" }, "What the model looks at", h("span", { class: "row" }, headSel, bandSel)), cv,
      h("p", { class: "note" }, "Weight of each scalp region toward the highest class (red) or lowest (blue). Only single-layer models can be drawn this way."));
    requestAnimationFrame(paint);
  }

  /** Windows from finished protocol blocks, labelled by the participant's SAM ratings. */
  labeledWindows(target: "valence" | "arousal", classes: number[]): LabeledWindow[] {
    const out: LabeledWindow[] = [];
    let block = 0;
    for (const r of this.runs) for (const b of r.blocks) {
      block++;
      if (!b.sam || b.task === "rest") continue;
      const s = target === "valence" ? b.sam.valence : b.sam.arousal;
      const y = classes.length === 2 ? (s > 5 ? classes[1] : classes[0]) : samClass(s);
      if (!classes.includes(y)) continue;
      for (const w of this.engine.featureLog) if (w.t >= b.t0 + 2 && w.t <= b.t1) out.push({ x: Float64Array.from(w.x), f: Float64Array.from(w.f), y, block });
    }
    return out;
  }

  renderPersonal() {
    const box = $("personal");
    if (!box) return;
    const entry = this.models.find(m => m.id === this.modelId);
    const blocks = this.runs.flatMap(r => r.blocks.filter(b => b.sam && b.task !== "rest"));
    const run = h("button", { class: "btn primary", disabled: blocks.length < 2, onclick: () => this.personalize() }, "Fit to my ratings");
    const kids: Node[] = [
      h("p", { class: "note" }, `Fits a logistic model to this session's EEG, labelled by the participant's own ratings, starting from ${entry?.model ? `"${entry.name}"` : "scratch"} and staying close to it unless the data disagree. Accuracy is checked by leaving out one block at a time.`),
      h("p", { class: "note" }, `${blocks.length} rated block${blocks.length === 1 ? "" : "s"} available.${blocks.length < 2 ? " Run a study session first." : ""}`),
      h("div", { class: "row" }, run),
    ];
    if (this.personal) {
      const p = this.personal;
      kids.push(h("p", { class: "note" }, p.report),
        h("div", { class: "row" },
          h("button", { class: "btn", onclick: () => this.usePersonal() }, "Use personal model"),
          this.link.available ? h("button", { class: "btn", onclick: () => this.savePersonal() }, "Save to server") : null,
          window.self === window.top ? h("button", { class: "btn ghost", onclick: () => download(`emotif_personal_${$<HTMLInputElement>("pcode").value || "P"}.json`, JSON.stringify(p.model)) }, "Download") : null));
    }
    box.replaceChildren(...kids);
  }

  personalize() {
    const entry = this.models.find(m => m.id === this.modelId);
    const base = entry?.model ?? null;
    const heads: ModelFile["heads"] = {};
    const lines: string[] = [];
    let windowsUsed = 0;
    for (const target of ["valence", "arousal"] as const) {
      const prior = base?.heads[target];
      const classes = prior ? prior.classes : [-1, 0, 1];
      // the personal head reads what the base head reads (graph features, and the face input on brain-only heads)
      const data = this.labeledWindows(target, classes).map(d => ({ ...d, x: base && prior ? headInput(base, prior, d.x, d.f ?? null) : base ? modelInput(base, d.x) : d.x }));
      const present = Array.from(new Set(data.map(d => d.y))).sort((a, b) => a - b);
      if (present.length < 2) { lines.push(`${target}: ratings all fell in one class, skipped`); continue; }
      const cls = prior ? classes : present;
      const priorLayer = prior && prior.layers.length === 1 ? prior.layers[0] : null;
      const cv = personalizationCv(data, cls, priorLayer);
      heads[target] = { classes: cls, layers: [fitLogistic(data, cls, priorLayer)], ...(prior?.face ? { face: prior.face } : {}) };
      windowsUsed = Math.max(windowsUsed, data.length);
      lines.push(Number.isNaN(cv.personal)
        ? `${target}: fitted on ${data.length} windows; accuracy can't be checked yet (needs at least two rated blocks for each level)`
        : `${target}: ${data.length} windows, leave-one-block-out balanced accuracy ${cv.personal.toFixed(2)}${priorLayer && !Number.isNaN(cv.prior) ? ` (before: ${cv.prior.toFixed(2)})` : ""}, chance ${(1 / cls.length).toFixed(2)}`);
    }
    if (!Object.keys(heads).length) { this.personal = null; this.showErr("modelErr", lines.join(". ")); this.renderPersonal(); return; }
    const model: ModelFile = {
      format: "emotif-model-v2", name: `Personal · ${$<HTMLInputElement>("pcode").value || "participant"}`,
      notes: `Fitted in the app from SAM-rated protocol blocks, starting from ${entry?.name ?? "scratch"}.`,
      created: new Date().toISOString().slice(0, 10),
      feature_space: { kind: "region-v1", regions: REGIONS.slice(), n_features: N_FEATURES },
      transform: base?.transform ?? null,
      face_input: base?.face_input ?? null,
      model: "logreg", datasets: ["this session"],
      heads: { ...(base?.heads ?? {}), ...heads },
      personalized: { from: entry?.id ?? "markers", windows: windowsUsed, date: new Date().toISOString() },
    };
    this.personal = { model: validateModel(model), report: lines.join(". ") + "." };
    this.renderPersonal();
  }

  usePersonal() {
    if (!this.personal) return;
    const id = "personal";
    this.models = this.models.filter(x => x.id !== id);
    this.models.push({ id, name: this.personal.model.name, desc: this.personal.report, model: this.personal.model, origin: "this session" });
    this.selectModel(id);
  }

  async savePersonal() {
    if (!this.personal) return;
    try {
      const r = await this.link.api<{ ok: boolean; file: string }>("/api/models/save", { name: this.personal.model.name, model: this.personal.model });
      this.showErr("modelErr", "");
      this.personal.report += ` Saved as models/${r.file}.`;
      this.renderPersonal();
    } catch (e) { this.showErr("modelErr", (e as Error).message); }
  }

  // ---------------------------------------------------------------- sessions
  async refreshSessions() {
    const box = $("sessionList");
    if (!this.link.available) {
      box.replaceChildren(h("p", { class: "note" }, "Recording needs the Emotif server. On this computer run ", h("code", {}, "python -m emotif.server"), " and open http://localhost:8000."));
      return;
    }
    try {
      const r = await this.link.api<{ sessions: SessionMeta[] }>("/api/sessions");
      if (!r.sessions.length) { box.replaceChildren(h("p", { class: "note" }, "No sessions yet. Record one from Study.")); return; }
      const tb = h("tbody");
      for (const s of r.sessions) {
        const link = (f: string) => h("a", { href: `/api/sessions/${encodeURIComponent(s.id)}/${f}` }, f.split(".")[0]);
        tb.append(h("tr", {},
          h("td", { class: "num" }, s.started.replace("T", " ").slice(0, 16)), h("td", {}, s.participant), h("td", {}, s.protocol || "free"),
          h("td", {}, s.device), h("td", { class: "n" }, s.duration_s !== undefined ? mmss(s.duration_s) : "recording"),
          h("td", {}, h("span", { class: "row" }, link("eeg.csv"), link("events.csv"), link("affect.csv"), link("meta.json"))),
          h("td", {}, h("span", { class: "row" },
            h("button", { class: "btn small", onclick: () => this.showReport(s) }, "Report"),
            h("button", { class: "btn small ghost", onclick: () => this.replay(s) }, "Replay")))));
      }
      box.replaceChildren(h("table", { class: "data" },
        h("thead", {}, h("tr", {}, h("th", {}, "Started"), h("th", {}, "Participant"), h("th", {}, "Protocol"), h("th", {}, "Device"), h("th", { class: "n" }, "Length"), h("th", {}, "Files"), h("th", {}))), tb));
    } catch (e) { box.replaceChildren(h("p", { class: "err" }, (e as Error).message)); }
  }

  async replay(s: SessionMeta) {
    try {
      const text = await (await fetch(`/api/sessions/${encodeURIComponent(s.id)}/eeg.csv`)).text();
      this.csv.load(text, s.fs, `${s.id}`);
      this.useSource("csv");
      this.go("live");
    } catch (e) { this.showErr("srcErr", (e as Error).message); this.go("setup"); }
  }

  // ---------------------------------------------------------------- montage
  renderMontage() {
    const tbl = $("montageTable");
    if (!tbl) return;
    if (!this.engine.C) { tbl.replaceChildren(h("p", { class: "note" }, "No signal yet.")); return; }
    const inputs: HTMLInputElement[] = [];
    const tb = h("tbody");
    this.engine.channels.forEach((c, i) => {
      const inp = h("input", { type: "text", value: c, list: "electrodeNames", spellcheck: "false", "aria-label": `Name for channel ${i + 1}` });
      const st = h("span", { class: `chip ${canonical(c) ? "ok" : "warn"}` }, canonical(c) ? "placed" : "unknown");
      inp.addEventListener("input", () => { const ok = !!canonical(inp.value); st.className = `chip ${ok ? "ok" : "warn"}`; st.textContent = ok ? "placed" : "unknown"; this.paintMontageTopo(inputs.map(x => x.value)); });
      inputs.push(inp);
      tb.append(h("tr", {}, h("td", { class: "num" }, String(i + 1)), h("td", { class: "num" }, this.rawChannels[i] ?? ""), h("td", {}, inp), h("td", {}, st)));
    });
    tbl.replaceChildren(h("table", { class: "data montage-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "#"), h("th", {}, "From device"), h("th", {}, "10-20 name"), h("th", {}))), tb),
      h("div", { class: "row", style: "margin-top:8px" },
        h("button", { class: "btn primary small", onclick: () => {
          const names = inputs.map(x => x.value.trim() || "?");
          this.engine.renameChannels(names);
          if (this.intent) { this.intent = new IntentDetector(this.engine.fs, names); this.intent.listeners.add(ev => this.onIntent(ev.kind)); }
          store.set(this.montageKey(this.streamLabel, this.rawChannels), names);
          this.marker("montage", names);
          this.renderMontage();
        } }, "Apply names"),
        h("button", { class: "btn small ghost", onclick: () => { inputs.forEach((x, i) => { x.value = this.rawChannels[i] ?? x.value; }); this.paintMontageTopo(inputs.map(x => x.value)); } }, "Reset to device names")));
    this.paintMontageTopo(this.engine.channels);
  }

  paintMontageTopo(names: string[]) {
    const cv = $<HTMLCanvasElement>("montageTopo");
    if (!cv) return;
    requestAnimationFrame(() => drawTopomap(cv, names, names.map(() => null), names.map(() => "ok"), { showRegions: true }));
    const cov = coverage(names);
    $("montageCov").textContent = `${cov.observed.length} of ${REGIONS.length} scalp regions covered${cov.missing.length ? ` (missing ${cov.missing.join(", ")})` : ""}.${cov.unknown.length ? ` Not placed: ${cov.unknown.join(", ")}.` : ""}`;
  }

  // ---------------------------------------------------------------- cleaning, fit, demo, method, reports
  setCleaning() {
    const k = Math.max(3, Math.min(50, Number($<HTMLInputElement>("cleanK").value) || 10));
    $<HTMLInputElement>("cleanK").value = String(k);
    const cl: Cleaning = { asr: $<HTMLInputElement>("cleanAsr").checked, asrK: k, detector: $<HTMLInputElement>("cleanDet").checked };
    this.engine.settings.cleaning = cl;
    store.set("settings", this.engine.settings);
    this.engine.resetCalibration();
    this.world.clearStrata();
    this.marker("cleaning", cl);
  }

  paintCleaning(ev: WindowEvent) {
    const pill = $("stClean");
    const e = this.engine;
    if (!ev.calibrated) { pill.hidden = true; }
    else {
      pill.hidden = false;
      const parts: string[] = [];
      if (e.asrActive) parts.push(`ASR ${Math.round(ev.repaired * 100)}%`);
      const bad = ev.masked.filter(Boolean).length;
      if (e.detectorActive) parts.push(bad ? `${bad} ch held out` : "all ch clean");
      pill.textContent = parts.length ? parts.join(" · ") : "Cleaning off";
      pill.className = `pill ${bad > ev.masked.length / 2 ? "warn" : ""}`;
    }
    const row = $("cleanRow");
    if (!row) return;
    const names = e.channels;
    const chips = names.map((n, c) => {
      const v = ev.verdicts?.[c];
      const off = !ev.good[c];
      const label = off ? "no signal" : v ? { clean: "clean", ocular: "eye", muscle: "muscle" }[v.label] : "—";
      const cls = off ? "bad" : v?.bad ? "warn" : v ? "ok" : "";
      return h("span", { class: `chip ${cls}`, title: v ? `p(clean) ${v.probs[0].toFixed(2)} · p(eye) ${v.probs[1].toFixed(2)} · p(muscle) ${v.probs[2].toFixed(2)}` : "" }, `${n} ${label}`);
    });
    const head = h("span", { class: "note" }, ev.calibrated
      ? `${e.asrActive ? `Artifact repair active, ${Math.round(ev.repaired * 100)}% of the last 10 s repaired` : "Artifact repair off"}${e.detectorActive ? ". AI detector per channel:" : "."}`
      : "Cleaning starts after the baseline.");
    row.replaceChildren(head, ...(e.detectorActive && ev.calibrated ? chips : []));
  }

  fitAdvice(st: FitStatus): string {
    return {
      good: "Good contact.",
      noisy: "Noisy. Press the electrode gently onto the skin and keep still.",
      muscle: "Muscle activity. Relax your forehead, jaw and neck.",
      mains: "Electrical hum. Move away from chargers and power cables, and check the reference electrode.",
      drift: "Slow drift. The electrode is moving or drying out; reseat it.",
      "no contact": "No contact. Reseat the electrode and move hair out of the way.",
    }[st];
  }

  paintFit() {
    const e = this.engine;
    if (!e.C || e.total < e.fs * 2) return;
    const needBase = !e.calibrated && this.view === "live";
    if (this.view !== "setup" && !needBase) return;
    const fit = fitCheck(e.lastSamples(Math.round(e.fs * 2), "raw"), e.lastSamples(Math.round(e.fs * 2), "filtered"), e.fs);
    this.lastFit = fit;
    const cls = (st: FitStatus) => (st === "good" ? "ok" : st === "no contact" ? "bad" : "warn");
    if (this.view === "setup") {
      const cv = $<HTMLCanvasElement>("fitTopo");
      if (cv) drawTopomap(cv, e.channels, e.channels.map(() => null), fit.map(f => (f.status === "good" ? "ok" : f.status === "no contact" ? "off" : "noisy")));
      const ul = $("fitList");
      if (ul) ul.replaceChildren(...fit.map((f, c) => h("li", {}, h("span", { class: `chip ${cls(f.status)}` }, e.channels[c]), h("span", {}, this.fitAdvice(f.status)))));
    }
    if (needBase) {
      $("baseFit").replaceChildren(...fit.map((f, c) => h("span", { class: `chip ${cls(f.status)}`, title: this.fitAdvice(f.status) }, `${e.channels[c]} ${f.status}`)));
    }
  }

  paintBaseline() {
    const e = this.engine, card = $("baseCard");
    const show = !!e.C && !e.calibrated && !this.protocolActive && !this.demo;
    card.hidden = !show;
    if (!show) return;
    const p = e.lastEvent ? e.lastEvent.calibProgress : 0;
    ($("baseBar") as HTMLElement).style.width = `${(p * 100).toFixed(1)}%`;
    $("baseTime").textContent = `${Math.round(p * e.settings.calibSec)} / ${e.settings.calibSec} s`;
    const err = $("baseErr");
    err.textContent = e.calibError;
    err.hidden = !e.calibError;
  }

  toggleDemo() {
    if (this.demo) {
      const d = this.demo;
      this.demo = null;
      $("demoCaption").hidden = true;
      $("demoBtn").textContent = "Booth demo";
      document.body.classList.remove("demo");
      this.sim.mode = d.prevMode;
      $<HTMLSelectElement>("simMode").value = d.prevMode;
      this.setSetting("calibSec", d.prevCalib);
      if (d.prevSource !== "sim") this.useSource(d.prevSource);
      if (d.prevModel !== this.modelId) void this.selectModel(d.prevModel);
      return;
    }
    if (this.protocolActive) return;
    this.demo = { on: true, t0: 0, prevSource: this.sourceKind, prevMode: this.sim.mode, prevCalib: this.engine.settings.calibSec, prevModel: this.modelId };
    // the simulator's tour is written in neuro-marker terms, so the tour decodes with the markers
    if (this.modelId !== "markers") void this.selectModel("markers");
    this.go("live");
    this.uiDirector.mode = "mirror"; this.director.mode = "mirror"; setSeg("dirMode", "mirror");
    this.engine.settings.calibSec = 20;
    this.sim.mode = "tour";
    $<HTMLSelectElement>("simMode").value = "tour";
    this.useSource("sim");
    this.world.clearStrata();
    $("demoCaption").hidden = false;
    $("demoBtn").textContent = "End demo";
    document.body.classList.add("demo");
  }

  updateDemo() {
    const e = this.engine, t = e.t, base = e.settings.calibSec;
    let step = "", text = "";
    if (!e.calibrated) {
      step = "1 · Baseline";
      text = `Measuring a resting baseline (${Math.round(t)} of ${base} s). Emotif learns what this person's signal looks like at rest, then repairs blinks and muscle bursts against it.`;
    } else {
      const k = Math.floor((t - base) / 25) % 4;
      const cap = [
        ["2 · Happy", "More alpha over the right forehead than the left, plus extra beta: pleasant and alert. Colours warm and brighten, petals fly and the explorer speeds up."],
        ["3 · Calm", "Still pleasant, but beta falls and alpha grows: relaxed. The wind drops, the explorer slows and fireflies drift."],
        ["4 · Sad", "Alpha shifts to the left forehead: less pleasant. Mist rolls in, colour drains and snow falls."],
        ["5 · Stressed", "Unpleasant and highly aroused: cool, harsh light, rain and fast motion. In Guide mode the world would now lean toward calm."],
      ][k];
      step = cap[0]; text = cap[1];
    }
    $("demoStep").textContent = step;
    $("demoText").textContent = text;
  }

  viewMethod(): HTMLElement {
    return h("section", { class: "view", tabindex: "0", "data-view": "method", hidden: true },
      h("div", { class: "view-head" }, h("div", {}, h("h1", {}, "How Emotif works"),
        h("p", {}, "Every stage runs live in this browser. Numbers update from the current signal."))),
      h("div", { class: "pipeline", id: "pipeline" }),
      h("div", { class: "cols evidence", id: "evidence" }),
      h("section", { class: "panel" }, h("h3", {}, "Image credits"),
        h("p", { class: "note" }, "Ground photographs (grass, rock, dirt, sand and their normal maps) and the Tropical Sunny Day sky: Babylon.js Assets, CC BY 4.0. Sky photographs Quarry 01, Blouberg Sunrise 2 and Moonless Golf: Poly Haven (Greg Zaal), CC0, as distributed with the three.js examples. Leaves and bark are drawn by Emotif. Details: app/assets/CREDITS.md.")));
  }

  renderMethod() { this.renderPipeline(); this.renderEvidence(); }

  private methodStatsAt = 0;
  renderMethodStats() {
    const now = performance.now();
    if (now - this.methodStatsAt < 500) return;
    this.methodStatsAt = now;
    this.renderPipeline();
  }

  renderPipeline() {
    const e = this.engine, ev = e.lastEvent;
    const stage = (n: string, title: string, body: string, stat: string) =>
      h("div", { class: "stage-box" }, h("span", { class: "stage-n" }, n), h("b", {}, title), h("p", {}, body), h("span", { class: "stage-stat num" }, stat));
    const entry = this.models.find(m => m.id === this.modelId);
    const cov = e.C ? coverage(e.channels) : null;
    $("pipeline").replaceChildren(
      stage("1", "Acquire", "Any headset: Muse, OpenBCI, Emotiv, Neurosity, Unicorn, BrainBit, or a lab cap.", e.C ? `${e.C} ch at ${+e.fs.toFixed(1)} Hz` : "no signal"),
      stage("2", "Filter", "0.5 Hz high-pass removes drift; notches remove 50/60 Hz mains hum.", "causal, sample by sample"),
      stage("3", "Repair", "Artifact Subspace Reconstruction finds bursts that don't look like the baseline and rebuilds them from the clean channels.", e.asrActive ? `${Math.round((ev?.repaired ?? 0) * 100)}% of last 10 s repaired` : e.calibrated ? "off" : "calibrating"),
      stage("4", "Detect", "A neural network trained on 13,500 recordings (EEGdenoiseNet) labels each channel clean, eye or muscle, and holds out bad ones.", e.detectorActive && ev ? `${ev.masked.filter(Boolean).length} of ${e.C} held out` : e.calibrated ? "off" : "calibrating"),
      stage("5", "Map", "Differential entropy in 5 bands, z-scored to the baseline, pooled into 17 fixed scalp regions so any headset fits one model.", cov ? `${cov.observed.length} / 17 regions covered` : "—"),
      stage("6", "Decode (brain only)", "Blink size and jaw-muscle power are measured in every window; the brain-only decoder removes everything they explain, then reads valence and arousal. A Kalman filter smooths them.",
        e.last ? `${isBrainOnly(e.model) ? "face removed · " : ""}${entry?.name ?? "markers"} · V ${fmtSigned(e.last.v)} A ${fmtSigned(e.last.a)}` : "waiting"),
      stage("7", "Respond", "The 3-D world mirrors, guides or (as a control) ignores the state; the robot arm can embody it.", `world: ${this.director.mode}`));
  }

  renderEvidence() {
    const figs: [string, string, string][] = [
      ["denoise", "Artifact repair against ground truth", "Simulated recordings with real eye and muscle artifacts from EEGdenoiseNet. Lower error is better; the right panel shows clean data is left alone."],
      ["detector", "AI artifact detector", "Held-out test windows (never seen in training) at 256 and 128 Hz."],
      ["confound", "Brain or face?", "Real Muse recordings, new people. Three face numbers per electrode decode the state as well as the EEG; once the face is partialled out, cleaned EEG is not significantly above the permutation null."],
      ["montage", "One model, many headsets", "Leave-one-subject-out accuracy when the test headset has fewer electrodes, with and without montage augmentation."],
      ["brain", "Brain-only decoding", "GAMEEMO, 28 new people. Left: accuracy. Right: how much of each decoder's output the face can explain, measured with the audit's independent face indices."],
      ["brainmap", "What the brain-only decoder reads", "Weight of each scalp region and band after face removal. Dark outline: same sign in at least 95 % of 200 refits on resampled people."],
      ["states", "Stress, depression, anxiety, sleepiness on the Emotiv Insight layout", "A: within-person stress on 17 people recorded with a real Insight (horror video minus relaxing music, new people). B: one-label-per-person screens from 5 min of rest: brain-only is at chance; standard decoders' signal is largely face. C: the same data look 82-84 % decodable when windows of one person land in both training and test."],
    ];
    const ev2 = $("evidence");
    const have = figs.filter(([k]) => this.figures[k]);
    ev2.replaceChildren(...(have.length ? have.map(([k, t, c]) => h("figure", { class: "panel fig" }, h("img", { src: this.figures[k], alt: t }), h("figcaption", {}, h("b", {}, t), " ", c)))
      : [h("p", { class: "note" }, "Result figures are added when the app is built from a project with a results folder.")]));
  }

  async showReport(s: SessionMeta) {
    const box = $("report");
    box.hidden = false;
    box.replaceChildren(h("h3", {}, `Report · ${s.id}`), h("p", { class: "note" }, "Loading…"));
    try {
      const get = async (f: string) => { const r = await fetch(`/api/sessions/${encodeURIComponent(s.id)}/${f}`); if (!r.ok) throw new Error(`${f}: ${r.status}`); return r.text(); };
      const [aff, evs] = await Promise.all([get("affect.csv"), get("events.csv")]);
      const parse = (t: string) => t.trim().split(/\r?\n/).slice(1).map(parseCsvLine);
      const hist = parse(aff).map(r => ({ t: Number(r[0]), v: Number(r[1]), a: Number(r[2]), q: Number(r[5]) })).filter(p => Number.isFinite(p.t) && Number.isFinite(p.v));
      const blocks: { id: string; t0: number; t1: number; cond: string; sam?: { valence: number; arousal: number } }[] = [];
      for (const r of parse(evs)) {
        let v: Record<string, unknown> = {};
        try { v = JSON.parse(r[2] || "{}"); } catch { continue; }
        const t = Number(r[0]);
        if (r[1] === "block_start" && typeof v.block === "string") blocks.push({ id: v.block, t0: t, t1: 0, cond: String(v.condition ?? "") });
        if (r[1] === "block_end" && typeof v.block === "string") { const b = [...blocks].reverse().find(x => x.id === v.block); if (b) b.t1 = t; }
        if (r[1] === "sam" && typeof v.block === "string") { const b = [...blocks].reverse().find(x => x.id === v.block); if (b) b.sam = { valence: Number(v.valence), arousal: Number(v.arousal) }; }
      }
      const cv = h("canvas", { class: "plot report-chart" });
      const end = hist.length ? hist[hist.length - 1].t : 0;
      const tb = h("tbody");
      for (const b of blocks.filter(x => !x.id.startsWith("rest"))) {
        const seg = hist.filter(p => p.t >= b.t0 + 2 && p.t <= (b.t1 || end));
        const mv = seg.length ? seg.reduce((a, p) => a + p.v, 0) / seg.length : NaN, ma = seg.length ? seg.reduce((a, p) => a + p.a, 0) / seg.length : NaN;
        const mq = seg.length ? seg.reduce((a, p) => a + p.q, 0) / seg.length : NaN;
        tb.append(h("tr", {}, h("td", {}, b.id), h("td", {}, b.cond || "—"), h("td", { class: "n" }, b.sam ? String(b.sam.valence) : "—"), h("td", { class: "n" }, b.sam ? String(b.sam.arousal) : "—"),
          h("td", { class: "n" }, Number.isNaN(mv) ? "—" : fmtSigned(mv)), h("td", { class: "n" }, Number.isNaN(ma) ? "—" : fmtSigned(ma)), h("td", { class: "n" }, Number.isNaN(mq) ? "—" : `${Math.round(mq * 100)}%`)));
      }
      const meanQ = hist.length ? hist.reduce((a, p) => a + p.q, 0) / hist.length : 0;
      box.replaceChildren(h("div", { class: "panel-h" }, `Report · ${s.participant} · ${s.started.replace("T", " ").slice(0, 16)}`, h("button", { class: "btn small ghost", onclick: () => { box.hidden = true; } }, "Close")),
        h("dl", { class: "kv report-kv" },
          h("dt", {}, "Protocol"), h("dd", {}, s.protocol || "free recording"),
          h("dt", {}, "Device"), h("dd", {}, `${s.device} · ${s.channels.length} ch · ${s.fs} Hz`),
          h("dt", {}, "Length"), h("dd", {}, s.duration_s !== undefined ? mmss(s.duration_s) : "—"),
          h("dt", {}, "Decoded windows"), h("dd", {}, String(hist.length)),
          h("dt", {}, "Mean signal quality"), h("dd", {}, `${Math.round(meanQ * 100)}%`)),
        cv,
        blocks.length ? h("div", { class: "table-wrap" }, h("table", { class: "data" },
          h("thead", {}, h("tr", {}, h("th", {}, "Block"), h("th", {}, "World"), h("th", { class: "n" }, "SAM val."), h("th", { class: "n" }, "SAM aro."), h("th", { class: "n" }, "Decoded V"), h("th", { class: "n" }, "Decoded A"), h("th", { class: "n" }, "Quality"))), tb))
          : h("p", { class: "note" }, "No protocol blocks in this session."));
      requestAnimationFrame(() => drawTimeline(cv, hist, [], Math.max(10, end), end, blocks.map(b => ({ t0: b.t0, t1: b.t1, label: b.id }))));
      box.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { box.replaceChildren(h("h3", {}, "Report"), h("p", { class: "err" }, `Couldn't load this session: ${(e as Error).message}`)); }
  }

  // ---------------------------------------------------------------- frame loop
  loop() {
    let last = performance.now(), chartAcc = 0;
    const frame = (t: number) => {
      const dt = Math.min(0.25, (t - last) / 1000);
      last = t;
      const decoded = this.engine.last ? { v: this.engine.last.v, a: this.engine.last.a } : null;
      const env = this.director.update(dt, decoded);
      this.world.target = env;
      this.sim.envState = env;
      const tgt = this.arm.update(dt, decoded);
      this.world.armPose = this.arm.pose;
      this.world.armTarget = this.arm.enabled ? tgt : null;
      const drawing = this.arm.mode === "draw";
      if (drawing !== this.lastDrawMode || this.arm.trail.length !== this.lastTrailLen || drawing) {
        this.world.setDrawing(drawing, this.arm.draw, this.arm.trail);
        this.lastDrawMode = drawing; this.lastTrailLen = this.arm.trail.length;
      }
      if (decoded && this.engine.t - this.lastStrata >= 1) { this.lastStrata = this.engine.t; this.world.pushStrata(decoded.v, decoded.a); }
      chartAcc += dt;
      if (chartAcc >= 1 / 15) { chartAcc = 0; this.paintCharts(); this.paintBaseline(); if (this.view === "method") this.renderMethodStats(); }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  paintCharts() {
    const ev = this.engine.lastEvent;
    const status = (i: number): "ok" | "noisy" | "off" => {
      if (!ev) return "ok";
      const s = ev.stats[i];
      if (!s || s.std < 0.1 || s.ptp > 500) return "off";
      if (ev.verdicts && ev.verdicts[i]?.bad) return "noisy";
      return this.engine.settings.rejectUv > 0 && s.ptp > this.engine.settings.rejectUv ? "noisy" : "ok";
    };
    this.paintFit();
    if (this.demo) this.updateDemo();
    if (this.view === "live" && !document.body.classList.contains("immersive")) {
      drawCircumplex($("circ") as HTMLCanvasElement, this.trail, this.sourceKind === "sim" ? this.sim.cur : null,
        this.director.mode !== "mirror" ? this.director.env : null, this.director.mode === "guide" ? this.director.target : null);
      $("dirNote").textContent = { mirror: "Mirror: the world shows the decoded state.", guide: "Guide: the world leans toward the target, nudging the person there.", sham: "Sham: the world follows a random path that ignores the brain (control condition)." }[this.director.mode];
      if (ev && this.engine.C) {
        const vals = ev.zde ? ev.zde.map(z => (z ? z[this.bandView] : null)) : this.engine.channels.map(() => null);
        drawTopomap($("topo") as HTMLCanvasElement, this.engine.channels, vals, this.engine.channels.map((_, i) => status(i)));
        const n = Math.round(this.engine.fs * 4);
        drawTraces($("traces") as HTMLCanvasElement, this.engine.lastSamples(n, this.traceMode), this.engine.channels, this.engine.channels.map((_, i) => status(i)));
      }
      drawTimeline($("timeline") as HTMLCanvasElement, this.engine.history, this.events, 120, this.engine.t, this.blocks);
      this.paintCompare();
    }
    if (this.view === "arm") {
      const p = this.arm.pose;
      $("armPose").textContent = `x ${p.x.toFixed(0)}  y ${p.y.toFixed(0)}  z ${p.z.toFixed(0)} mm`;
      const it = this.intent;
      if (this.arm.mode === "intent") {
        $("intentAction").textContent = `${this.arm.action}${this.arm.moving ? " ▸" : ""}`;
        $("actionWheel").querySelectorAll("span").forEach((s, i) => s.classList.toggle("on", i === this.arm.intentIndex));
        $("armSub").textContent = `${this.arm.action}${this.arm.moving ? " · moving" : ""}`;
        if (it) {
          const rest = it.cal.restRms || 1;
          const cm = $("clenchMeter");
          (cm.firstElementChild as HTMLElement).style.width = `${Math.min(100, it.levels.clenchRms / (rest * it.cal.clenchRatio * 1.5) * 100)}%`;
          cm.classList.toggle("hot", it.levels.clenching);
          ($("blinkMeter").firstElementChild as HTMLElement).style.width = `${Math.max(0, Math.min(100, it.levels.frontal / (it.cal.blinkUv * 1.5) * 100))}%`;
        }
      } else $("armSub").textContent = this.arm.mode === "draw" ? (this.arm.draw.lifted ? "pen up" : "pen down") : "";
      $("penBtn").textContent = this.arm.draw.lifted ? "Pen down" : "Pen up";
      const s = this.arm.connectedToReal ? this.arm.server : null;
      const rows: [string, string][] = [
        ["Pose", `${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}`],
        ["Gripper", `${p.t.toFixed(2)} rad`],
        ["Motion", this.arm.enabled ? "enabled" : "held"],
        ["Link", this.arm.connectedToReal ? `${s?.transport ?? ""}` : "twin only"],
      ];
      if (s) rows.push(["Clamped", s.clamped ? "yes" : "no"], ["Watchdog", s.last_cmd_age > 1 ? "holding" : "ok"], ...(s.error ? [["Error", s.error] as [string, string]] : []));
      else rows.push(["Clamped", this.arm.local.clamped ? "yes" : "no"]);
      $("armKv").replaceChildren(...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
      if (this.arm.connectedToReal && s && !s.enabled && this.arm.local.enabled) { this.arm.local.enabled = false; this.paintArmButtons(); }
    }
  }
}

/** One CSV line (RFC 4180): quoted fields may contain commas and doubled quotes. */
export function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (q) {
      if (ch === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else q = false; }
      else cur += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

function featureLabel(i: number): string {
  const R = REGIONS.length;
  if (i < R * 5) return `${BANDS[i % 5][0]} ${REGIONS[Math.floor(i / 5)]}`;
  if (i < R * 5 + REGION_PAIRS.length * 5) { const k = i - R * 5; const [l, r] = REGION_PAIRS[Math.floor(k / 5)]; return `${BANDS[k % 5][0]} ${l}−${r}`; }
  if (i < N_FEATURES) return `has ${REGIONS[i - R * 5 - REGION_PAIRS.length * 5]}`;
  const k = i - N_FEATURES, hop = Math.floor(k / (R * 5)) + 1, j = k % (R * 5);
  return `${BANDS[j % 5][0]} around ${REGIONS[Math.floor(j / 5)]} (graph, ${hop} hop${hop > 1 ? "s" : ""})`;
}

function boot() {
  const app = new App();
  (window as unknown as { emotif: App }).emotif = app;
  app.init().catch(e => {
    console.error(e);
    document.body.append(h("div", { class: "panel", style: "margin:24px" }, h("h3", {}, "Emotif couldn't start"), h("p", { class: "err" }, String((e as Error).message ?? e))));
  });
}
// the script may be placed in <head>; wait for <body>
if (document.body) boot(); else document.addEventListener("DOMContentLoaded", boot, { once: true });
