// Study protocols: baseline, validated emotion-induction tasks, closed-loop regulation blocks, and
// Self-Assessment Manikin (SAM, Bradley & Lang 1994) ratings after each block.
import { mulberry32, type DirectorMode, type VA } from "./director";

export type TaskKind = "recall_positive" | "recall_negative" | "breathing" | "arithmetic" | "rest" | "watch" | "anticipation" | "recall_anger" | "drowsy" | "eyes_closed" | "blinks" | "clench";
export type Step =
  | { kind: "instruction"; title: string; body: string }
  | { kind: "baseline"; seconds: number }
  | { kind: "task"; id: string; title: string; seconds: number; task: TaskKind; design: VA; condition: DirectorMode; target?: string; state?: string }
  | { kind: "sam"; block: string }
  | { kind: "intensity"; block: string; word: string };

export type Protocol = { id: string; name: string; minutes: number; summary: string; build: (code: string) => { steps: Step[]; order: string[] } };

const TASK_TEXT: Record<TaskKind, { title: string; body: string }> = {
  recall_positive: { title: "Happy memory", body: "Think of a time you felt really happy or proud. Picture where you were, who was there and what you could hear. Stay inside that memory until the timer ends." },
  recall_negative: { title: "Disappointing memory", body: "Think of a time you felt mildly disappointed or let down. Pick something small, not something painful. You can stop at any time." },
  breathing: { title: "Slow breathing", body: "Breathe in as the ring grows and out as it shrinks. Six breaths a minute." },
  arithmetic: { title: "Count backward", body: "Count backward from the number shown, subtracting 13 each time. Say each answer out loud, as fast as you can." },
  rest: { title: "Rest", body: "Look at the cross and let your thoughts settle." },
  watch: { title: "Watch the world", body: "Watch the landscape and let yourself relax. There's nothing to do." },
  anticipation: { title: "Waiting for a test", body: "Imagine you're about to sit an important test you haven't prepared for. Picture waiting outside the room, the clock, what could go wrong. Keep your face relaxed and still." },
  recall_anger: { title: "Something unfair", body: "Think of a time someone treated you unfairly and it made you annoyed or angry. Pick something small. Replay what happened and what you wanted to say, without moving your face." },
  eyes_closed: { title: "Eyes closed", body: "Close your eyes and stay relaxed until you hear the tone." },
  blinks: { title: "Blink", body: "Blink firmly about once a second until the timer ends." },
  clench: { title: "Clench", body: "Clench your jaw gently for two seconds, relax for two seconds, and repeat until the timer ends." },
  drowsy: { title: "Get sleepy", body: "Keep your eyes open and resting on the fading dot. Breathe slowly and let yourself feel as sleepy as you can without closing your eyes." },
};

/** Calibration states (states.ts) and the task that brings each one on. */
export const STATE_TASKS: [string, TaskKind, VA, string][] = [
  ["calm", "breathing", { v: 1, a: -1 }, "calm"], ["stress", "arithmetic", { v: -1, a: 1 }, "stressed"],
  ["anxiety", "anticipation", { v: -1, a: 1 }, "anxious or worried"], ["anger", "recall_anger", { v: -1, a: 1 }, "annoyed or angry"],
  ["low_mood", "recall_negative", { v: -1, a: -1 }, "sad or down"], ["drowsy", "drowsy", { v: 0, a: -1 }, "sleepy"],
];

/** Hash a participant code to a stable integer so condition orders are reproducible. */
export function codeSeed(code: string): number {
  let h = 2166136261;
  for (const ch of code) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
function shuffle<T>(a: T[], seed: number): T[] {
  const r = mulberry32(seed), out = a.slice();
  for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
  return out;
}

const task = (id: string, t: TaskKind, seconds: number, design: VA, condition: DirectorMode = "mirror", target?: string, state?: string): Step =>
  ({ kind: "task", id, title: TASK_TEXT[t].title, seconds, task: t, design, condition, target, state });

export const PROTOCOLS: Protocol[] = [
  {
    id: "quick", name: "Quick demo", minutes: 5,
    summary: "One-minute baseline, then three short tasks with a rating after each. Good for a science-fair demo.",
    build: () => ({
      order: ["happy", "calm", "stress"],
      steps: [
        { kind: "instruction", title: "Sit comfortably", body: "Relax your jaw and shoulders, keep still, and look at the screen. The first minute measures your resting signal." },
        { kind: "baseline", seconds: 60 },
        task("happy", "recall_positive", 60, { v: 1, a: 1 }), { kind: "sam", block: "happy" },
        task("calm", "breathing", 60, { v: 1, a: -1 }), { kind: "sam", block: "calm" },
        task("stress", "arithmetic", 60, { v: -1, a: 1 }), { kind: "sam", block: "stress" },
      ],
    }),
  },
  {
    id: "induction", name: "Emotion induction", minutes: 14,
    summary: "Baseline plus four 2-minute inductions, one per quadrant, in an order set by the participant code. Builds a personal training set.",
    build: code => {
      const blocks: [string, TaskKind, VA][] = [["happy", "recall_positive", { v: 1, a: 1 }], ["calm", "breathing", { v: 1, a: -1 }],
        ["stress", "arithmetic", { v: -1, a: 1 }], ["sad", "recall_negative", { v: -1, a: -1 }]];
      const order = shuffle(blocks, codeSeed(code));
      const steps: Step[] = [
        { kind: "instruction", title: "Emotion induction", body: "You'll do four short tasks. After each one, rate how you felt. Keep your head and jaw still while a task is running." },
        { kind: "baseline", seconds: 90 },
      ];
      order.forEach(([id, t, va], i) => {
        steps.push(task(id, t, 120, va), { kind: "sam", block: id });
        if (i < order.length - 1) steps.push(task(`rest${i + 1}`, "rest", 20, { v: 0, a: 0 }));
      });
      return { steps, order: order.map(o => o[0]) };
    },
  },
  {
    id: "check", name: "Headset check", minutes: 4,
    summary: "Run this first on any new headset or wearer: eyes open, eyes closed (alpha should rise at the back of the head), blinks and jaw clenches (to confirm the face measures work). research/session_qa.py scores it.",
    build: () => ({
      order: ["eyes_open", "eyes_closed", "blinks", "clench"],
      steps: [
        { kind: "instruction", title: "Headset check", body: "Four short steps to check the headset is reading your brain and your face correctly. Sit still unless a step asks you to move." },
        { kind: "baseline", seconds: 30 },
        task("eyes_open", "rest", 60, { v: 0, a: 0 }),
        task("eyes_closed", "eyes_closed", 60, { v: 0, a: 0 }),
        task("blinks", "blinks", 20, { v: 0, a: 0 }),
        task("clench", "clench", 20, { v: 0, a: 0 }),
      ],
    }),
  },
  {
    id: "face", name: "Blink and jaw check", minutes: 2,
    summary: "The 1-minute participant check: eyes open, blinks, jaw clenches (confirms the face measures work, so the brain-only decoder can remove them). Use Headset check instead if your approved protocol includes an eyes-closed minute. research/session_qa.py scores it.",
    build: () => ({
      order: ["eyes_open", "blinks", "clench"],
      steps: [
        { kind: "instruction", title: "Blink and jaw check", body: "Three 20-second steps so the software can tell your face's signals apart from your brain's. Sit still unless a step asks you to move." },
        { kind: "baseline", seconds: 30 },
        task("eyes_open", "rest", 20, { v: 0, a: 0 }),
        task("blinks", "blinks", 20, { v: 0, a: 0 }),
        task("clench", "clench", 20, { v: 0, a: 0 }),
      ],
    }),
  },
  {
    id: "states", name: "Personal state calibration", minutes: 28,
    summary: "Teaches the headset your own brain's version of six states (calm, stress, anxiety, anger, low mood, drowsiness): three rounds of six 1-minute tasks in an order set by the participant code, a 0-10 rating after each. Each state is then checked on rounds the model didn't see, and only states that pass are shown.",
    build: code => {
      const steps: Step[] = [
        { kind: "instruction", title: "Personal state calibration", body: "Three rounds of six one-minute tasks. After each, rate how strongly you felt the state. Keep your head, eyes and jaw as still as you can while a task runs: the decoder ignores the face, so stillness helps it learn your brain's signal. You can stop at any time." },
        { kind: "baseline", seconds: 90 },
      ];
      const order: string[] = [];
      for (let round = 1; round <= 3; round++) {
        const blocks = shuffle(STATE_TASKS, codeSeed(code) ^ (round * 0x2545f491));
        blocks.forEach(([state, t, va, word], i) => {
          const id = `${state}_${round}`;
          order.push(id);
          steps.push(task(id, t, 60, va, "mirror", undefined, state), { kind: "intensity", block: id, word });
          if (i < blocks.length - 1 || round < 3) steps.push(task(`rest_${round}_${i + 1}`, "rest", 15, { v: 0, a: 0 }));
        });
      }
      // always end calm (not used for calibration: it has no state label)
      steps.push(task("cooldown", "breathing", 60, { v: 1, a: -1 }));
      return { steps, order };
    },
  },
  {
    id: "checkin", name: "Daily check-in", minutes: 3,
    summary: "Two quiet minutes and three questions (mood, hours slept, sleep quality), once a day at the same time. Over days, Emotif compares your brain's low-mood and drowsiness readings with your own answers. It tracks change in you; it can't diagnose.",
    build: () => ({
      order: ["checkin"],
      steps: [
        { kind: "instruction", title: "Daily check-in", body: "Sit comfortably with your eyes open on the cross for two minutes, then answer three questions." },
        { kind: "baseline", seconds: 30 },
        task("checkin", "rest", 120, { v: 0, a: 0 }, "mirror", undefined, "checkin"),
        { kind: "intensity", block: "checkin", word: "mood" },
      ],
    }),
  },
  {
    id: "closedloop", name: "Closed-loop regulation", minutes: 20,
    summary: "Three stress-then-recover blocks. During recovery the world mirrors the brain, guides it toward calm, or follows a sham path. Condition order is counterbalanced and hidden from the participant.",
    build: code => {
      const conds = shuffle<DirectorMode>(["mirror", "guide", "sham"], codeSeed(code) ^ 0x9e3779b9);
      const steps: Step[] = [
        { kind: "instruction", title: "Stress and recovery", body: "Three rounds: a minute of fast mental arithmetic, then a few minutes of watching the world and relaxing. Rate how you feel after each round." },
        { kind: "baseline", seconds: 90 },
      ];
      conds.forEach((c, i) => {
        steps.push(task(`stress${i + 1}`, "arithmetic", 60, { v: -1, a: 1 }));
        steps.push(task(`recover${i + 1}`, "watch", 150, { v: 1, a: -1 }, c, "calm"));
        steps.push({ kind: "sam", block: `recover${i + 1}` });
      });
      return { steps, order: conds };
    },
  },
];

export type SamRating = { valence: number; arousal: number };
export type CheckinAnswers = { mood: number; sleepHours: number; sleepQuality: number };
export type BlockRecord = { id: string; task: TaskKind; condition: DirectorMode; t0: number; t1: number; design: VA; sam?: SamRating;
  state?: string; intensity?: number; checkin?: CheckinAnswers };
export type RunResult = { protocol: string; code: string; order: string[]; blocks: BlockRecord[]; aborted: boolean };

/** SAM 1-9 -> -1 / 0 / +1. */
export const samClass = (x: number): number => (x <= 3 ? -1 : x >= 7 ? 1 : 0);

export type RunnerDeps = {
  now: () => number;                                     // engine time (s)
  event: (type: string, value: unknown) => void;
  label: (va: VA | null) => void;
  condition: (mode: DirectorMode, target: string | null) => void;
  recalibrate: (seconds: number) => void;
  immersive: (on: boolean) => void;
  covered: (on: boolean) => void;       // a full-screen card hides the world, so it can stop rendering
  finish: (r: RunResult) => void;
};

/** A short soft tone (the eyes-closed step ends on it). */
function beep() {
  try {
    const Ctx = (window as unknown as { AudioContext?: typeof AudioContext }).AudioContext;
    if (!Ctx) return;
    const ctx = new Ctx(), o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = 660; g.gain.value = 0.08;
    o.connect(g); g.connect(ctx.destination);
    o.start(); o.stop(ctx.currentTime + 0.35);
    o.onended = () => { void ctx.close(); };
  } catch { /* no audio: the clock still shows the end */ }
}

const svgNS = "http://www.w3.org/2000/svg";
/** Simple original pictograms for the SAM anchors (face for valence, burst for arousal). */
export function samIcon(kind: "valence" | "arousal", level: number): SVGSVGElement {
  const s = document.createElementNS(svgNS, "svg");
  s.setAttribute("viewBox", "0 0 40 40");
  s.setAttribute("aria-hidden", "true");
  const el = (tag: string, attrs: Record<string, string | number>) => {
    const e = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
    s.appendChild(e);
    return e;
  };
  const u = (level - 1) / 8;
  if (kind === "valence") {
    el("circle", { cx: 20, cy: 20, r: 15, fill: "none", stroke: "currentColor", "stroke-width": 2 });
    el("circle", { cx: 14.5, cy: 16, r: 1.8, fill: "currentColor" });
    el("circle", { cx: 25.5, cy: 16, r: 1.8, fill: "currentColor" });
    const curve = (u - 0.5) * 14;
    el("path", { d: `M12 ${25 - curve / 4} Q20 ${25 + curve} 28 ${25 - curve / 4}`, fill: "none", stroke: "currentColor", "stroke-width": 2, "stroke-linecap": "round" });
  } else {
    const n = 6, rIn = 6, rOut = 8 + 10 * u;
    el("circle", { cx: 20, cy: 20, r: 4 + 2 * u, fill: "currentColor" });
    for (let i = 0; i < n; i++) {
      const ang = (i / n) * Math.PI * 2 + 0.3;
      el("line", { x1: 20 + Math.cos(ang) * rIn, y1: 20 + Math.sin(ang) * rIn, x2: 20 + Math.cos(ang) * rOut, y2: 20 + Math.sin(ang) * rOut,
        stroke: "currentColor", "stroke-width": 1.5 + 1.5 * u, "stroke-linecap": "round" });
    }
  }
  return s;
}

export class ProtocolRunner {
  private steps: Step[] = [];
  private i = -1;
  private result: RunResult | null = null;
  private timer: number | null = null;
  private stepEnds = 0;
  private overlay: HTMLElement;
  private block: BlockRecord | null = null;
  running = false;

  constructor(host: HTMLElement, private deps: RunnerDeps) {
    this.overlay = document.createElement("div");
    this.overlay.className = "runner";
    this.overlay.hidden = true;
    this.overlay.setAttribute("role", "dialog");
    this.overlay.setAttribute("aria-live", "polite");
    host.appendChild(this.overlay);
  }

  start(p: Protocol, code: string) {
    const { steps, order } = p.build(code);
    this.steps = steps;
    this.result = { protocol: p.id, code, order, blocks: [], aborted: false };
    this.running = true;
    this.i = -1;
    this.overlay.hidden = false;
    this.deps.event("protocol_start", { protocol: p.id, order });
    this.next();
  }

  abort() {
    if (!this.running) return;
    this.closeBlock();
    this.result!.aborted = true;
    this.deps.event("protocol_abort", { step: this.i });
    this.end();
  }

  private end() {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    this.running = false;
    this.overlay.hidden = true;
    this.overlay.replaceChildren();
    this.deps.immersive(false);
    this.deps.covered(false);
    this.deps.label(null);
    this.deps.condition("mirror", null);
    this.deps.finish(this.result!);
  }

  private closeBlock() {
    if (this.block) {
      this.block.t1 = this.deps.now();
      this.result!.blocks.push(this.block);
      this.deps.event("block_end", { block: this.block.id });
      this.block = null;
    }
  }

  private next() {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    this.closeBlock();
    this.i++;
    if (this.i >= this.steps.length) { this.deps.event("protocol_end", {}); this.end(); return; }
    const s = this.steps[this.i];
    this.deps.immersive(s.kind === "task" && s.task === "watch");
    this.deps.label(null);
    if (s.kind === "instruction") this.renderInstruction(s.title, s.body);
    else if (s.kind === "baseline") {
      this.deps.recalibrate(s.seconds);
      this.deps.event("baseline_start", { seconds: s.seconds });
      this.renderTimed("Resting baseline", "Keep your eyes open on the cross and stay still.", s.seconds, "rest");
    } else if (s.kind === "task") {
      this.block = { id: s.id, task: s.task, condition: s.condition, t0: this.deps.now(), t1: 0, design: s.design, ...(s.state ? { state: s.state } : {}) };
      this.deps.condition(s.condition, s.target ?? null);
      this.deps.label(s.task === "rest" ? null : s.design);
      this.deps.event("block_start", { block: s.id, task: s.task, condition: s.condition, design: s.design, ...(s.state ? { state: s.state } : {}) });
      this.renderTimed(TASK_TEXT[s.task].title, TASK_TEXT[s.task].body, s.seconds, s.task);
    } else if (s.kind === "intensity") {
      if (s.word === "mood") this.renderCheckin(s.block); else this.renderIntensity(s.block, s.word);
    } else this.renderSam(s.block);
  }

  private frame(title: string, body: string, kind: "card" | "caption"): { root: HTMLElement; slot: HTMLElement } {
    this.overlay.className = `runner ${kind}`;
    this.deps.covered(kind === "card");
    const root = document.createElement("div");
    root.className = "runner-box";
    const top = document.createElement("div");
    top.className = "runner-top";
    const prog = document.createElement("span");
    prog.className = "runner-progress";
    prog.textContent = `Step ${this.i + 1} of ${this.steps.length}`;
    const stop = document.createElement("button");
    stop.className = "btn ghost small";
    stop.textContent = "Stop session";
    stop.onclick = () => this.confirmAbort(stop);
    top.append(prog, stop);
    const h = document.createElement("h2");
    h.textContent = title;
    const p = document.createElement("p");
    p.textContent = body;
    const slot = document.createElement("div");
    slot.className = "runner-slot";
    root.append(top, h, p, slot);
    this.overlay.replaceChildren(root);
    return { root, slot };
  }

  private confirmAbort(btn: HTMLButtonElement) {
    if (btn.dataset.confirm) { this.abort(); return; }
    btn.dataset.confirm = "1";
    btn.textContent = "Click again to stop";
    btn.classList.add("danger");
    window.setTimeout(() => { if (btn.isConnected) { delete btn.dataset.confirm; btn.textContent = "Stop session"; btn.classList.remove("danger"); } }, 3000);
  }

  private renderInstruction(title: string, body: string) {
    const { slot } = this.frame(title, body, "card");
    const go = document.createElement("button");
    go.className = "btn primary large";
    go.textContent = "Continue";
    go.onclick = () => this.next();
    slot.append(go);
    go.focus();
  }

  private renderTimed(title: string, body: string, seconds: number, kind: TaskKind) {
    const { slot } = this.frame(title, body, kind === "watch" ? "caption" : "card");
    const vis = document.createElement("div");
    vis.className = `task-vis task-${kind}`;
    const clock = document.createElement("div");
    clock.className = "runner-clock";
    slot.append(vis, clock);
    let n = 1022;
    if (kind === "rest") vis.textContent = "+";
    if (kind === "drowsy") vis.classList.add("task-dot");
    if (kind === "arithmetic") {
      n = 1022 + Math.floor(Math.random() * 400);
      vis.textContent = String(n);
      const step = document.createElement("button");
      step.className = "btn small";
      step.textContent = "Next number";
      step.onclick = () => { n -= 13; vis.textContent = String(n); };
      slot.append(step);
    }
    const t0 = performance.now();
    this.stepEnds = t0 + seconds * 1000;
    const tick = () => {
      const left = Math.max(0, (this.stepEnds - performance.now()) / 1000);
      clock.textContent = `${Math.floor(left / 60)}:${String(Math.floor(left % 60)).padStart(2, "0")}`;
      if (kind === "breathing") {
        const ph = ((performance.now() - t0) / 1000) % 10;            // 4 s in, 6 s out
        const s = ph < 4 ? ph / 4 : 1 - (ph - 4) / 6;
        vis.style.setProperty("--s", (0.45 + 0.55 * s).toFixed(3));
        vis.textContent = ph < 4 ? "In" : "Out";
      }
      if (kind === "arithmetic") vis.style.setProperty("--p", String(1 - left / seconds));
      if (kind === "drowsy") vis.style.setProperty("--s", (0.25 + 0.75 * (0.5 + 0.5 * Math.cos(((performance.now() - t0) / 1000) * 2 * Math.PI / 12))).toFixed(3));
      if (left <= 0) {
        if (kind === "eyes_closed") beep();
        this.next();
      }
    };
    tick();
    this.timer = window.setInterval(tick, 100);
  }

  private blockById(id: string): BlockRecord | undefined { return [...this.result!.blocks].reverse().find(b => b.id === id); }

  /** 0-10 scale as eleven buttons (keyboard and screen-reader friendly). */
  private scale0to10(legend: string, lo: string, hi: string, onPick: (v: number) => void, max = 10, unit = ""): HTMLFieldSetElement {
    const wrap = document.createElement("fieldset");
    wrap.className = "sam";
    const lg = document.createElement("legend");
    lg.textContent = legend;
    const row = document.createElement("div");
    row.className = "sam-row";
    for (let k = 0; k <= max; k++) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "sam-pt";
      b.setAttribute("aria-label", `${legend}: ${k}${unit}`);
      b.setAttribute("aria-pressed", "false");
      const num = document.createElement("small");
      num.textContent = String(k);
      b.append(num);
      b.onclick = () => { row.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b))); onPick(k); };
      row.append(b);
    }
    const ends = document.createElement("div");
    ends.className = "sam-ends";
    const l = document.createElement("span"); l.textContent = lo;
    const r = document.createElement("span"); r.textContent = hi;
    ends.append(l, r);
    wrap.append(lg, row, ends);
    return wrap;
  }

  private renderIntensity(blockId: string, word: string) {
    const { slot } = this.frame("How strongly?", `During the task you just finished, how ${word} did you feel?`, "card");
    const done = document.createElement("button");
    done.className = "btn primary large";
    done.textContent = "Save rating";
    done.disabled = true;
    let v = -1;
    done.onclick = () => {
      const blk = this.blockById(blockId);
      if (blk) blk.intensity = v;
      this.deps.event("intensity", { block: blockId, value: v });
      this.next();
    };
    slot.append(this.scale0to10(`How ${word}`, "Not at all", "Extremely", k => { v = k; done.disabled = false; }), done);
  }

  private renderCheckin(blockId: string) {
    const { slot } = this.frame("Three questions", "Answer for today. There are no right answers.", "card");
    const a: Partial<CheckinAnswers> = {};
    const done = document.createElement("button");
    done.className = "btn primary large";
    done.textContent = "Save";
    done.disabled = true;
    const ready = () => { done.disabled = a.mood === undefined || a.sleepHours === undefined || a.sleepQuality === undefined; };
    done.onclick = () => {
      const blk = this.blockById(blockId);
      if (blk) blk.checkin = a as CheckinAnswers;
      this.deps.event("checkin", a);
      this.next();
    };
    slot.append(
      this.scale0to10("Mood right now", "Very low", "Very good", k => { a.mood = k; ready(); }),
      this.scale0to10("Hours slept last night", "0", "12 or more", k => { a.sleepHours = k; ready(); }, 12, " hours"),
      this.scale0to10("Sleep quality last night", "Very poor", "Very good", k => { a.sleepQuality = k; ready(); }),
      done);
  }

  private renderSam(blockId: string) {
    const { slot } = this.frame("How did you feel?", "Rate the task you just finished. Pick the point that fits best.", "card");
    const vals: Partial<SamRating> = {};
    const done = document.createElement("button");
    done.className = "btn primary large";
    done.textContent = "Save rating";
    done.disabled = true;
    const scale = (kind: "valence" | "arousal", lo: string, hi: string) => {
      const wrap = document.createElement("fieldset");
      wrap.className = "sam";
      const lg = document.createElement("legend");
      lg.textContent = kind === "valence" ? "Pleasant or unpleasant" : "Calm or excited";
      const row = document.createElement("div");
      row.className = "sam-row";
      for (let k = 1; k <= 9; k++) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "sam-pt";
        b.setAttribute("aria-label", `${kind} ${k} of 9`);
        b.setAttribute("aria-pressed", "false");
        if (k === 1 || k === 5 || k === 9) b.append(samIcon(kind, k));
        else { const d = document.createElement("span"); d.className = "sam-dot"; b.append(d); }
        const num = document.createElement("small");
        num.textContent = String(k);
        b.append(num);
        b.onclick = () => {
          row.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
          vals[kind] = k;
          done.disabled = !(vals.valence && vals.arousal);
        };
        row.append(b);
      }
      const ends = document.createElement("div");
      ends.className = "sam-ends";
      const l = document.createElement("span"); l.textContent = lo;
      const r = document.createElement("span"); r.textContent = hi;
      ends.append(l, r);
      wrap.append(lg, row, ends);
      return wrap;
    };
    done.onclick = () => {
      const rating = { valence: vals.valence!, arousal: vals.arousal! };
      const blk = [...this.result!.blocks].reverse().find(b => b.id === blockId);
      if (blk) blk.sam = rating;
      this.deps.event("sam", { block: blockId, ...rating });
      this.next();
    };
    slot.append(scale("valence", "Unpleasant", "Pleasant"), scale("arousal", "Calm", "Excited"), done);
  }
}
