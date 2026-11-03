// Canvas charts. Colours come from the CSS theme tokens so they follow light/dark mode.
import { position, project, REGIONS, anchor } from "./montage";
import { affectCss } from "./world";

export function token(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

let fontCache: { mono: string; sans: string; at: number } | null = null;
/** Canvas can't resolve CSS variables, so read the font stacks once per second. */
export function fonts(): { mono: string; sans: string } {
  const now = performance.now();
  if (!fontCache || now - fontCache.at > 1000) fontCache = { mono: token("--font-mono"), sans: token("--font-sans"), at: now };
  return fontCache;
}

/** Size the backing store to the element's CSS size at device pixel ratio; returns the 2D context and size. */
export function fit(cv: HTMLCanvasElement): { g: CanvasRenderingContext2D; w: number; h: number; dpr: number } | null {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
  if (w <= 0 || h <= 0) return null;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const g = cv.getContext("2d");
  if (!g) return null;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h, dpr };
}

export function drawCircumplex(cv: HTMLCanvasElement, trail: { v: number; a: number }[], truth: { v: number; a: number } | null,
                               env: { v: number; a: number } | null, target: { v: number; a: number } | null) {
  const f = fit(cv);
  if (!f) return;
  const { g, w, dpr } = f;
  const m = w * 0.1, R = (w - 2 * m) / 2, cx = w / 2, cy = w / 2;
  const P = (v: number, a: number): [number, number] => {
    const r = Math.hypot(v, a), k = r > 1 ? 1 / r : 1;          // keep every mark inside the unit disk
    return [cx + v * k * R, cy - a * k * R];
  };
  const line = token("--line"), muted = token("--muted"), ink = token("--ink");
  g.lineWidth = dpr;
  g.strokeStyle = line;
  g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();
  g.setLineDash([3 * dpr, 4 * dpr]);
  g.beginPath(); g.arc(cx, cy, R * 0.25, 0, Math.PI * 2); g.stroke();
  g.setLineDash([]);
  g.beginPath(); g.moveTo(cx - R, cy); g.lineTo(cx + R, cy); g.moveTo(cx, cy - R); g.lineTo(cx, cy + R); g.stroke();
  g.font = `${11 * dpr}px ${fonts().mono}`;
  g.fillStyle = muted; g.textBaseline = "middle";
  g.textAlign = "center";
  g.fillText("High arousal", cx, m * 0.45); g.fillText("Low arousal", cx, w - m * 0.45);
  g.save(); g.translate(m * 0.45, cy); g.rotate(-Math.PI / 2); g.fillText("Unpleasant", 0, 0); g.restore();
  g.save(); g.translate(w - m * 0.45, cy); g.rotate(Math.PI / 2); g.fillText("Pleasant", 0, 0); g.restore();
  g.font = `${10.5 * dpr}px ${fonts().sans}`;
  ([["Happy", 0.6, 0.6], ["Calm", 0.6, -0.6], ["Sad", -0.6, -0.6], ["Stressed", -0.6, 0.6]] as const).forEach(([t, v, a]) => {
    const [x, y] = P(v, a); g.fillStyle = affectCss(v, a); g.globalAlpha = 0.9; g.fillText(t, x, y); g.globalAlpha = 1;
  });
  if (target) {
    const [x, y] = P(target.v, target.a);
    g.strokeStyle = ink; g.setLineDash([2 * dpr, 3 * dpr]);
    g.beginPath(); g.arc(x, y, 12 * dpr, 0, Math.PI * 2); g.stroke(); g.setLineDash([]);
  }
  if (truth) {
    const [x, y] = P(truth.v, truth.a);
    g.strokeStyle = ink; g.lineWidth = 1.5 * dpr;
    g.beginPath(); g.arc(x, y, 8 * dpr, 0, Math.PI * 2); g.stroke();
  }
  if (env) {
    const [x, y] = P(env.v, env.a);
    g.fillStyle = muted;
    g.beginPath(); g.moveTo(x, y - 6 * dpr); g.lineTo(x + 6 * dpr, y); g.lineTo(x, y + 6 * dpr); g.lineTo(x - 6 * dpr, y); g.closePath(); g.fill();
  }
  if (trail.length) {
    g.lineWidth = 1.5 * dpr;
    for (let i = 1; i < trail.length; i++) {
      const [x0, y0] = P(trail[i - 1].v, trail[i - 1].a), [x1, y1] = P(trail[i].v, trail[i].a);
      g.strokeStyle = affectCss(trail[i].v, trail[i].a);
      g.globalAlpha = 0.15 + 0.6 * (i / trail.length);
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    }
    g.globalAlpha = 1;
    const last = trail[trail.length - 1], [x, y] = P(last.v, last.a);
    g.fillStyle = affectCss(last.v, last.a);
    g.strokeStyle = token("--surface"); g.lineWidth = 2 * dpr;
    g.beginPath(); g.arc(x, y, 6.5 * dpr, 0, Math.PI * 2); g.fill(); g.stroke();
  }
}

function diverging(t: number): string {   // t in [-1, 1]; blue - neutral - red
  const c = Math.max(-1, Math.min(1, t));
  const neu = [236, 236, 232], blue = [49, 106, 160], red = [196, 72, 52];
  const to = c < 0 ? blue : red, k = Math.abs(c);
  const ch = (i: number) => Math.round(neu[i] + (to[i] - neu[i]) * k);
  return `rgb(${ch(0)},${ch(1)},${ch(2)})`;
}

/** Scalp map: inverse-distance interpolation of per-channel values (z-scores) on the projected head disk. */
export type DotStatus = "ok" | "noisy" | "off" | "plain";
export function drawTopomap(cv: HTMLCanvasElement, channels: string[], values: (number | null)[], status: DotStatus[],
                            opts: { showRegions?: boolean; range?: number } = {}) {
  const f = fit(cv);
  if (!f) return;
  const { g, w, h, dpr } = f;
  const S = Math.min(w, h), cx = w / 2, cy = h / 2 + S * 0.02, R = S * 0.4;
  const pts: { x: number; y: number; v: number | null; s: DotStatus; name: string }[] = [];
  channels.forEach((c, i) => {
    const p = position(c);
    if (!p) return;
    const [px, py] = project(p);
    pts.push({ x: cx + px * R, y: cy + py * R, v: values[i], s: status[i], name: c });
  });
  const range = opts.range ?? 2;
  const valued = pts.filter(p => p.v !== null && Number.isFinite(p.v));
  // interpolated field
  if (valued.length) {
    const step = Math.max(2, Math.round(S / 70));
    for (let y = cy - R; y <= cy + R; y += step) {
      for (let x = cx - R; x <= cx + R; x += step) {
        const dx = x - cx, dy = y - cy;
        if (dx * dx + dy * dy > R * R) continue;
        let num = 0, den = 0;
        for (const p of valued) {
          const d2 = (p.x - x) ** 2 + (p.y - y) ** 2 + 1;
          const wgt = 1 / (d2 * d2);
          num += wgt * p.v!; den += wgt;
        }
        g.fillStyle = diverging(num / den / range);
        g.fillRect(x, y, step + 0.5, step + 0.5);
      }
    }
  }
  const ink = token("--ink"), muted = token("--muted");
  g.strokeStyle = ink; g.lineWidth = 1.2 * dpr;
  g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();
  g.beginPath(); g.moveTo(cx - R * 0.09, cy - R * 0.99); g.lineTo(cx, cy - R * 1.1); g.lineTo(cx + R * 0.09, cy - R * 0.99); g.stroke();
  g.beginPath(); g.ellipse(cx - R * 1.03, cy, R * 0.05, R * 0.14, 0, 0, Math.PI * 2); g.stroke();
  g.beginPath(); g.ellipse(cx + R * 1.03, cy, R * 0.05, R * 0.14, 0, 0, Math.PI * 2); g.stroke();
  if (opts.showRegions) {
    g.fillStyle = muted;
    for (const r of REGIONS) {
      const [px, py] = project(anchor(r));
      g.beginPath(); g.arc(cx + px * R, cy + py * R, 1.5 * dpr, 0, Math.PI * 2); g.fill();
    }
  }
  const col: Record<DotStatus, string> = { ok: token("--ok"), noisy: token("--warn"), off: token("--danger"), plain: token("--line-strong") };
  g.font = `600 ${10 * dpr}px ${fonts().mono}`;
  g.textAlign = "center"; g.textBaseline = "top";
  for (const p of pts) {
    g.fillStyle = token("--surface");
    g.strokeStyle = col[p.s as DotStatus] ?? muted;
    g.lineWidth = 2.5 * dpr;
    g.beginPath(); g.arc(p.x, p.y, 5.5 * dpr, 0, Math.PI * 2); g.fill(); g.stroke();
    g.fillStyle = ink;
    g.fillText(p.name, p.x, p.y + 8 * dpr);
  }
  if (!pts.length) {
    g.fillStyle = muted; g.textAlign = "center"; g.textBaseline = "middle";
    g.font = `${12 * dpr}px ${fonts().sans}`;
    g.fillText("Name the channels in Setup to place them on the scalp", cx, cy);
  }
}

export function drawTraces(cv: HTMLCanvasElement, data: Float64Array[], names: string[], status: ("ok" | "noisy" | "off")[], scaleUv = 100) {
  const f = fit(cv);
  if (!f) return;
  const { g, w, h, dpr } = f;
  if (!data.length || !data[0].length) return;
  const n = data[0].length, lane = h / data.length, step = Math.max(1, Math.floor(n / w));
  g.lineWidth = dpr;
  g.font = `${10 * dpr}px ${fonts().mono}`;
  g.textBaseline = "middle"; g.textAlign = "left";
  const ink = token("--ink-2"), muted = token("--muted"), warn = token("--warn"), bad = token("--danger");
  data.forEach((x, c) => {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    const y0 = lane * (c + 0.5), k = lane * 0.48 / scaleUv;
    g.strokeStyle = status[c] === "ok" ? ink : status[c] === "noisy" ? warn : bad;
    g.beginPath();
    for (let i = 0; i < n; i += step) {
      const X = i / n * w, Y = y0 - Math.max(-scaleUv, Math.min(scaleUv, x[i] - mean)) * k;
      if (i) g.lineTo(X, Y); else g.moveTo(X, Y);
    }
    g.stroke();
    g.fillStyle = muted;
    g.fillText(names[c], 4 * dpr, y0 - lane * 0.32);
  });
}

export type TimelineEvent = { t: number; label: string };
export function drawTimeline(cv: HTMLCanvasElement, hist: { t: number; v: number; a: number; q: number }[], events: TimelineEvent[],
                             span: number, now: number, blocks: { t0: number; t1: number; label: string }[] = []) {
  const f = fit(cv);
  if (!f) return;
  const { g, w, h, dpr } = f;
  const t0 = Math.max(0, now - span), X = (t: number) => (t - t0) / span * w;
  const top = 14 * dpr, bottom = h - 4 * dpr, mid = (top + bottom) / 2, amp = (bottom - top) / 2 * 0.9;
  g.fillStyle = token("--surface-2");
  for (const b of blocks) {
    if (b.t1 < t0) continue;
    const x0 = Math.max(0, X(b.t0)), x1 = X(b.t1 || now);
    g.fillRect(x0, top, Math.max(1, x1 - x0), bottom - top);
    g.fillStyle = token("--muted"); g.font = `${9.5 * dpr}px ${fonts().mono}`; g.textAlign = "left"; g.textBaseline = "top";
    g.fillText(b.label, x0 + 3 * dpr, top + 2 * dpr);
    g.fillStyle = token("--surface-2");
  }
  g.strokeStyle = token("--line"); g.lineWidth = dpr;
  g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
  const vis = hist.filter(p => p.t >= t0);
  const series = (key: "v" | "a", color: string) => {
    g.strokeStyle = color; g.lineWidth = 1.6 * dpr; g.beginPath();
    vis.forEach((p, i) => { const x = X(p.t), y = mid - p[key] * amp; if (i) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
  };
  for (const p of vis) if (p.q < 0.6) { g.fillStyle = token("--warn"); g.globalAlpha = 0.12; g.fillRect(X(p.t), top, 2 * dpr, bottom - top); g.globalAlpha = 1; }
  series("v", token("--accent"));
  series("a", token("--ink-2"));
  g.font = `${9.5 * dpr}px ${fonts().mono}`; g.textBaseline = "top";
  for (const e of events) {
    if (e.t < t0) continue;
    const x = X(e.t);
    g.strokeStyle = token("--muted"); g.setLineDash([2 * dpr, 2 * dpr]);
    g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke(); g.setLineDash([]);
  }
  g.fillStyle = token("--muted"); g.textAlign = "left";
  g.fillText(`−${Math.round(span)} s`, 4 * dpr, 1 * dpr);
  g.textAlign = "right";
  g.fillText("now", w - 4 * dpr, 1 * dpr);
}

/** Brain-only vs standard decoder over the last `span` seconds for one dimension, with facial activity underneath. */
export function drawCompare(cv: HTMLCanvasElement, hist: { t: number; brain: number; std: number | null }[],
                            face: { t: number; eye: number; jaw: number }[], span: number, now: number) {
  const f = fit(cv);
  if (!f) return;
  const { g, w, h, dpr } = f;
  const t0 = Math.max(0, now - span), X = (t: number) => (t - t0) / span * w;
  const strip = 22 * dpr, top = 14 * dpr, bottom = h - strip - 6 * dpr, mid = (top + bottom) / 2, amp = (bottom - top) / 2 * 0.9;
  g.strokeStyle = token("--line"); g.lineWidth = dpr;
  g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
  const vis = hist.filter(p => p.t >= t0);
  const line = (val: (p: typeof vis[number]) => number | null, color: string, dash: number[], width: number) => {
    g.strokeStyle = color; g.lineWidth = width * dpr; g.setLineDash(dash.map(d => d * dpr)); g.beginPath();
    let started = false;
    for (const p of vis) {
      const v = val(p);
      if (v === null) { started = false; continue; }
      const x = X(p.t), y = mid - Math.max(-1, Math.min(1, v)) * amp;
      if (started) g.lineTo(x, y); else { g.moveTo(x, y); started = true; }
    }
    g.stroke(); g.setLineDash([]);
  };
  line(p => p.std, token("--ink-2"), [4, 3], 1.4);
  line(p => p.brain, token("--accent"), [], 2);
  // facial activity strip: eye (top half) and jaw (bottom half), above the person's baseline only
  const y0 = h - strip;
  g.fillStyle = token("--surface-2"); g.fillRect(0, y0, w, strip);
  const fv = face.filter(p => p.t >= t0), bw = Math.max(dpr, w / (span * 4));
  for (const p of fv) {
    const e = Math.tanh(Math.max(0, p.eye) / 2), j = Math.tanh(Math.max(0, p.jaw) / 2), x = X(p.t);
    g.fillStyle = token("--warn"); g.globalAlpha = 0.25 + 0.75 * e; g.fillRect(x, y0, bw, (strip / 2) * e);
    g.fillStyle = token("--danger"); g.globalAlpha = 0.25 + 0.75 * j; g.fillRect(x, y0 + strip / 2, bw, (strip / 2) * j);
  }
  g.globalAlpha = 1;
  g.font = `${9.5 * dpr}px ${fonts().mono}`; g.fillStyle = token("--muted"); g.textBaseline = "top"; g.textAlign = "left";
  g.fillText(`−${Math.round(span)} s`, 4 * dpr, 1 * dpr);
  g.textAlign = "right"; g.fillText("now", w - 4 * dpr, 1 * dpr);
  g.textAlign = "left"; g.textBaseline = "middle";
  g.fillText("eye", 4 * dpr, y0 + strip * 0.25); g.fillText("jaw", 4 * dpr, y0 + strip * 0.75);
}
