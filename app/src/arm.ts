// RoArm-M2-S: kinematics (same as emotif/roarm.py and the Waveshare firmware), a local simulated arm used
// when no server is running, and the control policies that turn brain signals into end-effector targets.
import type { ServerLink, ArmSnapshot } from "./server";

export const L2A = 236.82, L2B = 30.0, L3A = 280.15, L3B = 1.73;
export const L2 = Math.hypot(L2A, L2B), L3 = Math.hypot(L3A, L3B), T2 = Math.atan2(L2B, L2A);
export const INIT = { x: L3A + L2B, y: 0, z: L2A - L3B, t: Math.PI };
export const GRIP_OPEN = 1.08, GRIP_CLOSED = Math.PI;
export type Pose = { x: number; y: number; z: number; t: number };
export type Joints = { b: number; s: number; e: number };

export function forward(j: Joints): { x: number; y: number; z: number } {
  const a1 = Math.PI / 2 - (j.s + T2), a2 = Math.PI / 2 - (j.e + j.s);
  const r = L2 * Math.cos(a1) + L3 * Math.cos(a2), z = L2 * Math.sin(a1) + L3 * Math.sin(a2);
  return { x: r * Math.cos(j.b), y: r * Math.sin(j.b), z };
}
export function inverse(x: number, y: number, z: number): Joints | null {
  const b = Math.atan2(y, x), r = Math.hypot(x, y);
  const c = (r * r + z * z - L2 * L2 - L3 * L3) / (2 * L2 * L3);
  if (c < -1 || c > 1) return null;
  const g = Math.acos(c), phi = Math.atan2(z, r);
  const a1 = phi + Math.atan2(L3 * Math.sin(g), L2 + L3 * Math.cos(g)), a2 = a1 - g;
  const s = Math.PI / 2 - a1 - T2, e = Math.PI / 2 - a2 - s;
  if (!(s >= -Math.PI / 2 - 1e-6 && s <= Math.PI / 2 + 1e-6 && e >= -1e-6 && e <= Math.PI + 1e-6)) return null;
  return { b, s, e };
}

export type Limits = { r_min: number; r_max: number; z_min: number; z_max: number; base_max_deg: number; speed_mm_s: number };
export const DEFAULT_LIMITS: Limits = { r_min: 150, r_max: 430, z_min: -80, z_max: 380, base_max_deg: 150, speed_mm_s: 120 };

export function clampPose(p: Pose, lim: Limits): Pose & { clamped: boolean } {
  let r = Math.hypot(p.x, p.y);
  let ang = r > 1e-9 ? Math.atan2(p.y, p.x) : 0;
  const amax = lim.base_max_deg * Math.PI / 180;
  ang = Math.max(-amax, Math.min(amax, ang));
  r = Math.max(lim.r_min, Math.min(lim.r_max, r));
  let z = Math.max(lim.z_min, Math.min(lim.z_max, p.z));
  for (let i = 0; i < 60 && !inverse(r * Math.cos(ang), r * Math.sin(ang), z); i++) {
    r = Math.max(lim.r_min, r - 5);
    if (r === lim.r_min) z = z > 0 ? z - 5 : z + 5;
  }
  const out = { x: r * Math.cos(ang), y: r * Math.sin(ang), z, t: Math.max(GRIP_OPEN, Math.min(GRIP_CLOSED, p.t)) };
  const clamped = Math.abs(out.x - p.x) > 1e-6 || Math.abs(out.y - p.y) > 1e-6 || Math.abs(out.z - p.z) > 1e-6 || Math.abs(out.t - p.t) > 1e-6;
  return { ...out, clamped };
}

/** Local stand-in for the server governor so the digital twin works anywhere. */
export class LocalArm {
  pose: Pose = { ...INIT };
  target: Pose = { ...INIT };
  enabled = false; estop = false; clamped = false;
  lim: Limits = { ...DEFAULT_LIMITS };
  private lastTarget = 0;
  setTarget(p: Pose) { const c = clampPose(p, this.lim); this.target = c; this.clamped = c.clamped; this.lastTarget = performance.now(); }
  step(dt: number) {
    if (!this.enabled || this.estop || performance.now() - this.lastTarget > 1000) return;
    const dx = this.target.x - this.pose.x, dy = this.target.y - this.pose.y, dz = this.target.z - this.pose.z;
    const d = Math.hypot(dx, dy, dz), step = this.lim.speed_mm_s * dt, f = d <= step || d === 0 ? 1 : step / d;
    this.pose = { x: this.pose.x + dx * f, y: this.pose.y + dy * f, z: this.pose.z + dz * f,
      t: this.pose.t + Math.max(-2 * dt, Math.min(2 * dt, this.target.t - this.pose.t)) };
  }
}

export type ArmMode = "embody" | "draw" | "intent" | "manual";
export const INTENT_ACTIONS = ["Forward", "Back", "Left", "Right", "Up", "Down", "Open grip", "Close grip"] as const;
export type IntentAction = typeof INTENT_ACTIONS[number];

/** Turns affect / intent / manual input into a stream of targets, and sends them to the real arm (server)
 * or the local simulation. */
export class ArmController {
  mode: ArmMode = "embody";
  local = new LocalArm();
  server: ArmSnapshot | null = null;
  useServer = false;
  manual: Pose = { ...INIT };
  draw = { cx: 280, cy: 0, size: 140, penZ: -60, lifted: true };
  trail: { x: number; y: number; z: number }[] = [];
  intentIndex = 0;
  moving = false;          // jaw clench held in intent mode
  speedMmS = 40;
  private time = 0;
  private sendAcc = 0;

  constructor(private link: ServerLink) {
    link.on("arm", (s: ArmSnapshot) => { this.server = s; });
  }

  get connectedToReal(): boolean { return this.useServer && !!this.server?.connected; }
  get pose(): Pose { return this.connectedToReal && this.server ? this.server : this.local.pose; }
  get enabled(): boolean { return this.connectedToReal ? !!this.server?.enabled : this.local.enabled; }
  get action(): IntentAction { return INTENT_ACTIONS[this.intentIndex]; }

  nextIntent() { this.intentIndex = (this.intentIndex + 1) % INTENT_ACTIONS.length; }

  /** dt seconds; affect in [-1, 1]. Returns the commanded target. */
  update(dt: number, affect: { v: number; a: number } | null): Pose {
    this.time += dt;
    const cur = this.pose;
    let tgt: Pose;
    const v = affect?.v ?? 0, a = affect?.a ?? 0;
    switch (this.mode) {
      case "embody": {
        // posture follows affect: open and lifted when positive, low and closed when negative;
        // sway rate follows arousal
        const ar = (a + 1) / 2, val = (v + 1) / 2;
        const sway = (12 + 18 * ar) * Math.sin(2 * Math.PI * (0.04 + 0.22 * ar) * this.time);
        const r = 250 + 70 * val, z = 40 + 200 * val;
        const ang = sway * Math.PI / 180;
        tgt = { x: r * Math.cos(ang), y: r * Math.sin(ang), z, t: GRIP_CLOSED - (GRIP_CLOSED - 1.6) * val };
        break;
      }
      case "draw": {
        // plots the affect trajectory on paper: arousal -> forward, valence -> right (y negative)
        const d = this.draw, h = d.size / 2;
        tgt = { x: d.cx + a * h, y: d.cy - v * h, z: d.penZ + (d.lifted ? 30 : 0), t: GRIP_CLOSED };
        if (!d.lifted) {
          const last = this.trail[this.trail.length - 1];
          if (!last || Math.hypot(last.x - cur.x, last.y - cur.y) > 1.5) this.trail.push({ x: cur.x, y: cur.y, z: cur.z });
          if (this.trail.length > 6000) this.trail.shift();
        }
        break;
      }
      case "intent": {
        const s = this.moving ? this.speedMmS * dt : 0;
        const p = { ...this.manual };
        const r = Math.hypot(p.x, p.y), ang = Math.atan2(p.y, p.x);
        switch (this.action) {
          case "Forward": p.x += s * Math.cos(ang); p.y += s * Math.sin(ang); break;
          case "Back": p.x -= s * Math.cos(ang); p.y -= s * Math.sin(ang); break;
          case "Left": { const na = ang + s / Math.max(r, 1); p.x = r * Math.cos(na); p.y = r * Math.sin(na); break; }
          case "Right": { const na = ang - s / Math.max(r, 1); p.x = r * Math.cos(na); p.y = r * Math.sin(na); break; }
          case "Up": p.z += s; break;
          case "Down": p.z -= s; break;
          case "Open grip": if (this.moving) p.t -= 1.2 * dt; break;
          case "Close grip": if (this.moving) p.t += 1.2 * dt; break;
        }
        const lim = this.connectedToReal && this.server ? this.server.limits : this.local.lim;
        this.manual = clampPose(p, lim);
        tgt = { ...this.manual };
        break;
      }
      default:
        tgt = { ...this.manual };
    }
    this.local.setTarget(tgt);
    this.local.step(dt);
    if (this.connectedToReal) {
      this.sendAcc += dt;
      if (this.sendAcc >= 0.1) {
        this.sendAcc = 0;
        if (this.server?.enabled) this.link.send({ type: "arm_target", ...tgt });
      }
    }
    return tgt;
  }

  /** Re-anchor manual/intent targets at the current pose (after enabling or switching modes). */
  syncManual() { const p = this.pose; this.manual = { x: p.x, y: p.y, z: p.z, t: p.t }; }
}
