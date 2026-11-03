// Client for the optional Python backend (python -m emotif.server). The app works without it.
import type { Source, SourceSink } from "./sources";

export type ArmSnapshot = {
  connected: boolean; transport: string; enabled: boolean; estop: boolean;
  x: number; y: number; z: number; t: number; target: { x: number; y: number; z: number; t: number };
  clamped: boolean; error: string; last_cmd_age: number;
  limits: { r_min: number; r_max: number; z_min: number; z_max: number; base_max_deg: number; speed_mm_s: number };
};
export type SessionMeta = { id: string; participant: string; device: string; fs: number; channels: string[]; started: string; duration_s?: number; protocol?: string; n_samples?: number };
export type ServerState = { stream: { fs: number; channels: string[]; origin: string; label: string } | null; recording: { id: string; seconds: number } | null; arm: ArmSnapshot | null; error: string };

type Handler = (msg: any) => void; // eslint-disable-line @typescript-eslint/no-explicit-any

export class ServerLink {
  available = false;
  ws: WebSocket | null = null;
  private handlers = new Map<string, Set<Handler>>();
  private queue: string[] = [];

  async detect(): Promise<boolean> {
    if (location.protocol !== "http:" && location.protocol !== "https:") return false;
    if (!document.querySelector('meta[name="emotif-server"]')) return false;   // static copy: no backend
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), 1500);
      const r = await fetch("/api/health", { signal: ctl.signal, cache: "no-store" });
      clearTimeout(timer);
      const j = r.ok ? await r.json() : null;
      this.available = !!j && j.app === "emotif";
    } catch { this.available = false; }
    if (this.available) this.connect();
    return this.available;
  }

  private connect() {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onopen = () => { for (const m of this.queue.splice(0)) ws.send(m); this.fire("open", {}); };
    ws.onmessage = ev => {
      let msg: { type?: string };
      try { msg = JSON.parse(String(ev.data)); } catch { return; }
      if (msg.type) this.fire(msg.type, msg);
    };
    ws.onclose = () => {
      if (this.ws === ws) { this.ws = null; this.fire("close", {}); setTimeout(() => this.connect(), 1500); }
    };
  }

  on(type: string, h: Handler): () => void {
    let s = this.handlers.get(type);
    if (!s) { s = new Set(); this.handlers.set(type, s); }
    s.add(h);
    return () => s!.delete(h);
  }
  private fire(type: string, msg: unknown) { this.handlers.get(type)?.forEach(h => h(msg)); }

  send(msg: object) {
    if (!this.available) return;
    const s = JSON.stringify(msg);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(s);
    else if (this.queue.length < 200) this.queue.push(s);
  }

  async api<T = { ok: boolean; error?: string }>(path: string, body?: object): Promise<T> {
    const r = await fetch(path, body === undefined ? { cache: "no-store" } : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    let j: unknown = null;
    try { j = await r.json(); } catch { /* non-JSON error page */ }
    if (!r.ok) {
      const msg = (j as { error?: string } | null)?.error ?? `Request failed (${r.status})`;
      throw new Error(msg);
    }
    return j as T;
  }
}

/** EEG acquired by the server (BrainFlow boards, LSL). */
export class ServerDeviceSource implements Source {
  readonly kind = "server";
  private off: (() => void)[] = [];
  constructor(private link: ServerLink, private cfg: Record<string, unknown>) {}

  async start(sink: SourceSink) {
    this.detach();
    this.off.push(this.link.on("meta", m => sink.onMeta(Number(m.fs), m.channels, `${m.label}`)));
    this.off.push(this.link.on("data", m => sink.onRows(m.samples)));
    this.off.push(this.link.on("device_closed", () => { sink.onStatus("Device disconnected", "warn"); sink.onEnd(); }));
    sink.onStatus("Opening the device… (Bluetooth headsets can take 10–20 s)", "warn");
    try {
      await this.link.api("/api/device/connect", this.cfg);
      sink.onStatus("Device streaming", "ok");
    } catch (e) {
      sink.onStatus((e as Error).message, "error");
      this.detach();
    }
  }
  private detach() { for (const f of this.off) f(); this.off = []; }
  stop() {
    const wasActive = this.off.length > 0;
    this.detach();
    if (wasActive && this.link.available) this.link.api("/api/device/disconnect", {}).catch(() => undefined);
  }
}
