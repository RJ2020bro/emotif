// Emotiv headsets (Insight, EPOC X, MN8, ...) through the EMOTIV Cortex service on this computer
// (wss://localhost:6868, JSON-RPC 2.0). Raw EEG needs an Emotiv licence that includes the "eeg" stream; without one
// Cortex refuses the subscription and this source says so.
import type { Source, SourceSink } from "./sources";

export const CORTEX_URL = "wss://localhost:6868";
const META_COLS = new Set(["COUNTER", "INTERPOLATED", "RAW_CQ", "MARKER_HARDWARE", "MARKERS", "TIMESTAMP"]);

/** Minimal WebSocket surface, so tests can pass a fake. */
export interface WsLike {
  readyState: number;
  send(data: string): void;
  close(): void;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onclose: ((ev: unknown) => void) | null;
}

export type CortexConfig = { clientId: string; clientSecret: string; headsetId?: string; license?: string };

/** EEG columns of an "eeg" subscription: names and positions of the electrode values (everything that isn't
 * a counter, quality or marker column). `sensors` (from queryHeadsets) is preferred when given. */
export function eegColumns(cols: string[], sensors?: string[]): { names: string[]; index: number[] } {
  const want = sensors && sensors.length ? new Set(sensors.map(s => s.toUpperCase())) : null;
  const names: string[] = [], index: number[] = [];
  cols.forEach((c, i) => {
    const u = c.toUpperCase();
    if (META_COLS.has(u) || u.startsWith("CQ_") || u.startsWith("EQ_")) return;
    if (want && !want.has(u)) return;
    names.push(c); index.push(i);
  });
  return { names, index };
}

export class CortexError extends Error {
  constructor(message: string, readonly code?: number) { super(message); }
}

/** Plain-language message for a Cortex error or refused subscription. */
export function explainCortex(code: number | undefined, message: string): string {
  if (code === -32016 || /unavailable|unsupported/i.test(message))
    return "Cortex refused the raw EEG stream. Raw EEG needs an Emotiv licence that includes it (EmotivPRO); check your plan in the EMOTIV Launcher.";
  if (code === -32002 || /license|licence/i.test(message))
    return `Licence problem: ${message}`;
  if (code === -32021 || (/client/i.test(message) && /invalid/i.test(message)))
    return "Cortex didn't accept the client ID or secret. Copy them again from your Cortex app at emotiv.com (My account → Cortex apps).";
  return message;
}

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: ReturnType<typeof setTimeout> };

export class CortexSource implements Source {
  readonly kind = "emotiv";
  private ws: WsLike | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private token = "";
  private session = "";
  private stopped = false;
  private colIndex: number[] = [];
  private batch: number[][] = [];
  private sink: SourceSink | null = null;
  private flushTimer: ReturnType<typeof setInterval> | null = null;
  timeoutMs = 15000;
  pollMs = 1500;

  constructor(private cfg: CortexConfig, private makeWs: (url: string) => WsLike = url => new WebSocket(url) as unknown as WsLike) {}

  private rpc<T = unknown>(method: string, params?: Record<string, unknown>): Promise<T> {
    const ws = this.ws;
    if (!ws || ws.readyState !== 1) return Promise.reject(new CortexError("Not connected to Cortex"));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new CortexError(`Cortex did not answer ${method}`)); }, this.timeoutMs);
      this.pending.set(id, { resolve: v => resolve(v as T), reject, timer });
      ws.send(JSON.stringify(params ? { id, jsonrpc: "2.0", method, params } : { id, jsonrpc: "2.0", method }));
    });
  }

  private onMessage(raw: unknown) {
    let msg: Record<string, unknown>;
    try { msg = JSON.parse(String(raw)); } catch { return; }
    if (typeof msg.id === "number" && this.pending.has(msg.id)) {
      const p = this.pending.get(msg.id)!;
      this.pending.delete(msg.id);
      clearTimeout(p.timer);
      const err = msg.error as { code?: number; message?: string } | undefined;
      if (err) p.reject(new CortexError(explainCortex(err.code, err.message ?? "Cortex error"), err.code));
      else p.resolve(msg.result);
      return;
    }
    if (Array.isArray(msg.eeg)) {
      const v = msg.eeg as unknown[];
      const row = this.colIndex.map(i => Number(v[i]));
      if (row.every(Number.isFinite)) this.batch.push(row);
      return;
    }
    const w = msg.warning as { code?: number; message?: unknown } | undefined;
    if (w && this.sink) {
      const text = typeof w.message === "string" ? w.message : (w.message && typeof w.message === "object" && "behavior" in w.message ? String((w.message as { behavior: unknown }).behavior) : "");
      if (w.code === 1 || w.code === 0) { this.sink.onStatus("Emotiv session ended", "warn"); this.sink.onEnd(); }
      else if (text) this.sink.onStatus(`Emotiv: ${text}`, "warn");
    }
  }

  private open(): Promise<void> {
    return new Promise((resolve, reject) => {
      let ws: WsLike;
      try { ws = this.makeWs(CORTEX_URL); } catch (e) { reject(new CortexError((e as Error).message)); return; }
      this.ws = ws;
      let opened = false;
      ws.onopen = () => { opened = true; resolve(); };
      ws.onerror = () => { if (!opened) reject(new CortexError("Can't reach Cortex at wss://localhost:6868. Install and open the EMOTIV Launcher, sign in, and try again.")); };
      ws.onclose = () => {
        for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new CortexError("Cortex closed the connection")); }
        this.pending.clear();
        if (opened && !this.stopped && this.sink) { this.sink.onStatus("Lost the connection to Cortex", "error"); this.sink.onEnd(); }
      };
      ws.onmessage = ev => this.onMessage(ev.data);
    });
  }

  private wait(ms: number) { return new Promise(r => setTimeout(r, ms)); }

  async start(sink: SourceSink) {
    this.sink = sink;
    this.stopped = false;
    if (!this.cfg.clientId || !this.cfg.clientSecret) {
      sink.onStatus("Enter the client ID and secret of your Cortex app (create one free at emotiv.com → My account → Cortex apps).", "error");
      return;
    }
    try {
      sink.onStatus("Connecting to the EMOTIV Launcher…", "warn");
      await this.open();
      const creds = { clientId: this.cfg.clientId, clientSecret: this.cfg.clientSecret };
      for (let t = 0; ; t++) {
        const r = await this.rpc<{ accessGranted: boolean }>("requestAccess", creds);
        if (r.accessGranted) break;
        if (this.stopped) return;
        if (t * this.pollMs > 120000) throw new CortexError("Emotif was not approved in the EMOTIV Launcher within 2 minutes.");
        sink.onStatus("Approve Emotif in the EMOTIV Launcher (a request is waiting there)…", "warn");
        await this.wait(this.pollMs);
      }
      const auth = await this.rpc<{ cortexToken: string }>("authorize",
        { ...creds, debit: 1, ...(this.cfg.license ? { license: this.cfg.license } : {}) });
      this.token = auth.cortexToken;
      type Headset = { id: string; status: string; sensors?: string[]; settings?: { eegRate?: number } };
      const pick = (hs: Headset[]) => hs.find(h => this.cfg.headsetId ? h.id === this.cfg.headsetId : /^INSIGHT/i.test(h.id)) ?? (this.cfg.headsetId ? undefined : hs[0]);
      let hs = pick(await this.rpc<Headset[]>("queryHeadsets"));
      if (!hs) {
        sink.onStatus("Searching for the headset (turn it on)…", "warn");
        await this.rpc("controlDevice", { command: "refresh" });
        for (let t = 0; !hs && t * this.pollMs < 30000 && !this.stopped; t++) {
          await this.wait(this.pollMs);
          hs = pick(await this.rpc<Headset[]>("queryHeadsets"));
        }
        if (!hs) throw new CortexError("No Emotiv headset found. Turn it on and pair it in the EMOTIV Launcher.");
      }
      if (hs.status !== "connected") {
        sink.onStatus(`Connecting to ${hs.id}…`, "warn");
        await this.rpc("controlDevice", { command: "connect", headset: hs.id });
        for (let t = 0; hs && hs.status !== "connected"; t++) {
          if (this.stopped) return;
          if (t * this.pollMs > 30000) throw new CortexError(`${hs.id} did not connect within 30 s.`);
          await this.wait(this.pollMs);
          const id: string = hs.id;
          hs = (await this.rpc<Headset[]>("queryHeadsets", { id })).find(h => h.id === id) ?? hs;
        }
      }
      if (this.stopped || !hs) return;
      const fs = Number(hs.settings?.eegRate) || 128;
      const ses = await this.rpc<{ id: string }>("createSession", { cortexToken: this.token, headset: hs.id, status: "open" });
      this.session = ses.id;
      await this.rpc("updateSession", { cortexToken: this.token, session: this.session, status: "active" });
      const sub = await this.rpc<{ success: { streamName: string; cols: string[] }[]; failure: { streamName: string; code: number; message: string }[] }>(
        "subscribe", { cortexToken: this.token, session: this.session, streams: ["eeg"] });
      const ok = sub.success?.find(s => s.streamName === "eeg");
      if (!ok) {
        const f = sub.failure?.find(s => s.streamName === "eeg");
        throw new CortexError(explainCortex(f?.code, f?.message ?? "subscription refused"), f?.code);
      }
      const { names, index } = eegColumns(ok.cols, hs.sensors);
      if (names.length < 2) throw new CortexError(`The eeg stream has no electrode columns (${ok.cols.join(", ")}).`);
      this.colIndex = index;
      if (this.stopped) return;
      sink.onMeta(fs, names, `Emotiv · ${hs.id}`);
      sink.onStatus(`Streaming ${names.length} electrodes from ${hs.id} at ${fs} Hz`, "ok");
      this.flushTimer = setInterval(() => {
        if (this.batch.length) { const b = this.batch; this.batch = []; this.sink?.onRows(b); }
      }, 50);
    } catch (e) {
      if (this.stopped) return;           // stopped by the user: not an error
      sink.onStatus((e as Error).message, "error");
      this.stop();
    }
  }

  stop() {
    this.stopped = true;
    if (this.flushTimer) { clearInterval(this.flushTimer); this.flushTimer = null; }
    this.batch = [];
    const ws = this.ws;
    if (ws && ws.readyState === 1 && this.session) {
      try { ws.send(JSON.stringify({ id: this.nextId++, jsonrpc: "2.0", method: "updateSession", params: { cortexToken: this.token, session: this.session, status: "close" } })); } catch { /* closing */ }
    }
    this.session = "";
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(new CortexError("stopped")); }
    this.pending.clear();
    if (ws) { ws.onclose = null; ws.onmessage = null; try { ws.close(); } catch { /* already closed */ } }
    this.ws = null;
  }
}
