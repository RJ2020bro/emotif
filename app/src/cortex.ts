// Emotiv headsets (Insight, EPOC X, MN8, ...) through the EMOTIV Cortex service on this computer
// (wss://localhost:6868, JSON-RPC 2.0). Raw EEG needs an Emotiv licence that includes the "eeg" stream (and a debited,
// activated session). Without one, the source falls back to the "pow" band-power stream, which a free Cortex account
// gets: the world still runs, but there is no raw signal to clean, record or read commands from.
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

/** Emotiv pow bands, in column order per sensor: theta 4-8, alpha 8-12, betaL 12-16, betaH 16-25, gamma 25-45 Hz. */
const POW_BANDS = ["theta", "alpha", "betaL", "betaH", "gamma"];
/** Band columns of a "pow" subscription ("AF3/theta", ...): sensor names and, per sensor, the column of each band. */
export function powColumns(cols: string[], sensors?: string[]): { names: string[]; index: number[][] } {
  const want = sensors && sensors.length ? sensors.map(x => x.toUpperCase()) : null;
  const by = new Map<string, { name: string; idx: number[] }>();
  cols.forEach((c, i) => {
    const [sensor, band] = c.split("/");
    const b = POW_BANDS.indexOf(band ?? "");
    if (!sensor || b < 0) return;
    const k = sensor.toUpperCase();
    if (want && !want.includes(k)) return;
    if (!by.has(k)) by.set(k, { name: sensor, idx: [-1, -1, -1, -1, -1] });
    by.get(k)!.idx[b] = i;
  });
  const full = [...by.values()].filter(v => v.idx.every(i => i >= 0));
  return { names: full.map(v => v.name), index: full.map(v => v.idx) };
}
/** One pow sample to the engine's bands: [delta (none: NaN), theta, alpha, beta = betaL + betaH, gamma]. */
export function powFrame(v: unknown[], index: number[][]): number[] | null {
  const out: number[] = [];
  for (const [th, al, bl, bh, ga] of index) {
    const x = [th, al, bl, bh, ga].map(i => Number(v[i]));
    if (!x.every(Number.isFinite)) return null;
    out.push(NaN, x[0], x[1], x[2] + x[3], x[4]);
  }
  return out;
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
  private powIndex: number[][] | null = null;
  private bandBatch: number[][][] = [];
  private bucket = { key: NaN, sum: [] as number[], n: 0 };
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
    if (Array.isArray(msg.pow) && this.powIndex) {
      const f = powFrame(msg.pow as unknown[], this.powIndex);
      if (!f) return;
      // average the samples of each 0.25 s step into one frame (the engine decodes every 0.25 s)
      const t = typeof msg.time === "number" ? msg.time : Date.now() / 1000;
      const key = Math.floor(t / 0.25), b = this.bucket;
      if (key !== b.key) {
        if (b.n) this.bandBatch.push(this.splitFrame(b.sum.map(x => x / b.n)));
        b.key = key; b.sum = f.slice(); b.n = 1;
      } else { for (let i = 0; i < f.length; i++) b.sum[i] += f[i]; b.n++; }
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

  private splitFrame(flat: number[]): number[][] {
    const out: number[][] = [];
    for (let c = 0; c < flat.length; c += 5) out.push(flat.slice(c, c + 5));
    return out;
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
      // debit one session from the licence so the session can be activated for raw EEG; an account without a
      // licence can't debit, so authorize again without (band power only)
      let canDebit = true;
      let auth: { cortexToken: string };
      try {
        auth = await this.rpc<{ cortexToken: string }>("authorize", { ...creds, debit: 1, ...(this.cfg.license ? { license: this.cfg.license } : {}) });
      } catch (e) {
        if (this.cfg.license || (e as CortexError).code === -32021) throw e;
        canDebit = false;
        auth = await this.rpc<{ cortexToken: string }>("authorize", creds);
      }
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
      let active = false;
      if (canDebit) {
        try { await this.rpc("updateSession", { cortexToken: this.token, session: this.session, status: "active" }); active = true; }
        catch { /* no licence to activate with: the open session still gives band power */ }
      }
      type Sub = { success: { streamName: string; cols: string[] }[]; failure: { streamName: string; code: number; message: string }[] };
      let refusal: { code?: number; message: string } = { code: -32016, message: "raw EEG needs an activated session" };
      let ok: { streamName: string; cols: string[] } | undefined;
      if (active) {
        const sub = await this.rpc<Sub>("subscribe", { cortexToken: this.token, session: this.session, streams: ["eeg"] });
        ok = sub.success?.find(s => s.streamName === "eeg");
        const f = sub.failure?.find(s => s.streamName === "eeg");
        if (f) refusal = { code: f.code, message: f.message };
      }
      if (!ok) {
        if (!sink.onBands || !sink.onBandMeta) throw new CortexError(explainCortex(refusal.code, refusal.message), refusal.code);
        if (this.stopped) return;
        await this.startBands(hs.id, hs.sensors);
        return;
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

  /** Free-tier path: subscribe to band power and pass 0.25 s frames on. */
  private async startBands(headset: string, sensors?: string[]) {
    const sink = this.sink!;
    type Sub = { success: { streamName: string; cols: string[] }[]; failure: { streamName: string; code: number; message: string }[] };
    const sub = await this.rpc<Sub>("subscribe", { cortexToken: this.token, session: this.session, streams: ["pow"] });
    const ok = sub.success?.find(s => s.streamName === "pow");
    if (!ok) {
      const f = sub.failure?.find(s => s.streamName === "pow");
      throw new CortexError(`Cortex refused both raw EEG and band power: ${explainCortex(f?.code, f?.message ?? "subscription refused")}`, f?.code);
    }
    const { names, index } = powColumns(ok.cols, sensors);
    if (names.length < 2) throw new CortexError(`The band-power stream has no usable sensors (${ok.cols.join(", ")}).`);
    this.powIndex = index;
    this.bucket = { key: NaN, sum: [], n: 0 };
    if (this.stopped) return;
    sink.onBandMeta!(names, `Emotiv · ${headset} · band power`);
    sink.onStatus(`No raw-EEG licence: running on band power from ${names.length} sensors. The world works; artifact repair, blink/clench commands and recording need raw EEG.`, "warn");
    this.flushTimer = setInterval(() => {
      if (this.bandBatch.length) { const b = this.bandBatch; this.bandBatch = []; this.sink?.onBands?.(b); }
    }, 100);
  }

  stop() {
    this.stopped = true;
    if (this.flushTimer) { clearInterval(this.flushTimer); this.flushTimer = null; }
    this.batch = []; this.bandBatch = []; this.powIndex = null;
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
