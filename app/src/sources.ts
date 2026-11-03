// Signal sources. Each calls onMeta(fs, channels, label) once, then onRows(rows) with raw µV samples.
import { DEVICES, position } from "./montage";
import { parseEdf } from "./edf";

export type SourceSink = {
  onMeta: (fs: number, channels: string[], label: string) => void;
  onRows: (rows: number[][]) => void;
  onStatus: (text: string, kind?: "ok" | "warn" | "error") => void;
  onEnd: () => void;
};
export interface Source { readonly kind: string; start(sink: SourceSink): Promise<void> | void; stop(): void }

// ------------------------------------------------------------------ simulator / virtual participant
let spare: number | null = null;
function gauss(): number {
  if (spare !== null) { const g = spare; spare = null; return g; }
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const r = Math.sqrt(-2 * Math.log(u));
  spare = r * Math.sin(2 * Math.PI * v);
  return r * Math.cos(2 * Math.PI * v);
}
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

export type SimMode = "tour" | "manual" | "participant";
export const SIM_MONTAGES: Record<string, { label: string; channels: string[] }> = {
  lab8: { label: "8-channel cap (Fp1 Fp2 F7 F3 F4 F8 O1 O2)", channels: ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "O1", "O2"] },
  muse: { label: DEVICES.muse.label, channels: DEVICES.muse.channels },
  emotiv_insight: { label: DEVICES.emotiv_insight.label, channels: DEVICES.emotiv_insight.channels },
  emotiv_epoc: { label: DEVICES.emotiv_epoc.label, channels: DEVICES.emotiv_epoc.channels },
  openbci_cyton: { label: DEVICES.openbci_cyton.label, channels: DEVICES.openbci_cyton.channels },
};
const TOUR = [{ v: 0.7, a: 0.6 }, { v: 0.65, a: -0.6 }, { v: -0.65, a: -0.55 }, { v: -0.7, a: 0.65 }];

type ChState = { ar1: number; ar2: number; ph: number[]; env: number[]; fj: number; frontal: number; side: number; occ: number; temporal: number };

/** Synthetic EEG whose spectra follow a valence/arousal state:
 * valence -> frontal alpha asymmetry, arousal -> alpha down / beta up, plus 1/f background, mains hum,
 * spontaneous blinks, and on-demand double blinks and jaw clenches for testing arm control.
 * In "participant" mode the simulated person's affect drifts toward what the environment shows,
 * so closed-loop protocols can be rehearsed without a human. */
export class Simulator implements Source {
  readonly kind = "sim";
  fs = 256;
  mode: SimMode = "tour";
  target = { v: 0, a: 0 };
  cur = { v: 0, a: 0 };
  envState = { v: 0, a: 0 };      // set by the environment director (participant mode)
  responsiveness = 0.08;          // per second
  calibSec = 30;
  montage = "lab8";
  private t = 0; private timer: number | null = null; private t0 = 0; private emitted = 0;
  private st: ChState[] = []; private channels: string[] = [];
  private nextBlink = 5; private blinkTimes: number[] = []; private clenchUntil = -1;

  start(sink: SourceSink) {
    this.stop();
    this.channels = SIM_MONTAGES[this.montage]?.channels ?? SIM_MONTAGES.lab8.channels;
    this.t = 0; this.cur = { v: 0, a: 0 }; this.target = { v: 0, a: 0 }; this.blinkTimes = []; this.clenchUntil = -1;
    this.nextBlink = 4 + Math.random() * 6;
    this.st = this.channels.map(name => {
      const p = position(name) ?? [0, 0, 1];
      return {
        ar1: 0, ar2: 0, ph: [0, 1, 2, 3, 4].map(() => Math.random() * 6.283), env: [1, 1, 1, 1, 1], fj: 0,
        frontal: clamp((p[1] - 0.2) / 0.5, 0, 1), side: Math.abs(p[0]) > 0.1 ? Math.sign(p[0]) : 0,
        occ: Math.max(0, (-p[1] - 0.6) / 0.4), temporal: Math.abs(p[0]) > 0.7 && p[2] < 0.45 ? 1 : 0,
      };
    });
    sink.onMeta(this.fs, this.channels, `Simulator · ${SIM_MONTAGES[this.montage]?.label ?? ""}`);
    sink.onStatus("Simulated signal", "ok");
    this.t0 = performance.now(); this.emitted = 0;
    this.timer = window.setInterval(() => {
      const due = Math.floor((performance.now() - this.t0) / 1000 * this.fs);
      let n = due - this.emitted;
      if (n <= 0) return;
      if (n > this.fs * 2) { n = this.fs * 2; }
      const rows: number[][] = new Array(n);
      for (let i = 0; i < n; i++) rows[i] = this.sample();
      this.emitted = due;
      sink.onRows(rows);
    }, 40);
  }
  stop() { if (this.timer !== null) window.clearInterval(this.timer); this.timer = null; }

  doubleBlink() { this.blinkTimes.push(this.t + 0.05, this.t + 0.4); }
  clench(seconds = 1.2) { this.clenchUntil = this.t + seconds; }

  private sample(): number[] {
    const dt = 1 / this.fs;
    this.t += dt;
    if (this.mode === "tour") {
      this.target = this.t < this.calibSec ? { v: 0, a: 0 } : TOUR[Math.floor((this.t - this.calibSec) / 25) % TOUR.length];
    } else if (this.mode === "participant" && this.t >= this.calibSec) {
      const g = this.responsiveness * dt;
      this.target.v = clamp(this.target.v + g * (this.envState.v - this.target.v) + 0.02 * Math.sqrt(dt) * gauss(), -1, 1);
      this.target.a = clamp(this.target.a + g * (this.envState.a - this.target.a) + 0.02 * Math.sqrt(dt) * gauss(), -1, 1);
    }
    const k = dt / 3;
    this.cur.v += (this.target.v - this.cur.v) * k;
    this.cur.a += (this.target.a - this.cur.a) * k;
    const v = this.cur.v, a = this.cur.a;
    if (this.t >= this.nextBlink) { this.blinkTimes.push(this.t); this.nextBlink = this.t + 6 + Math.random() * 8; }
    let blink = 0;
    for (const bt of this.blinkTimes) {
      const d = this.t - bt - 0.15;
      if (Math.abs(d) < 0.4) blink += 190 * Math.exp(-(d * d) / (2 * 0.06 * 0.06));
    }
    if (this.blinkTimes.length && this.t - this.blinkTimes[0] > 1) this.blinkTimes.shift();
    const clenching = this.t < this.clenchUntil;
    const freqs = [6, 10, 18, 24, 38];
    return this.st.map(s => {
      s.ar1 = 0.995 * s.ar1 + 1.1 * gauss();
      s.ar2 = 0.9 * s.ar2 + 2.5 * gauss();
      s.fj = 0.999 * s.fj + 0.002 * gauss();
      let alpha = 9 * (1 - 0.4 * a) * (1 + 0.6 * s.occ);
      alpha *= 1 + 0.35 * v * s.side * s.frontal;         // right (side > 0) alpha up when positive
      const amps = [4 * (1 - 0.15 * a), alpha, 2.5 * (1 + 0.6 * a), 2.0 * (1 + 0.6 * a), 0.8 * (1 + 0.4 * a)];
      let x = s.ar1 + s.ar2 + gauss();
      for (let b = 0; b < 5; b++) {
        s.env[b] = clamp(s.env[b] + 0.004 * gauss() + 0.0005 * (1 - s.env[b]), 0.3, 1.7);
        s.ph[b] += 2 * Math.PI * freqs[b] * (1 + s.fj) * dt;
        if (s.ph[b] > 6.283185307179586) s.ph[b] -= 6.283185307179586;
        x += amps[b] * s.env[b] * Math.sin(s.ph[b]);
      }
      x += 2 * Math.sin(2 * Math.PI * 60 * this.t);
      x += blink * Math.max(0, (s.frontal - 0.3) / 0.7) ** 2;   // eye field falls off quickly behind the brow
      if (clenching && s.temporal) x += 22 * gauss();
      return x;
    });
  }
}

// ------------------------------------------------------------------ CSV playback
export class CsvPlayback implements Source {
  readonly kind = "csv";
  data: { rows: number[][]; fs: number; channels: string[]; name: string } | null = null;
  private timer: number | null = null; private pos = 0;

  load(text: string, fallbackFs: number, name: string) {
    const lines = text.split(/\r?\n/).filter(l => l.trim() !== "");
    if (lines.length < 3) throw new Error("The file has fewer than two data rows.");
    const header = lines[0].split(",").map(h => h.trim());
    const low = header.map(h => h.toLowerCase());
    const tcol = low.findIndex(h => ["time", "time_s", "timestamp", "timestamps", "t"].includes(h));
    const skip = (h: string) => ["valence_label", "arousal_label", "right aux", "aux"].includes(h) || /^(pred_|event|marker)/.test(h);
    const chIdx = header.map((_, i) => i).filter(i => i !== tcol && !skip(low[i]));
    if (!chIdx.length) throw new Error("No channel columns found.");
    const rows: number[][] = [], times: number[] = [];
    for (let li = 1; li < lines.length; li++) {
      const p = lines[li].split(",");
      if (p.length !== header.length) continue;
      const r = chIdx.map(i => Number(p[i]));
      if (r.some(v => !Number.isFinite(v))) throw new Error(`Row ${li + 1} has a non-numeric value in a channel column.`);
      rows.push(r);
      if (tcol >= 0) times.push(Number(p[tcol]));
    }
    let fs = fallbackFs;
    if (tcol >= 0) {
      const d: number[] = [];
      for (let i = 1; i < times.length; i++) d.push(times[i] - times[i - 1]);
      d.sort((x, y) => x - y);
      const med = d[d.length >> 1];
      if (!(med > 0)) throw new Error("The time column isn't increasing.");
      fs = 1 / med;
      if (Math.abs(fs - Math.round(fs)) < 0.02 * fs) fs = Math.round(fs);
    }
    if (!(fs >= 100)) throw new Error(`Sampling rate ${fs.toFixed(1)} Hz is too low; Emotif needs at least 100 Hz.`);
    if (rows.length < 2 * fs) throw new Error("The recording is shorter than one 2-second window.");
    this.data = { rows, fs, channels: chIdx.map(i => header[i]), name };
  }

  /** EDF, EDF+ or BDF (any headset's export). Voltage signals at the most common rate are kept. */
  loadEdf(buf: ArrayBuffer, name: string): string[] {
    const e = parseEdf(buf);
    if (!(e.fs >= 100)) throw new Error(`Sampling rate ${e.fs.toFixed(1)} Hz is too low; Emotif needs at least 100 Hz.`);
    if (e.rows.length < 2 * e.fs) throw new Error("The recording is shorter than one 2-second window.");
    this.data = { rows: e.rows, fs: e.fs, channels: e.channels, name };
    return e.skipped;
  }

  start(sink: SourceSink) {
    this.stop();
    const d = this.data;
    if (!d) { sink.onStatus("Choose a recording file first", "warn"); return; }
    sink.onMeta(d.fs, d.channels, `File · ${d.name}`);
    sink.onStatus(`Playing ${d.name}`, "ok");
    this.pos = 0;
    const t0 = performance.now();
    let emitted = 0;
    this.timer = window.setInterval(() => {
      const due = Math.floor((performance.now() - t0) / 1000 * d.fs);
      let n = Math.min(due - emitted, Math.round(d.fs * 2));
      if (n <= 0) return;
      emitted = due;
      const out: number[][] = [];
      while (n-- > 0) { out.push(d.rows[this.pos]); this.pos = (this.pos + 1) % d.rows.length; }
      sink.onRows(out);
    }, 40);
  }
  stop() { if (this.timer !== null) window.clearInterval(this.timer); this.timer = null; }
}

// ------------------------------------------------------------------ OpenBCI Cyton over Web Serial (Chrome / Edge)
/** Cyton (8 channels, 250 Hz) binary stream: 33-byte packets, 0xA0 | sample no. | 8 x 24-bit big-endian
 * signed | 6 aux bytes | 0xC0-0xCF. Default gain 24, so 1 count = 4.5 V / 24 / (2^23 - 1). */
export const CYTON_UV_PER_COUNT = 4.5 / 24 / (2 ** 23 - 1) * 1e6;
export class CytonParser {
  private buf = new Uint8Array(0);
  dropped = 0;           // bytes skipped while resynchronising
  lost = 0;              // missing sample numbers (radio drop-outs)
  private lastSample = -1;
  push(chunk: Uint8Array): number[][] {
    const b = new Uint8Array(this.buf.length + chunk.length);
    b.set(this.buf); b.set(chunk, this.buf.length);
    const rows: number[][] = [];
    let i = 0;
    while (i + 33 <= b.length) {
      if (b[i] !== 0xa0 || (b[i + 32] & 0xf0) !== 0xc0) { i++; this.dropped++; continue; }
      const sn = b[i + 1];
      if (this.lastSample >= 0) this.lost += (sn - this.lastSample - 1 + 256) % 256;
      this.lastSample = sn;
      const r = new Array(8);
      for (let c = 0; c < 8; c++) {
        const p = i + 2 + 3 * c;
        let v = (b[p] << 16) | (b[p + 1] << 8) | b[p + 2];
        if (v & 0x800000) v -= 0x1000000;
        r[c] = v * CYTON_UV_PER_COUNT;
      }
      rows.push(r);
      i += 33;
    }
    this.buf = b.slice(i);
    return rows;
  }
}

type SerialPortLike = {
  open(o: { baudRate: number }): Promise<void>;
  close(): Promise<void>;
  readable: ReadableStream<Uint8Array> | null;
  writable: WritableStream<Uint8Array> | null;
};

export class OpenBciSerial implements Source {
  readonly kind = "cyton";
  channels: string[] = DEVICES.openbci_cyton.channels.slice();
  private port: SerialPortLike | null = null;
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private running = false;

  static supported(): boolean { return typeof navigator !== "undefined" && "serial" in navigator; }

  private async send(cmd: string) {
    const w = this.port?.writable?.getWriter();
    if (!w) return;
    try { await w.write(new TextEncoder().encode(cmd)); } finally { w.releaseLock(); }
  }

  async start(sink: SourceSink) {
    if (!OpenBciSerial.supported()) {
      sink.onStatus("This browser has no Web Serial. Use Chrome or Edge on a desktop, or connect through the Emotif server.", "error");
      return;
    }
    const serial = (navigator as unknown as { serial: { requestPort(): Promise<SerialPortLike> } }).serial;
    sink.onStatus("Choose the OpenBCI dongle in the port dialog…", "warn");
    try {
      this.port = await serial.requestPort();
      await this.port.open({ baudRate: 115200 });
      await this.send("v");                                // soft reset: default gain 24, all channels on
      await new Promise(r => setTimeout(r, 1500));
      await this.send("b");                                // start streaming
    } catch (e) {
      sink.onStatus(`Couldn't open the OpenBCI port: ${(e as Error).message}`, "error");
      await this.port?.close().catch(() => undefined);
      this.port = null;
      return;
    }
    sink.onMeta(250, this.channels, "OpenBCI Cyton · Web Serial");
    sink.onStatus("Streaming from OpenBCI Cyton", "ok");
    const parser = new CytonParser();
    this.running = true;
    this.reader = this.port.readable!.getReader();
    (async () => {
      try {
        while (this.running) {
          const { value, done } = await this.reader!.read();
          if (done) break;
          const rows = value ? parser.push(value) : [];
          if (rows.length) sink.onRows(rows);
        }
      } catch (e) {
        if (this.running) sink.onStatus(`OpenBCI stream stopped: ${(e as Error).message}`, "error");
      }
      if (this.running) sink.onEnd();
    })();
  }

  stop() {
    this.running = false;
    const port = this.port, reader = this.reader;
    this.port = null; this.reader = null;
    if (!port) return;
    (async () => {
      try { await reader?.cancel(); reader?.releaseLock(); } catch { /* already released */ }
      const w = port.writable?.getWriter();
      if (w) { try { await w.write(new TextEncoder().encode("s")); } catch { /* closing */ } w.releaseLock(); }
      await port.close().catch(() => undefined);
    })();
  }
}

// ------------------------------------------------------------------ Muse over Web Bluetooth (Chrome / Edge)
export class MuseBluetooth implements Source {
  readonly kind = "muse";
  private client: import("muse-js").MuseClient | null = null;
  private sub: { unsubscribe(): void } | null = null;

  static supported(): boolean { return typeof navigator !== "undefined" && "bluetooth" in navigator; }

  async start(sink: SourceSink) {
    if (!MuseBluetooth.supported()) {
      sink.onStatus("This browser has no Web Bluetooth. Use Chrome or Edge on desktop or Android, or connect through the Emotif server.", "error");
      return;
    }
    const { MuseClient, zipSamples } = await import("muse-js");
    const client = new MuseClient();
    this.client = client;
    sink.onStatus("Choose your Muse in the Bluetooth dialog…", "warn");
    try {
      await client.connect();
      await client.start();
    } catch (e) {
      this.client = null;
      sink.onStatus(`Couldn't connect to the Muse: ${(e as Error).message}`, "error");
      return;
    }
    sink.onMeta(256, ["TP9", "AF7", "AF8", "TP10"], `Muse · ${client.deviceName ?? "Bluetooth"}`);
    sink.onStatus(`Connected to ${client.deviceName ?? "Muse"}`, "ok");
    let batch: number[][] = [];
    let last = performance.now();
    this.sub = zipSamples(client.eegReadings).subscribe(s => {
      batch.push(s.data.slice(0, 4));
      const now = performance.now();
      if (batch.length >= 12 || now - last > 50) { sink.onRows(batch); batch = []; last = now; }
    });
    client.connectionStatus.subscribe(up => { if (!up) { sink.onStatus("Muse disconnected", "error"); sink.onEnd(); } });
  }
  stop() {
    this.sub?.unsubscribe(); this.sub = null;
    try { this.client?.disconnect(); } catch { /* already closed */ }
    this.client = null;
  }
}
