// Node tests for app modules that have no Python twin: intent detection, environment director,
// in-app personalization, and the live engine end to end. Run after `node app/build.mjs --test-bundles`.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const D = join(here, "..", "app", "dist-test");
const { StreamFilter } = require(join(D, "dsp.cjs"));
const { IntentDetector } = require(join(D, "intent.cjs"));
const { Director, mulberry32 } = require(join(D, "director.cjs"));
const { fitLogistic, personalizationCv, forward } = require(join(D, "model.cjs"));
const { N_FEATURES } = require(join(D, "montage.cjs"));
const { Engine, markers } = require(join(D, "engine.cjs"));

let seed = 7;
const rnd = mulberry32(seed);
function gauss() { const u = Math.max(1e-12, rnd()), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

// ---------------------------------------------------------------- intent detector
{
  const fs = 256, ch = ["TP9", "AF7", "AF8", "TP10"];
  const det = new IntentDetector(fs, ch);
  assert.deepEqual(det.frontalIdx, [1, 2]);
  assert.deepEqual(det.temporalIdx, [0, 3]);
  const f = new StreamFilter(fs, 4);
  const events = [];
  det.listeners.add(e => events.push(e));
  const blinkAt = [5.0, 5.35, 12.0];            // a double blink, then a single blink
  const T = 25 * fs;
  for (let n = 0; n < T; n++) {
    const t = n / fs;
    let blink = 0;
    for (const b of blinkAt) { const d = t - b; if (Math.abs(d) < 0.4) blink += 160 * Math.exp(-(d * d) / (2 * 0.05 * 0.05)); }
    const emg = t >= 17 && t < 18.5 ? 35 : 0;
    const row = [0, 1, 2, 3].map(c => 8 * gauss() + 10 * Math.sin(2 * Math.PI * 10 * t + c) + (c === 1 || c === 2 ? blink : 0) + ((c === 0 || c === 3) && emg ? emg * gauss() : 0));
    det.process(f.process(row));
  }
  const kinds = events.map(e => `${e.kind}@${e.t.toFixed(1)}`);
  assert.equal(events.filter(e => e.kind === "double_blink").length, 1, kinds.join(" "));
  assert.ok(events.find(e => e.kind === "double_blink").t > 5.3 && events.find(e => e.kind === "double_blink").t < 5.8, kinds.join(" "));
  assert.equal(events.filter(e => e.kind === "blink").length, 2, kinds.join(" "));   // first half of the pair + the single one
  assert.equal(events.filter(e => e.kind === "clench_start").length, 1, kinds.join(" "));
  assert.equal(events.filter(e => e.kind === "clench_end").length, 1, kinds.join(" "));
  const cs = events.find(e => e.kind === "clench_start").t, ce = events.find(e => e.kind === "clench_end").t;
  assert.ok(cs > 17 && cs < 17.6 && ce > 18.5 && ce < 19.2, kinds.join(" "));
  console.log(`intent OK: ${kinds.join(", ")}`);
}

// ---------------------------------------------------------------- director
{
  const a = new Director(), b = new Director();
  a.mode = b.mode = "sham";
  a.setSham(42); b.setSham(42);
  for (let i = 0; i < 600; i++) { a.update(0.05, { v: 1, a: 1 }); b.update(0.05, { v: -1, a: -1 }); }
  assert.deepEqual(a.env, b.env, "sham must ignore the brain and be reproducible from its seed");
  const g = new Director();
  g.mode = "guide"; g.target = { v: 0.6, a: -0.6 };
  for (let i = 0; i < 400; i++) g.update(0.05, { v: -0.6, a: 0.6 });
  assert.ok(Math.abs(g.env.v - 0) < 1e-9 && Math.abs(g.env.a - 0) < 1e-9, JSON.stringify(g.env));  // halfway at gain 0.5
  const m = new Director();
  m.update(0.1, { v: 1, a: 1 });
  assert.ok(Math.abs(m.env.v - 0.025) < 1e-12, "rate limit 0.25/s");
  console.log("director OK: sham reproducible and brain-independent, guide = halfway to target, rate-limited");
}

// ---------------------------------------------------------------- personalization
{
  const data = [];
  for (let blk = 0; blk < 6; blk++) {
    const y = blk % 2 ? 1 : -1;
    for (let k = 0; k < 40; k++) {
      const x = new Float64Array(N_FEATURES);
      for (let i = 0; i < N_FEATURES; i++) x[i] = 0.5 * gauss();
      x[12] += y * 1.2; x[40] -= y * 0.8;
      data.push({ x, y, block: blk });
    }
  }
  const one = personalizationCv(data.filter(d => d.block < 3), [-1, 1], null);
  assert.ok(Number.isNaN(one.personal), "one block of a class -> no estimate");
  const cv = personalizationCv(data, [-1, 1], null);
  assert.ok(cv.personal > 0.85, JSON.stringify(cv));
  const prior = { W: Array.from({ length: N_FEATURES }, () => [0.1, -0.1]), b: [0.2, -0.2], act: "softmax" };
  const stay = fitLogistic(data, [-1, 1], prior, { lambda: 1e6, iters: 50, lr: 1e-7 });
  assert.ok(Math.abs(stay.W[3][0] - 0.1) < 1e-4, "strong prior keeps the pretrained weights");
  const p = forward([fitLogistic(data, [-1, 1], null)], data[41].x);
  assert.ok(Math.abs(p[0] + p[1] - 1) < 1e-12);
  console.log(`personalization OK: leave-one-block-out balanced accuracy ${cv.personal.toFixed(2)} on separable data`);
}

// ---------------------------------------------------------------- engine end to end
function runEngine(channels, model, seconds, va, artifactsEvery = 0) {
  const eng = new Engine();
  eng.settings.calibSec = 10;
  eng.setStream(256, channels);
  if (model) eng.setModel(model);
  const fs = 256;
  for (let n = 0; n < seconds * fs; n++) {
    const t = n / fs;
    const cur = t < 10 ? { v: 0, a: 0 } : va;
    const row = channels.map((c, i) => {
      const right = /[02468]$/.test(c) || /10$/.test(c), frontal = /^(Fp|AF|F)\d/.test(c);
      let alpha = 9 * (1 - 0.4 * cur.a);
      if (frontal) alpha *= right ? 1 + 0.3 * cur.v : 1 - 0.3 * cur.v;
      let x = 3 * gauss() + alpha * Math.sin(2 * Math.PI * 10 * t + i) + 2.5 * (1 + 0.6 * cur.a) * Math.sin(2 * Math.PI * 20 * t + 2 * i) + 4 * Math.sin(2 * Math.PI * 6 * t);
      if (artifactsEvery && Math.floor(t) % artifactsEvery === 0 && t % 1 < 0.2) x += 400;
      return x;
    });
    eng.push([row]);
  }
  return eng;
}
{
  const eng = runEngine(["Fp1", "Fp2", "F7", "F3", "F4", "F8", "O1", "O2"], null, 40, { v: 0.8, a: 0.7 });
  assert.ok(eng.windows > 100 && eng.last, "engine decodes");
  assert.ok(eng.last.v > 0.2 && eng.last.a > 0.2, JSON.stringify(eng.last));
  const neg = runEngine(["Fp1", "Fp2", "F7", "F3", "F4", "F8", "O1", "O2"], null, 40, { v: -0.8, a: -0.7 });
  assert.ok(neg.last.v < -0.2 && neg.last.a < -0.2, JSON.stringify(neg.last));
  const art = runEngine(["Fp1", "Fp2", "O1", "O2"], null, 30, { v: 0, a: 0 }, 3);
  assert.ok(art.rejected > 10, `artifacts should be rejected (${art.rejected})`);
  const muse = JSON.parse(readFileSync(join(here, "..", "models", "muse_mental_logreg.json"), "utf8"));
  const m1 = runEngine(["TP9", "AF7", "AF8", "TP10"], muse, 30, { v: 0, a: 0.8 });
  assert.equal(m1.last.source, "model");
  assert.ok(m1.last.conf > 0.3 && m1.last.conf <= 1, JSON.stringify(m1.last));
  const m2 = runEngine(["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"], muse, 30, { v: 0, a: 0.8 });
  assert.equal(m2.last.source, "model", "a Muse-trained model must still run on an OpenBCI layout");
  const x = m2.lastEvent.features;
  assert.equal(x.length, N_FEATURES);
  const mk = markers(x);
  assert.ok(Number.isFinite(mk.v) && Number.isFinite(mk.a));
  console.log(`engine OK: positive (${eng.last.v.toFixed(2)}, ${eng.last.a.toFixed(2)}), negative (${neg.last.v.toFixed(2)}, ${neg.last.a.toFixed(2)}), ` +
    `${art.rejected} artifact windows rejected, Muse model runs on Muse and on OpenBCI layouts`);
}
// ---------------------------------------------------------------- OpenBCI Cyton packet parser
{
  const { CytonParser, CYTON_UV_PER_COUNT } = require(join(D, "sources.cjs"));
  const packet = (sn, vals) => {
    const p = new Uint8Array(33);
    p[0] = 0xa0; p[1] = sn & 255;
    vals.forEach((v, c) => { const u = v < 0 ? v + 0x1000000 : v; p[2 + 3 * c] = (u >> 16) & 255; p[3 + 3 * c] = (u >> 8) & 255; p[4 + 3 * c] = u & 255; });
    p[32] = 0xc0;
    return p;
  };
  const truth = [], bytes = [Uint8Array.from([0x13, 0xa0, 0x00, 0xc0, 0x55])];     // leading garbage incl. a fake header
  for (let k = 0; k < 300; k++) {
    if (k === 150) continue;                                                       // one packet lost on the radio
    const vals = Array.from({ length: 8 }, (_, c) => Math.round(Math.sin(k / 7 + c) * 4e6) - (c === 7 ? 8388608 - 4e6 : 0));
    truth.push(vals.map(v => v * CYTON_UV_PER_COUNT));
    bytes.push(packet(k, vals));
  }
  const all = new Uint8Array(bytes.reduce((n, b) => n + b.length, 0));
  let o = 0; for (const b of bytes) { all.set(b, o); o += b.length; }
  const parser = new CytonParser(), rows = [];
  for (let i = 0; i < all.length;) { const n = 1 + Math.floor(rnd() * 70); rows.push(...parser.push(all.slice(i, i + n))); i += n; }
  assert.equal(rows.length, truth.length);
  rows.forEach((r, i) => r.forEach((v, c) => assert.ok(Math.abs(v - truth[i][c]) < 1e-9)));
  assert.equal(parser.lost, 1);
  assert.ok(Math.abs(8388607 * CYTON_UV_PER_COUNT - 187500) < 1e-6);   // full scale = 4.5 V / gain 24
  console.log(`cyton OK: ${rows.length} samples decoded across random chunk boundaries, garbage skipped, 1 lost packet counted`);
}

// ---------------------------------------------------------------- photosensitive safety
{
  const { slewToward, maxFlashesPerSecond, TAU } = require(join(D, "safety.cjs"));
  const fps = 60, dt = 1 / fps;
  // adaptive adversary on both axes at once: flip the targets as soon as the displayed luminance has moved
  // 0.1. Luminance model: L = 0.79 * ((v + 1) / 2 + (a + 1) / 2) / 2, i.e. the full range of either axis spans
  // 40 % of the screen's luminance range (steeper than the real scene) and the darker end stays below 0.8.
  const run = step => {
    let v = 0, a = 0, tv = 1, ta = 1, anchor = null;
    const luma = [];
    for (let i = 0; i < fps * 10; i++) {
      v = step(v, tv); a = step(a, ta);
      const L = 0.79 * ((v + 1) / 2 + (a + 1) / 2) / 2;
      if (anchor === null) anchor = L;
      if (Math.abs(L - anchor) >= 0.1) { tv = -tv; ta = -ta; anchor = L; }
      luma.push(L);
    }
    return maxFlashesPerSecond(luma, fps);
  };
  const unlimited = run((c, t) => c + (t - c) * (1 - Math.exp(-dt / TAU)));
  const limited = run((c, t) => slewToward(c, t, dt));
  assert.ok(limited <= 3, `slew-limited world flashes ${limited} times per second`);
  // square-wave target at 5 Hz for 10 s
  let c = 0; const sq = [];
  for (let i = 0; i < fps * 10; i++) { c = slewToward(c, Math.floor(i / (fps / 10)) % 2 ? 1 : -1, dt); sq.push((c + 1) / 2 * 0.75); }
  assert.equal(maxFlashesPerSecond(sq, fps), 0);
  // the counter itself: a 5 Hz full-contrast flicker is 5 flashes per second
  const flicker = Array.from({ length: fps * 2 }, (_, i) => (Math.floor(i / 6) % 2 ? 0.7 : 0.05));
  assert.equal(maxFlashesPerSecond(flicker, fps), 5);
  console.log(`photosafety OK: worst-case adversary ${limited} flashes/s with the limiter (${unlimited}/s with smoothing only); WCAG limit is 3`);
}

// ---------------------------------------------------------------- inner world: prompt -> spec -> terrain
{
  const W = require(join(D, "worldspec.cjs"));
  const T = require(join(D, "terrain.cjs"));
  // validation never throws and always yields renderable values
  for (const junk of [null, 42, "x", [], { style: "photoreal", seed: -3.7, terrain: { height: 1e9, water: "lots", islands: 1 }, palette: { sky: "red", low: "#12345G", high: "#ABCDEF" }, features: ["trees", { kind: "dragons" }, { kind: "trees" }, "rocks", "flowers", "cacti", "palms"], weather: "hail", time: "noon", mood: 9 }]) {
    const s = W.validateSpec(junk);
    assert.ok(W.STYLES.includes(s.style) && W.WEATHERS.includes(s.weather) && W.TIMES.includes(s.time));
    assert.ok(s.terrain.height >= 2 && s.terrain.height <= 48 && s.terrain.water >= 0 && s.terrain.water <= 1 && typeof s.terrain.islands === "boolean");
    assert.ok(Object.values(s.palette).every(c => /^#[0-9a-f]{6}$/.test(c)));
    assert.ok(s.features.length <= 4 && new Set(s.features.map(f => f.kind)).size === s.features.length && s.features.every(f => W.FEATURES.includes(f.kind)));
    assert.ok(s.mood >= 0 && s.mood <= 1 && Number.isInteger(s.seed) && s.seed >= 0);
  }
  const j = W.validateSpec({ style: "photoreal", terrain: { height: 1e9 }, palette: { high: "#ABCDEF" }, features: ["trees", { kind: "dragons" }, "trees", "rocks"] });
  assert.equal(j.terrain.height, 48); assert.equal(j.palette.high, "#abcdef"); assert.equal(j.style, W.DEFAULT_SPEC.style);
  assert.deepEqual(j.features.map(f => f.kind), ["trees", "rocks"]);
  // the on-device interpreter reads styles, biomes, time and weather
  const cases = [
    ["Minecraft forest with a lake", { style: "voxel", has: "trees" }], ["low poly desert canyon at sunset", { style: "lowpoly", time: "sunset", has: "cacti" }],
    ["dreamy cherry blossom islands", { style: "dream", weather: "petals", islands: true }], ["neon synthwave mountains", { style: "neon", has: "pines" }],
    ["snowy pine forest at night", { time: "night", weather: "snow" }], ["volcano", { weather: "embers" }],
  ];
  for (const [p, want] of cases) {
    const s = W.interpretPrompt(p);
    if (want.style) assert.equal(s.style, want.style, p);
    if (want.time) assert.equal(s.time, want.time, p);
    if (want.weather) assert.equal(s.weather, want.weather, p);
    if (want.islands) assert.equal(s.terrain.islands, true, p);
    if (want.has) assert.ok(s.features.some(f => f.kind === want.has), p);
    assert.deepEqual(W.interpretPrompt(p), s, "interpreter must be deterministic");
  }
  assert.equal(W.interpretPrompt("Minecraft forest", "neon").style, "neon", "chosen style overrides the prompt");
  assert.notEqual(W.interpretPrompt("a forest").seed, W.interpretPrompt("a quiet forest").seed);
  // terrain: deterministic, seamless across chunk borders, finite, and every style builds
  for (const style of W.STYLES) {
    const spec = { ...W.interpretPrompt("mountains and a lake with trees, flowers and lanterns"), style };
    const t1 = new T.Terrain(spec), t2 = new T.Terrain(JSON.parse(JSON.stringify(spec)));
    for (let i = 0; i < 50; i++) { const x = i * 13.7 - 300, z = i * -7.3 + 90; assert.equal(t1.height(x, z), t2.height(x, z)); }
    let land = 0;
    for (let i = 0; i < 400; i++) if (t1.height((i % 20) * 9, Math.floor(i / 20) * 9) > 0) land++;
    assert.ok(land > 40 && land < 400, `${style}: land share ${land}/400`);
    const a = T.buildChunk(t1, 0, 0), b = T.buildChunk(t2, 0, 0);
    assert.deepEqual(a.position, b.position);
    let feats = 0;
    for (const [cx, cz] of [[0, 0], [1, 0], [-2, 3], [40, -17]]) {
      const g = T.buildChunk(t1, cx, cz);
      const nv = g.position.length / 3;
      assert.equal(g.normal.length, nv * 3); assert.equal(g.color.length, nv * 3); assert.equal(g.extra.length, nv * 2);
      for (const arr of [g.position, g.normal, g.color, g.extra]) for (const v of arr) assert.ok(Number.isFinite(v));
      for (const i of g.index) assert.ok(i < nv);
      for (let i = 0; i < nv; i++) { const l = Math.hypot(g.normal[i * 3], g.normal[i * 3 + 1], g.normal[i * 3 + 2]); assert.ok(Math.abs(l - 1) < 1e-3, `${style}: normal length ${l}`); }
      for (const c of g.color) assert.ok(c >= 0 && c <= 1.5);
      feats += g.features;
    }
    assert.ok(feats > 0, `${style}: no features placed`);
    // seam: ground vertices on the shared edge x = CHUNK have the same height in both chunks
    if (style === "dream" || style === "neon" || style === "natural") {
      const edge = (g) => { const m = new Map(); for (let i = 0; i < g.position.length / 3; i++) if (g.extra[i * 2] === 0 && g.position[i * 3] === T.CHUNK) m.set(g.position[i * 3 + 2], g.position[i * 3 + 1]); return m; };
      // neighbours generated in opposite states still meet: the state changes what grows, never the land
      const L = edge(T.buildChunk(t1, 0, 0, { v: 1, a: 1 })), R = edge(T.buildChunk(t1, 1, 0, { v: -1, a: -1 }));
      assert.ok(L.size === T.CHUNK + 1);
      for (const [z, y] of L) assert.equal(R.get(z), y);
    }
    if (style === "natural") {   // the state at generation time shapes what grows
      const count = (mood, kind) => { let n = 0; for (const [cx, cz] of [[0, 0], [1, 0], [2, 1], [-1, 2], [3, -2], [-3, -1]]) { const g = T.buildChunk(t1, cx, cz, mood); for (let i = 0; i < g.extra.length; i += 2) if (g.extra[i] === kind) n++; } return n; };
      const happy = { v: 0.9, a: 0.2 }, sad = { v: -0.9, a: 0.2 };
      assert.ok(count(happy, T.KIND.grass) > 1.2 * count(sad, T.KIND.grass), "pleasant states grow denser grass");
      assert.ok(count(happy, T.KIND.bloom) > count(sad, T.KIND.bloom), "pleasant states grow more flowers");
      assert.ok(T.moodRates("rocks", sad) > T.moodRates("rocks", happy) && T.moodRates("crystals", { v: 0, a: 1 }) > T.moodRates("crystals", { v: 0, a: -1 }));
      assert.deepEqual(T.buildChunk(t1, 2, 1, sad).position, T.buildChunk(t2, 2, 1, sad).position, "same state, same chunk");
    }
    const sp = t1.spawn(0, 0);
    assert.ok(t1.surface(sp.x, sp.z) >= 1.5, `${style}: spawn on land`);
  }
  console.log(`inner world OK: spec validation, interpreter, ${W.STYLES.length} styles build seamless chunks`);
}

// ---------------------------------------------------------------- brain-only decoder in the live engine
{
  const { Simulator } = require(join(D, "sources.cjs"));
  const brain = JSON.parse(readFileSync(join(here, "..", "models", "gameemo_brain.json"), "utf8"));
  const std = JSON.parse(readFileSync(join(here, "..", "models", "gameemo_logreg.json"), "utf8"));
  const art = JSON.parse(readFileSync(join(here, "..", "models", "artifact_detector.json"), "utf8"));
  globalThis.window = globalThis.window ?? { setInterval: () => 0, clearInterval: () => {} };
  const run = holdZ => {
    const m = JSON.parse(JSON.stringify(brain));
    if (holdZ !== null) for (const h of Object.values(m.heads)) h.face.hold_z = holdZ;
    const sim = new Simulator(); sim.mode = "manual"; sim.montage = "muse";
    const e = new Engine(); e.setArtifactModel(art);
    sim.start({ onMeta: (fs, ch) => e.setStream(fs, ch), onStatus: () => {}, onRows: () => {} });
    e.setModel(m); e.setCompareModel(std); e.settings.calibSec = 20;
    let last = null;
    for (let i = 0; i < 256 * 40; i++) e.push([sim.sample()]);
    last = e.last;
    return { e, last };
  };
  const normal = run(null), always = run(1e-6);
  assert.ok(normal.last && normal.last.source === "model" && normal.last.cmp, "brain model decodes with the comparison running");
  const decoded = r => r.e.history.length;          // windows decoded after the baseline
  assert.ok(decoded(normal) > 25 && normal.e.faceHolds < 0.2 * decoded(normal), `holds ${normal.e.faceHolds} of ${decoded(normal)}`);
  assert.ok(always.e.faceHolds === decoded(always), "every window holds when the face is always out of range");
  assert.ok(Math.abs(always.last.v) < 1e-9 && Math.abs(always.last.a) < 1e-9, "a held output never moves");
  assert.ok(normal.e.lastEvent.face.length === 4 && normal.e.lastEvent.faceRegion.length === 51);
  // an extra head ("stress") is decoded, smoothed and held like the others
  {
    const m = JSON.parse(JSON.stringify(brain));
    m.heads.stress = JSON.parse(JSON.stringify(m.heads.arousal));
    const sim = new Simulator(); sim.mode = "manual"; sim.montage = "muse";
    const e = new Engine(); e.setArtifactModel(art);
    sim.start({ onMeta: (fs, ch) => e.setStream(fs, ch), onStatus: () => {}, onRows: () => {} });
    e.setModel(m); e.settings.calibSec = 20;
    for (let i = 0; i < 256 * 40; i++) e.push([sim.sample()]);
    const x = e.last.extra.stress;
    assert.ok(Number.isFinite(x) && Math.abs(x) <= 1, `stress head ${x}`);
    assert.ok(Math.abs(x - e.last.a) < 0.25, `a copy of the arousal head tracks arousal (${x} vs ${e.last.a})`);
    assert.deepEqual(Object.keys(normal.last.extra), [], "valence/arousal-only models report no extra heads");
    const bad = JSON.parse(JSON.stringify(brain)); bad.heads["Bad Name"] = bad.heads.arousal;
    assert.throws(() => new Engine().setModel(bad), /head name/);
  }
  console.log(`brain-only engine OK: decodes with face input, standard decoder alongside, holds ${normal.e.faceHolds}/${decoded(normal)} windows on the simulator, hold logic verified, extra heads decoded`);
}

// ---------------------------------------------------------------- Emotiv Cortex source against a scripted fake service
{
  const { CortexSource, eegColumns, explainCortex } = require(join(D, "cortex.cjs"));
  const cols = ["COUNTER", "INTERPOLATED", "AF3", "T7", "Pz", "T8", "AF4", "RAW_CQ", "MARKER_HARDWARE", "MARKERS"];
  assert.deepEqual(eegColumns(cols), { names: ["AF3", "T7", "Pz", "T8", "AF4"], index: [2, 3, 4, 5, 6] });
  assert.deepEqual(eegColumns(cols, ["AF3", "AF4"]).index, [2, 6]);
  assert.match(explainCortex(-32016, "x"), /licence/);
  const run = async ({ grantAfter = 1, connected = false, refuseEeg = false }) => {
    const calls = [];
    let ws;
    let accessAsks = 0, hsStatus = connected ? "connected" : "discovered";
    const reply = (id, result) => setTimeout(() => ws.onmessage?.({ data: JSON.stringify({ id, jsonrpc: "2.0", result }) }), 1);
    const make = () => {
      ws = { readyState: 0, onopen: null, onmessage: null, onerror: null, onclose: null,
        close() { this.readyState = 3; },
        send(raw) {
          const m = JSON.parse(raw); calls.push(m.method);
          const hs = [{ id: "INSIGHT-A1B2C3D4", status: hsStatus, sensors: ["AF3", "T7", "Pz", "T8", "AF4"], settings: { eegRate: 128 } }];
          if (m.method === "requestAccess") reply(m.id, { accessGranted: ++accessAsks >= grantAfter, message: "" });
          else if (m.method === "authorize") { assert.equal(m.params.clientId, "cid"); reply(m.id, { cortexToken: "tok" }); }
          else if (m.method === "queryHeadsets") reply(m.id, hs);
          else if (m.method === "controlDevice") { assert.equal(m.params.headset, "INSIGHT-A1B2C3D4"); hsStatus = "connected"; reply(m.id, { command: "connect" }); }
          else if (m.method === "createSession") { assert.equal(m.params.status, "open"); reply(m.id, { id: "ses1" }); }
          else if (m.method === "updateSession") reply(m.id, {});
          else if (m.method === "subscribe") {
            assert.deepEqual(m.params.streams, ["eeg"]); assert.equal(m.params.session, "ses1");
            reply(m.id, refuseEeg ? { success: [], failure: [{ streamName: "eeg", code: -32016, message: "The stream is unavailable or unsupported." }] }
              : { success: [{ streamName: "eeg", cols, sid: "ses1" }], failure: [] });
            if (!refuseEeg) for (let k = 0; k < 40; k++) setTimeout(() => ws.onmessage?.({ data: JSON.stringify({ eeg: [k % 128, 0, 4100 + k, 4200, 4300, 4400, 4500 - k, 0, 0, []], sid: "ses1", time: k / 128 }) }), 5 + k);
          }
        } };
      setTimeout(() => { ws.readyState = 1; ws.onopen({}); }, 1);
      return ws;
    };
    const src = new CortexSource({ clientId: "cid", clientSecret: "sec" }, make);
    src.pollMs = 5;
    const got = { meta: null, rows: [], status: [] };
    await src.start({ onMeta: (fs, ch, label) => { got.meta = { fs, ch, label }; }, onRows: r => got.rows.push(...r), onStatus: (t, k) => got.status.push([t, k]), onEnd: () => {} });
    await new Promise(r => setTimeout(r, 200));
    src.stop();
    return { calls, got };
  };
  const ok = await run({ grantAfter: 2 });
  assert.deepEqual(ok.got.meta, { fs: 128, ch: ["AF3", "T7", "Pz", "T8", "AF4"], label: "Emotiv · INSIGHT-A1B2C3D4" });
  assert.equal(ok.got.rows.length, 40);
  assert.deepEqual(ok.got.rows[3], [4103, 4200, 4300, 4400, 4497]);
  assert.ok(ok.calls.filter(c => c === "requestAccess").length === 2 && ok.calls.includes("controlDevice"));
  assert.equal(ok.calls.at(-1), "updateSession", "the session is closed on stop");
  const refused = await run({ connected: true, refuseEeg: true });
  assert.equal(refused.got.meta, null);
  assert.match(refused.got.status.at(-1)[0], /licence/);
  assert.ok(!refused.calls.includes("controlDevice"), "an already connected headset is not reconnected");
  console.log("Emotiv Cortex source OK: access approval wait, connect, session, eeg columns, licence refusal, close on stop");
}

// ---------------------------------------------------------------- personal state calibration: known answers
{
  const S = require(join(D, "states.cjs"));
  const { N_FEATURES: NF } = require(join(D, "montage.cjs"));
  const NR = 51, K = 6, per = 3, W = 30;
  const r = mulberry32(11);
  const g = () => { const u = Math.max(1e-12, r()), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  // features seen by an Insight-like layout: only some columns vary
  const live = Array.from({ length: 50 }, (_, i) => i * 2);
  const sig = Array.from({ length: K }, () => live.map(() => g()));      // each state's brain pattern
  const fsig = Array.from({ length: K }, () => Array.from({ length: NR }, () => g()));
  const mix = Array.from({ length: NR }, () => live.map(() => 0.6 * g()));   // how the face leaks into the features
  const make = (where) => {
    const data = [];
    let block = 0;
    for (let rep = 0; rep < per; rep++) for (let k = 0; k < K; k++) {
      block++;
      for (let w = 0; w < W; w++) {
        const f = new Float64Array(NR), x = new Float64Array(NF);
        for (let i = 0; i < NR; i++) f[i] = 0.8 * g() + (where === "face" ? 1.0 * fsig[k][i] : 0);
        live.forEach((c, t) => {
          let v = 1.0 * g() + (where === "brain" ? 0.9 * sig[k][t] : 0);
          for (let i = 0; i < NR; i++) v += 0.15 * mix[i][t] * f[i];
          x[c] = v;
        });
        data.push({ x, f, y: k, block });
      }
    }
    return data;
  };
  const t0 = Date.now();
  const brain = S.calibrateStates(make("brain"), S.STATES, { perms: 99 });
  const ms = Date.now() - t0;
  const face = S.calibrateStates(make("face"), S.STATES, { perms: 99 });
  const none = S.calibrateStates(make("none"), S.STATES, { perms: 99 });
  const on = m => S.STATES.filter(s => m.checks[s].on).length;
  assert.equal(on(brain), 6, `brain-coded states all pass: ${JSON.stringify(brain.checks)}`);
  assert.ok(brain.bacc > 0.8 && brain.baccP <= 0.02, `brain bacc ${brain.bacc} p ${brain.baccP}`);
  assert.ok(on(face) <= 1 && face.faceOnly.bacc > 0.8 && face.faceOnly.p <= 0.02, `face-coded: brain-only gate ${on(face)}, face-only ${face.faceOnly.bacc}`);
  assert.ok(on(none) <= 1 && none.baccP > 0.05, `no signal: ${on(none)} states on, p ${none.baccP}`);
  // the folded layer reproduces the model on new windows and validates
  const m = S.validateStates(JSON.parse(JSON.stringify(brain)));
  const test = make("brain").filter(d => d.block <= 6);
  let right = 0;
  for (const d of test) { const p = S.stateProbs(m, d.x, d.f); let a = 0; for (let k = 1; k < K; k++) if (p[k] > p[a]) a = k; if (a === d.y) right++; }
  assert.ok(right / test.length > 0.9, `fitted model on fresh windows ${right / test.length}`);
  assert.throws(() => S.calibrateStates(make("brain").filter(d => d.block !== 1 && d.block !== 7), S.STATES), /two calibration blocks/);
  assert.throws(() => S.validateStates({ format: "emotif-states-v1", states: ["a"], head: { layers: [{ W: [[1]], b: [0] }] } }), /shape/);
  // the live engine reports only the states that passed, as probabilities
  {
    const { Simulator } = require(join(D, "sources.cjs"));
    const art = JSON.parse(readFileSync(join(here, "..", "models", "artifact_detector.json"), "utf8"));
    const pm = JSON.parse(JSON.stringify(brain));
    S.STATES.forEach((n, i) => { pm.checks[n].on = i < 2; });
    const sim = new Simulator(); sim.mode = "manual"; sim.montage = "emotiv_insight";
    const e = new Engine(); e.setArtifactModel(art);
    sim.start({ onMeta: (fs, ch) => e.setStream(fs, ch), onStatus: () => {}, onRows: () => {} });
    e.settings.calibSec = 20;
    e.personalStates = S.validateStates(pm);
    for (let i = 0; i < 256 * 35; i++) e.push([sim.sample()]);
    const st = e.last.states;
    assert.deepEqual(Object.keys(st), ["calm", "stress"]);
    for (const v of Object.values(st)) assert.ok(v >= 0 && v <= 1);
  }
  console.log(`personal states OK: brain-coded 6/6 on (bacc ${brain.bacc.toFixed(2)}), face-coded ${on(face)}/6 on while face alone scores ${face.faceOnly.bacc.toFixed(2)}, no signal ${on(none)}/6; calibration with 99 permutations ${ms} ms`);
}

// ---------------------------------------------------------------- daily check-in markers and trends
{
  const Ck = require(join(D, "checkin.cjs"));
  assert.equal(Ck.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1);
  assert.equal(Ck.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1);
  assert.ok(Math.abs(Ck.spearman([1, 2, 2, 3], [1, 3, 2, 4]) - 0.9486832980505138) < 1e-12);   // ties: average ranks
  const acc = new Ck.CheckinAccumulator(["AF3", "T7", "Pz", "T8", "AF4"]);
  for (let w = 0; w < 30; w++) acc.add([[1, 1, 2, 1, 1], [1, 1, 1, 1, 1], [1, 1, 4, 1, 1], [1, 1, 1, 1, 1], [1, 1, 8, 1, 1]], [false, false, false, w % 2 === 0, false]);
  const sm = acc.summarize();
  assert.ok(Math.abs(sm.markers["frontal alpha asymmetry"] - (Math.log(8 / 12) - Math.log(2 / 6))) < 1e-12, "AF4 minus AF3 relative alpha");
  assert.ok(Math.abs(sm.markers["posterior alpha"] - Math.log(4 / 8)) < 1e-12, "Pz relative alpha");
  assert.equal(new Ck.CheckinAccumulator(["AF3"]).summarize(), null, "needs 20 windows");
  const r = mulberry32(3);
  const day = (i, link) => { const mood = Math.round(r() * 10); return { date: `2026-11-${String(i + 1).padStart(2, "0")}`, answers: { mood, sleepHours: 7, sleepQuality: Math.round(r() * 10) },
    markers: { "frontal alpha asymmetry": link ? mood * 0.1 + 0.05 * r() : r(), "relative theta": r(), "relative alpha": r(), "relative beta": r(), "posterior alpha": r() }, windows: 100, channels: [] }; };
  assert.deepEqual(Ck.trends(Array.from({ length: 6 }, (_, i) => day(i, true))), [], "nothing before 7 days");
  const linked = Ck.trends(Array.from({ length: 14 }, (_, i) => day(i, true)));
  const faa = linked.find(t => t.marker === "frontal alpha asymmetry" && t.answer === "mood");
  assert.ok(faa.rho > 0.9 && faa.pHolm < 0.05, `planted link found: rho ${faa.rho}, Holm p ${faa.pHolm}`);
  const constant = linked.find(t => t.answer === "sleepHours");
  assert.ok(Number.isNaN(constant.rho) && constant.p === 1, "a constant answer gives no correlation");
  const nulls = Ck.trends(Array.from({ length: 14 }, (_, i) => day(i, false)));
  assert.ok(nulls.every(t => t.pHolm > 0.05), "no planted link, nothing survives Holm");
  console.log("daily check-in OK: Spearman with ties, markers from named channels, 7-day minimum, planted link found, null survives nothing");
}

// ---------------------------------------------------------------- brain control of objects (alpha focus / relax)
{
  const I = require(join(D, "interact.cjs"));
  const Mo = require(join(D, "montage.cjs"));
  const R = Mo.REGIONS.length, x = new Float64Array(Mo.N_FEATURES);
  const set = (region, alpha) => { const r = Mo.REGIONS.indexOf(region); x[r * 5 + 2] = alpha; x[R * 5 + 35 + r] = 1; };
  set("Pz", -2); set("Fp1", 5);
  assert.equal(I.alphaZ(x), -2, "posterior alpha preferred when Pz is observed");
  const y = new Float64Array(Mo.N_FEATURES); y[Mo.REGIONS.indexOf("Fp1") * 5 + 2] = 1; y[R * 5 + 35 + Mo.REGIONS.indexOf("Fp1")] = 1;
  assert.equal(I.alphaZ(y), 1, "falls back to the observed regions");
  assert.equal(I.alphaZ(new Float64Array(Mo.N_FEATURES)), null);
  const c = new I.AlphaControl(0.7, 1.0, 1.5);
  let st;
  for (let t = 0; t < 0.5; t += 0.25) st = c.update(x, false, 0.25);
  assert.equal(st.command, "idle", "a command needs the dwell time");
  for (let t = 0; t < 6; t += 0.25) st = c.update(x, false, 0.25);
  assert.equal(st.command, "focus", "alpha below baseline -> focus");
  for (let t = 0; t < 4; t += 0.25) st = c.update(x, true, 0.25);
  assert.equal(st.command, "focus", "held (face activity) keeps the last command and level");
  const relaxed = new Float64Array(x); relaxed[Mo.REGIONS.indexOf("Pz") * 5 + 2] = 2.5;
  for (let t = 0; t < 1.0; t += 0.25) st = c.update(relaxed, false, 0.25);
  assert.notEqual(st.command, "relax", "no jump straight from focus to relax");
  for (let t = 0; t < 8; t += 0.25) st = c.update(relaxed, false, 0.25);
  assert.equal(st.command, "relax", "alpha above baseline -> relax");
  const n = new I.AlphaControl(); let flips = 0, prev = "idle";
  const r2 = mulberry32(5);
  for (let t = 0; t < 120; t += 0.25) {          // noise around the baseline alone
    const z = new Float64Array(x); z[Mo.REGIONS.indexOf("Pz") * 5 + 2] = (r2() - 0.5) * 1.2;
    const s2 = n.update(z, false, 0.25); if (s2.command !== prev) { flips++; prev = s2.command; }
  }
  assert.ok(flips <= 2, `baseline noise rarely issues commands (${flips} changes in 2 min)`);
  console.log(`brain control OK: posterior alpha, dwell, hold on face activity, hysteresis, ${flips} command changes from 2 min of baseline noise`);
}

console.log("all app tests passed");
