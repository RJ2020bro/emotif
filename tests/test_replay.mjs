// Real-data replay: real Muse recordings (Bird et al. 2018) streamed through the app's live engine,
// exactly as a headset would deliver them. Checks the full pipeline on real noise and measures speed.
// Skips if data/external/bird_muse is missing (scripts/get_bird_muse.sh).
import { createRequire } from "node:module";
import { readFileSync, existsSync } from "node:fs";
import assert from "node:assert/strict";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const D = join(here, "..", "app", "dist-test");
const { Engine } = require(join(D, "engine.cjs"));
const dir = join(here, "..", "data", "external", "bird_muse");
if (!existsSync(dir)) { console.log("replay skipped: no data/external/bird_muse"); process.exit(0); }

function load(name) {
  const lines = readFileSync(join(dir, name), "utf8").trim().split(/\r?\n/);
  const h = lines[0].split(",");
  const idx = ["TP9", "AF7", "AF8", "TP10"].map(c => h.indexOf(c));
  return lines.slice(1).map(l => { const p = l.split(","); return idx.map(i => Number(p[i])); });
}

const artifact = JSON.parse(readFileSync(join(here, "..", "models", "artifact_detector.json"), "utf8"));
const model = JSON.parse(readFileSync(join(here, "..", "models", "muse_mental_logreg.json"), "utf8"));
const results = [];
for (const subj of ["a", "b", "c", "d"]) {
  const relaxed = load(`subject${subj}-relaxed-1.csv`);
  const conc = load(`subject${subj}-concentrating-1.csv`);
  const eng = new Engine();
  eng.settings.calibSec = 30;
  eng.setArtifactModel(artifact);
  eng.setStream(256, ["TP9", "AF7", "AF8", "TP10"]);
  eng.setModel(model);
  const out = { relaxed: [], conc: [] };
  const labels = { clean: 0, ocular: 0, muscle: 0 };
  let phase = "relaxed", fired = false;
  const winMs = [];
  eng.listeners.add(ev => {
    fired = true;
    if (ev.out && ev.calibrated) out[phase].push(ev.out.a);
    if (ev.verdicts) for (const v of ev.verdicts) labels[v.label]++;
  });
  const feed = rows => {
    for (let i = 0; i < rows.length; i += 12) {
      fired = false;
      const a = performance.now();
      eng.push(rows.slice(i, i + 12));
      if (fired && eng.calibrated) winMs.push(performance.now() - a);
    }
  };
  const t0 = performance.now();
  feed(relaxed);
  phase = "conc";
  feed(conc);
  const ms = performance.now() - t0;
  const secs = (relaxed.length + conc.length) / 256;
  assert.ok(eng.calibrated || eng.calibError, "baseline completes or reports why");
  if (!eng.calibrated) { results.push({ subj, note: eng.calibError }); continue; }
  assert.ok(eng.asrActive && eng.detectorActive, "cleaning active on real data");
  assert.ok(out.conc.length > 50 && out.conc.every(Number.isFinite), "decodes the concentrating recording");
  const mean = a => a.reduce((s, x) => s + x, 0) / Math.max(1, a.length);
  winMs.sort((x, y) => x - y);
  const q = p => winMs[Math.min(winMs.length - 1, Math.floor(p * winMs.length))];
  results.push({ subj, speed: secs / (ms / 1000), relaxedA: mean(out.relaxed), concA: mean(out.conc), labels,
                 p50: q(0.5), p99: q(0.99), delayMs: 1000 * eng.delaySec });
}
for (const r of results) {
  if (r.note) { console.log(`  subject ${r.subj}: ${r.note}`); continue; }
  const tot = r.labels.clean + r.labels.ocular + r.labels.muscle;
  console.log(`  subject ${r.subj}: ${r.speed.toFixed(0)}x real time, ${r.p50.toFixed(2)} ms per decoded window (p99 ${r.p99.toFixed(2)} ms), ASR delay ${r.delayMs.toFixed(0)} ms; decoded arousal relaxed ${r.relaxedA.toFixed(2)} vs concentrating ${r.concA.toFixed(2)}; ` +
    `channel-windows clean ${(100 * r.labels.clean / tot).toFixed(0)}% eye ${(100 * r.labels.ocular / tot).toFixed(0)}% muscle ${(100 * r.labels.muscle / tot).toFixed(0)}%`);
}
const ok = results.filter(r => !r.note);
assert.ok(ok.length >= 3, "at least 3 of 4 real recordings calibrate");
assert.ok(ok.every(r => r.speed > 5), "runs well faster than real time");
console.log(`replay OK: ${ok.length} real Muse sessions through the live pipeline`);
