// Builds the personal-state model for the demo's virtual participant (simulated Emotiv Insight, seed 4): the same
// in-app calibration a real person goes through ("states" protocol, three rounds of six states), run on the
// synthetic stream. The demo loads it so the six states are decoded live from the generated EEG.
//   (cd app && node build.mjs --test-bundles) && node scripts/make_demo_states.mjs
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url)), require = createRequire(import.meta.url), D = join(here, "..", "app", "dist-test");
globalThis.window = globalThis.window ?? { setInterval: () => 0, clearInterval: () => {} };
const So = require(join(D, "sources.cjs")), S = require(join(D, "states.cjs")), { Engine } = require(join(D, "engine.cjs"));
const art = JSON.parse(readFileSync(join(here, "..", "models", "artifact_detector.json"), "utf8"));
export const DEMO_SEED = 4;
const sim = new So.Simulator(); sim.montage = "emotiv_insight"; sim.seed = DEMO_SEED; sim.mode = "participant"; sim.calibSec = 40;
sim.start({ onMeta: () => {}, onStatus: () => {}, onRows: () => {}, onEnd: () => {} }); sim.stop();
const e = new Engine(); e.setArtifactModel(art); e.settings.calibSec = 40;
e.setStream(sim.fs, So.SIM_MONTAGES.emotiv_insight.channels);
const run = sec => { for (let i = 0; i < sim.fs * sec; i++) e.push([sim.sample()]); };
run(40);
const data = [];
let block = 0;
for (let round = 0; round < 3; round++) for (const st of S.STATES) {
  block++; sim.block = So.simBlockFor({ task: "x", state: st });
  const n0 = e.featureLog.length; run(30);
  e.featureLog.slice(n0 + 20).forEach(w => data.push({ x: Array.from(w.x), f: Array.from(w.f), y: S.STATES.indexOf(st), block }));
  sim.block = So.simBlockFor({ task: "rest" }); run(10); sim.block = null;
}
const m = S.calibrateStates(data, S.STATES, { perms: 199, seed: DEMO_SEED });
S.validateStates(JSON.parse(JSON.stringify(m)));
const out = join(here, "..", "models", "demo_insight_states.json");
writeFileSync(out, JSON.stringify({ ...m, note: "SYNTHETIC: virtual participant (simulated Emotiv Insight, seed 4) for the demo only; not a research result" }));
console.log(`demo states model: bacc ${m.bacc.toFixed(2)}, on: ${S.STATES.filter(s => m.checks[s].on).join(", ")} -> ${out}`);
