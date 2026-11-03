// Builds web/index.html (one self-contained file: app + three.js + muse-js + CSS) and, with
// --test-bundles, CommonJS bundles of the pure modules for the Python/Node parity tests.
import { build } from "esbuild";
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "web", "index.html");

const js = await build({
  entryPoints: [join(here, "src", "main.ts")], bundle: true, write: false, format: "iife", target: "es2020",
  minify: true, legalComments: "none", define: { "process.env.NODE_ENV": '"production"' }, logLevel: "warning",
});
const code = js.outputFiles[0].text.replace(/<\/script/gi, "<\\/script");
const css = readFileSync(join(here, "src", "styles.css"), "utf8");
// result figures shown in the Method view, embedded as data URIs (skipped if not generated yet)
const figFiles = {
  denoise: "results/denoise/fig_denoise.png",
  detector: "results/artifact_detector/fig_confusion.png",
  confound: "results/audit/fig_audit.png",
  montage: "results/muse_mental/fig_montage_arousal_logreg.png",
  brain: "results/brain_gameemo_v2/fig_brain_decoder.png",
  brainmap: "results/brain_gameemo_v2/fig_brain_map.png",
  states: "results/states/fig_states.png",
};
const figs = {};
for (const [k, rel] of Object.entries(figFiles)) {
  try { figs[k] = "data:image/png;base64," + readFileSync(join(here, "..", rel)).toString("base64"); } catch { /* not generated */ }
}
// image library for the Natural world (scripts/build_assets.py; credits in app/assets/CREDITS.md)
const assets = {};
for (const f of readdirSync(join(here, "assets"))) {
  const m = f.match(/^([a-z0-9_]+)\.(jpg|png)$/);
  if (m) assets[m[1]] = `data:image/${m[2] === "jpg" ? "jpeg" : "png"};base64,` + readFileSync(join(here, "assets", f)).toString("base64");
}
const tpl = readFileSync(join(here, "src", "index.html"), "utf8");
const figJs = `window.__EMOTIF_FIGS=${JSON.stringify(figs)};window.__EMOTIF_ASSETS=${JSON.stringify(assets)};`;
const html = tpl.replace("/*CSS*/", () => css).replace("/*JS*/", () => figJs + "\n" + code);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, html);
console.log(`web/index.html ${(html.length / 1024).toFixed(0)} KB`);

if (process.argv.includes("--test-bundles")) {
  const dir = join(here, "dist-test");
  mkdirSync(dir, { recursive: true });
  for (const m of ["dsp", "montage", "model", "intent", "arm", "director", "engine", "asr", "artifact", "edf", "sources", "safety", "worldspec", "terrain", "face", "cortex", "states", "checkin", "interact"]) {
    await build({ entryPoints: [join(here, "src", `${m}.ts`)], bundle: true, format: "cjs", platform: "node", outfile: join(dir, `${m}.cjs`), logLevel: "warning" });
  }
  console.log("test bundles in app/dist-test/");
}
