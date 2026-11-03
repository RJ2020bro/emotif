// The world's scenery. A WorldSpec is what a short theme turns into: the art style, the land, the colours, what grows
// on it and the weather. interpretPrompt builds it on this computer (no network, no AI service); validateSpec
// clamps any spec (a saved one, or a hand-written one) so the renderer only ever sees safe values. The decoded
// emotion then drives how the world looks and moves in real time (world.ts).

export const STYLES = ["natural", "voxel", "lowpoly", "dream", "neon"] as const;
export type Style = typeof STYLES[number];
export const STYLE_LABELS: Record<Style, string> = {
  natural: "Natural (realistic)", voxel: "Voxel (block world)", lowpoly: "Low-poly", dream: "Dreamscape (soft, painterly)", neon: "Neon (synthwave)",
};
export const FEATURES = ["trees", "pines", "palms", "crystals", "ruins", "mushrooms", "flowers", "lanterns", "rocks", "cacti"] as const;
export type Feature = typeof FEATURES[number];
export const WEATHERS = ["mood", "none", "rain", "snow", "fireflies", "petals", "embers", "stars"] as const;
export type Weather = typeof WEATHERS[number];
export const TIMES = ["dawn", "day", "sunset", "night"] as const;
export type TimeOfDay = typeof TIMES[number];

export type Palette = { sky: string; horizon: string; low: string; high: string; peak: string; water: string; accent: string; foliage: string };
export type WorldSpec = {
  name: string;
  style: Style;
  seed: number;
  terrain: {
    scale: number;       // size of hills in blocks, 16-160
    height: number;      // mountain height in blocks, 2-48
    roughness: number;   // 0 smooth .. 1 jagged
    water: number;       // 0 dry .. 1 mostly sea
    terrace: number;     // 0 none .. 1 strong plateaus / mesas
    islands: boolean;    // land rises from a sea in separate islands
  };
  palette: Palette;
  features: { kind: Feature; density: number }[];   // up to 4, density 0-1
  weather: Weather;
  time: TimeOfDay;
  mood: number;         // 0-1: how strongly the decoded emotion recolours and moves the world
};

export const DEFAULT_SPEC: WorldSpec = {
  name: "Inner meadow", style: "natural", seed: 1,
  terrain: { scale: 64, height: 18, roughness: 0.35, water: 0.3, terrace: 0, islands: false },
  palette: { sky: "#6fa8dc", horizon: "#f3e2c3", low: "#7aa65a", high: "#a7b86a", peak: "#f2f2ef", water: "#3b7bbf", accent: "#f2a541", foliage: "#4f8a3a" },
  features: [{ kind: "trees", density: 0.5 }, { kind: "flowers", density: 0.5 }],
  weather: "mood", time: "day", mood: 0.8,
};

const HEX = /^#[0-9a-f]{6}$/i;
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
const num = (x: unknown, lo: number, hi: number, d: number) => (typeof x === "number" && Number.isFinite(x) ? clamp(x, lo, hi) : d);
const pick = <T extends string>(x: unknown, set: readonly T[], d: T): T => (typeof x === "string" && (set as readonly string[]).includes(x) ? x as T : d);

/** Coerce anything (e.g. JSON from a language model) into a valid spec; unknown or out-of-range values fall back
 * to `base`. Never throws. */
export function validateSpec(raw: unknown, base: WorldSpec = DEFAULT_SPEC): WorldSpec {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const t = (r.terrain && typeof r.terrain === "object" ? r.terrain : {}) as Record<string, unknown>;
  const p = (r.palette && typeof r.palette === "object" ? r.palette : {}) as Record<string, unknown>;
  const pal = {} as Palette;
  (Object.keys(base.palette) as (keyof Palette)[]).forEach(k => { pal[k] = typeof p[k] === "string" && HEX.test(p[k] as string) ? (p[k] as string).toLowerCase() : base.palette[k]; });
  const feats: WorldSpec["features"] = [];
  if (Array.isArray(r.features)) {
    for (const f of r.features) {
      const o = (f && typeof f === "object" ? f : {}) as Record<string, unknown>;
      const kind = pick(typeof f === "string" ? f : o.kind, FEATURES, "trees");
      if ((typeof f === "string" ? f : o.kind) !== kind || feats.some(x => x.kind === kind)) continue;
      feats.push({ kind, density: num(o.density, 0, 1, 0.5) });
      if (feats.length === 4) break;
    }
  }
  return {
    name: typeof r.name === "string" && r.name.trim() ? r.name.trim().slice(0, 60) : base.name,
    style: pick(r.style, STYLES, base.style),
    seed: typeof r.seed === "number" && Number.isFinite(r.seed) ? Math.abs(Math.floor(r.seed)) % 1_000_000 : base.seed,
    terrain: {
      scale: num(t.scale, 16, 160, base.terrain.scale), height: num(t.height, 2, 48, base.terrain.height),
      roughness: num(t.roughness, 0, 1, base.terrain.roughness), water: num(t.water, 0, 1, base.terrain.water),
      terrace: num(t.terrace, 0, 1, base.terrain.terrace),
      islands: typeof t.islands === "boolean" ? t.islands : base.terrain.islands,
    },
    palette: pal,
    features: Array.isArray(r.features) ? feats : base.features.map(f => ({ ...f })),
    weather: pick(r.weather, WEATHERS, base.weather),
    time: pick(r.time, TIMES, base.time),
    mood: num(r.mood, 0, 1, base.mood),
  };
}

export function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0) % 1_000_000;
}

// ---------------------------------------------------------------- on-device interpreter
type Patch = { terrain?: Partial<WorldSpec["terrain"]>; palette?: Partial<Palette>; features?: Feature[]; weather?: Weather; time?: TimeOfDay; style?: Style };
type Rule = { words: string[]; patch: Patch; name?: string };

// Biomes come first; later rules (colour, time, weather, style) refine them.
const BIOMES: Rule[] = [
  { name: "Forest", words: ["forest", "woods", "woodland", "grove", "trees"], patch: { terrain: { height: 14, roughness: 0.35, water: 0.2 }, palette: { low: "#4f7d3a", high: "#6e9a48", horizon: "#e3ecd6", foliage: "#3f7f2f" }, features: ["trees", "mushrooms", "flowers"] } },
  { name: "Pine mountains", words: ["mountain", "mountains", "peaks", "alps", "cliffs", "alpine"], patch: { terrain: { height: 42, roughness: 0.65, scale: 90, water: 0.15 }, palette: { low: "#4d6b45", high: "#7d7f78", peak: "#f4f6f8" }, features: ["pines", "rocks"] } },
  { name: "Desert", words: ["desert", "dune", "dunes", "sahara", "sand"], patch: { terrain: { height: 10, roughness: 0.15, water: 0, scale: 110 }, palette: { low: "#d9b779", high: "#c99a5b", sky: "#7fb2e0", horizon: "#f5dfb5", peak: "#e8cf9c" }, features: ["cacti", "rocks", "ruins"] } },
  { name: "Canyon", words: ["canyon", "mesa", "badlands", "grand canyon"], patch: { terrain: { height: 30, roughness: 0.4, water: 0.05, terrace: 0.8 }, palette: { low: "#b8653f", high: "#d48a52", peak: "#e9c08a", horizon: "#f2c9a0" }, features: ["rocks", "cacti"] } },
  { name: "Archipelago", words: ["island", "islands", "ocean", "sea", "beach", "tropical", "lagoon", "archipelago"], patch: { terrain: { islands: true, water: 0.65, height: 16, roughness: 0.3 }, palette: { low: "#e6d39c", high: "#5f9e4f", water: "#2a9fd6", horizon: "#e6f4f7" }, features: ["palms", "rocks", "flowers"] } },
  { name: "Snowfield", words: ["snow", "snowy", "ice", "icy", "arctic", "frozen", "winter", "tundra", "glacier"], patch: { terrain: { height: 20, roughness: 0.4, water: 0.25 }, palette: { low: "#e9eef2", high: "#d5dee6", peak: "#ffffff", water: "#7fb3d5", sky: "#9cc2e6", horizon: "#eef3f7", foliage: "#2f5a46" }, features: ["pines", "crystals"], weather: "snow" } },
  { name: "Volcano", words: ["volcano", "volcanic", "lava", "magma", "hell", "inferno"], patch: { terrain: { height: 38, roughness: 0.7, water: 0.2 }, palette: { low: "#2b2523", high: "#3d3330", peak: "#5a4a44", water: "#ff5a1f", sky: "#2a1714", horizon: "#a8432a", accent: "#ff7a2f" }, features: ["rocks", "crystals"], weather: "embers" } },
  { name: "Jungle", words: ["jungle", "rainforest", "swamp", "marsh"], patch: { terrain: { height: 16, roughness: 0.5, water: 0.35 }, palette: { low: "#2f6b34", high: "#4f8f3a", water: "#3f7f6a", horizon: "#cfe5c4", foliage: "#2e7a2a" }, features: ["trees", "palms", "flowers", "mushrooms"], weather: "rain" } },
  { name: "Meadow", words: ["meadow", "field", "fields", "grassland", "prairie", "hills", "countryside", "garden"], patch: { terrain: { height: 10, roughness: 0.2, water: 0.2 }, palette: { low: "#7fb35a", high: "#a5c46a" }, features: ["flowers", "trees"] } },
  { name: "Crystal caverns", words: ["crystal", "crystals", "gem", "gems", "cave", "caves", "cavern"], patch: { terrain: { height: 24, roughness: 0.6, water: 0.2 }, palette: { low: "#3b3552", high: "#56507a", accent: "#7fe7ff", sky: "#1f1b33", horizon: "#4a4170" }, features: ["crystals", "rocks", "mushrooms"], time: "night" } },
  { name: "Ancient ruins", words: ["ruin", "ruins", "temple", "temples", "castle", "ancient", "kingdom", "city", "village", "medieval"], patch: { terrain: { height: 14, roughness: 0.3, terrace: 0.4 }, features: ["ruins", "trees", "lanterns"] } },
  { name: "Alien world", words: ["alien", "planet", "space", "mars", "moon", "galaxy", "cosmic", "sci-fi", "scifi"], patch: { terrain: { height: 26, roughness: 0.55, water: 0.15 }, palette: { low: "#5a2f6b", high: "#8a4f8f", peak: "#d6a6e8", water: "#36e0c2", sky: "#120b24", horizon: "#4b2a6b", accent: "#36e0c2", foliage: "#b05bd6" }, features: ["crystals", "mushrooms", "rocks"], time: "night", weather: "stars" } },
  { name: "Mushroom kingdom", words: ["mushroom", "mushrooms", "fungi", "fairy", "fairytale", "enchanted", "magic", "magical"], patch: { terrain: { height: 14, roughness: 0.35 }, palette: { low: "#5d8f5a", high: "#86b36b", accent: "#ff6fae" }, features: ["mushrooms", "flowers", "lanterns"], weather: "fireflies" } },
  { name: "Cherry valley", words: ["cherry", "sakura", "blossom", "blossoms", "japan", "zen"], patch: { terrain: { height: 18, roughness: 0.3, water: 0.3 }, palette: { low: "#8fb36b", high: "#b0c27a", accent: "#ffb7d2", horizon: "#fbe3ec", foliage: "#f4a6c6" }, features: ["trees", "flowers", "lanterns"], weather: "petals" } },
];
const MODIFIERS: Rule[] = [
  { words: ["autumn", "fall", "maple", "orange leaves"], patch: { palette: { foliage: "#d9822b", accent: "#e8b04a", horizon: "#f6e3c4" } } },
  { words: ["night", "midnight", "dark", "starry", "stars"], patch: { time: "night" } },
  { words: ["sunset", "dusk", "evening", "golden hour"], patch: { time: "sunset" } },
  { words: ["dawn", "sunrise", "morning"], patch: { time: "dawn" } },
  { words: ["rain", "rainy", "storm", "stormy"], patch: { weather: "rain" } },
  { words: ["fireflies", "firefly", "glowing", "glow"], patch: { weather: "fireflies" } },
  { words: ["flat", "plains", "calm", "peaceful", "gentle"], patch: { terrain: { roughness: 0.15, height: 8 } } },
  { words: ["huge", "giant", "epic", "towering", "steep", "jagged"], patch: { terrain: { height: 44, roughness: 0.7 } } },
  { words: ["lake", "lakes", "river", "rivers", "water"], patch: { terrain: { water: 0.45 } } },
  { words: ["lantern", "lanterns", "lights", "lamps"], patch: { features: ["lanterns"] } },
  { words: ["flower", "flowers", "floral"], patch: { features: ["flowers"] } },
  { words: ["pink"], patch: { palette: { accent: "#ff8fc0", horizon: "#fbe0ec" } } },
  { words: ["purple", "violet"], patch: { palette: { accent: "#b07cff", sky: "#4b3a8a", horizon: "#c7b5ef" } } },
  { words: ["gold", "golden", "yellow"], patch: { palette: { accent: "#ffcc4d", horizon: "#f8e3b0" } } },
  { words: ["blue", "azure"], patch: { palette: { accent: "#5fb0ff", sky: "#3f86d6" } } },
  { words: ["red", "crimson"], patch: { palette: { accent: "#ff5a4f" } } },
  { words: ["green", "emerald", "lush"], patch: { palette: { low: "#3f8f45", high: "#69ad4f" } } },
];
const STYLE_WORDS: [string[], Style][] = [
  [["realistic", "photoreal", "photorealistic", "lifelike", "natural", "cinematic", "real life", "real world"], "natural"],
  [["minecraft", "voxel", "blocky", "block", "blocks", "cube", "cubes", "pixel", "8-bit", "lego"], "voxel"],
  [["low poly", "low-poly", "lowpoly", "polygon", "polygonal", "geometric", "faceted"], "lowpoly"],
  [["dream", "dreamy", "dreamscape", "pastel", "watercolor", "watercolour", "painterly", "ghibli", "soft", "cozy", "cosy", "storybook"], "dream"],
  [["neon", "synthwave", "vaporwave", "cyber", "cyberpunk", "tron", "retro", "outrun", "80s", "wireframe", "hologram"], "neon"],
];

function has(text: string, w: string): boolean {
  return new RegExp(`(^|[^a-z])${w.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&")}($|[^a-z])`).test(text);
}
function applyPatch(s: WorldSpec, p: Patch) {
  if (p.terrain) s.terrain = { ...s.terrain, ...p.terrain };
  if (p.palette) s.palette = { ...s.palette, ...p.palette };
  if (p.weather) s.weather = p.weather;
  if (p.time) s.time = p.time;
  if (p.style) s.style = p.style;
  if (p.features) for (const k of p.features) if (!s.features.some(f => f.kind === k)) s.features.push({ kind: k, density: 0.5 });
}

/** On-device prompt interpreter: matches words for biomes, styles, colours, time and weather. Deterministic
 * (the seed comes from the text), instant and offline. `style` overrides any style word in the prompt. */
export function interpretPrompt(prompt: string, style?: Style): WorldSpec {
  const text = ` ${prompt.toLowerCase()} `;
  const s: WorldSpec = { ...DEFAULT_SPEC, terrain: { ...DEFAULT_SPEC.terrain }, palette: { ...DEFAULT_SPEC.palette }, features: [], seed: hashString(prompt.trim().toLowerCase() || "emotif") };
  const biomes = BIOMES.filter(r => r.words.some(w => has(text, w)));
  for (const b of biomes.slice(0, 2)) applyPatch(s, b.patch);
  for (const m of MODIFIERS) if (m.words.some(w => has(text, w))) applyPatch(s, m.patch);
  for (const [words, st] of STYLE_WORDS) if (words.some(w => has(text, w))) { s.style = st; break; }
  if (style) s.style = style;
  if (s.style === "neon" && !MODIFIERS.some(m => m.patch.palette?.accent && m.words.some(w => has(text, w)))) {
    s.palette = { ...s.palette, accent: "#ff2bd6", sky: "#120626", horizon: "#3a1050", water: "#120a2a" };
  }
  if (!s.features.length) s.features = DEFAULT_SPEC.features.map(f => ({ ...f }));
  s.features = s.features.slice(0, 4);
  const named = prompt.trim();
  s.name = named ? named.slice(0, 60) : biomes[0]?.name ?? DEFAULT_SPEC.name;
  return validateSpec(s);
}
