// Builds terrain chunks off the main thread so streaming new land never stalls a frame. world.ts sends the world
// spec once (and again whenever it changes, with a new generation number) and then chunk jobs; each reply carries
// the geometry arrays as transferables. The Terrain built here from the same spec is identical to the main
// thread's (everything is seeded by the spec), so worker-built and main-thread-built chunks are interchangeable.
import { Terrain, buildChunk, type ChunkMood } from "./terrain";
import type { WorldSpec } from "./worldspec";

type Msg = { type: "spec"; spec: WorldSpec; gen: number } | { type: "job"; gen: number; key: string; cx: number; cz: number; mood: ChunkMood; lod: number };

let terrain: Terrain | null = null;
let gen = -1;

self.onmessage = (e: MessageEvent<Msg>) => {
  const m = e.data;
  if (m.type === "spec") { terrain = new Terrain(m.spec); gen = m.gen; return; }
  if (!terrain || m.gen !== gen) return;          // a job for a world that has since been replaced
  const g = buildChunk(terrain, m.cx, m.cz, m.mood, m.lod);
  (self as unknown as Worker).postMessage({ gen: m.gen, key: m.key, cx: m.cx, cz: m.cz, lod: m.lod, g },
    [g.position.buffer, g.normal.buffer, g.color.buffer, g.extra.buffer, g.uv.buffer, g.index.buffer]);
};
