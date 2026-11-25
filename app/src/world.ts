// Two scenes on one renderer.
// Inner world (Live view): an endless open world generated from a WorldSpec (a prompt, through Claude or the
//   on-device interpreter) in one of four art styles. A glowing explorer drifts across it; the decoded emotion
//   sets its speed, the light, colour grading, wind, mist and weather. Each second leaves a beacon on its path,
//   so the journey is a record of the session (time as the fourth dimension).
// Arm scene (Arm view): the digital twin of the RoArm-M2 on its pedestal, with the paper and pen trail.
// Every brightness change passes the photosensitive slew limit (safety.ts, WCAG 2.3.1).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { slewToward } from "./safety";
import { forward, inverse, INIT, L2, T2, type Pose } from "./arm";
import { CHUNK, REGION, Terrain, buildChunk, type ChunkGeometry, type ChunkMood, type Culture, type Landmark } from "./terrain";
import { DEFAULT_SPEC, validateSpec, type Weather, type WorldSpec } from "./worldspec";

const PAL = { // [sky top, horizon, ground low, ground high, accent]
  happy: ["#4c86c9", "#f6d3a1", "#7c9a52", "#c9c27a", "#f2a541"],
  calm: ["#5d87a8", "#d4e6e1", "#3f7a6c", "#9cc4b2", "#5fc2b0"],
  stressed: ["#1d1416", "#7d3326", "#3a2422", "#7a4a3c", "#ef5a3c"],
  sad: ["#1c2430", "#4e5b6b", "#2a323c", "#56606c", "#7f9cc0"],
} as const;
const PALC = Object.fromEntries(Object.entries(PAL).map(([k, v]) => [k, v.map(h => new THREE.Color(h))])) as Record<keyof typeof PAL, THREE.Color[]>;
const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

export function paletteAt(v: number, a: number): THREE.Color[] {
  const u = (clamp(v, -1, 1) + 1) / 2, w = (clamp(a, -1, 1) + 1) / 2;
  return [0, 1, 2, 3, 4].map(i => PALC.sad[i].clone().lerp(PALC.calm[i], u).lerp(PALC.stressed[i].clone().lerp(PALC.happy[i], u), w));
}
export function affectCss(v: number, a: number): string { return "#" + paletteAt(v, a)[4].getHexString(); }

// ---------------------------------------------------------------- shaders
const GRADE = /* glsl */`
uniform vec3 uFogColor; uniform float uFogDensity; uniform vec3 uTint; uniform float uSat; uniform float uBright;
vec3 grade(vec3 c) { float l = dot(c, vec3(0.2126, 0.7152, 0.0722)); return max(mix(vec3(l), c, uSat) * uTint * uBright, 0.0); }
float fogF(vec3 wp) { float d = length(wp - cameraPosition) * uFogDensity; return clamp(1.0 - exp(-d * d), 0.0, 1.0); }
uniform float uFar;   // land fades into the haze before the edge of the loaded world, so no chunk edge ever shows
uniform float uSink;
float farF(vec3 wp) { return (1.0 - uSink) * smoothstep(uFar * 0.6, uFar, length(wp.xz - cameraPosition.xz)); }
float h13(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
float h12(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float vn(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h12(i), h12(i + vec2(1.0, 0.0)), f.x), mix(h12(i + vec2(0.0, 1.0)), h12(i + vec2(1.0, 1.0)), f.x), f.y); }
float fbm3(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 3; i++) { s += a * vn(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; } return s / 0.875; }
float fbm4(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vn(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; } return s / 0.9375; }
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
uniform float uNatural;
// natural style: aerial perspective (fog brightens toward the sun) and less fog high up
vec3 airColor(vec3 wp, vec3 fogCol, vec3 sunDir, vec3 sunCol) {
  vec3 vd = normalize(wp - cameraPosition);
  return fogCol + sunCol * 0.35 * pow(max(dot(vd, sunDir), 0.0), 6.0);
}
float airF(vec3 wp) { float f = fogF(wp); return f * mix(1.0, exp(-max(wp.y, 0.0) * 0.015), 0.6); }
`;
const CHUNK_VS = /* glsl */`
attribute vec3 aColor; attribute vec2 aExtra; attribute vec2 aUv;
uniform float uTime, uSway, uFar, uSink;
varying vec3 vColor; varying vec3 vNormal; varying vec3 vWorld; varying float vKind; varying float vLift; varying vec2 vUv;
varying float vM1;
float h12v(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float vnv(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(h12v(i), h12v(i + vec2(1.0, 0.0)), f.x), mix(h12v(i + vec2(0.0, 1.0)), h12v(i + vec2(1.0, 1.0)), f.x), f.y); }
void main() {
  vec3 p = position;
  vUv = aUv;
  vM1 = 0.0;
  if (aExtra.x < 0.5) {   // ground: the slow (22-unit) patchiness is smooth enough to work out per vertex, not per pixel
    vec2 q = (modelMatrix * vec4(p, 1.0)).xz * 0.045;
    vM1 = (vnv(q) * 0.5 + vnv(q * 2.03 + vec2(17.1, 9.2)) * 0.25 + vnv(q * 4.1209 + vec2(51.8, 28.0)) * 0.125) / 0.875;
  }
  if (abs(aExtra.x - 1.0) < 0.5 || abs(aExtra.x - 6.0) < 0.5) {
    float lift = aExtra.y;
    p.x += sin(uTime * 1.6 + p.x * 0.37 + p.z * 0.29) * uSway * lift * 0.05;
    p.z += cos(uTime * 1.3 + p.x * 0.23) * uSway * lift * 0.035;
  }
  vLift = aExtra.y;
  if (abs(aExtra.x - 5.0) < 0.5) {          // grass blade: shrinks to nothing far away, bends with the wind
    float lift = aExtra.y;
    float fade = 1.0 - smoothstep(38.0, 62.0, length(p.xz - cameraPosition.xz));
    p.y -= lift * (1.0 - fade);
    float wv = sin(uTime * 2.1 + p.x * 0.42 + p.z * 0.31) * 0.65 + sin(uTime * 3.3 + p.x * 1.7 - p.z * 0.9) * 0.25;
    p.x += wv * uSway * lift * lift * 0.32;
    p.z += wv * uSway * lift * lift * 0.18;
    vLift = lift * fade;
  }
  vec4 w = modelMatrix * vec4(p, 1.0);
  // at sea, distant islands sink below the horizon (like the curve of the Earth) before the loaded world ends
  w.y -= uSink * 40.0 * smoothstep(uFar * 0.62, uFar, length(w.xz - cameraPosition.xz));
  vWorld = w.xyz; vNormal = normal; vColor = aColor; vKind = aExtra.x;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const CHUNK_FS = /* glsl */`
uniform vec3 uSunDir, uSunColor, uAmbSky, uAmbGround, uAccent;
uniform float uGlow, uBloom, uBlock, uToon, uNeon, uSnow, uWaterY;
uniform sampler2D tGrass, tGrassN, tRock, tRockN, tDirt, tSand, tBark, tLeaves;
varying vec3 vColor; varying vec3 vNormal; varying vec3 vWorld; varying float vKind; varying float vLift; varying vec2 vUv;
varying float vM1;
${GRADE}
// photo texture sampled at two scales with a rotated second layer, blended by noise, so tiling doesn't show
vec3 tex2(sampler2D t, vec2 q, float s) {
  vec2 a = q * s, b = mat2(0.8, -0.6, 0.6, 0.8) * q * s * 0.37 + 17.3;
  float m = smoothstep(0.3, 0.7, vn(q * 0.07));
  return mix(texture2D(t, a).rgb, texture2D(t, b).rgb, 0.35 + 0.3 * m);
}
vec3 nrm2(sampler2D t, vec2 q, float s) { return texture2D(t, q * s).rgb * 2.0 - 1.0; }
float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 naturalShade(vec3 n, float kind, out float alpha) {
  alpha = 1.0;
  vec3 base = vColor;
  vec2 q = vWorld.xz;
  if (kind == 0.0) {
    // ground from photographs: grass, dry dirt patches, sand by the water, rock on slopes, snow on peaks;
    // the theme's palette (vColor) tints the grass so every world keeps its own colours
    // only the layers that show at this point are sampled (most of the ground is grass alone)
    float m1 = vM1, m2 = vn(q * 0.6);
    float slope = 1.0 - n.y;
    float wr = smoothstep(0.2, 0.36, slope + (m2 - 0.5) * 0.12);
    float wd = smoothstep(0.58, 0.78, m1 + 0.25 * (m2 - 0.5)) * (1.0 - wr);
    float ws = (1.0 - smoothstep(uWaterY + 0.4, uWaterY + 1.4, vWorld.y + (m2 - 0.5) * 0.6)) * (1.0 - wr);
    vec3 grass = tex2(tGrass, q, 0.16);
    vec3 tint = vColor / max(lum(vColor), 0.02);
    base = grass * mix(vec3(1.0), clamp(tint * lum(grass) / max(lum(grass), 1e-3), 0.0, 2.0), 0.45);
    vec3 tn = nrm2(tGrassN, q, 0.16);
    if (wd > 0.003) base = mix(base, texture2D(tDirt, q * 0.12).rgb, wd);
    if (ws > 0.003) base = mix(base, texture2D(tSand, q * 0.14).rgb, ws);
    if (wr > 0.003) {
      vec2 rq = vec2(q.x + q.y, vWorld.y);
      base = mix(base, tex2(tRock, rq, 0.11) * vec3(0.95, 0.93, 0.9), wr);
      tn = mix(tn, nrm2(tRockN, rq, 0.11), wr);
    }
    n = normalize(n + vec3(tn.x, 0.0, tn.y) * mix(0.35, 0.8, wr));
    float snow = smoothstep(uSnow - 1.5, uSnow + 1.5, vWorld.y + (m1 - 0.5) * 4.0) * smoothstep(0.55, 0.25, slope);
    base = mix(base, vec3(0.82, 0.85, 0.9), snow);
    base *= 0.85 + 0.3 * m1;
  } else if (kind == 5.0) {
    base *= mix(0.35, 1.15, clamp(vLift / 0.9, 0.0, 1.0));
    n = normalize(mix(vec3(0.0, 1.0, 0.0), n, 0.3));
  } else if (kind == 6.0) {                    // leaf card: drawn leaves, coloured by the foliage (and its mood)
    vec4 lf = texture2D(tLeaves, vUv);
    if (lf.a < 0.12) discard;
    alpha = smoothstep(0.2, 0.6, lf.a);
    base = vColor / max(lum(vColor), 0.02) * lum(lf.rgb) * 0.5;
  } else if (kind == 1.0) {                    // inner crown volume and conifer needles
    vec3 w = vWorld * 3.3;
    float leaf = vn(w.xz + w.y * 0.7) * 0.6 + vn(w.zy * 1.9 + 3.1) * 0.4;
    base *= 0.55 + 0.75 * leaf;
    n = normalize(n + (vec3(vn(w.xy), vn(w.yz + 1.7), vn(w.zx + 5.3)) - 0.5) * 0.9);
  } else if (kind == 7.0) {                    // bark: vertical texture around the trunk
    vec2 uv = vec2((vWorld.x * n.z - vWorld.z * n.x) * 1.4, vWorld.y * 0.45);
    base = texture2D(tBark, uv).rgb * (vColor / max(lum(vColor), 0.02)) * 0.35;
  } else if (kind == 8.0) {                    // stones: rock photograph, triplanar
    vec3 an = abs(n);
    vec3 r = texture2D(tRock, vWorld.zy * 0.35).rgb * an.x + texture2D(tRock, vWorld.xz * 0.35).rgb * an.y + texture2D(tRock, vWorld.xy * 0.35).rgb * an.z;
    base = r / (an.x + an.y + an.z) * (vColor / max(lum(vColor), 0.02)) * 1.05;
  } else {
    base *= 0.9 + 0.2 * vn(vWorld.xz * 2.7 + vWorld.y * 1.3);
  }
  float ndl = dot(n, uSunDir);
  float diff = kind == 6.0 ? clamp(ndl * 0.6 + 0.4, 0.0, 1.0) : max(ndl, 0.0);
  vec3 amb = mix(uAmbGround, uAmbSky, n.y * 0.5 + 0.5);
  float ao = kind == 0.0 ? 1.0 : mix(0.55, 1.0, clamp(vLift / 3.0, 0.0, 1.0));
  vec3 col = base * (amb * ao + uSunColor * diff * 1.15);
  if (kind == 5.0 || kind == 1.0 || kind == 6.0) {   // light through leaves and blades when looking toward the sun
    vec3 vd = normalize(vWorld - cameraPosition);
    col += base * uSunColor * 0.6 * pow(max(dot(vd, uSunDir), 0.0), 4.0);
  }
  return col;
}
void main() {
  vec3 n = normalize(vNormal);
  float kind = floor(vKind + 0.5);
  if (uNatural > 0.5 && kind != 2.0 && kind != 3.0) {
    float alpha;
    vec3 c = naturalShade(n, kind, alpha);
    c = mix(c, airColor(vWorld, uFogColor, uSunDir, uSunColor), max(airF(vWorld), farF(vWorld)));
    gl_FragColor = vec4(aces(grade(c) * 1.05), alpha);
    #include <colorspace_fragment>
    return;
  }
  vec3 base = vColor;
  if (kind == 3.0) { float l = dot(base, vec3(0.2126, 0.7152, 0.0722)); base = max(mix(vec3(l), base, uBloom), 0.0); }
  float ndl = dot(n, uSunDir);
  float diff = max(ndl, 0.0);
  if (uToon > 0.5) { float w = clamp((ndl + 0.35) / 1.35, 0.0, 1.0); diff = mix(w, floor(w * 3.0 + 0.5) / 3.0, 0.5); }
  vec3 amb = mix(uAmbGround, uAmbSky, n.y * 0.5 + 0.5);
  vec3 col = base * (amb + uSunColor * diff);
  if (uBlock > 0.5) {
    vec3 q = vWorld - n * 0.02;
    col *= (0.93 + 0.14 * h13(floor(q))) * (0.94 + 0.12 * h13(floor(q * 8.0) + 0.37));
    if (kind == 0.0) {
      vec3 d = 0.5 - abs(fract(vWorld) - 0.5) + abs(n);
      float e = min(min(d.x, d.y), d.z);
      col *= 1.0 - 0.22 * (1.0 - smoothstep(0.0, 0.02 + fwidth(e) * 1.5, e));
    }
  }
  if (kind == 2.0) col = mix(col, base * 1.4, clamp(0.45 + 0.55 * uGlow, 0.0, 1.0));
  if (uNeon > 0.5) {
    vec2 gp = vWorld.xz * 0.25;
    vec2 g = abs(fract(gp - 0.5) - 0.5) / max(fwidth(gp), vec2(1e-4));
    float near = 1.0 - smoothstep(30.0, 90.0, length(vWorld - cameraPosition));
    float line = (1.0 - min(min(g.x, g.y), 1.0)) * (0.35 + 0.65 * near);
    float cy = vWorld.y * 0.25;
    float cl = (1.0 - min(abs(fract(cy - 0.5) - 0.5) / max(fwidth(cy), 1e-4), 1.0)) * 0.4 * near;
    float rim = pow(1.0 - abs(dot(n, normalize(cameraPosition - vWorld))), 3.0);
    col = col * 0.6 + uAccent * (max(line, cl) * 0.95 + rim * 0.3);
  }
  gl_FragColor = vec4(grade(mix(col, uFogColor, max(fogF(vWorld), farF(vWorld)))), 1.0);
  #include <colorspace_fragment>
}`;
const WATER_VS = /* glsl */`
varying vec3 vWorld;
void main() { vec4 w = modelMatrix * vec4(position, 1.0); vWorld = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }`;
const WATER_FS = /* glsl */`
uniform vec3 uWater, uSky, uSunDir, uSunColor, uAccent;
uniform float uFoam;
uniform float uWaterTime, uWave, uNeon;
varying vec3 vWorld;
${GRADE}
uniform vec3 uTop, uHorizon;
void main() {
  vec2 p = vWorld.xz; float t = uWaterTime;
  if (uNatural > 0.5) {
    vec2 d = vec2(0.0);
    float dist = length(vWorld - cameraPosition);
    for (int i = 0; i < 4; i++) {               // sum of travelling waves in four directions, longest first
      float fi = float(i), a = fi * 2.39996 + 0.4, k = 0.18 * pow(1.7, fi), amp = 0.5 / pow(1.9, fi);
      vec2 dir = vec2(cos(a), sin(a));
      d += dir * cos(dot(dir, p) * k + t * (1.2 + 0.35 * fi)) * k * amp;
    }
    d += (vec2(vn(p * 1.7 + t * 0.6), vn(p * 1.7 - t * 0.5 + 9.0)) - 0.5) * 0.35;
    d *= 1.0 - 0.85 * smoothstep(40.0, 320.0, dist);   // calm the far sea: small ripples there only shimmer (aliasing)
    vec3 n = normalize(vec3(-d.x * uWave, 1.0, -d.y * uWave));
    vec3 v = normalize(cameraPosition - vWorld);
    vec3 r = reflect(-v, n);
    vec3 refl = mix(mix(uHorizon, uTop, 0.45), uTop, pow(clamp(r.y, 0.0, 1.0), 0.5));
    float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    float spec = pow(max(dot(r, uSunDir), 0.0), 220.0) * 3.0 + pow(max(dot(r, uSunDir), 0.0), 24.0) * 0.12;
    vec3 col = mix(uWater * 0.22, refl, fres) + uSunColor * spec;
    if (uFoam > 0.01) {
      float crest = smoothstep(0.62, 0.9, vn(p * 0.9 + vec2(t * 0.7, -t * 0.4)) * 0.6 + length(d) * 0.9);
      col += vec3(0.85) * crest * uFoam * 0.35 * (1.0 - fres);
    }      // whitecaps: stormy (aroused, unpleasant) seas
    col = mix(col, airColor(vWorld, uFogColor, uSunDir, uSunColor), airF(vWorld));
    gl_FragColor = vec4(aces(grade(col) * 1.05), 0.94);
    #include <colorspace_fragment>
    return;
  }
  vec2 g = vec2(cos(p.x * 0.35 + t * 1.1) * 0.6 + cos((p.x + p.y) * 0.21 - t * 0.8) * 0.4,
                sin(p.y * 0.31 + t * 0.9) * 0.6 + cos((p.x - p.y) * 0.17 + t * 0.7) * 0.4) * uWave;
  vec3 n = normalize(vec3(-g.x * 0.12, 1.0, -g.y * 0.12));
  vec3 v = normalize(cameraPosition - vWorld);
  float fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  float spec = pow(max(dot(reflect(-v, n), uSunDir), 0.0), 80.0);
  vec3 col = mix(uWater, uSky, 0.15 + 0.6 * fres) + uSunColor * spec * 0.5;
  if (uNeon > 0.5) {
    vec2 gp = p * 0.25;
    vec2 gg = abs(fract(gp - 0.5) - 0.5) / max(fwidth(gp), vec2(1e-4));
    col = col * 0.3 + uAccent * (1.0 - min(min(gg.x, gg.y), 1.0)) * 0.8;
  }
  gl_FragColor = vec4(grade(mix(col, uFogColor, fogF(vWorld))), 0.86);
  #include <colorspace_fragment>
}`;
const SKY_VS = /* glsl */`
varying vec3 vDir;
void main() { vDir = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const SKY_FS = /* glsl */`
uniform vec3 uTop, uHorizon, uSunDir, uSunColor, uAccent;
uniform float uNight, uNeon, uCloud, uCloudT, uStorm, uSkyRot, uSkyOn;
uniform sampler2D tSky;
varying vec3 vDir;
${GRADE}
vec3 naturalSky(vec3 d) {
  float h = d.y, mu = max(dot(d, uSunDir), 0.0);
  vec3 col = mix(uHorizon, uTop, pow(clamp(h, 0.0, 1.0), 0.45));
  if (uSkyOn > 0.5) {                             // a photographed sky (image library), turned to match the sun
    float u = fract(atan(d.z, d.x) / 6.2831853 + 0.5 + uSkyRot);
    float v = clamp(asin(clamp(h, 0.0, 1.0)) / 1.5707963, 0.004, 0.996);
    // the wrap of u (0 -> 1) would make the GPU pick the smallest mip there and draw a seam line: take the
    // gradient from a copy of u that is continuous across the wrap
    vec2 uv = vec2(u, v), uw = vec2(fract(u + 0.5) - 0.5, v);
    vec2 gx = dFdx(uv), gy = dFdy(uv), gxw = dFdx(uw), gyw = dFdy(uw);
    if (abs(gxw.x) + abs(gyw.x) < abs(gx.x) + abs(gy.x)) { gx = gxw; gy = gyw; }
    vec3 photo = textureGrad(tSky, uv, gx, gy).rgb;
    col = mix(uHorizon, photo, smoothstep(0.015, 0.09, h));
  } else col += uSunColor * (0.22 * pow(mu, 6.0) + 0.5 * pow(mu, 48.0));
  col = mix(col, uHorizon * 0.8, smoothstep(0.0, -0.25, h));
  col += uSunColor * 6.0 * smoothstep(0.9993, 0.9997, mu) * (1.0 - uSkyOn);
  float extra = uSkyOn > 0.5 ? clamp((uCloud - 0.22) / 0.73, 0.0, 1.0) : uCloud;   // the photo has its own clouds
  if (h > 0.0 && extra > 0.001) {                 // clouds the brain adds: a layer of noise projected onto the sky
    vec2 uv = d.xz / (h + 0.18) * 0.55 + vec2(uCloudT, uCloudT * 0.37);
    float c = fbm4(uv) * 0.85 + vn(uv * 5.4 + 4.2) * 0.15;
    float cov = clamp(extra, 0.0, 1.0);
    float m = smoothstep(1.0 - cov - 0.08, 1.0 - cov + 0.32, c) * smoothstep(0.0, 0.18, h);
    float thick = smoothstep(0.45, 0.95, c);
    vec3 lit = mix(vec3(1.0), uHorizon, 0.25) * (0.75 + 0.25 * uSunDir.y) + uSunColor * 0.35 * pow(mu, 3.0);
    vec3 cloud = mix(lit, lit * mix(0.55, 0.28, uStorm), thick);
    col = mix(col, cloud, m * 0.92);
  }
  return col;
}
void main() {
  vec3 d = normalize(vDir);
  float h = d.y;
  if (uNatural > 0.5) {
    vec3 c = naturalSky(d);
    if (uNight > 0.0) { vec3 cc = floor(d * 220.0); c += vec3(step(0.9975, h13(cc)) * smoothstep(0.0, 0.15, h) * uNight * (1.0 - uCloud)); }
    gl_FragColor = vec4(aces(grade(c) * 1.05), 1.0);
    #include <colorspace_fragment>
    return;
  }
  vec3 col = h < 0.0 ? uHorizon : mix(uHorizon, uTop, pow(clamp(h * 1.3 + 0.05, 0.0, 1.0), 0.7));
  float sd = max(dot(d, uSunDir), 0.0);
  col += uSunColor * (pow(sd, 600.0) * 2.0 + pow(sd, 12.0) * 0.25);
  if (uNeon > 0.5 && sd > 0.985 && d.y > uSunDir.y - 0.18) {
    float k = clamp((d.y - uSunDir.y + 0.18) / 0.36, 0.0, 1.0);
    float stripe = d.y < uSunDir.y ? step(0.35, fract((uSunDir.y - d.y) * 55.0)) : 1.0;
    col = mix(col, mix(uAccent, vec3(1.0, 0.85, 0.35), k), stripe);
  }
  if (uNight > 0.0) {
    vec3 c = floor(d * 220.0);
    float star = step(0.9975, h13(c)) * smoothstep(0.0, 0.15, h);
    col += vec3(star * uNight * (0.6 + 0.4 * h13(c + 1.7)));
  }
  gl_FragColor = vec4(grade(col), 1.0);
  #include <colorspace_fragment>
}`;

// ---------------------------------------------------------------- time of day and weather
type Tod = { sun: THREE.Vector3; sunColor: string; sunI: number; amb: number; sky: [string, number]; hor: [string, number]; night: number; glow: number };
const TOD: Record<WorldSpec["time"], Tod> = {
  dawn: { sun: new THREE.Vector3(0.8, 0.18, 0.3), sunColor: "#ffd9b8", sunI: 0.9, amb: 0.5, sky: ["#6f78b8", 0.25], hor: ["#ffc6a8", 0.45], night: 0, glow: 0.4 },
  day: { sun: new THREE.Vector3(0.4, 0.85, 0.3), sunColor: "#fff4e0", sunI: 1.1, amb: 0.55, sky: ["#000000", 0], hor: ["#000000", 0], night: 0, glow: 0.2 },
  sunset: { sun: new THREE.Vector3(-0.8, 0.13, 0.35), sunColor: "#ffb27a", sunI: 0.9, amb: 0.45, sky: ["#5b5a9a", 0.3], hor: ["#ff9a5a", 0.55], night: 0, glow: 0.5 },
  night: { sun: new THREE.Vector3(0.3, 0.6, -0.5), sunColor: "#8ea6d8", sunI: 0.35, amb: 0.3, sky: ["#070b1a", 0.85], hor: ["#1b2340", 0.75], night: 1, glow: 1 },
};
Object.values(TOD).forEach(t => t.sun.normalize());

type WType = Exclude<Weather, "mood">;
const WX: Record<WType, { color: string; size: number; vy: number; drift: number; add: boolean; opacity: number }> = {
  none: { color: "#ffffff", size: 0.3, vy: 0, drift: 0, add: false, opacity: 0 },
  stars: { color: "#ffffff", size: 0.3, vy: 0, drift: 0, add: false, opacity: 0 },
  rain: { color: "#a9c4e0", size: 0.18, vy: -24, drift: 0.1, add: false, opacity: 0.55 },
  snow: { color: "#ffffff", size: 0.32, vy: -2.4, drift: 0.7, add: false, opacity: 0.9 },
  fireflies: { color: "#ffe680", size: 0.35, vy: 0, drift: 1, add: true, opacity: 0.95 },
  petals: { color: "#ffb3cf", size: 0.3, vy: -1.4, drift: 1.4, add: false, opacity: 0.9 },
  embers: { color: "#ff8a3d", size: 0.26, vy: 2.6, drift: 0.7, add: true, opacity: 0.9 },
};
/** Weather for a calibrated personal state (main.ts passes the dominant one). */
const STATE_WEATHER: Record<string, WType> = { calm: "fireflies", stress: "rain", anxiety: "none", anger: "embers", low_mood: "snow", drowsy: "stars" };
/** Weather the decoded state asks for, by quadrant of the affect circle. */
export function moodWeather(v: number, a: number): WType {
  if (v >= 0.2) return a >= 0.1 ? "petals" : "fireflies";
  if (v <= -0.2) return a >= 0.1 ? "rain" : "snow";
  return "none";
}

const MAP_SPAN = 320;     // world units across the chart

/** The six personal states each turn the voyage into a chapter of its own: a realm with its own sky, sea and weather. */
export type RealmName = "calm" | "stress" | "anxiety" | "anger" | "low_mood" | "drowsy";
type Realm = { title: string; line: string; culture: Culture; sky: string; hor: string; tint: [number, number, number]; sat: number; bright: number;
  fog: number; cloud: number; storm: number; foam: number; water: string; night: number; weather: WType };
export const REALMS: Record<RealmName, Realm> = {
  calm: { title: "Hawaiki", line: "Calm. Clear water and a steady wind.", culture: "polynesian", sky: "#76b2e6", hor: "#ffe0ad", tint: [1.05, 1, 0.92], sat: 1.12, bright: 1.05, fog: 0.8, cloud: 0.12, storm: 0, foam: 0, water: "#1fb5c0", night: 0, weather: "fireflies" },
  stress: { title: "The Strait of Scylla", line: "Stress. Rough water between the rocks.", culture: "greek", sky: "#3b4552", hor: "#878d93", tint: [0.92, 0.96, 1.05], sat: 0.75, bright: 0.85, fog: 1.6, cloud: 0.92, storm: 0.85, foam: 1, water: "#3e4d57", night: 0, weather: "rain" },
  anxiety: { title: "The Mists of Avalon", line: "Anxiety. Fog hides the way.", culture: "celtic", sky: "#8d969c", hor: "#c9cdd0", tint: [0.97, 0.98, 1], sat: 0.6, bright: 0.95, fog: 4.5, cloud: 0.6, storm: 0.1, foam: 0.2, water: "#5a6b73", night: 0, weather: "none" },
  anger: { title: "Muspelheim", line: "Anger. Fire on the water.", culture: "norse", sky: "#5a1d12", hor: "#ff7a3d", tint: [1.15, 0.88, 0.78], sat: 1.15, bright: 0.9, fog: 1.3, cloud: 0.55, storm: 0.4, foam: 0.6, water: "#5a2a22", night: 0, weather: "embers" },
  low_mood: { title: "The Island of the Ka", line: "Low mood. Grey sky, still water.", culture: "egyptian", sky: "#4a5260", hor: "#8d949c", tint: [0.9, 0.94, 1.04], sat: 0.45, bright: 0.8, fog: 1.8, cloud: 0.75, storm: 0.1, foam: 0.1, water: "#3b4a55", night: 0, weather: "snow" },
  drowsy: { title: "Ryūgū-jō", line: "Drowsy. Night, and the sea goes quiet.", culture: "japanese", sky: "#0d1430", hor: "#2b3560", tint: [0.85, 0.9, 1.1], sat: 0.8, bright: 0.62, fog: 1.2, cloud: 0.08, storm: 0, foam: 0, water: "#0f2440", night: 1, weather: "stars" },
};
/** Without a calibrated personal-state model, the realm comes from the affect map. */
export function realmFromVA(v: number, a: number): RealmName | null {
  if (a <= -0.55) return "drowsy";
  if (v >= 0.15 && a < 0.45) return "calm";
  if (v <= -0.2) return a >= 0.55 ? "anger" : a >= 0.1 ? "stress" : "low_mood";
  if (a >= 0.45 && v < 0.15) return "anxiety";
  return null;
}

// ---------------------------------------------------------------- image library (app/assets, embedded at build)
function blankTexture(): THREE.Texture {
  const t = new THREE.DataTexture(new Uint8Array([200, 200, 200, 255]), 1, 1);
  t.needsUpdate = true;
  return t;
}
function flatNormal(): THREE.Texture {
  const t = new THREE.DataTexture(new Uint8Array([128, 128, 255, 255]), 1, 1);
  t.needsUpdate = true;
  return t;
}
type SkyInfo = { tex: THREE.Texture; horizon: THREE.Color; top: THREE.Color; sunU: number; sunV: number };
/** Sky photographs by time of day; a session picks one of the candidates from its seed. */
export const SKIES: Record<WorldSpec["time"], string[]> = {
  day: ["sky_day", "sky_clouds"], dawn: ["sky_dawn"], sunset: ["sky_dawn"], night: ["sky_night"],
};
const assets = (): Record<string, string> => ((globalThis as { __EMOTIF_ASSETS?: Record<string, string> }).__EMOTIF_ASSETS ?? {});

const S = 0.06;           // world units per millimetre for the arm twin
const PEDESTAL = 6;       // pedestal height (world units)
const SHOULDER_MM = 126;  // base plate to shoulder joint (firmware ARM_L1_LENGTH_MM)
const RADIUS = 3;         // chunks loaded around the explorer in each direction (2 on slow GPUs)
const NEAR = 1;           // chunks within this ring get full detail (grass, fine crowns); the rest lod 1
const MAX_JOBS = 4;       // chunk jobs in flight to the worker
const NP = 1600, BOX_W = 70, BOX_H = 34;
const NB = 22;            // seabirds (most when the decoded state is pleasant)
const DISCOVER_R = 40;    // how close the explorer comes to a landmark to discover it
const MAST = 3.6;         // ship: water line to the lantern at the masthead

/** A landmark found on this voyage (the logbook). */
export type Discovery = { name: string; kind: Landmark["kind"]; x: number; z: number; t: number; v: number; a: number };

export class World {
  ok = false;
  renderer!: THREE.WebGLRenderer;
  cur = { v: 0, a: 0 };
  target = { v: 0, a: 0 };
  armPose: Pose = { ...INIT };
  armTarget: Pose | null = null;
  reducedMotion = false;
  autoRotate = true;            // camera swings behind the explorer when nobody is dragging it
  explore = true;               // explorer moves on its own
  paused = false;               // set while an opaque overlay covers the canvas
  private mode: "world" | "arm" = "world";
  private radius = RADIUS;
  private host: HTMLElement | null = null;
  private ro: ResizeObserver | null = null;
  private last = 0;
  private lastRender = 0;
  private time = 0;
  private waterTime = 0;
  // adaptive resolution: keep ~30+ fps on weak GPUs by lowering the render scale
  private maxPR = 1;
  private pr = 1;
  private frameEma = 1 / 60;
  private slowFor = 0;
  private fastFor = 0;
  private dummy = new THREE.Object3D();

  // inner world
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(60, 1, 0.3, 900);
  private controls!: OrbitControls;
  private specV: WorldSpec = validateSpec(DEFAULT_SPEC);
  private terrain = new Terrain(this.specV);
  private chunks = new Map<string, { mesh: THREE.Mesh; features: number; lod: number }>();
  // chunk building runs in a Web Worker when one can be started (chunkworker.ts); otherwise on this thread
  private worker: Worker | null = null;
  private gen = 0;                                    // world generation: replies for an older world are dropped
  private pending = new Map<string, number>();        // chunk key -> lod requested from the worker
  private U = {
    uTime: { value: 0 }, uSway: { value: 0.5 }, uWaterTime: { value: 0 }, uWave: { value: 0.6 },
    uSunDir: { value: new THREE.Vector3(0.4, 0.85, 0.3).normalize() }, uSunColor: { value: new THREE.Color() },
    uAmbSky: { value: new THREE.Color() }, uAmbGround: { value: new THREE.Color() }, uAccent: { value: new THREE.Color() },
    uFogColor: { value: new THREE.Color() }, uFogDensity: { value: 0.02 }, uTint: { value: new THREE.Vector3(1, 1, 1) },
    uSat: { value: 1 }, uBright: { value: 1 }, uGlow: { value: 0.3 }, uBloom: { value: 1 },
    uBlock: { value: 1 }, uToon: { value: 0 }, uNeon: { value: 0 },
    uWater: { value: new THREE.Color() }, uSky: { value: new THREE.Color() },
    uTop: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() }, uNight: { value: 0 },
    uNatural: { value: 1 }, uSnow: { value: 1e4 }, uCloud: { value: 0.3 }, uCloudT: { value: 0 }, uStorm: { value: 0 },
    uSkyRot: { value: 0 }, uSkyOn: { value: 0 }, uWaterY: { value: 0 }, uFoam: { value: 0 }, uFar: { value: 60 }, uSink: { value: 0 },
    tGrass: { value: blankTexture() }, tGrassN: { value: flatNormal() }, tRock: { value: blankTexture() }, tRockN: { value: flatNormal() },
    tDirt: { value: blankTexture() }, tSand: { value: blankTexture() }, tBark: { value: blankTexture() }, tLeaves: { value: blankTexture() },
    tSky: { value: blankTexture() },
  };
  private brainState: string | null = null;            // dominant personal state (states.ts), if any
  private skies = new Map<string, SkyInfo>();
  private skyInfo: SkyInfo | null = null;
  libraryLoaded = 0;
  private chunkMood = new Map<string, ChunkMood>();   // state when each chunk was first generated: the world remembers
  private cloudT = 0;
  private chunkMat!: THREE.ShaderMaterial;
  private water!: THREE.Mesh;
  private sky!: THREE.Mesh;
  private orb!: THREE.Mesh;
  private halo!: THREE.Sprite;
  // brain-controlled objects (interact.ts): crystals near the path that focus lifts and relaxing sets down
  private dotTex!: THREE.Texture;
  private carried: Landmark | null = null;
  private aim: Landmark | null = null;
  private ring!: THREE.Mesh;
  private lit = new Set<string>();
  private beacons2 = new Map<string, THREE.Sprite>();
  private spark!: THREE.Sprite;
  private realmName: RealmName | null = null;
  private realmPending: RealmName | null = null;
  private realmPendingFor = 0;
  private realmW = 0;
  private chapters: { realm: RealmName; title: string; t: number }[] = [];
  onChapter: ((realm: RealmName, title: string, line: string) => void) | null = null;
  private mapCanvas: HTMLCanvasElement | null = null;
  private mapLand: HTMLCanvasElement | null = null;
  private mapCx = 1e9;
  private mapCz = 1e9;
  private mapTick = 0;
  private camFloor = 0;
  private mapJob: { cx: number; cz: number; row: number; img: ImageData } | null = null;
  private control: { command: "idle" | "focus" | "relax"; level: number } = { command: "idle", level: 0 };
  interactive = true;
  onInteract: ((type: "dodge" | "hit" | "calm" | "lift" | "place", info: { x: number; z: number; t: number; placed: number }) => void) | null = null;
  get placedCount(): number { return this.lit.size; }
  get carrying(): boolean { return !!this.carried; }
  get aiming(): boolean { return !!this.hazardNow; }   // rocks ahead: focus to steer clear
  // voyage: in sea worlds the explorer is a spirit ship whose lantern is the glowing orb; it sails to landmarks
  private ship!: THREE.Group;
  private sail!: THREE.Mesh;
  private shipMode = false;
  private hemi!: THREE.HemisphereLight;
  private sunLight!: THREE.DirectionalLight;
  private birds!: THREE.InstancedMesh;
  private found = new Map<string, Discovery>();
  private log: Discovery[] = [];
  private goal: Landmark | null = null;
  private goalSince = 0;
  private skip = new Set<string>();
  private labels = new Map<string, THREE.Sprite>();
  private toast: HTMLDivElement | null = null;
  private toastTimer = 0;
  private stuck = 0;
  onDiscover: ((d: Discovery, total: number) => void) | null = null;
  /** Landmarks discovered in this world, in order. */
  get discoveries(): Discovery[] { return this.log.slice(); }
  /** The landmark the explorer is heading for, if any. */
  get goalName(): string | null { return this.goal?.name ?? null; }
  get sailing(): boolean { return this.shipMode; }
  /** Name of the landmark whose beacon is in reach (or whose flame the lantern carries). */
  get targetName(): string | null { return (this.carried ?? this.aim)?.name ?? null; }
  private beacons!: THREE.InstancedMesh;
  private beaconCount = 0;
  private readonly beaconMax = 1800;
  private points!: THREE.Points;
  private pMat!: THREE.PointsMaterial;
  private pVel = new Float32Array(NP);
  private wType: WType = "none";
  private wOpacity = 0;
  private wPending: WType = "none";
  private wPendingFor = 0;
  private pos = new THREE.Vector3();
  private heading = 0;
  private keys = new Set<string>();
  private lastInteract = -1e9;
  private tmpC = new THREE.Color();

  // arm scene
  private armScene = new THREE.Scene();
  private armCamera = new THREE.PerspectiveCamera(50, 1, 0.5, 600);
  private armControls!: OrbitControls;
  private arm = new THREE.Group();
  private parts: Record<string, THREE.Object3D> = {};
  private paper!: THREE.Mesh;
  private trailLine!: THREE.Line;
  private trailGeom = new THREE.BufferGeometry();
  private targetMarker!: THREE.Mesh;
  private pen!: THREE.Mesh;
  private penLenMm = 66;
  private moodDisc!: THREE.Mesh;

  init(): boolean {
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: (window.devicePixelRatio || 1) < 1.5, powerPreference: "high-performance" });
    } catch {
      return false;
    }
    this.maxPR = Math.min(window.devicePixelRatio || 1, 1.25);   // beyond this the cost grows faster than the gain
    this.pr = Math.min(this.maxPR, 1);         // start modest; adapt() raises it when frames are fast
    this.renderer.setPixelRatio(this.pr);
    this.startWorker();
    const el = this.renderer.domElement;
    el.className = "world-canvas";
    el.tabIndex = 0;
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", "3-D view. In the inner world, arrow keys or W A S D steer the explorer while this view has focus.");
    el.addEventListener("keydown", e => this.onKey(e, true));
    el.addEventListener("keyup", e => this.onKey(e, false));
    el.addEventListener("blur", () => this.keys.clear());

    this.controls = new OrbitControls(this.camera, el);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 90;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.controls.enablePan = false;
    this.controls.addEventListener("start", () => { this.lastInteract = this.time; });
    this.buildInnerWorld();

    this.armControls = new OrbitControls(this.armCamera, el);
    this.armControls.enableDamping = true;
    this.armControls.dampingFactor = 0.08;
    this.armControls.minDistance = 12;
    this.armControls.maxDistance = 160;
    this.armControls.maxPolarAngle = Math.PI * 0.49;
    this.armControls.enabled = false;
    this.buildArmScene();

    this.setSpec(this.specV);
    this.ok = true;
    this.last = performance.now();
    this.renderer.setAnimationLoop(t => this.frame(t));
    return true;
  }

  // ---------------------------------------------------------------- inner world
  /** Load the image library: ground photographs (repeating), leaves, bark, and the sky photographs (analysed for
   * their horizon and zenith colours and where the sun is). Missing images leave neutral placeholders. */
  private loadLibrary() {
    const A = assets(), U = this.U as unknown as Record<string, { value: THREE.Texture }>;
    const loader = new THREE.TextureLoader();
    const aniso = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    const tex: [string, string, boolean][] = [["tGrass", "grass", true], ["tGrassN", "grass_n", false], ["tRock", "rock", true], ["tRockN", "rock_n", false],
      ["tDirt", "dirt", true], ["tSand", "sand", true], ["tBark", "bark", true], ["tLeaves", "leaves", true]];
    for (const [u, name, color] of tex) {
      if (!A[name]) continue;
      loader.load(A[name], t => {
        t.wrapS = t.wrapT = THREE.RepeatWrapping;
        t.colorSpace = color ? THREE.SRGBColorSpace : THREE.NoColorSpace;
        t.anisotropy = aniso;
        t.needsUpdate = true;
        U[u].value = t;
        this.libraryLoaded++;
      });
    }
    for (const name of new Set(Object.values(SKIES).flat())) {
      if (!A[name]) continue;
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = 256; c.height = 64;
        const g = c.getContext("2d", { willReadFrequently: true });
        if (!g) return;
        g.drawImage(img, 0, 0, 256, 64);
        const d = g.getImageData(0, 0, 256, 64).data;
        const avg = (y0: number, y1: number) => {
          let r = 0, gg = 0, b = 0, n = 0;
          for (let y = y0; y < y1; y++) for (let x = 0; x < 256; x++) { const i = (y * 256 + x) * 4; r += d[i]; gg += d[i + 1]; b += d[i + 2]; n++; }
          return new THREE.Color().setRGB(r / n / 255, gg / n / 255, b / n / 255, THREE.SRGBColorSpace);
        };
        let best = -1, bu = 0.5, bv = 0.3;
        for (let y = 0; y < 60; y++) for (let x = 0; x < 256; x++) {
          const i = (y * 256 + x) * 4, l = d[i] + d[i + 1] + d[i + 2];
          if (l > best) { best = l; bu = (x + 0.5) / 256; bv = 1 - (y + 0.5) / 64; }
        }
        const t = new THREE.Texture(img);
        t.colorSpace = THREE.SRGBColorSpace;
        t.wrapS = THREE.RepeatWrapping;
        t.needsUpdate = true;
        this.skies.set(name, { tex: t, horizon: avg(52, 62), top: avg(0, 20), sunU: bu, sunV: bv });
        this.applySky();
      };
      img.src = A[name];
    }
  }

  /** The sky photograph for this world: one of the candidates for its time of day, chosen by its seed. */
  private applySky() {
    const list = SKIES[this.specV.time], name = list[this.specV.seed % list.length];
    const sky = this.specV.style === "natural" ? this.skies.get(name) : undefined;
    this.skyInfo = sky ?? null;
    this.U.uSkyOn.value = sky ? 1 : 0;
    if (sky) this.U.tSky.value = sky.tex;
  }

  private buildInnerWorld() {
    const U = this.U;
    this.loadLibrary();
    this.chunkMat = new THREE.ShaderMaterial({
      uniforms: U, vertexShader: CHUNK_VS, fragmentShader: CHUNK_FS, side: THREE.DoubleSide,
      alphaToCoverage: true,          // soft leaf edges with the renderer's multisampling (alpha is 1 everywhere else)
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(500, 32, 16), new THREE.ShaderMaterial({
      uniforms: U, vertexShader: SKY_VS, fragmentShader: SKY_FS, side: THREE.BackSide, depthWrite: false,
    }));
    this.sky.frustumCulled = false;
    this.sky.renderOrder = -1;
    this.scene.add(this.sky);
    const span = 1800;   // reaches the horizon (camera far plane 900), so the sea never shows an edge
    const wg = new THREE.PlaneGeometry(span, span);
    wg.rotateX(-Math.PI / 2);
    this.water = new THREE.Mesh(wg, new THREE.ShaderMaterial({
      uniforms: U, vertexShader: WATER_VS, fragmentShader: WATER_FS, transparent: true,
    }));
    this.water.frustumCulled = false;
    this.scene.add(this.water);

    const dot = document.createElement("canvas");
    dot.width = dot.height = 64;
    const dg = dot.getContext("2d")!;
    const grad = dg.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(255,255,255,1)"); grad.addColorStop(0.35, "rgba(255,255,255,0.5)"); grad.addColorStop(1, "rgba(255,255,255,0)");
    dg.fillStyle = grad; dg.fillRect(0, 0, 64, 64);
    const dotTex = new THREE.CanvasTexture(dot);
    this.dotTex = dotTex;

    // explorer: a soft light with a halo
    this.orb = new THREE.Mesh(new THREE.SphereGeometry(0.7, 24, 16), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    this.halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTex, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.halo.scale.set(5, 5, 1);
    this.orb.add(this.halo);
    this.scene.add(this.orb);
    this.buildShip();
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1);
    this.sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
    this.scene.add(this.hemi, this.sunLight, this.sunLight.target);
    const wing = new THREE.BufferGeometry();
    wing.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0.3, 0, 0, -0.35, 0, 0, -0.1, 0.05, 1.25]), 3));
    this.birds = new THREE.InstancedMesh(wing, new THREE.MeshBasicMaterial({ color: 0x2b3038, side: THREE.DoubleSide }), NB * 2);
    this.birds.frustumCulled = false;
    this.birds.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.birds.count = 0;
    this.scene.add(this.birds);
    this.ring = new THREE.Mesh(new THREE.TorusGeometry(1.6, 0.07, 8, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7, depthWrite: false }));
    this.ring.visible = false;
    this.spark = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTex, color: new THREE.Color("#ffc46b"), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    this.spark.scale.set(4, 4, 1);
    this.spark.visible = false;
    this.scene.add(this.spark);
    this.scene.add(this.ring);

    // beacons: one per second of session, left along the explorer's path
    this.beacons = new THREE.InstancedMesh(new THREE.SphereGeometry(0.32, 10, 8), new THREE.MeshBasicMaterial(), this.beaconMax);
    this.beacons.count = 0;
    this.beacons.frustumCulled = false;
    this.beacons.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.scene.add(this.beacons);

    // weather
    const pg = new THREE.BufferGeometry(), pp = new Float32Array(NP * 3);
    for (let i = 0; i < NP; i++) {
      pp[i * 3] = (Math.random() - 0.5) * BOX_W; pp[i * 3 + 1] = (Math.random() - 0.5) * BOX_H; pp[i * 3 + 2] = (Math.random() - 0.5) * BOX_W;
      this.pVel[i] = 0.7 + 0.6 * Math.random();
    }
    pg.setAttribute("position", new THREE.BufferAttribute(pp, 3));
    this.pMat = new THREE.PointsMaterial({ size: 0.3, map: dotTex, transparent: true, opacity: 0, depthWrite: false });
    this.points = new THREE.Points(pg, this.pMat);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this.scene.fog = new THREE.FogExp2(0xffffff, 0.02);
  }

  /** The spirit ship: a pointed hull, a mast and a billowing sail; the explorer's light is its masthead lantern. */
  private buildShip() {
    const g = new THREE.Group();
    const sh = new THREE.Shape();
    sh.moveTo(-2.5, -0.95); sh.lineTo(1.1, -1.1); sh.quadraticCurveTo(2.6, -0.95, 3.4, 0); sh.quadraticCurveTo(2.6, 0.95, 1.1, 1.1);
    sh.lineTo(-2.5, 0.95); sh.quadraticCurveTo(-2.95, 0, -2.5, -0.95);
    const hullG = new THREE.ExtrudeGeometry(sh, { depth: 0.9, bevelEnabled: true, bevelThickness: 0.35, bevelSize: 0.3, bevelSegments: 3, curveSegments: 10 });
    hullG.rotateX(-Math.PI / 2);
    hullG.translate(0, -0.75, 0);
    const wood = new THREE.MeshLambertMaterial({ color: 0x7a5434, emissive: 0x1e140c });
    const hull = new THREE.Mesh(hullG, wood);
    const deckG = new THREE.ShapeGeometry(sh, 10);
    deckG.rotateX(-Math.PI / 2);
    deckG.translate(0, 0.17, 0);
    const deck = new THREE.Mesh(deckG, new THREE.MeshLambertMaterial({ color: 0xb08a5c }));
    deck.scale.set(0.92, 1, 0.88);
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.1, MAST + 0.4, 8), wood);
    mast.position.set(0.3, MAST / 2, 0);
    const spar = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 3.2, 6), wood);
    spar.rotation.x = Math.PI / 2;
    spar.position.set(0.38, MAST - 0.55, 0);
    const sg = new THREE.PlaneGeometry(2.9, 2.5, 8, 6);
    const sp = sg.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < sp.count; i++) { const u = sp.getX(i) / 1.45, v = sp.getY(i) / 1.25; sp.setZ(i, 0.55 * (1 - u * u) * (1 - 0.5 * v * v)); }
    sg.computeVertexNormals();
    sg.rotateY(Math.PI / 2);
    this.sail = new THREE.Mesh(sg, new THREE.MeshLambertMaterial({ color: 0xf3ead6, emissive: 0x3d3a33, side: THREE.DoubleSide }));
    this.sail.position.set(0.45, MAST - 1.85, 0);
    const bow = new THREE.Mesh(new THREE.ConeGeometry(0.06, 1.6, 6), wood);
    bow.rotation.z = -Math.PI / 2 - 0.35;
    bow.position.set(3.6, 0.55, 0);
    g.add(hull, deck, mast, spar, this.sail, bow);
    g.visible = false;
    this.ship = g;
    this.scene.add(g);
  }

  /** The tradition new waters take on: the current realm's (Greek on the open sea). */
  private cultureNow(): Culture { return this.realmName ? REALMS[this.realmName].culture : "greek"; }

  /** Discover landmarks the explorer comes close to; choose the next one to head for. */
  private updateVoyage(dt: number) {
    const P = this.pos, T = this.terrain;
    T.claimRegions(P.x, P.z, 130, this.cultureNow());   // waters coming into view take the tradition of the state you are in
    if (this.specV.landmarks <= 0) { this.goal = null; return; }
    for (const L of T.landmarksNear(P.x, P.z, DISCOVER_R)) {
      if (this.found.has(L.key)) continue;
      const d: Discovery = { name: L.name, kind: L.kind, x: L.x, z: L.z, t: this.time, v: this.cur.v, a: this.cur.a };
      this.found.set(L.key, d);
      this.log.push(d);
      this.showToast(L.name, this.log.length);
      this.addLabel(L);
      this.onDiscover?.(d, this.log.length);
      if (this.goal?.key === L.key) this.goal = null;
    }
    if (this.goal && (this.found.has(this.goal.key) || this.time - this.goalSince > 50)) {   // reached, or out of reach
      if (!this.found.has(this.goal.key)) this.skip.add(this.goal.key);
      this.goal = null;
    }
    if (!this.goal) {
      let best: Landmark | null = null, bd = 1e9;
      for (const L of T.landmarksNear(P.x, P.z, 300)) {
        if (this.found.has(L.key) || this.skip.has(L.key)) continue;
        const d = Math.hypot(L.x - P.x, L.z - P.z);
        if (d < bd) { bd = d; best = L; }
      }
      if (best) { this.goal = best; this.goalSince = this.time; }
    }
    // labels: names float over discovered landmarks nearby
    for (const [k, sp] of this.labels) {
      const d = Math.hypot(sp.position.x - P.x, sp.position.z - P.z);
      if (d > 260) { this.scene.remove(sp); (sp.material as THREE.SpriteMaterial).map?.dispose(); sp.material.dispose(); this.labels.delete(k); continue; }
      (sp.material as THREE.SpriteMaterial).opacity = clamp(1.4 - d / 140, 0, 0.95) * clamp((d - 30) / 30, 0, 1);   // fades out up close
    }
    void dt;
  }

  private addLabel(L: Landmark) {
    if (this.labels.has(L.key) || typeof document === "undefined") return;
    const c = document.createElement("canvas");
    c.width = 512; c.height = 96;
    const g = c.getContext("2d");
    if (!g) return;
    g.font = "600 44px Georgia, 'Times New Roman', serif";
    g.textAlign = "center"; g.textBaseline = "middle";
    g.shadowColor = "rgba(0,0,0,0.85)"; g.shadowBlur = 10;
    g.fillStyle = "#fff4d8";
    g.fillText(L.name, 256, 48);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, fog: false }));
    sp.scale.set(17, 3.2, 1);
    sp.position.set(L.x, L.y + 22 * L.size, L.z);
    this.scene.add(sp);
    this.labels.set(L.key, sp);
  }

  private showToast(name: string, n: number, kicker = "Discovered", sub = `${n} landmark${n === 1 ? "" : "s"} in the logbook`) {
    if (!this.toast) return;
    this.toast.textContent = "";
    const a = document.createElement("div");
    a.textContent = kicker;
    a.style.cssText = "font:600 11px system-ui,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#ffd58a";
    const b = document.createElement("div");
    b.textContent = name;
    b.style.cssText = "font:600 20px Georgia,'Times New Roman',serif;margin-top:2px";
    const c = document.createElement("div");
    c.textContent = sub;
    c.style.cssText = "font:12px system-ui,sans-serif;opacity:.85;margin-top:2px";
    this.toast.append(a, b, c);
    this.toast.style.opacity = "1";
    this.toast.style.transform = "translate(-50%, 0)";
    this.toastTimer = 3.5;
  }

  /** Seabirds circle the explorer: a flock when the state is pleasant, a few when it is not, almost none at night. */
  private updateBirds(va: number) {
    const night = this.specV.time === "night" || this.specV.style === "neon";
    const n = Math.round(NB * clamp(0.15 + 0.85 * va, 0, 1) * (night ? 0.15 : 1) * (this.reducedMotion ? 0.5 : 1));
    this.birds.count = n * 2;
    if (!n) return;
    const T = this.terrain, P = this.pos, m = new THREE.Matrix4(), r = new THREE.Matrix4(), f = new THREE.Matrix4(), sc = new THREE.Matrix4();
    const base = Math.max(T.waterY, T.surface(P.x, P.z));
    for (let i = 0; i < n; i++) {
      const w = 0.18 + 0.05 * (i % 4), ph = i * 2.399, R = 12 + (i % 5) * 5, t = this.time;
      const a = t * w * (i % 2 ? 1 : -1) + ph;
      const x = P.x + Math.cos(a) * R + Math.sin(t * 0.07 + i) * 6, z = P.z + Math.sin(a) * R;
      const y = base + 24 + (i % 3) * 5 + Math.sin(t * 0.5 + ph) * 1.5;   // above the chase camera
      const dir = a + (i % 2 ? Math.PI / 2 : -Math.PI / 2);
      const flap = Math.sin(t * (7 + (i % 3)) + ph) * 0.55;
      for (const side of [1, -1]) {
        m.makeTranslation(x, y, z);
        r.makeRotationY(-dir);
        f.makeRotationX(flap);
        sc.makeScale(1.3, 1, side * 1.3);
        m.multiply(r).multiply(sc).multiply(f);
        this.birds.setMatrixAt(i * 2 + (side > 0 ? 0 : 1), m);
      }
    }
    this.birds.instanceMatrix.needsUpdate = true;
  }

  private startWorker() {
    const src = (globalThis as { __EMOTIF_CHUNK_WORKER?: string }).__EMOTIF_CHUNK_WORKER;
    if (!src || typeof Worker === "undefined" || typeof Blob === "undefined") return;
    try {
      const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
      const w = new Worker(url);
      w.onmessage = e => this.onChunk(e.data);
      w.onerror = () => { w.terminate(); if (this.worker === w) { this.worker = null; this.pending.clear(); } };   // fall back to this thread
      w.postMessage({ type: "spec", spec: this.specV, gen: this.gen });
      this.worker = w;
    } catch { this.worker = null; }
  }

  /** Replace the inner world. The explorer stays where it is if there is land there. */
  setSpec(spec: WorldSpec) {
    this.specV = validateSpec(spec);
    this.terrain = new Terrain(this.specV);
    for (const c of this.chunks.values()) { this.scene.remove(c.mesh); c.mesh.geometry.dispose(); }
    this.chunks.clear();
    this.gen++;
    this.pending.clear();
    this.arrived = [];
    this.worker?.postMessage({ type: "spec", spec: this.specV, gen: this.gen });
    const st = this.specV.style;
    this.U.uBlock.value = st === "voxel" ? 1 : 0;
    this.U.uToon.value = st === "dream" ? 1 : 0;
    this.U.uNeon.value = st === "neon" ? 1 : 0;
    this.U.uNatural.value = st === "natural" ? 1 : 0;
    this.U.uSnow.value = this.specV.terrain.height * 0.86;
    this.chunkMood.clear();
    this.applySky();
    if (this.ring) this.clearProps();
    this.water.position.y = this.terrain.waterY;
    this.U.uWaterY.value = this.terrain.waterY;
    // voyage state belongs to one world
    for (const sp of this.labels.values()) { this.scene.remove(sp); (sp.material as THREE.SpriteMaterial).map?.dispose(); sp.material.dispose(); }
    this.labels.clear(); this.found.clear(); this.log = []; this.skip.clear(); this.goal = null;
    this.mapLand = null; this.mapJob = null; this.chapters = [];
    const sea = this.specV.terrain.islands && this.specV.terrain.water >= 0.45 ? (this.terrain.spawnSea(this.pos.x, this.pos.z, 26) ?? this.terrain.spawnSea(this.pos.x, this.pos.z)) : null;
    this.shipMode = !!sea;
    if (this.ship) this.ship.visible = this.shipMode;
    const sp = sea ?? this.terrain.spawn(this.pos.x, this.pos.z);
    const old = this.pos.clone();
    this.pos.set(sp.x, this.terrain.surface(sp.x, sp.z) + (this.shipMode ? MAST : 4), sp.z);
    if (this.controls) {
      const off = this.camera.position.clone().sub(this.controls.target);
      // natural: a low, grounded view that shows the horizon and sky
      const [back, up] = this.shipMode ? [30, 12] : this.specV.style === "natural" ? [20, 6.5] : [24, 14];   // ship: a high chase view over the coast
      if (off.lengthSq() < 1 || old.lengthSq() === 0) off.set(-Math.cos(this.heading) * back, up, -Math.sin(this.heading) * back);
      this.controls.target.copy(this.pos);
      this.camera.position.copy(this.pos).add(off);
      this.controls.update();
    }
  }
  get spec(): WorldSpec { return this.specV; }
  /** The person's dominant calibrated state (or null). It picks the weather and thickens haze or cloud. */
  setBrainState(name: string | null, live = name !== null) { this.brainState = name; this.statesLive = live; }
  private statesLive = false;   // a personal-state model is decoding: the realm follows it alone
  get chunkCount(): number { return this.chunks.size; }
  get viewRadius(): number { return this.radius; }
  get featureCount(): number { let n = 0; for (const c of this.chunks.values()) n += c.features; return n; }

  private onKey(e: KeyboardEvent, down: boolean) {
    if (this.mode !== "world") return;
    const k = ({ ArrowUp: "f", KeyW: "f", ArrowDown: "b", KeyS: "b", ArrowLeft: "l", KeyA: "l", ArrowRight: "r", KeyD: "r" } as Record<string, string>)[e.code];
    if (!k) return;
    e.preventDefault();
    if (down) { this.keys.add(k); this.lastInteract = this.time; } else this.keys.delete(k);
  }

  /** Bring the chunks around the explorer up to date: missing chunks first (nearest first), then near chunks still
   * at low detail. With a worker the jobs are queued (at most MAX_JOBS in flight) and arrive in onChunk; without one,
   * or when sync is set, chunks are built here until budgetMs is spent. */
  private streamChunks(budgetMs: number, sync = false) {
    const ccx = Math.floor(this.pos.x / CHUNK), ccz = Math.floor(this.pos.z / CHUNK);
    for (const [k, c] of this.chunks) {
      const [x, z] = k.split(",").map(Number);
      if (Math.max(Math.abs(x - ccx), Math.abs(z - ccz)) > this.radius + 1) { this.scene.remove(c.mesh); c.mesh.geometry.dispose(); this.chunks.delete(k); }
    }
    const need: [number, number, number, number][] = [];
    for (let dz = -this.radius; dz <= this.radius; dz++) for (let dx = -this.radius; dx <= this.radius; dx++) {
      const key = `${ccx + dx},${ccz + dz}`, lod = Math.max(Math.abs(dx), Math.abs(dz)) <= NEAR ? 0 : 1, c = this.chunks.get(key);
      if (!c) need.push([dx * dx + dz * dz, ccx + dx, ccz + dz, lod]);
      else if (c.lod > lod) need.push([1000 + dx * dx + dz * dz, ccx + dx, ccz + dz, lod]);
    }
    if (!need.length) return;
    need.sort((p, q) => p[0] - q[0]);
    if (this.worker && !sync) {
      for (const [, cx, cz, lod] of need) {
        if (this.pending.size >= MAX_JOBS) break;
        const key = `${cx},${cz}`;
        if (this.pending.get(key) === lod) continue;
        this.pending.set(key, lod);
        this.worker.postMessage({ type: "job", gen: this.gen, key, cx, cz, mood: this.moodFor(key), lod });
      }
      return;
    }
    const t0 = performance.now();
    for (const [, cx, cz, lod] of need) {
      const key = `${cx},${cz}`;
      this.placeChunk(key, buildChunk(this.terrain, cx, cz, this.moodFor(key), lod), lod);
      if (performance.now() - t0 > budgetMs) break;
    }
  }

  /** The state a chunk grows from: fixed the first time the chunk is made, so revisited land looks the same. */
  private moodFor(key: string): ChunkMood {
    let mood = this.chunkMood.get(key);
    if (!mood) {   // rounded so a rebuild is identical
      mood = { v: Math.round(clamp(this.cur.v, -1, 1) * 10) / 10, a: Math.round(clamp(this.cur.a, -1, 1) * 10) / 10 };
      if (this.chunkMood.size > 20000) this.chunkMood.clear();
      this.chunkMood.set(key, mood);
    }
    // the region's tradition (fixed once its waters come into view) travels with the job, so the worker dresses
    // its landmark the same way
    const [cx, cz] = key.split(",").map(Number), rx = Math.floor(cx * CHUNK / REGION), rz = Math.floor(cz * CHUNK / REGION);
    if (!this.terrain.hasCulture(rx, rz)) this.terrain.setCulture(rx, rz, this.cultureNow());
    return { ...mood, culture: this.terrain.cultureAt(rx, rz) };
  }

  private placeChunk(key: string, g: ChunkGeometry, lod: number) {
    const old = this.chunks.get(key);
    if (old) { this.scene.remove(old.mesh); old.mesh.geometry.dispose(); }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(g.position, 3));
    geom.setAttribute("normal", new THREE.BufferAttribute(g.normal, 3));
    geom.setAttribute("aColor", new THREE.BufferAttribute(g.color, 3));
    geom.setAttribute("aExtra", new THREE.BufferAttribute(g.extra, 2));
    geom.setAttribute("aUv", new THREE.BufferAttribute(g.uv, 2));
    geom.setIndex(new THREE.BufferAttribute(g.index, 1));
    geom.computeBoundingSphere();
    const mesh = new THREE.Mesh(geom, this.chunkMat);
    mesh.matrixAutoUpdate = false;
    this.scene.add(mesh);
    this.chunks.set(key, { mesh, features: g.features, lod });
  }

  private onChunk(m: { gen: number; key: string; cx: number; cz: number; lod: number; g: ChunkGeometry }) {
    if (m.gen !== this.gen) return;
    const drop = () => { if (this.pending.get(m.key) === m.lod) this.pending.delete(m.key); };
    const ccx = Math.floor(this.pos.x / CHUNK), ccz = Math.floor(this.pos.z / CHUNK);
    if (Math.max(Math.abs(m.cx - ccx), Math.abs(m.cz - ccz)) > this.radius + 1) { drop(); return; }   // explorer has moved on
    const have = this.chunks.get(m.key);
    if (have && have.lod <= m.lod) { drop(); return; }                                                // already as detailed
    this.arrived.push(m);   // placed by placeArrived (still counted as pending, so it is not requested again)
  }
  private arrived: { key: string; cx: number; cz: number; lod: number; g: ChunkGeometry }[] = [];
  private placeArrived() {
    for (let n = 0; n < 2 && this.arrived.length; n++) {
      const m = this.arrived.shift()!, have = this.chunks.get(m.key);
      if (this.pending.get(m.key) === m.lod) this.pending.delete(m.key);
      if (have && have.lod <= m.lod) { n--; continue; }
      this.placeChunk(m.key, m.g, m.lod);
    }
  }

  /** Whether chunks are being built in a worker (tests). */
  get workerActive(): boolean { return !!this.worker; }

  /** Build every chunk around the explorer now (tests and screenshots). */
  buildAll() { if (this.ok) this.streamChunks(1e9, true); }

  private moveExplorer(dt: number, motion: number, ar: number, va: number) {
    const T = this.terrain, m = this.specV.mood;
    const turn = (this.keys.has("l") ? 1 : 0) - (this.keys.has("r") ? 1 : 0);
    let speed = 0;
    if (this.explore) {
      // a steady course: the mind doesn't have to steer. Focus fills the sails; near an unlit beacon the ship heaves to
      speed = (4.5 + 2.5 * ar * m + (this.control.command === "focus" ? 3 : 0)) * motion;
      if (this.slowT > 0) speed *= 0.25;                 // scraped the rocks
      if (!this.goal) this.heading += (Math.sin(this.time * 0.13) * 0.5 + Math.sin(this.time * 0.051 + 1.3) * 0.5) * 0.25 * dt;
      if (!this.specV.terrain.islands) {         // keep to land: steer toward higher ground when the sea is ahead
        const ahead = (a: number) => T.surface(this.pos.x + Math.cos(a) * 14, this.pos.z + Math.sin(a) * 14);
        if (ahead(this.heading) <= T.waterY + 0.01) this.heading += (ahead(this.heading + 0.6) >= ahead(this.heading - 0.6) ? 1 : -1) * 0.9 * dt;
      }
    }
    const clearing = this.hazards.some(h => h.state === "dodged" && (h.x - this.pos.x) * Math.cos(this.heading) + (h.z - this.pos.z) * Math.sin(this.heading) > -6);
    if (this.explore && this.goal && !this.keys.size && !clearing) {   // set a course for the next landmark (not while passing rocks)
      const want = Math.atan2(this.goal.z - this.pos.z, this.goal.x - this.pos.x);
      const d = Math.atan2(Math.sin(want - this.heading), Math.cos(want - this.heading));
      this.heading += d * Math.min(1, dt * (this.shipMode ? 0.6 : 0.4));
    }
    const wet = (a: number, r: number) => T.surface(this.pos.x + Math.cos(a) * r, this.pos.z + Math.sin(a) * r) <= T.waterY + 0.05;
    if (this.shipMode && this.explore) {               // a ship keeps to the sea: turn toward open water before the shore
      if (!wet(this.heading, 8) || !wet(this.heading, 15)) {
        let side = 0;
        for (const s of [0.5, 1, 1.6]) { side += (wet(this.heading + s, 12) ? 1 : 0) - (wet(this.heading - s, 12) ? 1 : 0); }
        this.heading += (side >= 0 ? 1 : -1) * 1.3 * dt;
      }
    }
    if (this.keys.has("f")) speed = Math.max(speed, 6) * 1.8;
    if (this.keys.has("b")) speed = -4;
    this.heading -= turn * 1.4 * dt;
    if (this.dodgeT > 0) this.heading += this.dodgeDir * 0.4 * dt;   // steering clear of rocks (focus)
    const nx = this.pos.x + Math.cos(this.heading) * speed * dt, nz = this.pos.z + Math.sin(this.heading) * speed * dt;
    if (this.shipMode && T.surface(nx, nz) > T.waterY + 0.05) {   // aground: stop and come about
      this.stuck += dt;
      this.heading += 1.2 * dt;
    } else {
      this.stuck = 0;
      this.pos.x = nx; this.pos.z = nz;
    }
    if (this.shipMode && this.stuck > 6) {              // boxed into a cove: back out to open sea
      const sea = T.spawnSea(this.pos.x, this.pos.z);
      if (sea) { this.pos.x = sea.x; this.pos.z = sea.z; }
      this.stuck = 0;
    }
    const alt = this.shipMode ? T.waterY + MAST : T.surface(this.pos.x, this.pos.z) + 3.5 + 2.5 * va * m;
    this.pos.y += (alt - this.pos.y) * Math.min(1, dt * 2);
  }

  private updateWorld(dt: number, motion: number) {
    const v = this.cur.v, a = this.cur.a, ar = (a + 1) / 2, va = (v + 1) / 2, neg = Math.max(0, -v), pos = Math.max(0, v);
    const spec = this.specV, m = spec.mood, U = this.U, tod = TOD[spec.time], neon = spec.style === "neon";
    const [aTop, aHor, , , aAcc] = paletteAt(v, a);
    const P = spec.palette;

    // light and colour: the prompt's palette, the time of day, then the decoded state (scaled by spec.mood)
    const top = new THREE.Color(P.sky).lerp(new THREE.Color(tod.sky[0]), tod.sky[1]).lerp(aTop, m * 0.3);
    const hor = new THREE.Color(P.horizon).lerp(new THREE.Color(tod.hor[0]), tod.hor[1]).lerp(aHor, m * 0.3);
    if (neon) { top.lerp(new THREE.Color("#0a0418"), 0.85); hor.lerp(new THREE.Color(P.accent).lerp(new THREE.Color("#2a0f3f"), 0.55), 0.8); }
    if (spec.style === "natural" && this.skyInfo) {   // match the photo (at night only a little: its horizon is town glow)
      const k = spec.time === "night" ? 0.25 : 0.75;
      top.lerp(this.skyInfo.top, k); hor.lerp(this.skyInfo.horizon, k);
      if (spec.time === "night") hor.multiplyScalar(0.55);
    }
    this.updateRealm(dt);
    const R = this.realmName ? REALMS[this.realmName] : null, rw = R ? this.realmW * (0.4 + 0.6 * m) : 0;
    if (R) { top.lerp(new THREE.Color(R.sky), 0.75 * rw); hor.lerp(new THREE.Color(R.hor), 0.75 * rw); }
    U.uTop.value.copy(top);
    U.uHorizon.value.copy(hor);
    U.uFogColor.value.copy(hor);
    if (spec.style === "natural") U.uFogColor.value.lerp(top, 0.35);   // haze takes the sky's blue, not the horizon's glow
    if (neon) U.uFogColor.value.set("#12081f").lerp(new THREE.Color(P.accent), 0.08);
    U.uSky.value.copy(hor).lerp(top, 0.5);
    U.uSunDir.value.copy(tod.sun);
    if (spec.style === "natural" && spec.time === "day") U.uSunDir.value.set(0.62, 0.52, 0.58).normalize();   // lower sun: shaped light
    const sky = this.skyInfo;
    if (spec.style === "natural" && sky) {
      // light comes from where the photographed sun is (its elevation), the sky turned to put it at our azimuth
      const az = Math.atan2(U.uSunDir.value.z, U.uSunDir.value.x);
      if (spec.time !== "night") {
        const el = clamp(sky.sunV, 0.06, 0.95) * Math.PI / 2;
        U.uSunDir.value.set(Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az));
      }
      U.uSkyRot.value = sky.sunU - (az / (2 * Math.PI) + 0.5);
    }
    U.uSunColor.value.set(tod.sunColor).multiplyScalar(tod.sunI * (0.85 + 0.3 * va * m) * (neon ? 0.5 : 1));
    U.uAmbSky.value.copy(top).lerp(hor, 0.5).multiplyScalar(tod.amb * 1.2);
    U.uAmbGround.value.set(P.low).multiplyScalar(tod.amb * 0.5);
    U.uAccent.value.set(P.accent).lerp(aAcc, m * 0.25);
    // the sea follows the state: grey and rough when unpleasant and aroused, clear turquoise when pleasant
    U.uWater.value.set(P.water).lerp(new THREE.Color("#46545c"), clamp(m * neg * (0.45 + 0.55 * ar), 0, 0.8)).lerp(new THREE.Color("#22c7c4"), clamp(0.35 * m * pos, 0, 0.35));
    U.uFoam.value = clamp(m * neg * ar * 2 + (this.brainState === "anger" || this.brainState === "stress" ? 0.4 * m : 0), 0, 1);
    U.uNight.value = neon ? 1 : tod.night;
    U.uGlow.value = clamp(tod.glow + 0.3 * m * pos, 0, 1);
    U.uBloom.value = clamp(1 + 0.5 * m * v, 0.4, 1.5);
    U.uSat.value = clamp((spec.style === "dream" ? 0.92 : 1) + 0.4 * m * v, 0.55, 1.4);
    U.uBright.value = 1 + 0.12 * m * v;
    const warm = m * pos, cool = m * neg;
    U.uTint.value.set((1 + 0.06 * warm) * (1 - 0.1 * cool), (1) * (1 - 0.04 * cool), (1 - 0.1 * warm) * (1 + 0.08 * cool));
    U.uFogDensity.value = (1.15 / (this.radius * CHUNK)) * (1 + 0.9 * m * neg) * (spec.style === "natural" ? 0.5 : 1) * (this.shipMode ? 0.7 : 1);   // clear sea air
    U.uSway.value = (0.25 + 1.6 * ar * m) * motion;
    U.uWave.value = 0.3 + 0.9 * ar + 0.5 * U.uFoam.value;
    // sky: cloud cover follows unpleasantness, storms need unpleasant + aroused; clouds move with arousal
    U.uCloud.value = clamp(0.22 + 0.55 * m * neg + 0.12 * m * ar * neg, 0, 0.95);
    U.uStorm.value = clamp(m * neg * ar * 1.6, 0, 1);
    if (R) {   // the chapter's realm (one of the six states) reshapes light, sky and sea
      const mixv = (x: number, y: number) => x + (y - x) * rw;
      U.uTint.value.set(U.uTint.value.x * mixv(1, R.tint[0]), U.uTint.value.y * mixv(1, R.tint[1]), U.uTint.value.z * mixv(1, R.tint[2]));
      U.uSat.value = mixv(U.uSat.value, R.sat);
      U.uBright.value *= mixv(1, R.bright);
      U.uFogDensity.value *= mixv(1, R.fog);
      U.uCloud.value = mixv(U.uCloud.value, R.cloud);
      U.uStorm.value = mixv(U.uStorm.value, R.storm);
      const f0 = U.uFoam.value;
      U.uFoam.value = mixv(f0, R.foam);
      U.uWave.value += 0.5 * (U.uFoam.value - f0);
      U.uWater.value.lerp(new THREE.Color(R.water), 0.8 * rw);
      U.uNight.value = Math.max(U.uNight.value, R.night * rw);
      U.uSunColor.value.multiplyScalar(mixv(1, R.bright));
    }
    this.cloudT += dt * (0.004 + 0.03 * ar) * motion;
    U.uCloudT.value = this.cloudT;
    this.waterTime += dt * (0.4 + 1.2 * ar) * motion;
    U.uWaterTime.value = this.waterTime;
    U.uTime.value = this.time;
    // built-in materials (explorer, beacons, weather) use three's fog: give it the graded horizon
    const g = this.tmpC.copy(U.uFogColor.value), l = 0.2126 * g.r + 0.7152 * g.g + 0.0722 * g.b, s = U.uSat.value, tint = U.uTint.value, b = U.uBright.value;
    g.setRGB((l + (g.r - l) * s) * tint.x * b, (l + (g.g - l) * s) * tint.y * b, (l + (g.b - l) * s) * tint.z * b);
    (this.scene.fog as THREE.FogExp2).color.copy(g);
    (this.scene.fog as THREE.FogExp2).density = U.uFogDensity.value;
    U.uFar.value = (this.radius + 0.35) * CHUNK;
    U.uSink.value = this.shipMode ? 1 : 0;

    this.hemi.color.copy(U.uAmbSky.value).multiplyScalar(1.6);
    this.hemi.groundColor.copy(U.uAmbGround.value).multiplyScalar(1.6);
    this.sunLight.color.copy(U.uSunColor.value);
    this.sunLight.position.copy(this.pos).addScaledVector(U.uSunDir.value, 50);
    this.sunLight.target.position.copy(this.pos);
    this.updateVoyage(dt);
    this.moveExplorer(dt, motion, ar, va);
    const prev = this.controls.target.clone();
    this.controls.target.copy(this.pos);
    this.camera.position.add(this.pos.clone().sub(prev));
    if (this.autoRotate && this.time - this.lastInteract > 3 && (this.explore || this.keys.size)) {
      const off = this.camera.position.clone().sub(this.pos);
      const az = Math.atan2(off.z, off.x), want = this.heading + Math.PI;
      let d = want - az;
      d = Math.atan2(Math.sin(d), Math.cos(d));
      const rot = d * Math.min(1, dt * 0.6);
      const r = Math.hypot(off.x, off.z);
      off.x = Math.cos(az + rot) * r; off.z = Math.sin(az + rot) * r;
      this.camera.position.copy(this.pos).add(off);
    }
    // never inside trees or buildings: over land at sea the camera keeps above the canopy
    const gnd = this.terrain.surface(this.camera.position.x, this.camera.position.z);
    const floor = gnd + (this.shipMode && gnd > this.terrain.waterY + 0.3 ? 17 : 1.5);
    this.camFloor = floor > this.camFloor ? this.camFloor + (floor - this.camFloor) * Math.min(1, dt * 4) : floor;   // rise smoothly at the coast
    if (this.camera.position.y < this.camFloor) this.camera.position.y = this.camFloor;
    if (this.mode === "world") {   // keep the explorer in sight: rise over any hill between the camera and it
      const t = this.controls.target, c = this.camera.position;
      let need = -Infinity;
      for (let k = 1; k <= 8; k++) {
        const f = k / 9, gs = this.terrain.surface(t.x + (c.x - t.x) * f, t.z + (c.z - t.z) * f), g = gs + (this.shipMode && gs > this.terrain.waterY + 0.3 ? 14 : 1.2);
        need = Math.max(need, t.y + (g - t.y) / f);
      }
      need = Math.min(need, t.y + 70);
      if (need > c.y) c.y += (need - c.y) * Math.min(1, dt * 5);
    }
    this.controls.update();
    this.sky.position.copy(this.camera.position);
    this.water.position.x = Math.round(this.camera.position.x / CHUNK) * CHUNK;
    this.water.position.z = Math.round(this.camera.position.z / CHUNK) * CHUNK;

    (this.orb.material as THREE.MeshBasicMaterial).color.copy(U.uAccent.value).lerp(new THREE.Color(0xffffff), 0.35);
    (this.halo.material as THREE.SpriteMaterial).color.copy(U.uAccent.value);
    this.orb.position.copy(this.pos);
    if (this.shipMode) {   // the ship rides the waves; the orb is its masthead lantern
      const w = U.uWave.value * motion, t = this.time;
      const roll = Math.sin(t * 1.1) * 0.05 * w, pitch = Math.sin(t * 0.8 + 1) * 0.04 * w, bob = Math.sin(t * 1.3) * 0.12 * w;
      this.ship.position.set(this.pos.x, this.terrain.waterY + bob, this.pos.z);
      this.ship.rotation.set(roll, -this.heading, pitch, "YXZ");
      this.ship.updateMatrixWorld();
      this.orb.position.set(0.3, MAST + 0.35, 0).applyMatrix4(this.ship.matrixWorld);
      this.orb.scale.setScalar(0.55);
      (this.sail.material as THREE.MeshLambertMaterial).color.set("#f3ead6").lerp(U.uAccent.value, 0.18);
    } else {
      this.orb.scale.setScalar(1);
      this.orb.position.y += Math.sin(this.time * 1.4) * 0.25;
    }
    this.updateBirds(va);
    if (this.mapCanvas && (this.mapTick++ % 6 === 0)) { this.mapCanvas.hidden = this.mode !== "world"; this.drawMap(); }
    if (this.toast && this.toastTimer > 0) {
      this.toastTimer -= dt;
      if (this.toastTimer <= 0) { this.toast.style.opacity = "0"; this.toast.style.transform = "translate(-50%, -8px)"; }
    }

    this.updateWeather(dt, motion, v, a);
    this.updateProps(dt, motion);
  }

  private updateWeather(dt: number, motion: number, v: number, a: number) {
    const spec = this.specV;
    let want: WType = spec.weather === "mood" ? (this.realmName ? REALMS[this.realmName].weather : (this.brainState && STATE_WEATHER[this.brainState]) || moodWeather(v, a)) : spec.weather;
    if (spec.weather === "mood") {   // the state must hold for 3 s before the weather changes
      if (want !== this.wPending) { this.wPending = want; this.wPendingFor = 0; }
      this.wPendingFor += dt;
      if (this.wPendingFor < 3 && want !== this.wType) want = this.wType;
    }
    const p = WX[this.wType];
    if (want !== this.wType) {
      this.wOpacity = Math.max(0, this.wOpacity - 0.8 * dt);
      if (this.wOpacity === 0) {
        this.wType = want;
        const q = WX[want];
        this.pMat.color.set(q.color);
        this.pMat.size = q.size;
        this.pMat.blending = q.add ? THREE.AdditiveBlending : THREE.NormalBlending;
        this.pMat.needsUpdate = true;
      }
    } else {
      const intensity = spec.weather === "mood" ? clamp(0.35 + Math.hypot(v, a), 0.35, 1) : 1;
      const goal = p.opacity * intensity;
      this.wOpacity += clamp(goal - this.wOpacity, -0.8 * dt, 0.8 * dt);
    }
    this.pMat.opacity = this.wOpacity;
    this.points.visible = this.wOpacity > 0.005;
    if (!this.points.visible) return;
    const arr = this.points.geometry.attributes.position.array as Float32Array;
    const c = this.camera.position, t = this.time, wind = (0.4 + 1.2 * (a + 1) / 2) * motion;
    const wrap = (x: number, w: number) => ((x % w) + w * 1.5) % w - w / 2;
    for (let i = 0; i < NP; i++) {
      const k = this.pVel[i], ph = i * 0.618;
      let x = arr[i * 3] + Math.sin(t * 0.7 + ph) * p.drift * wind * dt;
      let y = arr[i * 3 + 1] + (p.vy * k + (this.wType === "fireflies" ? Math.sin(t * 0.9 + ph * 3) * 0.6 : 0)) * motion * dt;
      let z = arr[i * 3 + 2] + Math.cos(t * 0.5 + ph) * p.drift * wind * dt;
      x = c.x + wrap(x - c.x, BOX_W); y = c.y + wrap(y - c.y, BOX_H); z = c.z + wrap(z - c.z, BOX_W);
      arr[i * 3] = x; arr[i * 3 + 1] = y; arr[i * 3 + 2] = z;
    }
    this.points.geometry.attributes.position.needsUpdate = true;
  }

  /** Leave one second of history on the explorer's path. */
  /** Brain command for the objects (from interact.AlphaControl, every decoded window). */
  setControl(command: "idle" | "focus" | "relax", level: number) { this.control = { command, level }; }

  // ---------------------------------------------------------------- beacons (brain-controlled objects)
  /** Each landmark has a fire. Near an unlit one, focusing (posterior alpha below baseline, interact.ts) kindles the
   * masthead lantern; relaxing (alpha above baseline) sends the flame across to light the landmark's beacon. */
  private beaconAt(L: Landmark): THREE.Vector3 { return new THREE.Vector3(L.beacon[0], L.beacon[1], L.beacon[2]); }

  // ---------------------------------------------------------------- play: steer clear with focus, calm the sea with relaxing
  /** Free play (not during a study protocol): rocks appear on the course, and unpleasant realms can be calmed. */
  gameOn = true;
  private hazards: { g: THREE.Group; x: number; z: number; state: "ahead" | "dodged" | "hit" }[] = [];
  private nextHazardAt = 25;
  private dodgeT = 0;
  private dodgeDir = 1;
  private slowT = 0;
  private lastCmd = "";
  private relaxCool = 0;
  private calmMeter = 0;
  private regFrom: RealmName | null = null;
  private regStart = 0;
  private rockGeo: THREE.BufferGeometry | null = null;
  private rockMat: THREE.Material | null = null;
  /** Rocks dodged and hit, and each recovery from an unpleasant realm (seconds from entering it to calm). */
  dodges = 0;
  hits = 0;
  recoveries: { realm: RealmName; seconds: number; t: number }[] = [];
  /** 0-1 progress of calming the current unpleasant realm (null when the realm is not one to calm). */
  get calmProgress(): number | null { return this.gameOn && this.regFrom ? this.calmMeter : null; }

  private hazardAhead(): { g: THREE.Group; x: number; z: number; state: string } | null {
    const P = this.pos, fx = Math.cos(this.heading), fz = Math.sin(this.heading);
    for (const h of this.hazards) {
      if (h.state !== "ahead") continue;
      const dx = h.x - P.x, dz = h.z - P.z, d = Math.hypot(dx, dz);
      if (d < 48 && (dx * fx + dz * fz) / Math.max(d, 1e-6) > 0.35) return h;
    }
    return null;
  }

  /** Put a cluster of sea rocks on the course ahead (also used by tests). Returns false if no open water there. */
  spawnHazard(dist = 55): boolean {
    const T = this.terrain, P = this.pos, a = this.heading + (Math.random() - 0.5) * 0.25;
    const x = P.x + Math.cos(a) * dist, z = P.z + Math.sin(a) * dist;
    for (let q = 0; q < 6; q++) if (T.surface(x + Math.cos(q) * 7, z + Math.sin(q) * 7) > T.waterY + 0.05) return false;
    if (!this.rockGeo) {
      this.rockGeo = new THREE.IcosahedronGeometry(1, 0);
      this.rockMat = new THREE.MeshLambertMaterial({ color: 0x5d6166, flatShading: true });
    }
    const g = new THREE.Group();
    const n = 3 + Math.floor(Math.random() * 3);
    for (let k = 0; k < n; k++) {
      const m = new THREE.Mesh(this.rockGeo, this.rockMat!);
      const s = 1.6 + Math.random() * 1.8;
      m.scale.set(s * (0.8 + Math.random() * 0.4), s * (1.4 + Math.random() * 1.6), s * (0.8 + Math.random() * 0.4));
      m.position.set((Math.random() - 0.5) * 6, T.waterY + m.scale.y * 0.35, (Math.random() - 0.5) * 6);
      m.rotation.set(Math.random() * 0.4, Math.random() * 6.28, Math.random() * 0.4);
      g.add(m);
    }
    g.position.set(x, 0, z);
    this.scene.add(g);
    this.hazards.push({ g, x, z, state: "ahead" });
    return true;
  }

  private updateProps(dt: number, motion: number) {
    const P = this.pos, cmd = this.control.command, edge = cmd !== this.lastCmd;
    this.lastCmd = cmd;
    this.relaxCool = Math.max(0, this.relaxCool - dt);
    this.dodgeT = Math.max(0, this.dodgeT - dt);
    this.slowT = Math.max(0, this.slowT - dt);
    const play = this.interactive && this.gameOn && this.shipMode;
    // rocks on the course
    if (play && this.explore && this.time > this.nextHazardAt && !this.hazards.some(h => h.state === "ahead")) {
      this.nextHazardAt = this.time + (this.spawnHazard() ? 22 + Math.random() * 16 : 3);
    }
    const ahead = play ? this.hazardAhead() : null;
    if (ahead && cmd === "focus" && edge) {                 // focus: steer clear
      const side = (ahead.x - P.x) * -Math.sin(this.heading) + (ahead.z - P.z) * Math.cos(this.heading);
      this.dodgeDir = side > 0 ? -1 : 1;
      this.dodgeT = 2.6;
      ahead.state = "dodged";
      this.dodges++;
      this.onInteract?.("dodge", { x: ahead.x, z: ahead.z, t: this.time, placed: this.dodges });
    }
    for (let i = this.hazards.length - 1; i >= 0; i--) {
      const h = this.hazards[i], d = Math.hypot(h.x - P.x, h.z - P.z);
      if (h.state === "ahead" && d < 7.5) {                  // too late: the hull scrapes the rocks
        h.state = "hit"; this.hits++; this.slowT = 2.2;
        this.onInteract?.("hit", { x: h.x, z: h.z, t: this.time, placed: this.hits });
      }
      const behind = (h.x - P.x) * Math.cos(this.heading) + (h.z - P.z) * Math.sin(this.heading) < -10;
      if ((h.state !== "ahead" && behind && d > 30) || d > 160 || !play) {
        this.scene.remove(h.g); this.hazards.splice(i, 1);
      }
    }
    this.aim = null;
    this.ring.visible = !!ahead;
    if (ahead) {
      this.ring.position.set(ahead.x, this.terrain.waterY + 0.3, ahead.z);
      this.ring.rotation.set(-Math.PI / 2, 0, 0);
      const s = 5 + 0.4 * Math.sin(this.time * 4 * motion);
      this.ring.scale.set(s, s, s);
      (this.ring.material as THREE.MeshBasicMaterial).color.set("#ffd27a");
    }
    this.hazardNow = ahead;
    // regulation: in an unpleasant realm, calm evidence fills the meter and visibly calms the sea
    const neg = this.realmName === "stress" || this.realmName === "anxiety" || this.realmName === "anger" || this.realmName === "low_mood";
    if (this.gameOn && neg) {
      if (this.regFrom !== this.realmName) { this.regFrom = this.realmName; this.regStart = this.time; this.calmMeter = 0; }
      const calmness = clamp(0.5 + 0.6 * this.cur.v - 0.5 * Math.max(0, this.cur.a), 0, 1);
      this.calmMeter = clamp(this.calmMeter + dt * 0.07 * (calmness - 0.35), 0, 1);
      if (cmd === "relax" && edge && this.relaxCool === 0) {   // relaxing (alpha up) helps at once
        this.calmMeter = Math.min(1, this.calmMeter + 0.12); this.relaxCool = 3;
        this.onInteract?.("calm", { x: P.x, z: P.z, t: this.time, placed: Math.round(this.calmMeter * 100) });
      }
    } else if (this.regFrom && this.realmName !== this.regFrom) {
      if (this.gameOn && (this.realmName === "calm" || this.realmName === null)) {   // made it through
        const R = REALMS[this.regFrom], secs = Math.round(this.time - this.regStart);
        this.recoveries.push({ realm: this.regFrom, seconds: secs, t: this.time });
        const near = this.terrain.landmarksNear(P.x, P.z, 160).filter(L => !this.lit.has(L.key))
          .sort((a, b) => Math.hypot(a.x - P.x, a.z - P.z) - Math.hypot(b.x - P.x, b.z - P.z))[0];
        if (near) this.lightBeacon(near, false);
        this.showToast(`${R.title} calmed`, this.recoveries.length, "Cleared", `in ${secs} s · ${this.recoveries.length} cleared`);
      }
      this.regFrom = null; this.calmMeter = 0;
    }
    // lit beacons pulse gently
    for (const b of this.beacons2.values()) {
      (b.material as THREE.SpriteMaterial).opacity = 0.75 + 0.15 * Math.sin(this.time * 1.7 * motion + b.position.x);
      b.visible = Math.hypot(b.position.x - P.x, b.position.z - P.z) < 420;
    }
  }
  private hazardNow: { x: number; z: number } | null = null;

  private lightBeacon(L: Landmark, toast = true) {
    this.lit.add(L.key);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.dotTex, color: new THREE.Color("#ffb347"), transparent: true, opacity: 0.85, depthWrite: false, blending: THREE.AdditiveBlending }));
    sp.position.copy(this.beaconAt(L));
    sp.scale.set(16, 16, 1);
    this.scene.add(sp);
    this.beacons2.set(L.key, sp);
    if (toast) this.showToast(L.name, this.lit.size, "Beacon lit", `${this.lit.size} burning`);
  }

  /** Put out every beacon (a new session or a new world). */
  clearProps() {
    for (const b of this.beacons2.values()) { this.scene.remove(b); b.material.dispose(); }
    this.beacons2.clear(); this.lit.clear();
    this.carried = null; this.aim = null;
    for (const h of this.hazards) this.scene.remove(h.g);
    this.hazards = []; this.dodges = 0; this.hits = 0; this.recoveries = []; this.regFrom = null; this.calmMeter = 0;
    if (this.spark) this.spark.visible = false;
  }

  // ---------------------------------------------------------------- realms: the six states retell the voyage
  private updateRealm(dt: number) {
    const want = (this.statesLive ? (this.brainState && REALMS[this.brainState as RealmName] ? this.brainState : this.realmPending)
      : realmFromVA(this.cur.v, this.cur.a)) as RealmName | null;
    if (want !== this.realmPending) { this.realmPending = want; this.realmPendingFor = 0; }
    this.realmPendingFor += dt;
    if (this.realmPendingFor > 3 && want !== this.realmName) {          // a state must hold 3 s to change the chapter
      if (this.realmW > 0.02 && this.realmName) this.realmW = Math.max(0, this.realmW - dt * 0.35);
      else {
        this.realmName = want;
        if (want) {
          const R = REALMS[want];
          this.chapters.push({ realm: want, title: R.title, t: this.time });
          this.showToast(R.title, this.chapters.length, `Chapter ${this.chapters.length}`, R.line);
          this.onChapter?.(want, R.title, R.line);
        }
      }
    } else if (this.realmName) this.realmW = Math.min(1, this.realmW + dt * 0.25);   // slow fade: no flashes
  }
  /** The realm (one of the six states) the voyage is in now, and how fully. */
  get realm(): { name: RealmName | null; title: string; line: string; weight: number } {
    const R = this.realmName ? REALMS[this.realmName] : null;
    return { name: this.realmName, title: R?.title ?? "The open sea", line: R?.line ?? "", weight: this.realmW };
  }
  get chapterLog(): { realm: RealmName; title: string; t: number }[] { return this.chapters.slice(); }

  // ---------------------------------------------------------------- chart (minimap)
  private drawMap() {
    const c = this.mapCanvas, g = c?.getContext("2d");
    if (!c || !g || !this.terrain) return;
    const T = this.terrain, P = this.pos, S = MAP_SPAN, W = c.width;
    // re-chart the land around the ship a few rows at a time (one full chart in one frame would stutter)
    if (!this.mapJob && (!this.mapLand || Math.hypot(P.x - this.mapCx, P.z - this.mapCz) > S * 0.22)) {
      const N = 72;
      this.mapJob = { cx: Math.round(P.x / 24) * 24, cz: Math.round(P.z / 24) * 24, row: 0, img: new ImageData(N, N) };
    }
    const job = this.mapJob;
    if (job) {
      const N = 72, sea = new THREE.Color(this.specV.palette.water), land = new THREE.Color(this.specV.palette.high), sand = new THREE.Color("#e6d6a4"), col = new THREE.Color();
      for (const end = Math.min(N, job.row + (this.mapLand ? 6 : N)); job.row < end; job.row++) {
        const j = job.row;
        for (let i = 0; i < N; i++) {
          const x = job.cx + (i / N - 0.5) * S * 1.5, z = job.cz + (j / N - 0.5) * S * 1.5;
          const hgt = T.height(x, z), k = (j * N + i) * 4;
          if (hgt <= T.waterY + 0.02) col.copy(sea).multiplyScalar(0.75 + 0.25 * clamp(1 + hgt / 8, 0, 1));
          else if (hgt < 1.2) col.copy(sand);
          else col.copy(land).multiplyScalar(0.8 + 0.5 * clamp(hgt / Math.max(2, this.specV.terrain.height), 0, 1));
          job.img.data[k] = col.r * 255; job.img.data[k + 1] = col.g * 255; job.img.data[k + 2] = col.b * 255; job.img.data[k + 3] = 255;
        }
      }
      if (job.row >= N) {
        const off = this.mapLand ?? document.createElement("canvas");
        off.width = off.height = N;
        off.getContext("2d")!.putImageData(job.img, 0, 0);
        this.mapLand = off; this.mapCx = job.cx; this.mapCz = job.cz; this.mapJob = null;
      }
    }
    if (!this.mapLand) return;
    const px = (x: number) => W / 2 + (x - P.x) / S * W, pz = (z: number) => W / 2 + (z - P.z) / S * W;
    g.save();
    g.clearRect(0, 0, W, W);
    g.beginPath(); g.arc(W / 2, W / 2, W / 2 - 2, 0, Math.PI * 2); g.clip();
    g.imageSmoothingEnabled = true;
    const span = S * 1.5;
    g.drawImage(this.mapLand, px(this.mapCx - span / 2), pz(this.mapCz - span / 2), span / S * W, span / S * W);
    // course to the next landmark
    const goal = this.goal;
    if (goal) {
      g.setLineDash([4, 4]); g.strokeStyle = "rgba(255,255,255,0.9)"; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(W / 2, W / 2); g.lineTo(px(goal.x), pz(goal.z)); g.stroke(); g.setLineDash([]);
    }
    for (const L of T.landmarksNear(P.x, P.z, S * 0.75)) {
      const x = px(L.x), z = pz(L.z), on = this.lit.has(L.key), seen = this.found.has(L.key);
      g.beginPath(); g.arc(x, z, on ? 5 : 4, 0, Math.PI * 2);
      g.fillStyle = on ? "#ffb347" : seen ? "#ffffff" : "rgba(255,255,255,0.45)";
      g.fill(); g.lineWidth = 1.5; g.strokeStyle = "#1b2430"; g.stroke();
      if (goal && L.key === goal.key) { g.beginPath(); g.arc(x, z, 8, 0, Math.PI * 2); g.strokeStyle = "#ffffff"; g.stroke(); }
    }
    // the ship (or explorer): an arrow along the heading
    g.translate(W / 2, W / 2); g.rotate(this.heading);
    g.beginPath(); g.moveTo(9, 0); g.lineTo(-6, -5); g.lineTo(-3, 0); g.lineTo(-6, 5); g.closePath();
    g.fillStyle = "#ffd58a"; g.fill(); g.strokeStyle = "#1b2430"; g.lineWidth = 1.5; g.stroke();
    g.restore();
    g.beginPath(); g.arc(W / 2, W / 2, W / 2 - 2, 0, Math.PI * 2); g.lineWidth = 3; g.strokeStyle = "rgba(255,255,255,0.85)"; g.stroke();
    g.fillStyle = "#ffffff"; g.font = "600 11px system-ui, sans-serif"; g.textAlign = "center"; g.fillText("N", W / 2, 13);
  }

  pushStrata(v: number, a: number) {
    if (!this.ok) return;
    const i = this.beaconCount % this.beaconMax;
    // the wake of the voyage: small floating lights, coloured by mood, larger when energy was higher
    const k = 0.6 + 0.8 * (clamp(a, -1, 1) + 1) / 2;
    const gy = this.terrain.surface(this.pos.x, this.pos.z);
    this.dummy.position.set(this.pos.x, gy + 0.3 * k, this.pos.z);
    this.dummy.scale.set(k, k, k);
    this.dummy.updateMatrix();
    this.beacons.setMatrixAt(i, this.dummy.matrix);
    this.beacons.setColorAt(i, paletteAt(v, a)[4]);
    this.beaconCount++;
    this.beacons.count = Math.min(this.beaconCount, this.beaconMax);
    this.beacons.instanceMatrix.needsUpdate = true;
    if (this.beacons.instanceColor) this.beacons.instanceColor.needsUpdate = true;
  }
  clearStrata() { this.beaconCount = 0; if (this.beacons) this.beacons.count = 0; }

  // ---------------------------------------------------------------- arm scene
  private buildArmScene() {
    const sc = this.armScene;
    sc.background = new THREE.Color("#d9dee2");
    sc.fog = new THREE.Fog("#d9dee2", 120, 320);
    sc.add(new THREE.HemisphereLight(0xffffff, 0x8a8f94, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.3);
    sun.position.set(60, 120, 40);
    sc.add(sun);
    const grid = new THREE.GridHelper(400, 80, 0xa9b0b6, 0xc3c9ce);
    sc.add(grid);
    this.moodDisc = new THREE.Mesh(new THREE.CircleGeometry(34, 64), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 }));
    this.moodDisc.rotation.x = -Math.PI / 2;
    this.moodDisc.position.y = 0.02;
    sc.add(this.moodDisc);
    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(20, 22, PEDESTAL, 64), new THREE.MeshStandardMaterial({ color: 0xeceeea, roughness: 0.9 }));
    pedestal.position.y = PEDESTAL / 2;
    sc.add(pedestal);
    this.paper = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshStandardMaterial({ color: 0xf7f6f1, roughness: 1, side: THREE.DoubleSide }));
    this.paper.rotation.x = -Math.PI / 2;
    this.paper.visible = false;
    sc.add(this.paper);
    this.trailGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6000 * 3), 3));
    this.trailGeom.setDrawRange(0, 0);
    this.trailLine = new THREE.Line(this.trailGeom, new THREE.LineBasicMaterial({ color: 0x1f2a36 }));
    this.trailLine.frustumCulled = false;
    sc.add(this.trailLine);
    this.pen = new THREE.Mesh(new THREE.CylinderGeometry(4 * S, 4 * S, 1, 12), new THREE.MeshStandardMaterial({ color: 0x2f5d8a, roughness: 0.5 }));
    this.pen.visible = false;
    sc.add(this.pen);
    this.buildArm();
    this.targetMarker = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 12), new THREE.MeshBasicMaterial({ color: 0x2f5d8a, transparent: true, opacity: 0.6 }));
    sc.add(this.targetMarker);
    this.armCamera.position.set(42, 30, 42);
    this.armControls.target.set(0, 16, 0);
    this.armControls.update();
  }

  private buildArm() {
    const shell = new THREE.MeshStandardMaterial({ color: 0xf1f2ef, roughness: 0.45, metalness: 0.05 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x2b2f33, roughness: 0.6, metalness: 0.2 });
    const base = new THREE.Mesh(new THREE.CylinderGeometry(70 * S, 78 * S, 40 * S, 48), dark);
    base.position.y = 20 * S;
    const turret = new THREE.Group();
    const column = new THREE.Mesh(new THREE.BoxGeometry(60 * S, (SHOULDER_MM - 40) * S, 70 * S), shell);
    column.position.y = 40 * S + (SHOULDER_MM - 40) * S / 2;
    const shoulder = new THREE.Mesh(new THREE.CylinderGeometry(24 * S, 24 * S, 80 * S, 32), dark);
    shoulder.rotation.x = Math.PI / 2;
    shoulder.position.y = SHOULDER_MM * S;
    turret.add(column, shoulder);
    const upper = new THREE.Mesh(new THREE.BoxGeometry(34 * S, 34 * S, 1), shell);
    const fore = new THREE.Mesh(new THREE.BoxGeometry(26 * S, 26 * S, 1), shell);
    const elbow = new THREE.Mesh(new THREE.CylinderGeometry(20 * S, 20 * S, 60 * S, 28), dark);
    const jawA = new THREE.Mesh(new THREE.BoxGeometry(8 * S, 40 * S, 14 * S), dark);
    const jawB = jawA.clone();
    this.arm.add(base, turret, upper, fore, elbow, jawA, jawB);
    this.arm.position.y = PEDESTAL;
    this.parts = { turret, upper, fore, elbow, jawA, jawB };
    this.armScene.add(this.arm);
  }

  /** Arm frame (mm, x fwd / y left / z up from the shoulder plane) -> world. */
  armToWorld(x: number, y: number, z: number): THREE.Vector3 {
    return new THREE.Vector3(x * S, PEDESTAL + (SHOULDER_MM + z) * S, -y * S);
  }

  private poseArm(p: Pose) {
    const j = inverse(p.x, p.y, p.z) ?? { b: 0, s: 0, e: Math.PI / 2 };
    const { turret, upper, fore, elbow, jawA, jawB } = this.parts;
    turret.rotation.y = j.b;
    const sh = this.armToWorld(0, 0, 0);
    const a1 = Math.PI / 2 - (j.s + T2);
    const r1 = L2 * Math.cos(a1), z1 = L2 * Math.sin(a1);
    const el = this.armToWorld(r1 * Math.cos(j.b), r1 * Math.sin(j.b), z1);
    const ee = forward(j);
    const end = this.armToWorld(ee.x, ee.y, ee.z);
    const place = (m: THREE.Object3D, a: THREE.Vector3, b: THREE.Vector3) => {
      const local = (v: THREE.Vector3) => v.clone().sub(this.arm.position);
      const la = local(a), lb = local(b);
      m.position.copy(la).add(lb).multiplyScalar(0.5);
      m.scale.z = la.distanceTo(lb);
      m.lookAt(b);
    };
    place(upper, sh, el);
    place(fore, el, end);
    elbow.position.copy(el.clone().sub(this.arm.position));
    elbow.rotation.order = "YXZ";
    elbow.rotation.set(Math.PI / 2, j.b, 0);
    const open = clamp((Math.PI - p.t) / (Math.PI - 1.08), 0, 1);
    const side = new THREE.Vector3(-Math.sin(j.b), 0, -Math.cos(j.b)).multiplyScalar((6 + 26 * open) * S);
    const tip = end.clone().sub(this.arm.position).add(new THREE.Vector3(0, -14 * S, 0));
    jawA.position.copy(tip).add(side);
    jawB.position.copy(tip).sub(side);
    jawA.rotation.y = jawB.rotation.y = j.b;
  }

  /** The paper lies on the table (the arm's base plate, z = -SHOULDER_MM); the pen hangs from the gripper
   * with its tip at the table when the end effector is at penZ. */
  setDrawing(show: boolean, d: { cx: number; cy: number; size: number; penZ: number }, trail: { x: number; y: number; z: number }[]) {
    if (!this.ok) return;
    this.paper.visible = show;
    this.pen.visible = show;
    this.penLenMm = Math.max(10, d.penZ + SHOULDER_MM);
    if (show) {
      this.paper.position.copy(this.armToWorld(d.cx, d.cy, -SHOULDER_MM + 0.6));
      this.paper.scale.set((d.size + 20) * S, (d.size + 20) * S, 1);
    }
    const arr = this.trailGeom.attributes.position.array as Float32Array;
    const n = Math.min(trail.length, 6000);
    for (let i = 0; i < n; i++) {
      const w = this.armToWorld(trail[trail.length - n + i].x, trail[trail.length - n + i].y, -SHOULDER_MM + 1.2);
      arr[i * 3] = w.x; arr[i * 3 + 1] = w.y; arr[i * 3 + 2] = w.z;
    }
    this.trailGeom.setDrawRange(0, show ? n : 0);
    this.trailGeom.attributes.position.needsUpdate = true;
  }

  private updateArm() {
    this.poseArm(this.armPose);
    if (this.pen.visible) {
      const tip = this.armToWorld(this.armPose.x, this.armPose.y, this.armPose.z - this.penLenMm);
      const top = this.armToWorld(this.armPose.x, this.armPose.y, this.armPose.z);
      this.pen.position.copy(tip).add(top).multiplyScalar(0.5);
      this.pen.scale.y = this.penLenMm * S;
    }
    if (this.armTarget) {
      this.targetMarker.visible = true;
      this.targetMarker.position.copy(this.armToWorld(this.armTarget.x, this.armTarget.y, this.armTarget.z));
    } else this.targetMarker.visible = false;
    (this.moodDisc.material as THREE.MeshBasicMaterial).color.copy(paletteAt(this.cur.v, this.cur.a)[4]);
    this.armControls.update();
  }

  // ---------------------------------------------------------------- shared
  attach(host: HTMLElement) {
    if (!this.ok || this.host === host) return;
    this.ro?.disconnect();
    this.host = host;
    host.appendChild(this.renderer.domElement);
    if (!this.toast) {
      const t = document.createElement("div");
      t.className = "world-toast";
      t.setAttribute("role", "status");
      t.setAttribute("aria-live", "polite");
      t.style.cssText = "position:absolute;left:50%;top:12px;transform:translate(-50%,-8px);opacity:0;transition:opacity .6s,transform .6s;" +
        "pointer-events:none;background:rgba(12,16,24,.82);color:#fff;padding:10px 18px;border-radius:12px;text-align:center;" +
        "border:1px solid rgba(255,213,138,.45);box-shadow:0 6px 24px rgba(0,0,0,.35);z-index:3;max-width:80%";
      this.toast = t;
    }
    if (!this.mapCanvas) {
      const m = document.createElement("canvas");
      m.width = m.height = 300;
      m.className = "world-map";
      m.setAttribute("role", "img");
      m.setAttribute("aria-label", "Chart: the ship, nearby landmarks (gold when their beacon is lit) and the course to the next one");
      m.style.cssText = "position:absolute;right:12px;top:56px;width:112px;height:112px;opacity:.92;border-radius:50%;z-index:2;pointer-events:none;box-shadow:0 6px 20px rgba(0,0,0,.35)";
      this.mapCanvas = m;
    }
    if (getComputedStyle(host).position === "static") host.style.position = "relative";
    host.appendChild(this.toast);
    host.appendChild(this.mapCanvas);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
  }

  private resize() {
    if (!this.host) return;
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.renderer.setSize(w, h, false);
    for (const c of [this.camera, this.armCamera]) { c.aspect = w / h; c.updateProjectionMatrix(); }
  }

  /** Arm view shows the arm scene; every other view shows the inner world. */
  focusArm(on: boolean) {
    if (!this.ok) return;
    this.mode = on ? "arm" : "world";
    this.controls.enabled = !on;
    this.armControls.enabled = on;
    this.keys.clear();
  }

  /** Put the camera back behind the explorer. */
  resumeAutoRotate() {
    if (!this.ok) return;
    this.autoRotate = true;
    this.lastInteract = -1e9;
    if (this.mode === "world") {
      this.camera.position.set(this.pos.x - Math.cos(this.heading) * 24, this.pos.y + 14, this.pos.z - Math.sin(this.heading) * 24);
      this.controls.target.copy(this.pos);
      this.controls.update();
    } else {
      this.armCamera.position.set(42, 30, 42);
      this.armControls.target.set(0, 16, 0);
      this.armControls.update();
    }
  }

  private frame(t: number) {
    const dt = Math.min(0.25, (t - this.last) / 1000);
    this.last = t;
    const motion = this.reducedMotion ? 0.3 : 1;
    this.time += dt * motion;
    // smoothing + slew limit: brightness can never flash (see safety.ts, WCAG 2.3.1)
    this.cur.v = slewToward(this.cur.v, this.target.v, dt);
    this.cur.a = slewToward(this.cur.a, this.target.a, dt);
    if (!this.host || this.host.clientWidth === 0 || document.hidden) return;   // view not on screen
    if (this.paused && t - this.lastRender < 1000) return;   // ~1 fps while hidden behind a card
    if (this.mode === "world") {
      this.updateWorld(dt, motion);
      this.streamChunks(this.chunks.size < 9 ? 40 : 6);
      this.placeArrived();
    } else this.updateArm();
    if (!this.paused) this.adapt(dt);
    this.lastRender = t;
    if (this.mode === "world") this.renderer.render(this.scene, this.camera);
    else this.renderer.render(this.armScene, this.armCamera);
  }

  private adapt(dt: number) {
    this.frameEma += (dt - this.frameEma) * 0.1;
    // aim for 40+ fps: lower the resolution quickly when slow, draw less world if that is not enough, raise slowly
    if (this.frameEma > 1 / 40) { this.slowFor += dt; this.fastFor = 0; }
    else if (this.frameEma < 1 / 55) { this.fastFor += dt; this.slowFor = 0; }
    else { this.slowFor = 0; this.fastFor = 0; }
    let next = this.pr;
    if (this.slowFor > 1.5 && this.pr <= 0.6 && this.radius > 2) { this.radius = 2; this.slowFor = 0; }
    else if (this.slowFor > 3 && this.pr <= 0.6 && this.radius <= 2 && this.pr > 0.45) next = 0.45;   // last resort
    if (this.slowFor > 0.8 && this.pr > 0.6) next = Math.max(0.6, this.pr * 0.8);   // below 0.6 the world turns to mush
    else if (this.fastFor > 4 && this.pr < this.maxPR) next = Math.min(this.maxPR, this.pr / 0.85);
    if (next !== this.pr) {
      this.pr = next;
      this.slowFor = 0; this.fastFor = 0;
      this.renderer.setPixelRatio(this.pr);
      this.resize();
    }
  }

  get renderScale(): number { return this.pr; }
}
