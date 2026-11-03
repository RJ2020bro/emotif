# Emotif → Unreal Engine 5 showcase

The browser app is the research instrument (tested, logged, sham-controlled). This folder connects the same decoded
brain state to an Unreal Engine 5 scene for a high-fidelity showcase. Unreal only renders; all decoding stays in
Emotif, so the numbers in the paper describe exactly what drives both renderers.

Status: the Emotif side (OSC sender) is tested (`tests/test_server.py`, `test_osc_bridge`). The Unreal side below is a
Blueprint recipe; it has not been built in this repository because Unreal Engine is not available here. Check each
node name in your editor version.

## 1. Send the state
```bash
python -m emotif.server --osc 127.0.0.1:8001
python scripts/osc_listen.py 8001        # optional: see the messages before opening Unreal
```
Open http://localhost:8000 and start any source. While the app decodes, the server sends, every window:

| Address | Arguments | Meaning |
|---|---|---|
| `/emotif/state` | float valence, float arousal, float confidence, float quality | valence and arousal in −1…1 (already Kalman-smoothed and slew-limited), confidence 0…1 (−1 = no model), signal quality 0…1 |
| `/emotif/hold` | int 0/1 | 1 while the brain-only decoder holds its value (facial activity outside its training range) |
| `/emotif/stress` | float −1…1 | only with a model that has a stress head ("Brain-only + stress (Insight)"); experimental |

## 2. Receive it in Unreal Engine 5
1. Edit → Plugins → Input Devices → enable **OSC (Open Sound Control)**; restart.
2. Content Browser → Materials → **Material Parameter Collection** `MPC_Emotif` with scalars `Valence`, `Arousal`,
   `Confidence`, `Hold`.
3. New Blueprint Actor `BP_EmotifReceiver`, place it in the level.
   - Event BeginPlay → **Create OSC Server** (Receive IP Address `127.0.0.1`, Port `8001`, Start Listening ✓) →
     promote the return value to a variable `Server`.
   - From `Server`: **Assign On Osc Message Received** → custom event `OnEmotif(Message)`.
   - In `OnEmotif`: **Get OSC Message Address** → **Get OSC Address Method** → **Switch on String**:
     - `state`: **Get Float** (index 0, 1, 2) → set variables `TargetV`, `TargetA`, `Confidence`.
     - `hold`: **Get Int32** (index 0) → set `Hold`.
   - Event Tick: `V = FInterp To(V, TargetV, DeltaTime, 1.5)`, same for `A`; then **Set Scalar Parameter Value**
     on `MPC_Emotif` for each. (The values are already smoothed by Emotif; this only removes network jitter.)
4. Keep the photosensitivity rule from the app: never change overall brightness by more than ~10 % per 0.33 s
   (the interp speed above does this; don't raise it above ~2).

## 3. Drive the scene (same rules as the browser world, `app/src/world.ts` and `app/src/terrain.ts`)
| Decoded | Browser world | Unreal Engine 5 |
|---|---|---|
| Valence + | warmer light, saturation up, more flowers, lusher grass | Directional Light temperature 5200→6500 K, Post Process saturation 0.9→1.15 |
| Valence − | haze and cloud cover up, foliage wilts, rock and ruins appear | Exponential Height Fog density ×(1 + 0.9·neg), Volumetric Cloud coverage 0.22 + 0.55·neg |
| Arousal + | wind, waves, faster clouds, faster explorer | Wind Directional Source strength, Water body wave amplitude, cloud wind speed |
| Valence − and Arousal + | storm | cloud darkness + rain Niagara system |
| New land | each chunk's vegetation is fixed by the state when it was first generated | PCG graph: World Partition / runtime generation ahead of the camera with `Density = 1 + 1.6·pos` (flowers), `1 + 1.5·neg` (rocks); store the value per tile so revisits match |
| Hold = 1 | output freezes | ignore new values while `Hold` is 1 |

Suggested assets (free): Quixel Megascans meadow/forest packs, the Water plugin, Volumetric Cloud, Sky Atmosphere and
the PCG framework (UE 5.2+).

## 4. In the study
Use the browser world for data collection (it logs every frame's state, supports the sham condition and is tested).
Use Unreal for demonstrations, the booth, and a secondary "immersion" check if your IRB approves it; report it as
the same decoder output rendered differently.
