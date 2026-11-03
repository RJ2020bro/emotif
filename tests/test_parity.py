"""Python reference vs the TypeScript app, on identical inputs:
filter, band power / DE, channel-quality rule, region features (many montages), model forward pass,
and RoArm kinematics. Requires `node app/build.mjs --test-bundles` first (run_tests.sh does this)."""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from emotif import artifact as A, asr, face as FACE, features as F, montage as M, models, roarm  # noqa: E402

DIST = os.path.join(ROOT, "app", "dist-test")


def node(code, payload):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "in.json")
        with open(p, "w") as fh:
            json.dump(payload, fh)
        r = subprocess.run(["node", "-e", code, DIST, p], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
    return json.loads(r.stdout)


PRE = "const D=process.argv[1];const I=JSON.parse(require('fs').readFileSync(process.argv[2]));"


def test_dsp(rng):
    chans = ["Fp1", "Fp2", "AF7", "AF8", "F3", "F4", "TP9", "TP10", "Cz", "O1", "O2"]
    cases = []
    for fs in (128.0, 200.0, 250.0, 256.0, 255.99, 500.0, 128.5, 300.0):
        N = F.window_len(fs)
        t = np.arange(N) / fs
        x = rng.normal(0, 8, (len(chans), N)).cumsum(axis=1) * 0.2 + rng.normal(0, 5, (len(chans), N))
        x += 10 * np.sin(2 * np.pi * 10 * t) + 3 * np.sin(2 * np.pi * 21 * t) + 40
        if fs == 250.0:
            x[0, N // 2] += 300            # artifact
        if fs == 200.0:
            x[3] = 0.0                      # flat channel
        if fs == 300.0:
            x[5, :N // 2] += 700            # lost contact
        cases.append({"fs": fs, "x": x.tolist()})
    out = node(PRE + "const S=require(D+'/dsp.cjs');console.log(JSON.stringify(I.map(c=>{"
               "const de=S.deMatrix(c.x,c.fs).map(r=>Array.from(r));const q=S.channelQuality(S.channelStats(c.x));"
               "const q0=S.channelQuality(S.channelStats(c.x),0);"
               "return {de,good:q.good,reject:q.reject,good0:q0.good,reject0:q0.reject,N:S.windowLen(c.fs)}})))", cases)
    worst = 0.0
    for c, o in zip(cases, out):
        x = np.array(c["x"])
        assert o["N"] == F.window_len(c["fs"])
        err = float(np.max(np.abs(F.de_matrix(x, c["fs"]) - np.array(o["de"]))))
        worst = max(worst, err)
        assert err < 1e-8, (c["fs"], err)
        g, r = F.channel_quality(x)
        assert list(g) == o["good"] and r == o["reject"], (c["fs"], list(g), o["good"], r, o["reject"])
        g0, r0 = F.channel_quality(x, 0)
        assert list(g0) == o["good0"] and r0 == o["reject0"]
    print(f"dsp parity OK: {len(cases)} sampling rates, max |dDE| = {worst:.1e}, quality rule identical")

    fcases = []
    for fs in (100.0, 128.0, 250.0, 256.0, 500.0):
        n = int(fs * 6)
        t = np.arange(n) / fs
        x = np.vstack([800 + 30 * t + 40 * np.sin(2 * np.pi * 60 * t) + 10 * np.sin(2 * np.pi * 10 * t),
                       -300 + rng.normal(0, 10, n) + 20 * np.sin(2 * np.pi * 50 * t)])
        fcases.append({"fs": fs, "x": x.tolist()})
    out = node(PRE + "const S=require(D+'/dsp.cjs');console.log(JSON.stringify(I.map(c=>{const f=new S.StreamFilter(c.fs,c.x.length);"
               "const T=c.x[0].length,o=c.x.map(()=>[]);for(let i=0;i<T;i++){const y=f.process(c.x.map(r=>r[i]));y.forEach((v,k)=>o[k].push(v));}"
               "return {n:S.filterCoeffs(c.fs).length,y:o}})))", fcases)
    worst = 0.0
    for c, o in zip(fcases, out):
        assert o["n"] == len(F.filter_coeffs(c["fs"]))
        err = float(np.max(np.abs(F.filter_signal(np.array(c["x"]), c["fs"]) - np.array(o["y"]))))
        worst = max(worst, err)
        assert err < 1e-6, (c["fs"], err)
    print(f"filter parity OK: max |diff| = {worst:.1e}")


def test_montage(rng):
    names = list(M.POSITIONS)
    cases = []
    for dev, chans in M.DEVICES.items():
        cases.append(chans)
    cases.append(["EEG Fp1-REF", "EEG Fp2-REF", "Ch3", "T3", "T4", "O1"])
    for _ in range(25):
        k = int(rng.integers(1, 20))
        cases.append([str(n).title() if n.startswith("FP") is False else "Fp" + n[2:] for n in rng.choice(names, size=k, replace=False)])
    payload = []
    for chans in cases:
        z = rng.normal(0, 1, (len(chans), 5))
        payload.append({"ch": chans, "z": z.tolist()})
    out = node(PRE + "const Mo=require(D+'/montage.cjs');console.log(JSON.stringify(I.map(c=>{const m=Mo.regionWeights(c.ch);"
               "return {f:Array.from(Mo.regionFeatures(c.z,m)),mask:Array.from(m.mask),canon:c.ch.map(Mo.canonical)}})))", payload)
    for c, o in zip(payload, out):
        W, mask = M.region_weights(c["ch"])
        f = M.region_features(np.array(c["z"]), W, mask)
        assert np.allclose(f, o["f"], atol=1e-9), c["ch"]
        assert list(mask) == o["mask"]
        assert [M.canonical(x) for x in c["ch"]] == o["canon"], (c["ch"], o["canon"])
    assert len(out[0]["f"]) == M.N_FEATURES
    print(f"montage parity OK: {len(payload)} layouts incl. every supported headset")


def test_model(rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    X = rng.normal(0, 1, (300, M.N_FEATURES))
    payload = []
    for ncls in (2, 3):
        y = rng.integers(0, ncls, 300) - (1 if ncls == 3 else 0)
        y = np.where(y == 0, -1, y) if ncls == 2 else y
        for clf in (LogisticRegression(max_iter=2000), MLPClassifier((16,), max_iter=200, random_state=0)):
            clf.fit(X, y)
            layers = models.export_layers(clf)
            assert np.allclose(models.forward(layers, X), clf.predict_proba(X), atol=1e-9)
            payload.append({"layers": layers, "classes": [int(c) for c in clf.classes_], "X": X[:5].tolist(), "p": clf.predict_proba(X[:5]).tolist()})
    out = node(PRE + "const Md=require(D+'/model.cjs');console.log(JSON.stringify(I.map(c=>c.X.map(x=>Array.from(Md.forward(c.layers,x))))))", payload)
    for c, o in zip(payload, out):
        assert np.allclose(c["p"], o, atol=1e-9)
    print("model parity OK: exported logreg/MLP heads give sklearn's probabilities in Python and in the app")


def test_bundled_models(rng):
    """Every model shipped in models/ gives the same calibrated probabilities in Python and in the app
    (including brain-only heads, which also read the face input)."""
    import glob
    payload = []
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*_*.json"))):
        mf = json.load(open(path))
        if mf.get("format") != "emotif-model-v2":
            continue
        X = rng.normal(0, 1, (6, M.N_FEATURES))
        X[:, -len(M.REGIONS):] = (rng.random((6, len(M.REGIONS))) < 0.5).astype(float)
        Fr = rng.normal(0, 1, (6, FACE.N_REGION_FACE))
        Z = M.graph_features(X) if mf.get("transform") else X
        for name, head in mf["heads"].items():
            Zi = Z
            if head.get("face"):
                q = (FACE.design(Fr) - np.array(head["face"]["mu"])) / np.array(head["face"]["sd"])
                Zi = np.hstack([Z, q])
            payload.append({"file": os.path.basename(path), "model": {"transform": mf.get("transform")}, "head": head,
                            "x": X.tolist(), "f": Fr.tolist(),
                            "p": models.forward(head["layers"], Zi, head.get("temperature", 1.0)).tolist()})
    out = node(PRE + "const Md=require(D+'/model.cjs');console.log(JSON.stringify(I.map(c=>c.x.map((x,i)=>Array.from(Md.forward(c.head.layers, Md.headInput(c.model, c.head, x, c.f[i]), c.head.temperature ?? 1))))))", payload)
    for c, o in zip(payload, out):
        assert np.allclose(c["p"], o, atol=1e-9), c["file"]
    print(f"bundled models parity OK: {sorted({c['file'] for c in payload})}, {len(payload)} heads")


def test_face(rng):
    """Face indices (causal eye filter over a stream, muscle band power), regional pooling and the brain-only head's
    input: identical in Python and in the app."""
    cases = []
    for fs, chans in ((256, ["TP9", "AF7", "AF8", "TP10"]), (128, ["AF3", "F7", "F3", "FC5", "T7", "O1"]), (250, ["Fp1", "Fp2", "C3", "C4"])):
        T = int(fs * 9)
        t = np.arange(T) / fs
        x = rng.normal(0, 8, (len(chans), T)) + 25 * np.sin(2 * np.pi * 10 * t)
        x[1, int(3 * fs):int(3.3 * fs)] += 180 * np.hanning(int(0.3 * fs))      # a blink
        x[0, int(5 * fs):int(6 * fs)] += rng.normal(0, 40, int(fs))            # jaw clench burst
        xf = F.filter_signal(x, fs)
        slow = FACE.slow_signal(xf, fs)
        N = F.window_len(fs)
        starts = list(range(0, T - N + 1, int(fs)))
        idx = np.array([FACE.window_indices(xf[:, s:s + N], slow[:, s:s + N], fs) for s in starts])
        W, _ = M.region_weights(chans)
        rel = idx - np.median(idx, axis=0)
        pooled = FACE.pool(rel, W)
        cases.append({"fs": fs, "ch": chans, "xf": xf.T.tolist(), "starts": starts, "N": N, "idx": idx.tolist(), "pooled": pooled.tolist(),
                      "rel": rel.tolist()})
    out = node(PRE + "const Fa=require(D+'/face.cjs');const Mo=require(D+'/montage.cjs');"
               "console.log(JSON.stringify(I.map(c=>{const C=c.ch.length;const e=new Fa.EyeFilter(c.fs,C);const slow=c.xf.map(r=>Array.from(e.process(r)));"
               "const col=(rows,s,N)=>Array.from({length:C},(_,k)=>rows.slice(s,s+N).map(r=>r[k]));"
               "const idx=c.starts.map(s=>Fa.windowIndices(col(c.xf,s,c.N),col(slow,s,c.N),c.fs).map(a=>Array.from(a)));"
               "const map=Mo.regionWeights(c.ch);const pooled=c.rel.map(r=>Array.from(Fa.pool(r,map)));return {idx,pooled};})))", cases)
    worst = 0.0
    for c, o in zip(cases, out):
        e1 = float(np.max(np.abs(np.array(c["idx"]) - np.array(o["idx"]))))
        e2 = float(np.max(np.abs(np.array(c["pooled"]) - np.array(o["pooled"]))))
        worst = max(worst, e1, e2)
        assert e1 < 1e-8 and e2 < 1e-10, (c["fs"], e1, e2)
        i = np.array(c["idx"])
        assert i[:, 1, 0].max() > np.median(i[:, 1, 0]) + 1.0, "the blink raises the eye index"
        assert i[:, 0, 2].max() > np.median(i[:, 0, 2]) + 1.0, "the clench raises the muscle index"
    # brain-only model: Python predict_proba == exported head on [features, face input] in the app
    n = 400
    X = rng.normal(0, 1, (n, M.N_FEATURES))
    Fr = rng.normal(0, 1, (n, FACE.N_REGION_FACE))
    X[:, :20] += 0.8 * Fr[:, :20]                     # features contaminated by the face
    y = (X[:, 30] + rng.normal(0, 1, n) > 0).astype(int) * 2 - 1
    clf = models.fit_model("brain", X, y, Fr)
    head = {"classes": [int(v) for v in clf.classes_], "layers": models.export_layers(clf), "face": {"mu": clf.q_mu.tolist(), "sd": clf.q_sd.tolist()}}
    payload = {"head": head, "x": X[:8].tolist(), "f": Fr[:8].tolist()}
    o = node(PRE + "const Md=require(D+'/model.cjs');console.log(JSON.stringify(I.x.map((x,i)=>Array.from(Md.forward(I.head.layers, Md.headInput({transform:null}, I.head, x, I.f[i]))))))", payload)
    assert np.allclose(o, clf.predict_proba(X[:8], Fr[:8]), atol=1e-10)
    print(f"face parity OK: eye/jaw indices and regional pooling max |diff| {worst:.1e} at 128/250/256 Hz; brain-only head identical")


def test_graph_and_temperature(rng):
    """Graph-diffusion transform (with random headset coverage) and temperature-scaled heads match."""
    from sklearn.linear_model import LogisticRegression
    X = rng.normal(0, 1, (200, M.N_FEATURES))
    X[:, -len(M.REGIONS):] = (rng.random((200, len(M.REGIONS))) < 0.4).astype(float)
    X[:5, -len(M.REGIONS):] = 0.0
    X[0, -len(M.REGIONS)] = 1.0                      # one observed region: every region copies it
    y = rng.integers(-1, 2, 200)
    gm = models.make_model("graph").fit(X, y)
    layers = models.export_layers(gm)
    G = M.graph_features(X)
    assert G.shape[1] == M.N_FEATURES + M.GRAPH_HOPS * len(M.REGIONS) * M.N_BANDS
    assert np.allclose(models.forward(layers, G), gm.predict_proba(X), atol=1e-9)
    T = 2.7
    payload = {"x": X[:12].tolist(), "adj": M.region_adjacency().tolist(), "hops": M.GRAPH_HOPS,
               "layers": layers, "T": T}
    out = node(PRE + "const Mo=require(D+'/montage.cjs');const Md=require(D+'/model.cjs');"
               "const g=I.x.map(x=>Array.from(Mo.graphFeatures(x,I.adj,I.hops)));"
               "console.log(JSON.stringify({g, p:g.map(v=>Array.from(Md.forward(I.layers,v,I.T)))}))", payload)
    assert np.allclose(out["g"], G[:12], atol=1e-10)
    assert np.allclose(out["p"], models.forward(layers, G[:12], T), atol=1e-10)
    assert np.allclose(G[0, M.N_FEATURES:M.N_FEATURES + 85].reshape(17, 5), X[0, :5][None, :], atol=1e-12)
    print("graph + temperature parity OK: diffusion features and calibrated probabilities identical")


def test_edf(rng):
    """EDF and BDF written by emotif.edf: the app's reader, the Python reader and MNE (an independent reader)
    agree; mV/V units are converted; accelerometer (other rate) and Status channels are left out."""
    import mne
    from emotif import edf
    with tempfile.TemporaryDirectory() as d:
        payload = []
        for bdf in (False, True):
            fs = 250 if bdf else 256
            x = rng.normal(0, 40, (4, fs * 6)) + np.linspace(-300, 300, fs * 6)
            path = os.path.join(d, "t.bdf" if bdf else "t.edf")
            extra = [("Accel X", "g", rng.normal(0, 1, 6 * 25), 25)] + ([("Status", "", np.zeros(6 * fs), fs)] if bdf else [])
            edf.write_edf(path, x, fs, ["EEG Fp1-REF", "AF8", "TP9", "Oz"], bdf=bdf, units=["uV", "mV", "uV", "V"] if bdf else ["uV", "mV", "uV", "uV"], extra=extra)
            py = edf.read_edf(path)
            raw = (mne.io.read_raw_bdf if bdf else mne.io.read_raw_edf)(path, preload=True, verbose=False)
            ref = raw.get_data(picks=list(range(4))) * 1e6
            assert py["channels"] == ["EEG Fp1-REF", "AF8", "TP9", "Oz"] and py["fs"] == fs, py["channels"]
            assert np.abs(py["x"] - ref).max() < 1e-6, np.abs(py["x"] - ref).max()
            assert np.abs(py["x"] - x).max() < (0.02 if bdf else 0.05) * 40, np.abs(py["x"] - x).max()
            assert "Accel X" in py["skipped"] and ("Status" in py["skipped"]) == bdf
            payload.append({"path": path, "x": py["x"].tolist(), "fs": fs, "ch": py["channels"]})
        out = node(PRE + "const E=require(D+'/edf.cjs');const fsm=require('fs');console.log(JSON.stringify(I.map(c=>{"
                   "const b=fsm.readFileSync(c.path);const e=E.parseEdf(b.buffer.slice(b.byteOffset,b.byteOffset+b.length));"
                   "return {fs:e.fs,ch:e.channels,x:e.rows,sk:e.skipped,f:e.format}})))", payload)
    for c, o in zip(payload, out):
        assert o["fs"] == c["fs"] and o["ch"] == c["ch"], (o["fs"], o["ch"])
        assert np.abs(np.array(o["x"]).T - np.array(c["x"])).max() < 1e-9
    print("edf parity OK: EDF and BDF read identically by the app, emotif.edf and MNE; units converted, "
          "non-EEG channels left out")


def test_arm(rng):
    pts = []
    while len(pts) < 300:
        b, s, e = rng.uniform(-2.5, 2.5), rng.uniform(-1.5, 1.5), rng.uniform(0.05, 3.0)
        pts.append([b, s, e])
    reqs = rng.uniform(-600, 600, (200, 4)).tolist()
    out = node(PRE + "const A=require(D+'/arm.cjs');console.log(JSON.stringify({fk:I.j.map(([b,s,e])=>A.forward({b,s,e})),"
               "ik:I.j.map(([b,s,e])=>{const p=A.forward({b,s,e});return A.inverse(p.x,p.y,p.z)}),"
               "cl:I.r.map(([x,y,z,t])=>A.clampPose({x,y,z,t},A.DEFAULT_LIMITS))}))", {"j": pts, "r": reqs})
    for (b, s, e), fk, ik in zip(pts, out["fk"], out["ik"]):
        p = roarm.forward(b, s, e)
        assert np.allclose(p, (fk["x"], fk["y"], fk["z"]), atol=1e-9)
        pj = roarm.inverse(*p)
        assert (pj is None) == (ik is None)
        if pj:
            assert np.allclose(pj, (ik["b"], ik["s"], ik["e"]), atol=1e-6)
    lim = roarm.Limits()
    for (x, y, z, t), c in zip(reqs, out["cl"]):
        px, py, pz, pt, clamped = roarm.clamp_target(x, y, z, t, lim)
        assert np.allclose((px, py, pz, pt), (c["x"], c["y"], c["z"], c["t"]), atol=1e-6), ((x, y, z, t), c)
        assert clamped == c["clamped"]
        assert roarm.inverse(px, py, pz) is not None
    print("arm parity OK: forward/inverse kinematics and workspace clamp identical; every clamped target reachable")


def test_asr(rng):
    """ASR calibration + streaming repair: app (Jacobi eigensolver) vs Python (LAPACK) on multichannel data
    with bursts, at several rates and channel counts, including a rank-deficient (average-referenced) case."""
    worst = 0.0
    for fs, C, avgref in [(256, 4, False), (250, 8, False), (128, 14, False), (200, 2, False), (256, 8, True)]:
        N = int(fs * 36)
        A_ = rng.normal(0, 1, (C, C)) * 0.3 + np.eye(C)
        X = F.filter_signal(A_ @ rng.normal(0, 10, (C, N)), fs)
        if avgref:
            X = X - X.mean(axis=0, keepdims=True)
        Y = X.copy()
        for t0 in (22, 27, 31):
            Y[:, int(t0 * fs):int(t0 * fs) + fs // 2] += rng.normal(0, 1, (C, 1)) * 150 * np.hanning(fs // 2)
        cal = X[:, : int(fs * 15)]
        st = asr.calibrate(cal, fs)
        Z, removed = asr.clean(Y, st, fs)
        assert removed.max() > 0, "bursts should be repaired"
        out = np.array(node(PRE + "const S=require(D+'/asr.cjs');const st=S.calibrate(I.cal,I.fs);const s=new S.AsrStream(st,I.fs);"
                            "const out=[];const T=I.y[0].length;for(let t=0;t<T;t++){for(const o of s.push(I.y.map(c=>c[t])))out.push(Array.from(o));}"
                            "console.log(JSON.stringify(out));", {"cal": cal.tolist(), "y": Y.tolist(), "fs": fs})).T
        n = out.shape[1]
        assert n >= N - asr._sizes(fs)[1] - asr._sizes(fs)[2] - 1
        err = float(np.max(np.abs(out - Z[:, :n])) / np.max(np.abs(Z)))
        worst = max(worst, err)
        assert err < 1e-8, (fs, C, avgref, err)
        # stream == offline in Python too
        s = asr.ASRStream(st, fs)
        py = np.concatenate([s.push(Y[:, i:i + 37].T) for i in range(0, N, 37)]).T
        assert np.allclose(py, Z[:, :py.shape[1]], atol=1e-9)
        # repair never adds energy to a window
        assert np.sum(Z ** 2) <= np.sum(Y ** 2) * 1.0001
    print(f"asr parity OK: 5 configurations incl. average reference, max relative diff {worst:.1e}")


def test_artifact(rng):
    model = A.load()
    cases = []
    for fs in (128.0, 256.0, 250.0):
        N = F.window_len(fs)
        x = rng.normal(0, 10, (6, N)).cumsum(axis=1) * 0.1 + rng.normal(0, 5, (6, N))
        x[1, N // 3:N // 3 + 40] += 120
        x[2] += rng.normal(0, 30, N)
        cases.append({"fs": fs, "x": x.tolist()})
    base = rng.normal(0, 0.5, (6, A.N_FEATURES))
    out = node(PRE + "const Ar=require(D+'/artifact.cjs');const S=require(D+'/dsp.cjs');const m=I.model;"
               "console.log(JSON.stringify(I.cases.map(c=>{const bp=S.bandPowers(c.x,c.fs);"
               "const f=c.x.map((x,i)=>Ar.channelFeatures(x,c.fs,bp[i]));"
               "return {f:f.map(v=>Array.from(v)),v:f.map((v,i)=>{const r=Ar.classify(m,v,Float64Array.from(I.base[i]));return {p:Array.from(r.probs),bad:r.bad};})};})))",
               {"cases": cases, "model": model, "base": base.tolist()})
    worst = 0.0
    for c, o in zip(cases, out):
        x = np.array(c["x"])
        f = A.channel_features(x, c["fs"])
        err = float(np.max(np.abs(f - np.array(o["f"]))))
        worst = max(worst, err)
        assert err < 1e-9, (c["fs"], err)
        bad, p = A.bad_channels(model, f, base)
        assert np.allclose(p, [v["p"] for v in o["v"]], atol=1e-9)
        assert list(bad) == [v["bad"] for v in o["v"]]
    print(f"artifact detector parity OK: features max |diff| {worst:.1e}, probabilities and decisions identical")


def test_states(rng):
    """Personal state calibration: held-out AUCs, accuracy, face-only reference and the folded layer agree between
    emotif/states.py and app/src/states.ts (permutations use different generators, so they're not compared)."""
    from emotif import states as ST
    K, per, W, NR = 6, 3, 12, FACE.N_REGION_FACE
    live = np.arange(0, 100, 2)
    sig = rng.normal(0, 1, (K, len(live)))
    mix = rng.normal(0, 0.6, (NR, len(live)))
    X, F, y, blocks = [], [], [], []
    b = 0
    for rep in range(per):
        for k in range(K):
            b += 1
            for _ in range(W):
                f = rng.normal(0, 0.8, NR)
                x = np.zeros(M.N_FEATURES)
                x[live] = rng.normal(0, 1, len(live)) + 0.5 * sig[k] + 0.15 * f @ mix
                X.append(x); F.append(f); y.append(k); blocks.append(b)
    X, F, y, blocks = map(np.array, (X, F, y, blocks))
    py = ST.calibrate(X, F, y, blocks, ST_NAMES, perms=0)
    payload = {"d": [{"x": X[i].tolist(), "f": F[i].tolist(), "y": int(y[i]), "block": int(blocks[i])} for i in range(len(y))]}
    out = node(PRE + "const S=require(D+'/states.cjs');const m=S.calibrateStates(I.d,S.STATES,{perms:0});"
               "const X=I.d.slice(0,5);console.log(JSON.stringify({checks:m.checks,bacc:m.bacc,face:m.faceOnly.bacc,W:m.head.layers[0].W,b:m.head.layers[0].b,"
               "mu:m.head.face.mu,sd:m.head.face.sd,hz:m.head.face.hold_z,p:X.map(w=>Array.from(S.stateProbs(m,w.x,w.f)))}))", payload)
    a_py = np.array([py["checks"][s]["auc"] for s in ST_NAMES])
    a_ts = np.array([out["checks"][s]["auc"] for s in ST_NAMES])
    assert np.allclose(a_py, a_ts, atol=1e-9), (a_py, a_ts)
    assert abs(py["bacc"] - out["bacc"]) < 1e-12 and abs(py["face_only"]["bacc"] - out["face"]) < 1e-12
    err = max(np.abs(py["layer"]["W"] - np.array(out["W"])).max() / (np.abs(py["layer"]["W"]).max() + 1e-12),
              np.abs(py["layer"]["b"] - np.array(out["b"])).max() / (np.abs(py["layer"]["b"]).max() + 1e-12),
              np.abs(py["face"]["mu"] - np.array(out["mu"])).max(), np.abs(py["face"]["sd"] - np.array(out["sd"])).max(),
              abs(py["face"]["hold_z"] - out["hz"]))
    assert err < 1e-7, err
    pp = ST.probs(py, X[:5], F[:5])
    assert np.abs(pp - np.array(out["p"])).max() < 1e-7
    print(f"states parity OK: held-out AUCs, accuracy {py['bacc']:.2f}, face-only reference, folded layer (rel. diff {err:.1e}) and live probabilities agree")


ST_NAMES = ["calm", "stress", "anxiety", "anger", "low_mood", "drowsy"]


def main():
    rng = np.random.default_rng(1)
    test_dsp(rng)
    test_montage(rng)
    test_model(rng)
    test_graph_and_temperature(rng)
    test_bundled_models(rng)
    test_face(rng)
    test_edf(rng)
    test_arm(rng)
    test_asr(rng)
    test_artifact(rng)
    test_states(rng)


if __name__ == "__main__":
    main()
