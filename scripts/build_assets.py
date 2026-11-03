#!/usr/bin/env python3
"""Build the world's image library (app/assets/) from public, openly licensed photographs.

  python scripts/build_assets.py            # downloads (checksums pinned), resizes, writes app/assets + CREDITS.md

Ground photographs: Babylon.js Assets (CC BY 4.0, attribution in app/assets/CREDITS.md and in the app).
Sky photographs: Poly Haven HDRIs (CC0), as shipped with three.js r160 examples, tone-mapped to 8-bit panoramas; plus
the Babylon.js "TropicalSunnyDay" skybox side (CC BY 4.0).
Leaf and bark: drawn here, procedurally (no third-party image).
"""
import hashlib
import io
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rgbe import read_hdr  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "app", "assets")
BAB = "https://raw.githubusercontent.com/BabylonJS/Assets/master/"
THREE = "https://raw.githubusercontent.com/mrdoob/three.js/r160/examples/textures/"

# name: (url, sha256 of the downloaded file, licence, credit)
SOURCES = {
    "grass": (BAB + "textures/grass.png", None, "CC BY 4.0", "Babylon.js Assets, textures/grass.png"),
    "grass_n": (BAB + "textures/grassn.png", None, "CC BY 4.0", "Babylon.js Assets, textures/grassn.png"),
    "rock": (BAB + "textures/rock.png", None, "CC BY 4.0", "Babylon.js Assets, textures/rock.png"),
    "rock_n": (BAB + "textures/rockn.png", None, "CC BY 4.0", "Babylon.js Assets, textures/rockn.png"),
    "dirt": (BAB + "textures/dirt.jpg", None, "CC BY 4.0", "Babylon.js Assets, textures/dirt.jpg"),
    "sand": (BAB + "textures/ground.jpg", None, "CC BY 4.0", "Babylon.js Assets, textures/ground.jpg"),
    "sky_day": (THREE + "equirectangular/quarry_01_1k.hdr", None, "CC0", "Poly Haven, Quarry 01 (Greg Zaal)"),
    "sky_dawn": (THREE + "equirectangular/blouberg_sunrise_2_1k.hdr", None, "CC0", "Poly Haven, Blouberg Sunrise 2 (Greg Zaal)"),
    "sky_night": (THREE + "equirectangular/moonless_golf_1k.hdr", None, "CC0", "Poly Haven, Moonless Golf (Greg Zaal)"),
    "sky_clouds": (BAB + "skyboxes/TropicalSunnyDay/TropicalSunnyDay_px.jpg", None, "CC BY 4.0", "Babylon.js Assets, skyboxes/TropicalSunnyDay"),
}
PINS = os.path.join(HERE, "assets_sha256.json")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "emotif-assets/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def save_jpg(im, name, q=84):
    p = os.path.join(OUT, name + ".jpg")
    im.convert("RGB").save(p, "JPEG", quality=q, optimize=True, progressive=True)
    return p


def tonemap(x, exposure):
    x = x * exposure
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def sky_band(img):
    """Upper half of an equirectangular panorama (horizon at the bottom row), as 2048 x 512."""
    w, h = img.size
    return img.crop((0, 0, w, h // 2)).resize((2048, 512), Image.LANCZOS)


# elevation (degrees) below which a sky photo shows buildings, pylons or trees; replaced by the sky just above it
CLEAN_BELOW = {"sky_dawn": 19, "sky_night": 15}


def clean_horizon(band, deg):
    """Replace the rows below `deg` elevation with the row at `deg`, blurred along the horizon (no structures), slightly
    brightening toward the horizon as skies do. band: 2048 x 512, bottom row = horizon."""
    a = np.asarray(band, dtype=np.float32)
    h = a.shape[0]
    y0 = int(round(h * (1 - deg / 90)))
    ref = np.median(a[y0 - 6:y0].reshape(-1, 3), axis=0)[None, :].repeat(a.shape[1], 0)   # structure-free sky colour
    for y in range(y0, h):
        t = (y - y0) / max(1, h - 1 - y0)
        a[y] = np.clip(ref * (1 + 0.18 * t), 0, 255)
    # soften the seam
    for k in range(1, 7):
        w = k / 7
        a[y0 - k] = a[y0 - k] * w + a[y0] * (1 - w)
    return Image.fromarray(a.astype(np.uint8))


def seamless(im):
    """Blend opposite edges so a photo tiles without a visible seam (a cross-fade over the outer 1/8)."""
    a = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w, _ = a.shape
    m = w // 8
    out = a.copy()
    for k in range(m):
        t = 0.5 * (1 - k / m)
        out[:, k] = (1 - t) * a[:, k] + t * a[:, w - 1 - k]
        out[:, w - 1 - k] = (1 - t) * a[:, w - 1 - k] + t * a[:, k]
    a = out.copy()
    for k in range(m):
        t = 0.5 * (1 - k / m)
        out[k] = (1 - t) * a[k] + t * a[h - 1 - k]
        out[h - 1 - k] = (1 - t) * a[h - 1 - k] + t * a[k]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def leaves(seed=3, size=512):
    """A cluster of leaves with alpha (drawn here): used on the crown cards of every tree."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for _ in range(1100):
        cx, cy = rng.normal(size / 2, size / 5, 2)
        r = np.hypot(cx - size / 2, cy - size / 2)
        if r > size * 0.47:
            continue
        L, W = rng.uniform(16, 30), rng.uniform(7, 12)
        a = rng.uniform(0, np.pi)
        tone = rng.uniform(0.55, 1.0)
        g = int(150 * tone + 40)
        col = (int(70 * tone + 20), g, int(40 * tone + 10), 255)
        pts = [(cx + np.cos(a + s) * L * f, cy + np.sin(a + s) * L * f) for s, f in ((0, 1), (np.pi / 2, W / L), (np.pi, 1), (3 * np.pi / 2, W / L))]
        d.polygon(pts, fill=col)
        d.line([(cx - np.cos(a) * L * 0.9, cy - np.sin(a) * L * 0.9), (cx + np.cos(a) * L * 0.9, cy + np.sin(a) * L * 0.9)],
               fill=(int(col[0] * 0.7), int(col[1] * 0.75), int(col[2] * 0.7), 255), width=1)
    return im.filter(ImageFilter.GaussianBlur(0.6))


def bark(seed=5, size=256):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, size, endpoint=False)
    img = np.zeros((size, size))
    for k in range(1, 9):
        ph = rng.uniform(0, 2 * np.pi, 3)
        img += (0.6 ** k) * np.sin(2 * np.pi * (k * 3) * x[None, :] + ph[0] + 0.8 * np.sin(2 * np.pi * k * x[:, None] + ph[1]))
    img += 0.25 * rng.normal(0, 1, (size, size))
    img = (img - img.min()) / (img.max() - img.min())
    base = np.array([96, 72, 52], np.float32)
    rgb = base[None, None] * (0.55 + 0.7 * img[..., None])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))


def main():
    os.makedirs(OUT, exist_ok=True)
    pins = json.load(open(PINS)) if os.path.exists(PINS) else {}
    credits = []
    for name, (url, _, lic, credit) in SOURCES.items():
        raw = fetch(url)
        h = hashlib.sha256(raw).hexdigest()
        if name in pins and pins[name] != h:
            raise SystemExit(f"{name}: checksum changed ({h}); check the source before accepting it")
        pins[name] = h
        if url.endswith(".hdr"):
            tmp = os.path.join(OUT, "_tmp.hdr")
            open(tmp, "wb").write(raw)
            x = read_hdr(tmp)
            os.remove(tmp)
            target = 0.08 if name == "sky_night" else 0.55          # night stays dark
            exp = target / (np.percentile(x[: x.shape[0] // 2], 75) + 1e-6)
            img = Image.fromarray((tonemap(x, exp) * 255).astype(np.uint8))
            band = sky_band(img)
            save_jpg(clean_horizon(band, CLEAN_BELOW[name]) if name in CLEAN_BELOW else band, name, 86)
        else:
            img = Image.open(io.BytesIO(raw))
            if name.startswith("sky_"):
                if name == "sky_clouds":      # a cube face: sky is its upper half; stretch to a panorama band
                    w, hh = img.size
                    band = img.convert("RGB").crop((0, 0, w, hh // 2)).resize((1024, 512), Image.LANCZOS)
                    pano = Image.new("RGB", (2048, 512))
                    pano.paste(band, (0, 0)); pano.paste(band.transpose(Image.FLIP_LEFT_RIGHT), (1024, 0))
                    save_jpg(pano, name, 86)
                else:
                    band = sky_band(img.convert("RGB"))
                    save_jpg(clean_horizon(band, CLEAN_BELOW[name]) if name in CLEAN_BELOW else band, name, 86)
            else:
                size = 512
                im = img.convert("RGB").resize((size, size), Image.LANCZOS)
                save_jpg(im if name.endswith("_n") else seamless(im), name, 88)
        credits.append(f"- `{name}`: {credit}. Licence: {lic}. Source: {url}")
    leaves().save(os.path.join(OUT, "leaves.png"), optimize=True)
    save_jpg(bark(), "bark", 88)
    credits.append("- `leaves`, `bark`: drawn procedurally by scripts/build_assets.py. Licence: same as Emotif.")
    json.dump(pins, open(PINS, "w"), indent=1)
    with open(os.path.join(OUT, "CREDITS.md"), "w") as fh:
        fh.write("# Image credits\n\nGround and sky photographs used by the Natural world style. CC BY 4.0 items require this attribution.\n\n"
                 + "\n".join(credits) + "\n\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/ · CC0: https://creativecommons.org/publicdomain/zero/1.0/\n")
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"wrote {len(os.listdir(OUT))} files to {OUT} ({total / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
