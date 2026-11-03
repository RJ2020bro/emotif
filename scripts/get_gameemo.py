#!/usr/bin/env python3
"""Download GAMEEMO (Alakus et al. 2020, CC BY 4.0) CSV files from Mendeley Data, checking each SHA-256.

  python scripts/get_gameemo.py [--out data/external/gameemo] [--kinds raw,preprocessed]

Keeps the dataset's own folder layout: GAMEEMO/(S01)/Raw EEG Data/.csv format/S01G1AllRawChannels.csv
"""
import argparse
import hashlib
import json
import os
import urllib.request

API = "https://data.mendeley.com/public-api/datasets/b3pn4kwpmn"
VERSION = 3
KIND_DIR = {"raw": "Raw EEG Data", "preprocessed": "Preprocessed EEG Data"}


HEADERS = {"User-Agent": "emotif-dataset-fetch/1.0 (+https://data.mendeley.com)"}   # Mendeley rejects Python's default agent


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return json.load(r)


def download(url, dest):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=300) as r, open(dest, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/external/gameemo")
    p.add_argument("--kinds", default="raw,preprocessed")
    a = p.parse_args()
    kinds = [KIND_DIR[k.strip()] for k in a.kinds.split(",") if k.strip()]
    folders = get_json(f"{API}/folders/{VERSION}")
    by = {f["id"]: f for f in folders}

    def path(f):
        parts = [f["name"]]
        while f.get("parent_id") in by:
            f = by[f["parent_id"]]
            parts.append(f["name"])
        return list(reversed(parts))

    targets = [f for f in folders if len(path(f)) == 4 and path(f)[2] in kinds and path(f)[3] == ".csv format"]
    n = 0
    for f in sorted(targets, key=lambda x: path(x)):
        d = os.path.join(a.out, *path(f))
        os.makedirs(d, exist_ok=True)
        for item in get_json(f"{API}/files?folder_id={f['id']}&version={VERSION}"):
            dest = os.path.join(d, item["filename"])
            want = item["content_details"]["sha256_hash"]
            if os.path.isfile(dest) and sha256(dest) == want:
                continue
            tmp = dest + ".part"
            download(item["content_details"]["download_url"], tmp)
            if sha256(tmp) != want:
                os.remove(tmp)
                raise SystemExit(f"checksum mismatch: {item['filename']}")
            os.replace(tmp, dest)
            n += 1
        print("ok", "/".join(path(f)))
    print(f"downloaded {n} files into {a.out}")


if __name__ == "__main__":
    main()
