#!/usr/bin/env python3
"""Download a public Mendeley Data dataset (any folder depth), checking every file's SHA-256.

  python scripts/get_mendeley.py DATASET_ID --out data/external/NAME [--include REGEX] [--exclude REGEX]
                                 [--max-mb N] [--list]

--include / --exclude match the file's path inside the dataset ("folder/sub/file.edf"). --list prints the tree with
sizes and downloads nothing. Files already present with the right checksum are skipped, so a stopped download
resumes. The datasets Emotif uses are listed in docs/DATASETS.md with their IDs and licences.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://data.mendeley.com/public-api/datasets"
HEADERS = {"User-Agent": "emotif-dataset-fetch/1.0 (+https://data.mendeley.com)"}   # Mendeley rejects Python's default agent


def get_json(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError):
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    return None


def download(url, dest, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=600) as r, open(dest, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree(ds, version):
    """[(path inside dataset, file record)] for every file in every folder."""
    folders = get_json(f"{API}/{ds}/folders/{version}") or []
    by = {f["id"]: f for f in folders}

    def fpath(f):
        parts = [f["name"]]
        while f.get("parent_id") in by:
            f = by[f["parent_id"]]
            parts.append(f["name"])
        return "/".join(reversed(parts))

    out = [(item["filename"], item) for item in get_json(f"{API}/{ds}/files?folder_id=root&version={version}")]
    for f in folders:
        for item in get_json(f"{API}/{ds}/files?folder_id={f['id']}&version={version}"):
            out.append((f"{fpath(f)}/{item['filename']}", item))
    return sorted(out, key=lambda x: x[0])


def safe_join(root, rel):
    dest = os.path.normpath(os.path.join(root, rel))
    if not dest.startswith(os.path.normpath(root) + os.sep):
        raise SystemExit(f"refusing a path outside the output folder: {rel!r}")
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--out", required=False)
    p.add_argument("--version", type=int, default=None, help="default: the latest")
    p.add_argument("--include", default=None)
    p.add_argument("--exclude", default=None)
    p.add_argument("--max-mb", type=float, default=None, help="skip single files larger than this")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    meta = get_json(f"{API}/{a.dataset}")
    version = a.version or meta["version"]
    lic = (meta.get("data_licence") or {}).get("short_name", "?")
    print(f"{meta['name']} (v{version}, licence {lic})")
    files = tree(a.dataset, version)
    inc = re.compile(a.include) if a.include else None
    exc = re.compile(a.exclude) if a.exclude else None
    chosen = [(rel, it) for rel, it in files
              if (not inc or inc.search(rel)) and not (exc and exc.search(rel))
              and (a.max_mb is None or it.get("size", 0) <= a.max_mb * 1e6)]
    total = sum(it.get("size", 0) for _, it in chosen)
    if a.list or not a.out:
        for rel, it in chosen:
            print(f"{it.get('size', 0) / 1e6:10.2f} MB  {rel}")
        print(f"{len(chosen)} files, {total / 1e6:.1f} MB")
        return 0
    n = 0
    for rel, it in chosen:
        dest = safe_join(a.out, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        want = it["content_details"]["sha256_hash"]
        if os.path.isfile(dest) and sha256(dest) == want:
            continue
        tmp = dest + ".part"
        download(it["content_details"]["download_url"], tmp)
        if sha256(tmp) != want:
            os.remove(tmp)
            raise SystemExit(f"checksum mismatch: {rel}")
        os.replace(tmp, dest)
        n += 1
        print(f"ok {rel}", flush=True)
    print(f"downloaded {n} new files ({total / 1e6:.1f} MB selected) into {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
