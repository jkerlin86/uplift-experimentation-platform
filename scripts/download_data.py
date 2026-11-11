"""Verify the Criteo Uplift v2 file, downloading it only if absent.

The file is distributed by Criteo AI Lab through its official Hugging
Face organisation (linked from https://ailab.criteo.com/ressources/).
The expected SHA-256 in configs/default.json is the hash published on
the file's own page in that repository. The data are CC BY-NC-SA 4.0
and are never committed to git (see data/README.md).
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift_platform.dataio import sha256_file


def main() -> None:
    config = json.loads((ROOT / "configs/default.json").read_text())
    spec = config["data"]
    raw = ROOT / "data/raw" / spec["raw_file"]
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        print(f"downloading {spec['source_url']}")
        with urllib.request.urlopen(spec["source_url"]) as resp, open(raw, "wb") as fh:
            while chunk := resp.read(1 << 22):
                fh.write(chunk)
    digest = sha256_file(raw)
    size = raw.stat().st_size
    print(f"file: {raw.name} bytes={size} sha256={digest}")
    assert size == spec["bytes"], f"size mismatch: {size} != {spec['bytes']}"
    assert digest == spec["sha256"], "SHA-256 mismatch against the published hash"
    print("verified against the hash published by Criteo's official channel")


if __name__ == "__main__":
    main()
