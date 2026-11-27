#!/usr/bin/env python3
"""Download the released seed-1 checkpoints (Tab. 2 / Tab. 3) from Google Drive into checkpoints/.

  python scripts/download_checkpoints.py            # all five
  python scripts/download_checkpoints.py vlumen_only lora_vlumen
"""

import hashlib
import sys
from pathlib import Path

import gdown

# name -> (Google Drive file id, sha256)
CHECKPOINTS = {
    "vlumen_only":  ("1yu0z2WcvXVtJqxThXWnR8glZkkcBmwyo",
                    "327d80f4ad6bfde81a42247de04705c92c92cdce58b51f626b7035db6d20d488"),
    "lora_only":    ("1Ey1cine3NccTEheDwWrhNPufX6YDCy8z",
                    "b15e07682613a2e3899feec0727959cb90d645d2d1a88fb922066f1f681aa01a"),
    "lora_vlumen":  ("1FHdoGO4dBuXW4Xlp5kS2XbUfPlnvnMs3",
                    "ac611a2cd42e6e8d32d0438189ae36652bd11dfc0f42797f79a50118d1c763ef"),
    "moe_only":     ("1p-VLGHNwUB0Hrcnu_dwprMowfAwqOCiO",
                    "6d2c76a346a91e8cbeb2a6719d3928d3fa816ca3d200be504bc928c5ac15f798"),
    "moe_vlumen":   ("1euBBEyuXOhve6B9FFHwFh43UVPbf3_Rc",
                    "1bb0387a8e213994df03c5001dc892e3a1d52c1667171e3162a2bbb5a543f6d8"),
}
OUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(names):
    OUT_DIR.mkdir(exist_ok=True)
    for name in names or CHECKPOINTS:
        file_id, digest = CHECKPOINTS[name]
        dst = OUT_DIR / f"{name}_seed1.pt"
        if not (dst.exists() and sha256(dst) == digest):
            gdown.download(id=file_id, output=str(dst), quiet=False)
        ok = sha256(dst) == digest
        print(f"{dst}: {'OK' if ok else 'CHECKSUM MISMATCH'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
