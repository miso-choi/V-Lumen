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
    "vlumen_only":  ("TODO_FILE_ID",
                    "3d5d546de6195c9334d88f37aa1802e252526dc1ef46eb7d16304e4ded923677"),
    "lora_only":    ("TODO_FILE_ID",
                    "b2832611ddde8c9bc84ea9dc18c092a26f1882059b19186c6eb78110dc06fb0d"),
    "lora_vlumen":  ("TODO_FILE_ID",
                    "371d694572501bead98175a1b5a939d08328f080df9acbae9c6397926b87a7be"),
    "moe_only":     ("TODO_FILE_ID",
                    "7849cd58f9b440b88d41c2e816f0121dd12bd9bdaa98a4f18becf906aad3d896"),
    "moe_vlumen":   ("TODO_FILE_ID",
                    "bf8d86a4fbd4a00c225b130e77c0177bede7b5f03cc5acf312061e6aafc12408"),
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
