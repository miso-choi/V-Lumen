#!/usr/bin/env python3
"""Aggregate evaluation JSONs into a Markdown table (mean +- sample std over seeds).

  python scripts/summarize.py results/tab2/*.json
  python scripts/summarize.py runs/*/eval_reasoning.json

Avg = mean accuracy over the benchmarks; Avg Delta = mean of (model - vanilla) from the same pass.
"""

import json
import statistics
import sys
from collections import defaultdict

NAMES = {"vlumen": "V-LUMEN-only", "lora": "LoRA-only", "lora_vlumen": "LoRA + V-LUMEN",
         "moe": "MoE-only", "moe_vlumen": "MoE + V-LUMEN"}
HEADERS = {"chartmuseum": "ChartMus.", "frieda": "FRIEDA", "visres": "VisRes", "mmmu": "MMMU",
           "mmmu_pro": "MMMU-Pro", "textvqa": "TextVQA", "ocrvqa": "OCR-VQA", "vizwiz": "VizWiz"}


def fmt(vals):
    vals = [100 * v for v in vals]
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def main(paths):
    runs = [json.load(open(p)) for p in paths]
    benches = list(runs[0]["benchmarks"])
    by_setting, vanilla = defaultdict(list), defaultdict(list)
    for r in runs:
        if list(r["benchmarks"]) != benches:
            raise SystemExit("all files must cover the same benchmarks")
        m = {b: r["benchmarks"][b]["metrics"] for b in benches}
        by_setting[r["config"]["train_target"]].append(m)
        for b in benches:
            if "vanilla_acc" in m[b]:
                vanilla[b].append(m[b]["vanilla_acc"])

    cols = [HEADERS.get(b, b) for b in benches] + ["Avg", "Avg Δ"]
    print("| Setting | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    if vanilla:
        v = [statistics.mean(vanilla[b]) for b in benches]
        print("| Vanilla | | " + " | ".join(f"{100 * x:.2f}" for x in v) + f" | {100 * statistics.mean(v):.2f} | – |")
    for key in [k for k in NAMES if k in by_setting]:
        ms = by_setting[key]
        cells = [fmt([m[b]["acc"] for m in ms]) for b in benches]
        cells.append(fmt([statistics.mean(m[b]["acc"] for b in benches) for m in ms]))
        if all("delta" in m[b] for m in ms for b in benches):
            deltas = [statistics.mean(m[b]["delta"] for b in benches) for m in ms]
            cells.append(("+" if statistics.mean(deltas) >= 0 else "") + fmt(deltas))
        else:
            cells.append("–")
        print(f"| {NAMES[key]} | {len(ms)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
