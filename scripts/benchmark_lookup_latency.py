#!/usr/bin/env python3
"""Tab. 1: memory lookup latency vs. memory-table size (one fusion layer, 64 visual tokens, bf16).

A lookup = key construction (SAH axial pooling + coordinate bias + TCH bin + per-cell
hashing; never touches the table) + gather (reading the rows). Both are timed
together and the gather alone. No trained weights are needed.

  python scripts/benchmark_lookup_latency.py --out results/tab1_lookup_latency.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vlumen.config import IMAGE_TOKEN_TYPE, TEXT_TOKEN_TYPE
from vlumen.memory_table import MultiHeadMemoryTable

DEFAULT_ROWS = 81920


def timed(fn, n_warmup: int, n_iter: int) -> dict:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": statistics.mean(ts), "median_ms": statistics.median(ts), "std_ms": statistics.stdev(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_per_cell", default="16384,81920,327680,655360")
    ap.add_argument("--memory_dim", type=int, default=512)
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--n_warmup", type=int, default=100)
    ap.add_argument("--out", default="results/tab1_lookup_latency.json")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    g = 8                                    # 224 x 224 image -> 8 x 8 = 64 visual tokens
    T = 16 + g * g + 32                      # text prefix + image block + question
    token_types = torch.full((1, T), TEXT_TOKEN_TYPE, dtype=torch.long, device=dev)
    token_types[:, 16:16 + g * g] = IMAGE_TOKEN_TYPE
    codes = torch.randint(0, 16384, (1, T), device=dev)
    text_bins = torch.randint(0, 8, (1, g, g), device=dev)

    report = {"config": vars(a) | {"visual_tokens": g * g, "seq_len": T, "dtype": "bfloat16",
                                   "gpu": torch.cuda.get_device_name()}, "rows": []}
    @torch.no_grad()
    def measure(rows: int) -> dict:
        table = MultiHeadMemoryTable(memory_dim=a.memory_dim, n_heads=8, ngram_orders=[2, 3], rows_per_cell=rows,
                                     visual_grid=g).to(dev).to(torch.bfloat16).eval()
        n_params = sum(t.weight.numel() for t in table.tables)
        full = timed(lambda: table(codes, token_types, text_bins), a.n_warmup, a.n_iter)
        addr = table.addresses(codes, token_types, text_bins)

        def gather():
            for o in range(table.n_orders):
                for h in range(table.n_heads):
                    table.tables[o * table.n_heads + h](addr[o][:, :, h])
        gat = timed(gather, a.n_warmup, a.n_iter)
        return {"rows_per_cell": rows, "table_params": n_params, "table_mb_bf16": n_params * 2 / 1e6,
                "lookup_total": full, "gather_only": gat}

    for rows in [int(x) for x in a.rows_per_cell.split(",")]:
        r = measure(rows)
        torch.cuda.empty_cache()
        report["rows"].append(r)
        print(f"rows/cell={rows:7d}  params={r['table_params']:12,d}  lookup={r['lookup_total']['mean_ms']:.2f}"
              f"+-{r['lookup_total']['std_ms']:.2f} ms  gather={r['gather_only']['mean_ms']:.3f} ms", flush=True)

    ref = next((r for r in report["rows"] if r["rows_per_cell"] == DEFAULT_ROWS), report["rows"][0])
    print(f"\n{'rows/cell':>10} {'scale':>6} {'params':>13} {'MB':>7} {'lookup (ms)':>20} {'gather (ms)':>16}")
    for r in report["rows"]:
        r["lookup_vs_default"] = r["lookup_total"]["mean_ms"] / ref["lookup_total"]["mean_ms"]
        r["gather_vs_default"] = r["gather_only"]["mean_ms"] / ref["gather_only"]["mean_ms"]
        print(f"{r['rows_per_cell']:>10,} {r['rows_per_cell'] / DEFAULT_ROWS:>5.1f}x {r['table_params']:>13,} "
              f"{r['table_mb_bf16']:>7.1f} {r['lookup_total']['mean_ms']:>8.2f}+-{r['lookup_total']['std_ms']:.2f} "
              f"({r['lookup_vs_default']:.3f}x) {r['gather_only']['mean_ms']:>7.3f} ({r['gather_vs_default']:.3f}x)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
