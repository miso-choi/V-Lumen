#!/usr/bin/env python3
"""Evaluate a V-LUMEN / LoRA / MoE checkpoint (and the vanilla backbone) on the paper benchmarks.

  # Tab. 2: reasoning benchmarks (held-out 90% split, up to 1,000 examples each)
  python scripts/evaluate.py --ckpt checkpoints/vlumen_only_seed1.pt --benchmarks reasoning

  # Tab. 3: zero-shot benchmarks
  python scripts/evaluate.py --ckpt checkpoints/vlumen_only_seed1.pt --benchmarks zero_shot
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vlumen.checkpoint import load_model, load_processor
from vlumen.data.benchmarks import BENCHMARKS, REASONING_BENCHMARKS, ZERO_SHOT_BENCHMARKS, load_benchmark
from vlumen.evaluation import evaluate

GROUPS = {"reasoning": REASONING_BENCHMARKS, "zero_shot": ZERO_SHOT_BENCHMARKS}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--benchmarks", default="reasoning",
                   help="'reasoning', 'zero_shot', or a comma-separated list of: " + ", ".join(BENCHMARKS))
    p.add_argument("--n_eval", type=int, default=1000, help="max examples per benchmark")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--no_vanilla", action="store_true", help="skip the vanilla-backbone pass (2x faster)")
    p.add_argument("--out", default=None, help="output JSON (default: results/<ckpt name>_<benchmarks>.json)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    names = GROUPS.get(args.benchmarks) or tuple(b.strip() for b in args.benchmarks.split(",") if b.strip())
    unknown = [b for b in names if b not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"unknown benchmarks: {unknown}")
    out = Path(args.out or f"results/{Path(args.ckpt).stem}_{args.benchmarks.replace(',', '+')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg, ckpt = load_model(args.ckpt, args.device)
    processor = load_processor(cfg.model_id)
    print(f"[eval] {args.ckpt}: setting={cfg.train_target} stage={ckpt['stage']} step={ckpt['step']}")

    report = {"ckpt": str(args.ckpt), "config": cfg.to_dict(), "n_eval": args.n_eval,
              "max_new_tokens": args.max_new_tokens, "benchmarks": {}}
    for name in names:
        t0 = time.time()
        samples = load_benchmark(name, args.n_eval, split="eval")
        print(f"[eval] {name}: {len(samples)} examples")
        results, metrics = evaluate(model, processor, samples, args.max_new_tokens, args.device,
                                    with_vanilla=not args.no_vanilla)
        metrics["seconds"] = round(time.time() - t0, 1)
        report["benchmarks"][name] = {"metrics": metrics, "results": results}
        print(f"[eval] {name}: {metrics}", flush=True)
        out.write_text(json.dumps(report, indent=2))

    accs = {n: r["metrics"] for n, r in report["benchmarks"].items()}
    report["summary"] = {"avg_acc": sum(m["acc"] for m in accs.values()) / len(accs)}
    if not args.no_vanilla:
        report["summary"]["avg_vanilla_acc"] = sum(m["vanilla_acc"] for m in accs.values()) / len(accs)
        report["summary"]["avg_delta"] = report["summary"]["avg_acc"] - report["summary"]["avg_vanilla_acc"]
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "benchmark".ljust(14) + "acc(%)".rjust(9) + ("vanilla(%)".rjust(12) if not args.no_vanilla else ""))
    for n, m in accs.items():
        line = n.ljust(14) + f"{m['acc'] * 100:9.2f}"
        if not args.no_vanilla:
            line += f"{m['vanilla_acc'] * 100:12.2f}"
        print(line)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
