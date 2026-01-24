# plots.py
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_eval_curve(eval_jsonl_path) :
    xs, ys = [], []
    with open(eval_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "step" in rec and "eval/mean_return" in rec:
                xs.append(int(rec["step"]))
                ys.append(float(rec["eval/mean_return"]))
    return xs, ys


def parse_args():
    p = argparse.ArgumentParser("Plot evaluation curves (Fig.5 style)")
    p.add_argument("--log_dir", type=str, required=True, help=".../logs/ccfdm/<env_tag>/seed_X")
    p.add_argument("--out", type=str, default=None, help="output png path (default: <log_dir>/fig5_eval_curve.png)")
    p.add_argument("--title", type=str, default="Evaluation Score Curve")
    return p.parse_args()


def main_plots():
    args = parse_args()

    eval_path = os.path.join(args.log_dir, "eval.jsonl")
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(f"eval.jsonl not found: {eval_path}")

    xs, ys = read_eval_curve(eval_path)

    if len(xs) == 0:
        raise RuntimeError("No eval points found in eval.jsonl (missing eval/mean_return logs).")

    out_path = args.out or os.path.join(args.log_dir, "fig5_eval_curve.png")

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Test Return")
    plt.title(args.title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved plot: {out_path}")


if __name__ == "__main__":
    main_plots()
