# main.py
from __future__ import annotations

import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser("CCFDM project entrypoint")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Run training (train_ccfdm.py)")
    sub.add_parser("eval", help="Run evaluation (eval.py)")
    sub.add_parser("video", help="Generate videos (video.py)")
    sub.add_parser("plots", help="Generate plots (plots.py)")

    # run = multiple actions in sequence (episode-based workflows)
    run = sub.add_parser("run", help="Run eval + video + plots sequentially")
    run.add_argument("--do_eval", action="store_true")
    run.add_argument("--do_video", action="store_true")
    run.add_argument("--do_plots", action="store_true")

    return p.parse_args()


def _forward(module_name: str, argv0: str, remaining: list[str]):
    mod = __import__(module_name, fromlist=["main"])
    sys.argv = [argv0] + remaining
    # each module must expose a main_* function
    fn = getattr(mod, f"main_{module_name.split('.')[-1]}", None)
    if fn is None:
        # fallback: common "main"
        fn = getattr(mod, "main", None)
    if fn is None:
        raise RuntimeError(f"{module_name} has no entry function.")
    fn()


def main():
    args = parse_args()
    remaining = sys.argv[2:]

    if args.cmd == "train":
        _forward("train_ccfdm", "train_ccfdm.py", remaining)
        return

    if args.cmd == "eval":
        _forward("eval", "eval.py", remaining)
        return

    if args.cmd == "video":
        _forward("video", "video.py", remaining)
        return

    if args.cmd == "plots":
        _forward("plots", "plots.py", remaining)
        return

    if args.cmd == "run":
        # example: python main.py run --do_eval --do_video -- --model_path ... (forwarded)
        # We reuse the SAME remaining args for each module.
        if args.do_eval:
            _forward("eval", "eval.py", remaining)
        if args.do_video:
            _forward("video", "video.py", remaining)
        if args.do_plots:
            _forward("plots", "plots.py", remaining)
        return


if __name__ == "__main__":
    main()