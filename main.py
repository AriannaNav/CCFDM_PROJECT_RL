# main.py
from __future__ import annotations

import argparse
import sys


KNOWN_SUBCOMMANDS = {"train", "eval", "video", "plots", "run"}


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


def parse_args_and_remaining(argv: list[str]):
    """
    Supports BOTH:
      - subcommands:   python main.py eval --model_dir ...
      - legacy flags:  python main.py --eval --render --model_dir ...

    Returns: (mode, args, remaining)
      mode in {"subcommand","legacy"}
    """
    if len(argv) >= 2 and argv[1] in KNOWN_SUBCOMMANDS:
        # ---- subcommand mode (existing behavior)
        p = argparse.ArgumentParser("CCFDM project entrypoint")
        sub = p.add_subparsers(dest="cmd", required=True)

        sub.add_parser("train", help="Run training (train_ccfdm.py)")
        sub.add_parser("eval", help="Run evaluation (eval.py)")
        sub.add_parser("video", help="Generate videos (video.py)")
        sub.add_parser("plots", help="Generate plots (plots.py)")

        run = sub.add_parser("run", help="Run eval + video + plots sequentially")
        run.add_argument("--do_eval", action="store_true")
        run.add_argument("--do_video", action="store_true")
        run.add_argument("--do_plots", action="store_true")

        args, remaining = p.parse_known_args(argv[1:])
        # remaining already excludes the "cmd" token because argparse consumed it.
        # But we want to forward exactly what comes after the subcommand in original argv.
        # Example: argv = ["main.py","eval","--model_dir","X"]
        # args consumes "eval", remaining becomes ["--model_dir","X"]
        return "subcommand", args, remaining

    # ---- legacy mode: python main.py --eval --render ...
    p = argparse.ArgumentParser("CCFDM project entrypoint (legacy flags mode)")
    p.add_argument("--train", action="store_true", help="Run training (train_ccfdm.py)")
    p.add_argument("--eval", action="store_true", help="Run evaluation (eval.py)")
    p.add_argument("--video", action="store_true", help="Run video rollout (video.py)")
    p.add_argument("--render", action="store_true", help="Alias for --video")
    p.add_argument("--plots", action="store_true", help="Run plots (plots.py)")

    # convenience: run sequentially (like your desired eval+render)
    p.add_argument("--run", action="store_true", help="Run selected actions sequentially")

    args, remaining = p.parse_known_args(argv[1:])

    # normalize render -> video
    if args.render:
        args.video = True

    # if user did not specify any action, print help and exit
    if not (args.train or args.eval or args.video or args.plots):
        p.print_help()
        raise SystemExit(2)

    return "legacy", args, remaining


def main():
    mode, args, remaining = parse_args_and_remaining(sys.argv)

    # ---------------- subcommand mode ----------------
    if mode == "subcommand":
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
            # reuse the SAME forwarded args for each module.
            if args.do_eval:
                _forward("eval", "eval.py", remaining)
            if args.do_video:
                _forward("video", "video.py", remaining)
            if args.do_plots:
                _forward("plots", "plots.py", remaining)
            return

        raise RuntimeError(f"Unknown cmd: {args.cmd}")

    # ---------------- legacy flags mode ----------------
    # Run in a sensible order: train -> eval -> video -> plots
    if getattr(args, "train", False):
        _forward("train_ccfdm", "train_ccfdm.py", remaining)

    if getattr(args, "eval", False):
        _forward("eval", "eval.py", remaining)

    if getattr(args, "video", False):
        _forward("video", "video.py", remaining)

    if getattr(args, "plots", False):
        _forward("plots", "plots.py", remaining)


if __name__ == "__main__":
    main()