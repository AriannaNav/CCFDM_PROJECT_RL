# main.py
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple


KNOWN_SUBCOMMANDS = {"train", "eval", "video", "plots", "run"}


def _import_and_call(module_name: str, argv0: str, forwarded_argv: List[str]):
    mod = __import__(module_name, fromlist=["main"])
    sys.argv = [argv0] + forwarded_argv

    # each module must expose main_<module>
    fn = getattr(mod, f"main_{module_name.split('.')[-1]}", None)
    if fn is None:
        fn = getattr(mod, "main", None)
    if fn is None:
        raise RuntimeError(f"{module_name} has no entry function.")
    fn()


def _filter_forward_args(remaining: List[str], allowed_flags: dict) -> List[str]:
    """
    Keep only args whose flag is in allowed_flags.
    allowed_flags: dict(flag -> takes_value: bool)
    Works with:
      --flag value
      --flag=value
      boolean flags (takes_value=False)
    """
    out: List[str] = []
    i = 0
    n = len(remaining)

    while i < n:
        tok = remaining[i]

        if tok.startswith("--"):
            # --k=v style
            if "=" in tok:
                k, v = tok.split("=", 1)
                if k in allowed_flags:
                    out.append(tok)  # keep as-is
                i += 1
                continue

            # --k value style
            k = tok
            if k in allowed_flags:
                out.append(k)
                if allowed_flags[k]:  # consumes next token as value
                    if i + 1 < n and not remaining[i + 1].startswith("--"):
                        out.append(remaining[i + 1])
                        i += 2
                        continue
                    else:
                        # missing value; let the target parser raise a nice error
                        i += 1
                        continue
                else:
                    i += 1
                    continue

        # non-flag token (positional) -> drop (we don't use positionals)
        i += 1

    return out


def _infer_log_dir_from_model_dir(model_dir: str, models_root: str, logs_root: str) -> str:
    """
    If model_dir = models/.../ccfdm/<env_tag>/seed_X
    infer log_dir = logs/.../ccfdm/<env_tag>/seed_X
    """
    model_dir_abs = os.path.abspath(model_dir)
    models_root_abs = os.path.abspath(models_root)
    logs_root_abs = os.path.abspath(logs_root)

    # If model_dir is inside models_root, keep relative suffix
    if model_dir_abs.startswith(models_root_abs + os.sep):
        suffix = model_dir_abs[len(models_root_abs) + 1 :]
        return os.path.join(logs_root_abs, suffix)

    # otherwise try a simple replacement of "/models/" -> "/logs/"
    # (best-effort)
    return model_dir_abs.replace(os.sep + "models" + os.sep, os.sep + "logs" + os.sep)


def parse_args_and_remaining(argv: List[str]) -> Tuple[str, argparse.Namespace, List[str]]:
    """
    Supports BOTH:
      - subcommands:   python main.py eval --model_dir ...
      - legacy flags:  python main.py --eval --render --model_dir ...
    """
    if len(argv) >= 2 and argv[1] in KNOWN_SUBCOMMANDS:
        # ---------- subcommand mode ----------
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
        return "subcommand", args, remaining

    # ---------- legacy flags mode ----------
    p = argparse.ArgumentParser("CCFDM project entrypoint (legacy flags mode)")
    p.add_argument("--train", action="store_true", help="Run training (train_ccfdm.py)")
    p.add_argument("--eval", action="store_true", help="Run evaluation (eval.py)")
    p.add_argument("--video", action="store_true", help="Run video rollout (video.py)")
    p.add_argument("--render", action="store_true", help="Alias for --video")
    p.add_argument("--plots", action="store_true", help="Run plots (plots.py)")

    # common convenience for legacy
    p.add_argument("--model_dir", type=str, default=None, help="Model directory (used by eval/video; can infer log_dir for plots)")
    p.add_argument("--log_dir", type=str, default=None, help="Log directory (used by plots). If missing, inferred from model_dir.")
    p.add_argument("--models_root", type=str, default="models", help="Used only to infer log_dir from model_dir")
    p.add_argument("--logs_root", type=str, default="logs", help="Used only to infer log_dir from model_dir")

    args, remaining = p.parse_known_args(argv[1:])

    # normalize render -> video
    if args.render:
        args.video = True

    if not (args.train or args.eval or args.video or args.plots):
        p.print_help()
        raise SystemExit(2)

    return "legacy", args, remaining


def main():
    mode, args, remaining = parse_args_and_remaining(sys.argv)

    # ---------------- subcommand mode ----------------
    if mode == "subcommand":
        if args.cmd == "train":
            _import_and_call("train_ccfdm", "train_ccfdm.py", remaining)
            return

        if args.cmd == "eval":
            _import_and_call("eval", "eval.py", remaining)
            return

        if args.cmd == "video":
            _import_and_call("video", "video.py", remaining)
            return

        if args.cmd == "plots":
            _import_and_call("plots", "plots.py", remaining)
            return

        if args.cmd == "run":
            if args.do_eval:
                _import_and_call("eval", "eval.py", remaining)
            if args.do_video:
                _import_and_call("video", "video.py", remaining)
            if args.do_plots:
                _import_and_call("plots", "plots.py", remaining)
            return

        raise RuntimeError(f"Unknown cmd: {args.cmd}")

    # ---------------- legacy flags mode ----------------
    # Define which flags each module should receive (to avoid argparse errors).
    EVAL_FLAGS = {
        "--device": True,
        "--seed": True,
        "--deterministic": False,
        "--model_dir": True,
        "--episodes": True,
    }

    VIDEO_FLAGS = {
        "--device": True,
        "--seed": True,
        "--deterministic_seed": False,
        "--model_dir": True,
        "--ckpt": True,
        "--out_dir": True,
        "--episodes": True,
        "--fps": True,
        "--deterministic": False,
        "--max_steps": True,
    }

    PLOTS_FLAGS = {
        "--log_dir": True,
        "--out": True,
        "--title": True,
    }

    # Base forwarded args from user
    eval_argv = _filter_forward_args(remaining, EVAL_FLAGS)
    video_argv = _filter_forward_args(remaining, VIDEO_FLAGS)
    plots_argv = _filter_forward_args(remaining, PLOTS_FLAGS)

    # Inject model_dir if user provided it at main level but didn't repeat it
    if args.model_dir:
        if "--model_dir" not in eval_argv and getattr(args, "eval", False):
            eval_argv += ["--model_dir", args.model_dir]
        if "--model_dir" not in video_argv and getattr(args, "video", False):
            video_argv += ["--model_dir", args.model_dir]

    # Inject/Infer log_dir for plots if needed
    if getattr(args, "plots", False):
        log_dir = args.log_dir
        if log_dir is None and args.model_dir is not None:
            log_dir = _infer_log_dir_from_model_dir(args.model_dir, args.models_root, args.logs_root)

        if log_dir is not None and "--log_dir" not in plots_argv:
            plots_argv += ["--log_dir", log_dir]

    # Run in desired order for legacy: eval -> video -> plots
    if getattr(args, "train", False):
        _import_and_call("train_ccfdm", "train_ccfdm.py", remaining)

    if getattr(args, "eval", False):
        _import_and_call("eval", "eval.py", eval_argv)

    if getattr(args, "video", False):
        _import_and_call("video", "video.py", video_argv)

    if getattr(args, "plots", False):
        _import_and_call("plots", "plots.py", plots_argv)


if __name__ == "__main__":
    main()