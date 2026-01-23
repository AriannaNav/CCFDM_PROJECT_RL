from __future__ import annotations

import argparse
import torch

from utils import get_device, device_info, set_seed

def parse_args():
    parser = argparse.ArgumentParser("CCFDM training entry point")

    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps"],
                        help="Device selection")

    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")

    parser.add_argument("--deterministic", action="store_true",
                        help="Force deterministic execution")

    parser.add_argument("--test_utils", action="store_true",
                        help="Only test device + seed and exit")

    parser.add_argument("--dry_run", action="store_true",
                        help="Run a minimal sanity check and exit")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Device
    device = get_device(args.device)
    print(f"[INFO] Device selected: {device_info(device)}")

    # 2) Seed
    set_seed(
        seed=args.seed,
        deterministic=args.deterministic,
        device=device,
    )
    print(f"[INFO] Seed set to {args.seed} (deterministic={args.deterministic})")

    # 3) Very simple sanity check
    x = torch.randn(3, 3, device=device)
    print("[INFO] Sanity tensor device:", x.device)

    # Test-only mode
    if args.test_utils:
        print("[OK] utils.py test passed.")
        return

    # Dry run mode (future hook)
    if args.dry_run:
        print("[OK] Dry run completed. No training executed.")
        return

    # ---------------------------------------------------
    # From here: REAL TRAINING (later)
    # ---------------------------------------------------
    print("[INFO] Starting training (not implemented yet).")

