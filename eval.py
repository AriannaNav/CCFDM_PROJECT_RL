# eval.py
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch

from utils import get_device, device_info, set_seed, load_json, save_json
from make_env import EnvSpec, make_env
from ccfdm_agent import CCFDMAgent


def env_tag(spec):
    name = spec.name.lower().strip()
    if name == "dmc":
        return f"dmc_{spec.domain}_{spec.task}"
    if name == "minigrid":
        return f"minigrid_{spec.env_id}"
    return f"env_{name}"


def parse_args():
    p = argparse.ArgumentParser("CCFDM evaluation (loads best.pt)")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--model_dir", type=str, required=True, help=".../models/ccfdm/<env_tag>/seed_X")
    p.add_argument("--episodes", type=int, default=10)

    return p.parse_args()


@torch.no_grad()
def run_eval(agent, env, episodes) :
    agent.train(False)
    agent.critic_target.eval()

    rets = []
    lens = []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        while not done:
            a = agent.select_action(obs)
            a = np.clip(a, env.action_low, env.action_high).astype(np.float32)
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += float(r)
            ep_len += 1
            done = bool(term or trunc)
        rets.append(ep_ret)
        lens.append(ep_len)

    rets = np.asarray(rets, dtype=np.float32)
    lens = np.asarray(lens, dtype=np.int32)

    return {
        "episodes": int(episodes),
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std()),
        "mean_length": float(lens.mean()),
        "std_length": float(lens.std()),
    }


def main_eval():
    args = parse_args()

    device = get_device(args.device)
    print(f"[INFO] Device: {device_info(device)}")
    set_seed(args.seed, args.deterministic, device=device)

    cfg_path = os.path.join(args.model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in model_dir: {args.model_dir}")

    cfg = load_json(cfg_path)
    spec = EnvSpec(**cfg["env_spec"])

    env = make_env(spec)
    print(f"[INFO] Env: {env_tag(spec)}")

    best_path = os.path.join(args.model_dir, "best.pt")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"best.pt not found: {best_path}")

    agent = CCFDMAgent(obs_shape=env.obs_shape, action_shape=env.action_shape, device=device)
    agent.load(best_path)

    stats = run_eval(agent, env, episodes=args.episodes)
    print("[EVAL]", stats)

    save_json(os.path.join(args.model_dir, "eval_result.json"), {"env_spec": asdict(spec), **stats})

    env.close()


if __name__ == "__main__":
    main_eval()