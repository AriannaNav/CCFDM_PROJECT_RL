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


def env_tag(spec: EnvSpec) -> str:
    name = spec.name.lower().strip()
    if name == "dmc":
        return f"dmc_{spec.domain}_{spec.task}"
    if name == "minigrid":
        return f"minigrid_{spec.env_id}"
    return f"env_{name}"


def parse_args():
    p = argparse.ArgumentParser("CCFDM evaluation (loads best.pt)")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=12345, help="seed for eval env + rng")
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--model_dir", type=str, required=True, help=".../models/ccfdm/<env_tag>/seed_X")
    p.add_argument("--episodes", type=int, default=10)

    return p.parse_args()


@torch.no_grad()
def run_eval(agent, env, episodes):
    agent.train(False)
    agent.critic_target.eval()

    rets, lens = [], []
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

    # seed rng
    set_seed(args.seed, args.deterministic, device=device)

    cfg_path = os.path.join(args.model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in model_dir: {args.model_dir}")

    cfg = load_json(cfg_path)
    spec = EnvSpec(**cfg["env_spec"])
    train_args = cfg.get("train_args", {})

    # IMPORTANT: seed eval env explicitly
    spec.seed = int(args.seed)

    env = make_env(spec)
    print(f"[INFO] Env: {env_tag(spec)} seed={spec.seed}")

    best_path = os.path.join(args.model_dir, "best.pt")
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"best.pt not found: {best_path}")

    agent_kwargs = dict(
        obs_shape=env.obs_shape,
        action_shape=env.action_shape,
        device=device,

        # architecture (MUST match train)
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        encoder_feature_dim=int(train_args.get("encoder_feature_dim", 50)),
        num_layers=int(train_args.get("num_layers", 4)),
        num_filters=int(train_args.get("num_filters", 32)),

        # rl/ccfdm hparams
        discount=float(train_args.get("discount", 0.99)),
        critic_tau=float(train_args.get("critic_tau", 0.01)),
        encoder_tau=float(train_args.get("encoder_tau", 0.01)),
        actor_update_freq=int(train_args.get("actor_update_freq", 2)),
        critic_target_update_freq=int(train_args.get("critic_target_update_freq", 2)),
        ccfmd_update_freq=int(train_args.get("ccfmd_update_freq", 1)),
        contrastive_method=str(train_args.get("contrastive_method", "infonce")),
        temperature=float(train_args.get("temperature", 1.0)),
        normalize=bool(train_args.get("normalize", True)),
        triplet_margin=float(train_args.get("triplet_margin", 0.2)),
        curiosity_C=float(train_args.get("curiosity_C", 0.2)),
        curiosity_gamma=float(train_args.get("curiosity_gamma", 2e-5)),
        intrinsic_weight=float(train_args.get("intrinsic_weight", 0.2)),
        momentum_update_freq=int(train_args.get("momentum_update_freq", 1)),
    )

    agent = CCFDMAgent(**agent_kwargs)
    agent.load(best_path)

    stats = run_eval(agent, env, episodes=args.episodes)
    print("[EVAL]", stats)

    save_json(os.path.join(args.model_dir, "eval_result.json"), {"env_spec": asdict(spec), **stats})

    env.close()


if __name__ == "__main__":
    main_eval()