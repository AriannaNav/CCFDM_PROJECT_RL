# video.py
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import imageio.v2 as imageio

from make_env import EnvSpec, make_env
from utils import get_device, device_info, set_seed, make_dir
from ccfdm_agent import CCFDMAgent
from data import center_crop 


def load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_from_cfg(cfg: dict, seed_override: int | None = None):
    spec_dict = cfg.get("env_spec", None)
    if not isinstance(spec_dict, dict):
        raise ValueError("config.json must contain env_spec dict")

    spec = EnvSpec(**spec_dict)
    if seed_override is not None:
        spec.seed = int(seed_override)

    env = make_env(spec)
    return env, spec


def obs_to_rgb_frame(obs_chw_stack: np.ndarray) -> np.ndarray:
    """
    Expected obs: (3*k, H, W) uint8 (frame-stacked CHW).
    Render the LAST RGB frame => (H, W, 3) uint8.
    """
    if not isinstance(obs_chw_stack, np.ndarray):
        obs_chw_stack = np.asarray(obs_chw_stack)

    if obs_chw_stack.ndim != 3:
        raise ValueError(f"Expected obs with shape (C,H,W), got {obs_chw_stack.shape}")

    c, h, w = obs_chw_stack.shape
    if c % 3 != 0:
        raise ValueError(f"Expected stacked RGB with channels multiple of 3, got {obs_chw_stack.shape}")

    last = obs_chw_stack[-3:, :, :]  # (3,H,W)
    frame = np.transpose(last, (1, 2, 0))  # (H,W,3)

    # if obs is float (0..1), convert safely
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)

    return frame


def save_video_mp4(frames: list[np.ndarray], path: str, fps: int = 30):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=int(fps))
    except Exception as e:
        npz_path = os.path.splitext(path)[0] + ".npz"
        np.savez_compressed(npz_path, frames=np.stack(frames, axis=0))
        print(f"[WARN] MP4 failed ({e}). Saved frames to {npz_path} instead.")


def build_agent_from_cfg(env, device, cfg: dict) -> CCFDMAgent:
    train_args = cfg.get("train_args", {}) or {}

    # IMPORTANT: must match training architecture/hparams, otherwise load(best.pt) may fail
    agent_kwargs = dict(
        img_size = int(train_args.get("image_size", 84)),
        obs_shape=(env.obs_shape[0], img_size, img_size),
        action_shape=env.action_shape,
        device=device,

        # architecture
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        encoder_feature_dim=int(train_args.get("encoder_feature_dim", 50)),
        num_layers=int(train_args.get("num_layers", 4)),
        num_filters=int(train_args.get("num_filters", 32)),

        # rl/ccfdm hyperparams (not strictly needed for inference, but harmless and keeps consistency)
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

    return CCFDMAgent(**agent_kwargs)


@torch.no_grad()
def rollout_video(env, agent, episodes: int, deterministic: bool, max_steps: int | None):
    videos = []

    for _ep in range(int(episodes)):
        obs, _info = env.reset()
        done = False
        frames = [obs_to_rgb_frame(obs)]
        t = 0

        while not done:
            obs_in = center_crop(obs, out_size=84)

            if deterministic:
                action = agent.select_action(obs_in)
            else:
                action = agent.sample_action(obs_in)

            action = np.clip(action, env.action_low, env.action_high).astype(np.float32)
            obs, _reward, terminated, truncated, _info = env.step(action)

            frames.append(obs_to_rgb_frame(obs))

            done = bool(terminated or truncated)
            t += 1
            if max_steps is not None and t >= int(max_steps):
                break

        videos.append(frames)

    return videos


def parse_args():
    p = argparse.ArgumentParser("video.py — rollout and render best.pt")

    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint filename inside model_dir (default: best.pt)")
    p.add_argument("--out_dir", type=str, default="videos")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=12345, help="seed for env + rng (video reproducibility)")
    p.add_argument("--deterministic_seed", action="store_true")

    return p.parse_args()


def main_video():
    args = parse_args()

    device = get_device(args.device)
    print(f"[INFO] Device: {device_info(device)}")
    set_seed(args.seed, args.deterministic_seed, device=device)

    cfg = load_config(args.model_dir)

    # IMPORTANT: use a fixed seed for video env so the rollout is reproducible
    env, spec = build_env_from_cfg(cfg, seed_override=args.seed)

    agent = build_agent_from_cfg(env, device, cfg)
    agent.train(False)
    agent.critic_target.eval()

    ckpt_path = os.path.join(args.model_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    agent.load(ckpt_path)
    print(f"[INFO] Loaded: {ckpt_path}")

    out_dir = make_dir(args.out_dir)

    videos = rollout_video(
        env=env,
        agent=agent,
        episodes=args.episodes,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
    )

    # naming with env tag
    tag = spec.name
    if spec.name == "dmc":
        tag = f"dmc_{spec.domain}_{spec.task}"
    elif spec.name == "minigrid":
        tag = f"minigrid_{spec.env_id}"

    for i, frames in enumerate(videos, start=1):
        path = os.path.join(out_dir, f"{tag}_seed{spec.seed}_ep{i:03d}.mp4")
        save_video_mp4(frames, path, fps=args.fps)
        print(f"[OK] Saved: {path}")

    env.close()


if __name__ == "__main__":
    main_video()