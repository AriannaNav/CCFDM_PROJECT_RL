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


def load_config(model_dir):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_from_cfg(cfg):
    spec_dict = cfg.get("env_spec", None)
    if not isinstance(spec_dict, dict):
        raise ValueError("config.json must contain env_spec dict")
    spec = EnvSpec(**spec_dict)
    env = make_env(spec)
    return env, spec


def obs_to_rgb_frame(obs_chw_stack):
    """
    obs is (3*k, H, W) uint8.
    We render the LAST frame (last 3 channels) as HWC uint8.
    """
    c, h, w = obs_chw_stack.shape
    assert c % 3 == 0, f"Expected stacked RGB with channels multiple of 3, got {obs_chw_stack.shape}"
    last = obs_chw_stack[-3:, :, :]
    frame = np.transpose(last, (1, 2, 0))  # HWC
    return frame


def save_video_mp4(frames, path, fps= 30):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=int(fps))
    except Exception as e:
        # fallback: save raw frames
        npz_path = os.path.splitext(path)[0] + ".npz"
        np.savez_compressed(npz_path, frames=np.stack(frames, axis=0))
        print(f"[WARN] MP4 failed ({e}). Saved frames to {npz_path} instead.")


@torch.no_grad()
def rollout_video(env, agent, episodes, deterministic, max_steps):
    videos = []

    for ep in range(int(episodes)):
        obs, _info = env.reset()
        done = False
        frames = [obs_to_rgb_frame(obs)]
        t = 0

        while not done:
            if deterministic:
                action = agent.select_action(obs)
            else:
                action = agent.sample_action(obs)

            action = np.clip(action, env.action_low, env.action_high).astype(np.float32)
            obs, reward, terminated, truncated, _info = env.step(action)

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
    p.add_argument("--out_dir", type=str, default="videos")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deterministic_seed", action="store_true")

    return p.parse_args()


def main_video():
    args = parse_args()

    device = get_device(args.device)
    print(f"[INFO] Device: {device_info(device)}")
    set_seed(args.seed, args.deterministic_seed, device=device)

    cfg = load_config(args.model_dir)
    env, spec = build_env_from_cfg(cfg)

    agent = CCFDMAgent(
        obs_shape=env.obs_shape,
        action_shape=env.action_shape,
        device=device,
    )
    agent.train(False)
    agent.critic_target.eval()

    best_path = os.path.join(args.model_dir, "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Missing best.pt in {args.model_dir}")
    agent.load(best_path)
    print(f"[INFO] Loaded: {best_path}")

    out_dir = make_dir(args.out_dir)

    videos = rollout_video(
        env=env,
        agent=agent,
        episodes=args.episodes,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
    )

    # naming with env tag
    env_tag = spec.name
    if spec.name == "dmc":
        env_tag = f"dmc_{spec.domain}_{spec.task}"
    elif spec.name == "minigrid":
        env_tag = f"minigrid_{spec.env_id}"

    for i, frames in enumerate(videos, start=1):
        path = os.path.join(out_dir, f"{env_tag}_ep{i:03d}.mp4")
        save_video_mp4(frames, path, fps=args.fps)
        print(f"[OK] Saved: {path}")

    env.close()


if __name__ == "__main__":
    main_video()