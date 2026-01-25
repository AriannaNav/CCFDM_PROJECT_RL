# train_ccfdm.py
from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from utils import get_device, device_info, set_seed, make_dir, save_json
from logger import Logger
from make_env import EnvSpec, make_env
from data import ReplayBuffer
from ccfdm_agent import CCFDMAgent


def env_tag(spec):
    name = spec.name.lower().strip()
    if name == "dmc":
        return f"dmc_{spec.domain}_{spec.task}"
    if name == "minigrid":
        return f"minigrid_{spec.env_id}"
    return f"env_{name}"


def build_run_dirs(models_root, logs_root, algo, spec, seed):
    tag = env_tag(spec)
    run_name = f"{algo}/{tag}/seed_{seed}"
    model_dir = make_dir(os.path.join(models_root, run_name))
    log_dir = make_dir(os.path.join(logs_root, run_name))
    return {"model_dir": model_dir, "log_dir": log_dir}


def parse_args():
    p = argparse.ArgumentParser("CCFDM training")

    # system
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deterministic", action="store_true")

    # env
    p.add_argument("--env", type=str, required=True, choices=["dmc", "minigrid"])
    p.add_argument("--dmc_domain", type=str, default=None)
    p.add_argument("--dmc_task", type=str, default=None)
    p.add_argument("--minigrid_id", type=str, default=None)
    p.add_argument("--image_size", type=int, default=84)
    p.add_argument("--frame_stack", type=int, default=3)
    p.add_argument("--action_repeat", type=int, default=1)
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--max_episode_steps", type=int, default=None)

    # training
    p.add_argument("--total_steps", type=int, default=200_000)
    p.add_argument("--init_random_steps", type=int, default=5_000)
    p.add_argument("--update_after", type=int, default=1_000)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--replay_size", type=int, default=100_000)

    # saving/logging
    p.add_argument("--models_root", type=str, default="models")
    p.add_argument("--logs_root", type=str, default="logs")
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--log_every", type=int, default=1_000)

    # periodic eval INSIDE training (paper-style)
    p.add_argument("--eval_every", type=int, default=10_000, help="Every N env steps run eval episodes (for Fig.5 + best.pt).")
    p.add_argument("--eval_episodes", type=int, default=10)

    # agent hyperparams
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--critic_tau", type=float, default=0.01)
    p.add_argument("--encoder_tau", type=float, default=0.01)
    p.add_argument("--actor_update_freq", type=int, default=2)
    p.add_argument("--critic_target_update_freq", type=int, default=2)
    p.add_argument("--ccfmd_update_freq", type=int, default=1)

    # intrinsic
    p.add_argument("--intrinsic_weight", type=float, default=1.0)
    p.add_argument("--intrinsic_decay", type=float, default=2e-5)

    # contrastive
    p.add_argument("--contrastive_method", type=str, default="infonce", choices=["infonce", "triplet", "byol"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no_normalize", action="store_false", dest="normalize")
    p.add_argument("--triplet_margin", type=float, default=0.2)

    # curiosity
    p.add_argument("--curiosity_C", type=float, default=0.2)
    p.add_argument("--curiosity_gamma", type=float, default=2e-5)

    return p.parse_args()


def make_spec(args):
    if args.env == "dmc":
        if not args.dmc_domain or not args.dmc_task:
            raise ValueError("--dmc_domain and --dmc_task required for dmc")
        return EnvSpec(
            name="dmc",
            domain=args.dmc_domain,
            task=args.dmc_task,
            camera_id=args.camera_id,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        if not args.minigrid_id:
            raise ValueError("--minigrid_id required for minigrid")
        return EnvSpec(
            name="minigrid",
            env_id=args.minigrid_id,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
        )


def sample_random_action(env):
    # used only for init_random_steps exploration
    low = env.action_low
    high = env.action_high
    a = np.random.uniform(low=low, high=high).astype(np.float32)
    return a


def save_checkpoint(agent, model_dir, name):
    path = os.path.join(model_dir, name)
    agent.save(path)
    return path


@torch.no_grad()
def run_eval_episodes(agent, env, episodes):
    """
    Deterministic evaluation: uses select_action (mu).
    Returns mean and std of episode returns.
    """
    agent.train(False)
    agent.critic_target.eval()

    rets = []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.select_action(obs)
            a = np.clip(a, env.action_low, env.action_high).astype(np.float32)
            obs, r, term, trunc, _ = env.step(a)
            ep_ret += float(r)
            done = bool(term or trunc)
        rets.append(ep_ret)

    agent.train(True)
    agent.critic_target.eval()

    rets = np.asarray(rets, dtype=np.float32)
    return float(rets.mean()), float(rets.std())


def main_train():
    args = parse_args()

    device = get_device(args.device)
    print(f"[INFO] Device: {device_info(device)}")

    set_seed(args.seed, args.deterministic, device=device)
    print(f"[INFO] Seed: {args.seed} deterministic={args.deterministic}")

    spec = make_spec(args)
    env = make_env(spec)
    print(f"[INFO] Env: {env_tag(spec)} obs_shape={env.obs_shape} action_shape={env.action_shape}")

    # dirs
    dirs = build_run_dirs(args.models_root, args.logs_root, algo="ccfdm", spec=spec, seed=args.seed)
    model_dir, log_dir = dirs["model_dir"], dirs["log_dir"]

    # save config
    cfg = {
        "algo": "ccfdm",
        "seed": args.seed,
        "device": str(device),
        "env_spec": asdict(spec),
        "train_args": vars(args),
    }
    save_json(os.path.join(model_dir, "config.json"), cfg)

    logger = Logger(log_dir, name="train")
    eval_logger = Logger(log_dir, name="eval")  # for Fig.5 curve

    print("[DBG] before replay buffer")

    rb = ReplayBuffer(
        obs_shape=env.obs_shape,
        action_shape=env.action_shape,
        capacity=args.replay_size,
        batch_size=args.batch_size,
        device=device,
        image_size=spec.image_size,
    )

    print("[DBG] after replay buffer")

    agent = CCFDMAgent(
        obs_shape=env.obs_shape,
        action_shape=env.action_shape,
        device=device,
        discount=args.discount,
        critic_tau=args.critic_tau,
        encoder_tau=args.encoder_tau,
        actor_update_freq=args.actor_update_freq,
        critic_target_update_freq=args.critic_target_update_freq,
        ccfmd_update_freq=args.ccfmd_update_freq,
        contrastive_method=args.contrastive_method,
        temperature=args.temperature,
        normalize=args.normalize,
        triplet_margin=args.triplet_margin,
        curiosity_C=args.curiosity_C,
        curiosity_gamma=args.curiosity_gamma,
        intrinsic_weight=args.intrinsic_weight,
        intrinsic_decay=args.intrinsic_decay,
    )

    print("[DBG] after agent")

    # important: target critic always eval (EMA only)
    agent.critic_target.eval()

    obs, _ = env.reset()
    ep_ret = 0.0
    ep_len = 0
    episode_idx = 0

    best_eval_return = -float("inf")
    best_step = 0

    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        if step == 1:
            print(f"[WARMUP] collecting {args.init_random_steps} random steps, updates start at step >= {args.update_after} and when replay >= {args.batch_size}")
        if step == args.update_after:
            print(f"[TRAIN] update_after reached at step={step} (will train only if replay >= batch_size)")
        # action
        if step <= args.init_random_steps:
            action = sample_random_action(env)
        else:
            action = agent.sample_action(obs)
            action = np.clip(action, env.action_low, env.action_high).astype(np.float32)

        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)

        rb.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
        )

        obs = next_obs
        ep_ret += float(reward)
        ep_len += 1

        # update
        if step >= args.update_after and rb.size >= args.batch_size:
            if step == args.update_after:
                print(f"[TRAIN] first possible update at step={step} (replay={rb.size})")
            for _ in range(args.update_every):
                agent.update(rb, logger=logger, step=step)

        # episode end logs
        if done:
            episode_idx += 1
            logger.log_dict(
                {
                    "train/episode_return": ep_ret,
                    "train/episode_length": ep_len,
                    "train/episode_idx": episode_idx,
                },
                step=step,
            )
            # --- HUMAN READABLE episode print
            print(f"[EP ] step={step} ep={episode_idx} return={ep_ret:.3f} len={ep_len}")
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0

        # periodic logs + print
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = step / max(1e-9, elapsed)

            last_critic = logger.last("train/critic_loss", None)
            last_actor = logger.last("train/actor_loss", None)
            last_alpha = logger.last("train/alpha", None)
            last_c = logger.last("train/contrastive_loss", None)

            last_r_ext = logger.last("train/reward_ext_mean", None)
            last_r_int = logger.last("train/reward_int_mean", None)
            last_r_tot = logger.last("train/reward_total_mean", None)

            last_ep_ret = logger.last("train/episode_return", None)
            last_ep_len = logger.last("train/episode_length", None)

            def _fmt(x, prec=4):
                if x is None:
                    return "NA"
                try:
                    return f"{float(x):.{prec}f}"
                except Exception:
                    return str(x)

            print(
                f"[LOG] step={step} replay={rb.size} sps={sps:.1f} | "
                f"ep_ret={_fmt(last_ep_ret,3)} ep_len={_fmt(last_ep_len,0)} | "
                f"r_ext={_fmt(last_r_ext,4)} r_int={_fmt(last_r_int,4)} r_tot={_fmt(last_r_tot,4)} | "
                f"critic={_fmt(last_critic,6)} actor={_fmt(last_actor,6)} alpha={_fmt(last_alpha,6)} ccfdm={_fmt(last_c,6)}"
            )

            logger.log_dict(
                {
                    "train/step": step,
                    "train/replay_size": rb.size,
                    "train/sps": sps,
                },
                step=step,
            )
            logger.flush()

        # periodic EVAL (paper-style) + best.pt
        if args.eval_every is not None and args.eval_every > 0 and (step % args.eval_every == 0):
            mean_ret, std_ret = run_eval_episodes(agent, env, episodes=args.eval_episodes)
            eval_logger.log_dict(
                {
                    "eval/mean_return": mean_ret,
                    "eval/std_return": std_ret,
                    "eval/episodes": int(args.eval_episodes),
                },
                step=step,
            )
            eval_logger.flush()

            print(f"[EVAL] step={step} mean_return={mean_ret:.3f} std={std_ret:.3f} (episodes={args.eval_episodes})")

            if mean_ret > best_eval_return:
                best_eval_return = mean_ret
                best_step = step
                save_checkpoint(agent, model_dir, "best.pt")
                save_json(
                    os.path.join(model_dir, "best.json"),
                    {"best_eval_return": best_eval_return, "best_step": best_step},
                )
                print(f"[BEST] new best.pt at step={best_step} return={best_eval_return:.3f}")

        # checkpointing (regular)
        if step % args.save_every == 0:
            save_checkpoint(agent, model_dir, f"ckpt_step_{step:09d}.pt")
            save_checkpoint(agent, model_dir, "last.pt")
            print(f"[INFO] Saved ckpt + last at step={step}")

    # final save
    save_checkpoint(agent, model_dir, "last.pt")
    logger.close()
    eval_logger.close()
    env.close()
    print("[OK] Training completed.")
    if best_step > 0:
        print(f"[OK] Best model: best.pt at step={best_step} return={best_eval_return:.3f}")

def main_train_ccfdm():
    main_train()

def main():
    main_train()

if __name__ == "__main__":
    main_train()
