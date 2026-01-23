# test_data.py
import numpy as np
import torch

from data import ReplayBuffer
from make_env import EnvSpec, make_env


def _rand_action(env):
    # I tuoi wrapper espongono action_low/high e action_shape
    low = env.action_low
    high = env.action_high
    a = np.random.uniform(low, high).astype(np.float32)

    # garantisco shape corretta
    return a.reshape(env.action_shape)


def _fill_buffer(env, rb: ReplayBuffer, n_steps: int):
    obs, _ = env.reset()
    done = False

    for _ in range(n_steps):
        action = _rand_action(env)
        next_obs, reward, done, info = env.step(action)

        rb.add(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
        )

        obs = next_obs
        if done:
            obs, _ = env.reset()
            done = False


def _check_batch(tag: str, batch, image_size: int):
    obs = batch.obs
    next_obs = batch.next_obs

    assert obs.ndim == 4, f"{tag}: obs should be (B,C,H,W), got {obs.shape}"
    assert next_obs.ndim == 4, f"{tag}: next_obs should be (B,C,H,W), got {next_obs.shape}"

    B, C, H, W = obs.shape
    assert H == image_size and W == image_size, f"{tag}: obs spatial {H,W} != {image_size}"
    assert obs.dtype == torch.float32, f"{tag}: obs dtype {obs.dtype} != float32"

    # range [0,1]
    mn = float(obs.min().cpu())
    mx = float(obs.max().cpu())
    assert mn >= 0.0 - 1e-6 and mx <= 1.0 + 1e-6, f"{tag}: obs range [{mn},{mx}] not in [0,1]"

    assert batch.action.shape[0] == B, f"{tag}: action batch mismatch"
    assert batch.reward.shape == (B, 1), f"{tag}: reward shape {batch.reward.shape} != (B,1)"
    assert batch.not_done.shape == (B, 1), f"{tag}: not_done shape {batch.not_done.shape} != (B,1)"

    if batch.cpc_kwargs is not None:
        assert "obs_anchor" in batch.cpc_kwargs and "obs_pos" in batch.cpc_kwargs, f"{tag}: missing cpc keys"
        oa = batch.cpc_kwargs["obs_anchor"]
        op = batch.cpc_kwargs["obs_pos"]
        assert oa.shape == obs.shape and op.shape == obs.shape, f"{tag}: cpc obs shapes mismatch"


def run_one_env(spec: EnvSpec, device: str = "cpu"):
    env = make_env(spec)

    print(f"\n=== Testing env: {spec.name} ===")
    print("obs_shape:", env.obs_shape)
    print("action_shape:", env.action_shape, "low/high:", env.action_low, env.action_high)

    rb = ReplayBuffer(
        obs_shape=env.obs_shape,
        action_shape=env.action_shape,
        capacity=10000,
        batch_size=32,
        device=torch.device(device),
        image_size=spec.image_size,
    )

    # riempi buffer (almeno batch_size)
    _fill_buffer(env, rb, n_steps=200)
    assert rb.can_sample(), "ReplayBuffer can't sample: not enough data"

    b1 = rb.sample()
    _check_batch("sample()", b1, spec.image_size)

    b2 = rb.sample_no_aug()
    _check_batch("sample_no_aug()", b2, spec.image_size)

    b3 = rb.sample_cpc()
    _check_batch("sample_cpc()", b3, spec.image_size)

    print("OK ✅  sample / sample_no_aug / sample_cpc all good.")


def main():
    # 1) MiniGrid (scegli un env_id che hai installato)
    spec_mg = EnvSpec(
        name="minigrid",
        env_id="MiniGrid-Empty-8x8-v0",
        image_size=84,
        frame_stack=3,
        action_repeat=1,
        seed=1,
    )
    run_one_env(spec_mg)

    # 2) DMC
    # attenzione: richiede dm_control installato e i task disponibili
    spec_dmc = EnvSpec(
        name="dmc",
        domain="cheetah",
        task="run",
        camera_id=0,
        image_size=84,
        frame_stack=3,
        action_repeat=1,
        seed=1,
    )
    run_one_env(spec_dmc)


if __name__ == "__main__":
    main()