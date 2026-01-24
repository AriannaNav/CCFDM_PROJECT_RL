# make_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from dmc import make_dmc_env


@dataclass
class EnvSpec:
    name: str  # "dmc" | "minigrid"

    # DMC
    domain: Optional[str] = None
    task: Optional[str] = None
    camera_id: Optional[int] = 0

    # MiniGrid
    env_id: Optional[str] = None

    # Common
    image_size: int = 84
    frame_stack: int = 3
    action_repeat: int = 1
    seed: int = 1
    max_episode_steps: Optional[int] = None


class UnifiedEnv:
    """
    Unified wrapper that guarantees a consistent Gymnasium-like API across envs:

      reset() -> (obs, info)
      step(a) -> (obs, reward, terminated, truncated, info)

    Obs is uint8 CHW with frame stacking: (3*frame_stack, H, W)
    """

    def __init__(self, env):
        self._env = env

        self.obs_shape: Tuple[int, int, int] = tuple(env.obs_shape)  # (C, H, W)
        self.action_shape: Tuple[int, ...] = tuple(env.action_shape)
        self.action_low: np.ndarray = np.asarray(env.action_low, dtype=np.float32)
        self.action_high: np.ndarray = np.asarray(env.action_high, dtype=np.float32)

        self.max_episode_steps: Optional[int] = getattr(env, "max_episode_steps", None)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        # passthrough: dmc.py and minigrid_env.py return (obs, reward, terminated, truncated, info)
        return self._env.step(action)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


def make_env(spec):
    name = spec.name.lower().strip()

    if name == "dmc":
        if not spec.domain or not spec.task:
            raise ValueError("For DMC you must provide spec.domain and spec.task.")

        env = make_dmc_env(
            domain=spec.domain,
            task=spec.task,
            image_size=spec.image_size,
            frame_stack=spec.frame_stack,
            action_repeat=spec.action_repeat,
            seed=spec.seed,
            camera_id=int(spec.camera_id or 0),
            max_episode_steps=spec.max_episode_steps,
        )
        return UnifiedEnv(env)

    if name == "minigrid":
        from minigrid_env import make_minigrid_env
        if not spec.env_id:
            raise ValueError("For MiniGrid you must provide spec.env_id.")

        env = make_minigrid_env(
            env_id=spec.env_id,
            image_size=spec.image_size,
            frame_stack=spec.frame_stack,
            action_repeat=spec.action_repeat,
            seed=spec.seed,
            max_episode_steps=spec.max_episode_steps,
        )
        return UnifiedEnv(env)

    raise ValueError(f"Unknown environment name: {spec.name}")