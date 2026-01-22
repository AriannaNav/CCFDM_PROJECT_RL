from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

from dmc import make_dmc_env
from minigrid_env import make_minigrid_env


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

    def __init__(self, env: Any):
        self._env = env


        self.obs_shape: Tuple[int, int, int] = env.obs_shape  # (C, H, W)

        self.action_shape: Tuple[int, ...] = env.action_shape
        self.action_low: np.ndarray = env.action_low
        self.action_high: np.ndarray = env.action_high

        # episode handling
        self._max_episode_steps = getattr(env, "max_episode_steps", None)
        self._t = 0

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        obs, info = self._env.reset()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    
        obs, reward, done, info = self._env.step(action)
        self._t += 1

        if self._max_episode_steps is not None and self._t >= self._max_episode_steps:
            done = True
            info = dict(info)
            info["TimeLimit.truncated"] = True

        return obs, float(reward), bool(done), info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()


def make_env(spec: EnvSpec) -> UnifiedEnv:

    name = spec.name.lower().strip()

    if name == "dmc":
        env = make_dmc_env(
            domain=spec.domain,
            task=spec.task,
            image_size=spec.image_size,
            frame_stack=spec.frame_stack,
            action_repeat=spec.action_repeat,
            seed=spec.seed,
            camera_id=spec.camera_id,
            max_episode_steps=spec.max_episode_steps,
        )
        return UnifiedEnv(env)

    if name == "minigrid":
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
