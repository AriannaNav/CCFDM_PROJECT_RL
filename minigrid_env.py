# minigrid_env.py
from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from PIL import Image


def to_uint8_chw_rgb(obs_rgb_hwc, out_size):
    if obs_rgb_hwc.dtype != np.uint8:
        obs_rgb_hwc = obs_rgb_hwc.astype(np.uint8)

    img = Image.fromarray(obs_rgb_hwc)
    img = img.resize((out_size, out_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)      # HWC
    chw = np.transpose(arr, (2, 0, 1))         # CHW
    return chw


class FrameStack:
    def __init__(self, k, obs_shape_chw):
        self.k = int(k)
        c, h, w = obs_shape_chw
        self._frames = np.zeros((self.k, c, h, w), dtype=np.uint8)
        self._filled = False

    def reset(self, first_frame):
        for i in range(self.k):
            self._frames[i] = first_frame
        self._filled = True
        return self.get()

    def push(self, frame):
        if not self._filled:
            return self.reset(frame)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = frame
        return self.get()

    def get(self) :
        # (k, C, H, W) -> (k*C, H, W)
        return self._frames.reshape(
            self.k * self._frames.shape[1],
            self._frames.shape[2],
            self._frames.shape[3],
        )


class MiniGridContinuousWrapper:
    """
    MiniGrid has Discrete action space, but SAC is continuous.
    We map a continuous scalar action in [-1,1] to a discrete action via binning.
    This is a pragmatic proxy for quick experimentation.
    """

    def __init__(
        self,
        env_id: str,
        image_size: int = 84,
        frame_stack: int = 3,
        action_repeat: int = 1,
        seed: int = 1,
        max_episode_steps: Optional[int] = None,
    ):
        self.env_id = env_id
        self.image_size = int(image_size)
        self.frame_stack = int(frame_stack)
        self.action_repeat = int(action_repeat)
        self.seed = int(seed)
        self.max_episode_steps = max_episode_steps
        self.render_size = self.image_size  

        env = gym.make(env_id)
        env = RGBImgObsWrapper(env)  # gives obs dict with "image" (HWC RGB)
        self._env = env

        if not isinstance(self._env.action_space, gym.spaces.Discrete):
            raise ValueError("MiniGrid action space must be Discrete.")

        self._n_actions = int(self._env.action_space.n)

        # Observation format for the rest of the project: uint8 CHW stacked
        self.obs_shape = (3 * self.frame_stack, self.render_size, self.render_size)

        # Continuous proxy action space for SAC
        self.action_shape = (1,)
        self.action_low = np.array([-1.0], dtype=np.float32)
        self.action_high = np.array([1.0], dtype=np.float32)

        self._fs = FrameStack(self.frame_stack, (3, self.render_size, self.render_size))

        self._t = 0
        self._episode_idx = 0

    def continuous_to_discrete(self, action):
        if isinstance(action, np.ndarray):
            a = float(action.reshape(-1)[0]) if action.size > 0 else 0.0
        else:
            a = float(action)

        a = max(-1.0, min(1.0, a))

        if self._n_actions <= 1:
            return 0

        scaled = (a + 1.0) * 0.5 * (self._n_actions - 1)
        idx = int(np.floor(scaled))
        return max(0, min(self._n_actions - 1, idx))

    def reset(self):
        self._t = 0
        seed = self.seed + self._episode_idx
        self._episode_idx += 1

        obs, info = self._env.reset(seed=seed)
        img = obs["image"] if isinstance(obs, dict) else obs  # HWC uint8
        frame = to_uint8_chw_rgb(img, self.render_size)
        stacked = self._fs.reset(frame)
        return stacked, info

    def step(self, action) :
        a_discrete = self.continuous_to_discrete(action)

        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        last_obs = None

        for _ in range(self.action_repeat):
            obs, reward, term, trunc, info = self._env.step(a_discrete)
            total_reward += float(reward)
            last_obs = obs

            self._t += 1
            terminated = bool(term)
            truncated = bool(trunc)

            # Apply external time limit in Gym-standard way
            if self.max_episode_steps is not None and self._t >= int(self.max_episode_steps) and not terminated:
                truncated = True
                info = dict(info)
                info["TimeLimit.truncated"] = True

            if terminated or truncated:
                break

        assert last_obs is not None
        img = last_obs["image"] if isinstance(last_obs, dict) else last_obs  # HWC
        frame = to_uint8_chw_rgb(img, self.render_size)
        stacked = self._fs.push(frame)

        return stacked, total_reward, terminated, truncated, info

    def close(self) :
        if hasattr(self._env, "close"):
            self._env.close()


def make_minigrid_env(
    env_id,
    image_size = 84,
    frame_stack = 3,
    action_repeat = 1,
    seed= 1,
    max_episode_steps = None,
):
    return MiniGridContinuousWrapper(
        env_id=env_id,
        image_size=image_size,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
#okok
