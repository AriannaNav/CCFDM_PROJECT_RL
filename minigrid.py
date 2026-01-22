from __future__ import annotations
from typing import Optional

import numpy as np
import gymnasium as gym


from gymnasium import spaces
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from PIL import Image


def _resize_rgb_uint8(img_hwc: np.ndarray, size: int) -> np.ndarray:
    
    #Resize HWC uint8 image to (size, size, 3) uint8.
   
    if img_hwc.shape[0] == size and img_hwc.shape[1] == size:
        return img_hwc
    im = Image.fromarray(img_hwc)
    im = im.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _hwc_to_chw(img_hwc: np.ndarray) -> np.ndarray:
    #HWC uint8 -> CHW uint8
    return np.transpose(img_hwc, (2, 0, 1)).astype(np.uint8)


class DiscreteToContinuousActionWrapper(gym.ActionWrapper):
    # Wrap MiniGrid discrete actions as continuous in [-1,1]
    # Maps continuous action in [-1,1] to discrete action index [0,n-1]
    # where n is number of discrete actions.
    # This allows using continuous-action algorithms on MiniGrid.


    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Discrete), "MiniGrid should be Discrete actions"
        self.n = env.action_space.n
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action: np.ndarray) -> int:
        # Ensure shape (1,)
        a = np.asarray(action, dtype=np.float32).reshape(1)
        x = float(np.clip(a[0], -1.0, 1.0))
        # Scale [-1,1] -> [0, n-1]
        idx = int(np.rint((x + 1.0) * 0.5 * (self.n - 1)))
        idx = int(np.clip(idx, 0, self.n - 1))
        return idx


class ImageObsCHWWrapper(gym.ObservationWrapper):
    # Convert MiniGrid RGB image obs to CHW and resize to fixed size.
    # Input: HWC uint8 RGB image
    # Output: CHW uint8 RGB image of shape (3, image_size, image_size)

    def __init__(self, env: gym.Env, image_size: int):
        super().__init__(env)
        self.image_size = int(image_size)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, self.image_size, self.image_size), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs expected HWC uint8
        obs = np.asarray(obs, dtype=np.uint8)
        if obs.ndim != 3 or obs.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape={obs.shape}")

        obs = _resize_rgb_uint8(obs, self.image_size)
        obs = _hwc_to_chw(obs)
        return obs


class FrameStackCHW(gym.Wrapper):
    # Stack k last frames along channel dimension for CHW images

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = int(k)
        assert self.k >= 1
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box)
        c, h, w = obs_space.shape
        assert c == 3, "Expected CHW RGB input before stacking"
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(c * self.k, h, w), dtype=np.uint8
        )
        self._frames = None  # will be deque-like list

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._frames = [obs.copy() for _ in range(self.k)]
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.pop(0)
        self._frames.append(obs.copy())
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self):
        return np.concatenate(self._frames, axis=0).astype(np.uint8)


def make_minigrid_env(
    env_id: str = "MiniGrid-Empty-8x8-v0",
    seed: int = 1,
    image_size: int = 84,
    frame_stack: int = 3,
    max_episode_steps: Optional[int] = None,
):

    # Create env (render_mode="rgb_array" makes wrappers safe/consistent)
    env = gym.make(env_id, render_mode="rgb_array")

    if max_episode_steps is not None:
        # Gymnasium TimeLimit wrapper: easiest is to re-wrap
        env = gym.wrappers.TimeLimit(env, max_episode_steps=int(max_episode_steps))

    # Seed handling (gymnasium)
    try:
        env.reset(seed=int(seed))
    except TypeError:
        try:
            env.seed(int(seed))
        except Exception:
            pass

    # 1) Convert obs dict -> RGB image
    env = RGBImgObsWrapper(env)   # adds 'rgb' image into obs
    env = ImgObsWrapper(env)      # returns only the image (HWC uint8)

    # 2) Make obs CHW uint8 and resize to fixed image_size
    env = ImageObsCHWWrapper(env, image_size=image_size)

    # 3) Frame stack in CHW
    if frame_stack and int(frame_stack) > 1:
        env = FrameStackCHW(env, k=int(frame_stack))

    # 4) Expose discrete actions as continuous float actions
    env = DiscreteToContinuousActionWrapper(env)

    return env