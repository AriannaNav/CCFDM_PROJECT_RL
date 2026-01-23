# minigrid.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from PIL import Image


def to_uint8_chw_rgb(obs_rgb_hwc: np.ndarray, image_size: int) -> np.ndarray:

    if obs_rgb_hwc.dtype != np.uint8:
        obs_rgb_hwc = obs_rgb_hwc.astype(np.uint8)
    img = Image.fromarray(obs_rgb_hwc)
    img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)  # HWC
    chw = np.transpose(arr, (2, 0, 1))     # CHW
    return chw


class FrameStack:

    def __init__(self, k: int, obs_shape_chw: Tuple[int, int, int]):
        self.k = int(k)
        c, h, w = obs_shape_chw
        self._frames = np.zeros((self.k, c, h, w), dtype=np.uint8)
        self._filled = False

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        for i in range(self.k):
            self._frames[i] = first_frame
        self._filled = True
        return self.get()

    def push(self, frame: np.ndarray) -> np.ndarray:
        if not self._filled:
            return self.reset(frame)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = frame
        return self.get()

    def get(self) -> np.ndarray:
        return self._frames.reshape(self.k * self._frames.shape[1], self._frames.shape[2], self._frames.shape[3])


class MiniGridContinuousWrapper:

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

      
        env = gym.make(env_id)

        env = RGBImgObsWrapper(env)

        self._env = env

        if not isinstance(self._env.action_space, gym.spaces.Discrete):
            raise ValueError("discrete (MiniGrid).")

        self._n_actions = int(self._env.action_space.n)

        # Output obs shape: (3*frame_stack, image_size, image_size)
        self.obs_shape = (3 * self.frame_stack, self.image_size, self.image_size)


        self.action_shape = (1,)
        self.action_low = np.array([-1.0], dtype=np.float32)
        self.action_high = np.array([1.0], dtype=np.float32)

        self._fs = FrameStack(self.frame_stack, (3, self.image_size, self.image_size))

        self._t = 0

    def _continuous_to_discrete(self, action: Union[float, int, np.ndarray]) -> int:
        
        if isinstance(action, np.ndarray):
            if action.size == 0:
                a = 0.0
            else:
                a = float(action.reshape(-1)[0])
        else:
            a = float(action)

        # clamp in [-1, 1]
        a = max(-1.0, min(1.0, a))

        # map [-1,1] -> [0, n-1] (discrete)
        if self._n_actions == 1:
            return 0

        scaled = (a + 1.0) * 0.5 * (self._n_actions - 1)
        idx = int(np.round(scaled))
        idx = max(0, min(self._n_actions - 1, idx))
        return idx

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        obs, info = self._env.reset(seed=self.seed)

        img = obs["image"]  
        frame = to_uint8_chw_rgb(img, self.image_size)
        stacked = self._fs.reset(frame)

        return stacked, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        action: np.ndarray shape (1,) in [-1,1]
        """
        a_discrete = self._continuous_to_discrete(action)

        total_reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        last_obs = None

        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(a_discrete)
            total_reward += float(reward)
            done = bool(terminated or truncated)

            last_obs = obs

            self._t += 1
            if self.max_episode_steps is not None and self._t >= int(self.max_episode_steps):
                done = True
                info = dict(info)
                info["TimeLimit.truncated"] = True

            if done:
                break

        assert last_obs is not None
        img = last_obs["image"]  # HWC
        frame = to_uint8_chw_rgb(img, self.image_size)
        stacked = self._fs.push(frame)

        return stacked, total_reward, done, info

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()


def make_minigrid_env(
    env_id: str,
    image_size: int = 84,
    frame_stack: int = 3,
    action_repeat: int = 1,
    seed: int = 1,
    max_episode_steps: Optional[int] = None,
) -> MiniGridContinuousWrapper:
   
    return MiniGridContinuousWrapper(
        env_id=env_id,
        image_size=image_size,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )