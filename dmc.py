# dmc.py
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import pixels
from PIL import Image


def resize_to_uint8_chw(img_hwc, image_size):
    if img_hwc.dtype != np.uint8:
        img_hwc = img_hwc.astype(np.uint8)

    if img_hwc.shape[0] == image_size and img_hwc.shape[1] == image_size:
        arr = img_hwc
    else:
        im = Image.fromarray(img_hwc)
        im = im.resize((image_size, image_size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)

    return np.transpose(arr, (2, 0, 1))


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

    def get(self):
        return self._frames.reshape(
            self.k * self._frames.shape[1],
            self._frames.shape[2],
            self._frames.shape[3],
        )


class DMCEnv:
    def __init__(
        self,
        domain,
        task,
        image_size,
        frame_stack,
        action_repeat,
        seed,
        camera_id,
        max_episode_steps,
    ):
        self.domain = domain
        self.task = task
        self.image_size = int(image_size)          # crop size target (84)
        self.frame_stack = int(frame_stack)
        self.action_repeat = int(action_repeat)
        self.seed = int(seed)
        self.camera_id = int(camera_id)
        self.max_episode_steps = max_episode_steps

        # render bigger so that random_crop(., 84) is not a no-op
        self.render_size = self.image_size + 16    # 84 -> 100

        env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})
        env = pixels.Wrapper(
            env,
            pixels_only=True,
            render_kwargs={
                "height": self.render_size,
                "width": self.render_size,
                "camera_id": self.camera_id,
            },
        )
        self._env = env

        action_spec = self._env.action_spec()
        self.action_shape = tuple(action_spec.shape)
        self.action_low = np.array(action_spec.minimum, dtype=np.float32)
        self.action_high = np.array(action_spec.maximum, dtype=np.float32)

        # IMPORTANT: env outputs render_size images (100x100), crop happens in ReplayBuffer
        self.obs_shape = (3 * self.frame_stack, self.render_size, self.render_size)
        self._fs = FrameStack(self.frame_stack, (3, self.render_size, self.render_size))

        self._t = 0

    def reset(self):
        self._t = 0
        ts = self._env.reset()
        obs = ts.observation
        img = obs["pixels"] if isinstance(obs, dict) else obs  # HWC
        frame = resize_to_uint8_chw(img, self.render_size)     # <-- render_size
        stacked = self._fs.reset(frame)
        return stacked, {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        a = np.clip(a, self.action_low, self.action_high)

        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        ts = None
        for _ in range(self.action_repeat):
            ts = self._env.step(a)

            r = ts.reward if ts.reward is not None else 0.0
            total_reward += float(r)

            terminated = bool(ts.last())
            self._t += 1

            if self.max_episode_steps is not None and self._t >= int(self.max_episode_steps) and not terminated:
                truncated = True
                info["TimeLimit.truncated"] = True

            if terminated or truncated:
                break

        assert ts is not None
        obs = ts.observation
        img = obs["pixels"] if isinstance(obs, dict) else obs
        frame = resize_to_uint8_chw(img, self.render_size)     # <-- render_size
        stacked = self._fs.push(frame)

        return stacked, total_reward, terminated, truncated, info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


def make_dmc_env(
    domain,
    task,
    image_size,
    frame_stack,
    action_repeat,
    seed,
    camera_id,
    max_episode_steps,
):
    return DMCEnv(
        domain=domain,
        task=task,
        image_size=image_size,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        seed=seed,
        camera_id=camera_id,
        max_episode_steps=max_episode_steps,
    )
#ok 