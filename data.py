#data.py 
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch


def random_crop(imgs, out_size):
    """
    imgs: (B, C, H, W)
    returns: (B, C, out_size, out_size)
    """
    b, c, h, w = imgs.shape
    if h == out_size and w == out_size:
        return imgs
    if out_size > h or out_size > w:
        raise ValueError(f"out_size={out_size} > image size {(h, w)}")

    max_y = h - out_size
    max_x = w - out_size

    ys = np.random.randint(0, max_y + 1, size=b)
    xs = np.random.randint(0, max_x + 1, size=b)

    out = np.empty((b, c, out_size, out_size), dtype=imgs.dtype)
    for i in range(b):
        y = ys[i]
        x = xs[i]
        out[i] = imgs[i, :, y:y + out_size, x:x + out_size]
    return out


def center_crop(imgs, out_size):
    """
    imgs: (B, C, H, W)
    returns: (B, C, out_size, out_size)
    """
    b, c, h, w = imgs.shape
    if h == out_size and w == out_size:
        return imgs
    if out_size > h or out_size > w:
        raise ValueError(f"out_size={out_size} > image size {(h, w)}")

    y0 = (h - out_size) // 2
    x0 = (w - out_size) // 2
    return imgs[:, :, y0:y0 + out_size, x0:x0 + out_size]


def random_shift(imgs, pad: int = 4):
    """
    DrQ-style random shift:
      - pad with edge values
      - random crop back to original HxW
    Works even when H=W=out_size (e.g. 84x84), unlike random_crop(84->84) which is a no-op.

    imgs: (B, C, H, W) uint8
    returns: (B, C, H, W) uint8
    """
    b, c, h, w = imgs.shape
    if pad <= 0:
        return imgs

    p = int(pad)
    imgs_pad = np.pad(
        imgs,
        pad_width=((0, 0), (0, 0), (p, p), (p, p)),
        mode="edge",
    )

    # choose top-left corner in [0..2p]
    max_off = 2 * p
    ys = np.random.randint(0, max_off + 1, size=b)
    xs = np.random.randint(0, max_off + 1, size=b)

    out = np.empty((b, c, h, w), dtype=imgs.dtype)
    for i in range(b):
        y = ys[i]
        x = xs[i]
        out[i] = imgs_pad[i, :, y:y + h, x:x + w]
    return out


def to_torch_obs(obs_u8, device):
    """
    Safe path on MPS: numpy -> CPU tensor -> .to(device)
    """
    if isinstance(obs_u8, torch.Tensor):
        t = obs_u8
    else:
        t = torch.from_numpy(obs_u8)  # CPU tensor

    t = t.to(device=device, dtype=torch.float32)  # move to device safely
    return t / 255.0


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    not_done: torch.Tensor
    cpc_kwargs: Optional[Dict[str, torch.Tensor]] = None


class ReplayBuffer:
    def __init__(
        self,
        obs_shape,            # (C,H,W)
        action_shape,         # (A,)
        capacity,
        batch_size,
        device,
        image_size,           # kept for compatibility; crop size if using random_crop/center_crop
        aug_pad: int = 4,     # NEW: pad for DrQ-style random shift
    ):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.device = device
        self.image_size = int(image_size)
        self.aug_pad = int(aug_pad)

        c, h, w = obs_shape
        self._obses = np.empty((self.capacity, c, h, w), dtype=np.uint8)
        self._next_obses = np.empty((self.capacity, c, h, w), dtype=np.uint8)

        self._actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self._rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self._not_dones = np.empty((self.capacity, 1), dtype=np.float32)

        self._idx = 0
        self._full = False

        self._obs_shape = (c, h, w)
        self._action_shape = tuple(action_shape)

    def __len__(self):
        return self.size

    @property
    def size(self):
        return self.capacity if self._full else self._idx

    def clear(self):
        self._idx = 0
        self._full = False

    def add(
        self,
        obs,             # (C,H,W) uint8
        action,          # (A,) float32
        reward,
        next_obs,        # (C,H,W) uint8
        terminated,
        truncated,
    ):
        if obs.dtype != np.uint8 or next_obs.dtype != np.uint8:
            raise TypeError(f"obs/next_obs must be uint8. Got {obs.dtype=} {next_obs.dtype=}")
        if obs.shape != self._obs_shape or next_obs.shape != self._obs_shape:
            raise ValueError(f"obs shape mismatch. Expected {self._obs_shape}, got {obs.shape=} {next_obs.shape=}")

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self._action_shape:
            raise ValueError(f"action shape mismatch. Expected {self._action_shape}, got {action.shape}")

        done = bool(terminated or truncated)

        np.copyto(self._obses[self._idx], obs)
        np.copyto(self._next_obses[self._idx], next_obs)
        np.copyto(self._actions[self._idx], action)

        self._rewards[self._idx, 0] = float(reward)
        self._not_dones[self._idx, 0] = 0.0 if done else 1.0

        self._idx = (self._idx + 1) % self.capacity
        if self._idx == 0:
            self._full = True

    def sample_idxs(self):
        n = self.capacity if self._full else self._idx
        if n < self.batch_size:
            raise RuntimeError(
                f"ReplayBuffer: not enough samples to draw a batch. "
                f"Have {n}, need batch_size={self.batch_size}."
            )
        return np.random.randint(0, n, size=self.batch_size)

    def sample(self):
        """
        Standard SAC batch.
        NEW: applies DrQ-style random shift augmentation to obs and next_obs.
        """
        idxs = self.sample_idxs()

        obs_u8 = self._obses[idxs]
        next_obs_u8 = self._next_obses[idxs]

        # NEW: effective augmentation even if frames are already 84x84
        obs_u8 = random_shift(obs_u8, pad=self.aug_pad)
        next_obs_u8 = random_shift(next_obs_u8, pad=self.aug_pad)

        obs = to_torch_obs(obs_u8, self.device)
        next_obs = to_torch_obs(next_obs_u8, self.device)

        action = torch.from_numpy(self._actions[idxs]).to(self.device, dtype=torch.float32)
        reward = torch.from_numpy(self._rewards[idxs]).to(self.device, dtype=torch.float32)
        not_done = torch.from_numpy(self._not_dones[idxs]).to(self.device, dtype=torch.float32)

        return ReplayBatch(obs=obs, action=action, reward=reward, next_obs=next_obs, not_done=not_done)

    def sample_no_aug(self):
        """
        Center-crop + no stochastic augmentation (useful for evaluation-like batches).
        """
        idxs = self.sample_idxs()

        obs_u8 = center_crop(self._obses[idxs], self.image_size)
        next_u8 = center_crop(self._next_obses[idxs], self.image_size)

        obs = to_torch_obs(obs_u8, self.device)
        next_obs = to_torch_obs(next_u8, self.device)

        action = torch.as_tensor(self._actions[idxs], device=self.device, dtype=torch.float32)
        reward = torch.as_tensor(self._rewards[idxs], device=self.device, dtype=torch.float32)
        not_done = torch.as_tensor(self._not_dones[idxs], device=self.device, dtype=torch.float32)

        return ReplayBatch(obs=obs, action=action, reward=reward, next_obs=next_obs, not_done=not_done)

    def sample_cpc(self):
        """
        Contrastive batch:
          - obs = obs_anchor (aug view)
          - next_obs = next_obs (aug view)
          - cpc_kwargs = { "obs_pos": second aug view of obs }
        NEW: uses DrQ-style random shift to ensure augmentation is never a no-op on 84x84.
        """
        idxs = self.sample_idxs()

        obs_u8 = self._obses[idxs]            # (B,C,H,W)
        next_obs_u8 = self._next_obses[idxs]

        # two different augmented views of current obs
        obs_anchor_u8 = random_shift(obs_u8, pad=self.aug_pad)
        obs_pos_u8 = random_shift(obs_u8, pad=self.aug_pad)

        # augmented next obs
        next_obs_u8 = random_shift(next_obs_u8, pad=self.aug_pad)

        obs_anchor = to_torch_obs(obs_anchor_u8, self.device)
        obs_pos = to_torch_obs(obs_pos_u8, self.device)
        next_obs = to_torch_obs(next_obs_u8, self.device)

        action = torch.from_numpy(self._actions[idxs]).to(self.device, dtype=torch.float32)
        reward = torch.from_numpy(self._rewards[idxs]).to(self.device, dtype=torch.float32)
        not_done = torch.from_numpy(self._not_dones[idxs]).to(self.device, dtype=torch.float32)

        cpc_kwargs = {"obs_pos": obs_pos}

        return ReplayBatch(
            obs=obs_anchor,
            action=action,
            reward=reward,
            next_obs=next_obs,
            not_done=not_done,
            cpc_kwargs=cpc_kwargs,
        )