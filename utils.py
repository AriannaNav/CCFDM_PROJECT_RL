# utils.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch



def get_device(requested):
    
    #requested: "auto" | "cpu" | "mps"

    requested = (requested or "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available in this PyTorch build.")

    # auto
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


def device_info(device):
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


# Reproducibility / seeding
def set_seed(seed,deterministic,device,) :

    seed = int(seed)

    # Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _ = device

def soft_update(
    target,
    source,
    tau,
) :
    """
    target = tau * source + (1 - tau) * target
    """
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * sp.data)

def hard_update(
    target,
    source,
) :
    target.load_state_dict(source.state_dict())

