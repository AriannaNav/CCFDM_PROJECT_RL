# utils.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch



def get_device(requested: str = "auto") -> torch.device:
    
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


def device_info(device: torch.device) -> str:
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


# Reproducibility / seeding
def set_seed(seed: int,deterministic: bool = False,device: Optional[torch.device] = None,) -> None:

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