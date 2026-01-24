# utils.py
import os
import json
import random
import numpy as np
import torch


def get_device(requested):
    requested = (requested or "auto").lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available in this PyTorch build.")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")

    return torch.device("cpu")


def device_info(device):
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


def set_seed(seed, deterministic, device):
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _ = device


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, obj):
    make_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def soft_update(target, source, tau):
    # EMA / Polyak averaging
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau)
            tp.data.add_(tau * sp.data)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())