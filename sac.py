import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 

import utils 
from encoder import make_encoder

"""
✅ SAC deve funzionare con:
	•	input: obs immagini (tu: uint8 CHW stackato)
	•	replay buffer: il tuo data.py (già ok)
	•	encoder: quello già pronto della collega (lo usi come feature extractor)
	•	actor/critic: prendono in input embedding (output encoder), non pixel direttamente
	•	loop di training: usa ReplayBuffer.sample() e agent.update()

❌ SAC NON deve includere:
	•	InfoNCE
	•	forward dynamics model
	•	intrinsic reward
	•	EMA key encoder

Queste cose arrivano solo in CCFDM.
"""


import torch
import torch.nn.functional as F
import math


def gaussian_logprob(noise: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    # noise, log_std: (B, action_dim)
    residual = -0.5 * (noise.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
    return residual.sum(dim=-1, keepdim=True)


def squash_action(mu: torch.Tensor,pi: torch.Tensor,log_pi: torch.Tensor,eps: float = 1e-6,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    mu = torch.tanh(mu)
    pi = torch.tanh(pi)

    # Jacobian correction: log |det d(tanh(u))/du|
    # = sum log(1 - tanh(u)^2)
    log_pi = log_pi - torch.log(1.0 - pi.pow(2) + eps).sum(dim=-1, keepdim=True)

    return mu, pi, log_pi

def weight_init(m):
    # Orthogonal initialization for layers with ReLU activations"
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

