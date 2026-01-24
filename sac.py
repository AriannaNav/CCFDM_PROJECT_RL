import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import make_encoder

LOG_FREQ = 1000


def gaussian_logprob(noise, log_std):
    # noise, log_std: (B, action_dim)
    residual = -0.5 * (noise.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
    return residual.sum(dim=-1, keepdim=True)


def squash_action(mu, pi, log_pi=None, eps=1e-6):
    # tanh squash
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)

    # Jacobian correction for tanh squashing (only if log_pi is provided)
    if log_pi is not None:
        log_pi = log_pi - torch.log(1.0 - pi.pow(2) + eps).sum(dim=-1, keepdim=True)

    return mu, pi, log_pi


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # delta-orthogonal init
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim,
        encoder_type,
        encoder_feature_dim,
        log_std_min,
        log_std_max,
        num_layers,
        num_filters,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type,
            obs_dim,
            encoder_feature_dim,
            num_layers,
            num_filters,
            output_logits=True,
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # outputs 2 * action_dim: mean and log_std
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim[0]),
        )

        self.outputs = {}
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        h = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(h).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1.0
        )

        self.outputs["mu"] = mu
        self.outputs["std"] = log_std.exp()

        pi = None
        log_pi = None
        noise = None

        # if we don't sample actions, we cannot compute log_pi
        if not compute_pi:
            compute_log_pi = False

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std

            if compute_log_pi:
                log_pi = gaussian_logprob(noise, log_std)

            mu, pi, log_pi = squash_action(mu, pi, log_pi)

            self.outputs["pi"] = pi
            if log_pi is not None:
                self.outputs["log_pi"] = log_pi
        else:
            # deterministic action
            mu = torch.tanh(mu)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            if v is None:
                continue
            L.log_histogram("train_actor/%s_hist" % k, v, step)

        # trunk: [Linear, ReLU, Linear, ReLU, Linear]
        L.log_param("train_actor/fc1", self.trunk[0], step)
        L.log_param("train_actor/fc2", self.trunk[2], step)
        L.log_param("train_actor/fc3", self.trunk[4], step)


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        # ensure action has shape (B, A)
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        # catch batch mismatch early
        assert obs.size(0) == action.size(0), f"batch mismatch: {obs.size(0)} vs {action.size(0)}"

        # obs: (B, D), action: (B, A)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)
    
class SACCritic(nn.Module):
   
    def __init__(
        self, obs_dim, action_dim, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_dim, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(self.encoder.feature_dim, action_dim[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_dim[0], hidden_dim)

        self.outputs = {}
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        h = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(h, action)
        q2 = self.Q2(h, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
            if step % log_freq != 0:
                return
            
            if hasattr(self.encoder, "log"):
                self.encoder.log(L, step, log_freq)

            for k, v in self.outputs.items():
                if v is None:
                    continue
                L.log_histogram("train_critic/%s_hist" % k, v, step)

            
            for i in range(3):
                L.log_param("train_critic/q1_fc%d" % (i + 1), self.Q1.trunk[i * 2], step)
                L.log_param("train_critic/q2_fc%d" % (i + 1), self.Q2.trunk[i * 2], step)

