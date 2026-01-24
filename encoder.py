from __future__ import annotations
from typing import Dict, Type
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def copy_weights(src, trg):
    assert type(src) is type(trg), f"Type mismatch: {type(src)} vs {type(trg)}"
    trg.weight.copy_(src.weight)
    if trg.bias is not None and src.bias is not None:
        trg.bias.copy_(src.bias)


class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, output_logits):
        super().__init__()
        assert len(obs_shape) == 3, f"obs_shape must be (C,H,W), got {obs_shape}"

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.outputs = dict()
        self.output_logits = output_logits

        C, H, W = obs_shape

        convs = []
        convs.append(nn.Conv2d(C, num_filters, kernel_size=3, stride=2))
        for _ in range(1, num_layers):
            convs.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1))
        self.convs = nn.ModuleList(convs)

        # compute flatten dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            h = self._forward_convs(dummy)
            flatten_dim = h.view(1, -1).shape[1]

        self.fc = nn.Linear(flatten_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

    def _forward_convs(self, obs):
        x = obs
        for conv in self.convs:
            x = F.relu(conv(x))
        return x

    def forward_conv(self, obs):
        # normalize to [0,1] if uint8 or if values look like 0..255
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        else:
            if obs.max() > 1.5:
                obs = obs / 255.0

        h = self._forward_convs(obs)
        h = h.view(h.size(0), -1)
        return h

    def forward(self, obs, detach = False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()

        z = self.fc(h)
        z = self.ln(z)

        if self.output_logits:
            return z
        return torch.tanh(z)

    def copy_conv_weights_from(self, source):
        assert isinstance(source, PixelEncoder), "source must be PixelEncoder"
        assert len(source.convs) == len(self.convs), "num_layers mismatch"

        for src_layer, trg_layer in zip(source.convs, self.convs):
            copy_weights(src_layer, trg_layer)

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


def make_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits):
    encoder_type = (encoder_type or "pixel").lower().strip()
    if encoder_type != "pixel":
        raise ValueError(f"Only encoder_type='pixel' is supported now. Got '{encoder_type}'")

    return PixelEncoder(
        obs_shape=obs_shape,
        feature_dim=feature_dim,
        num_layers=num_layers,
        num_filters=num_filters,
        output_logits=output_logits,
    )