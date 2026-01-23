import torch
import torch.nn as nn

#encoder.py 
#ref https://github.com/thanhkaist/CCFDM1/blob/main/encoder.py
# encoder.py
# Macro-TODO (Encoder only: PixelEncoder + IdentityEncoder + factory)

from __future__ import annotations

# TODO(Imports):
# - torch
# - torch.nn as nn
# - typing: Dict, Type, Optional

# =============================================================================
# TODO 0) Purpose
# =============================================================================
# - Define encoder modules ONLY:
#   - PixelEncoder: CNN -> latent z
#   - IdentityEncoder: passthrough for vector observations
# - Provide make_encoder factory
# - Provide optional conv weight copying (for tying conv between modules)
#
# IMPORTANT:
# - No EMA update here (belongs to ccfdm_agent.py)
# - No SAC logic here
# - No losses here

# =============================================================================
# TODO 1) Helper: tie/copy weights (optional)
# =============================================================================
# TODO:
# def tie_weights(src: nn.Module, trg: nn.Module) -> None:
#   - assert same type
#   - trg.weight = src.weight; trg.bias = src.bias
#
# NOTE:
# - Use ONLY for conv-weight tying between two encoders (e.g., actor vs critic if needed).
# - Do NOT use this to "EMA update" KE (EMA is separate, in agent).

# =============================================================================
# TODO 2) PixelEncoder (CNN)
# =============================================================================
# TODO:
# class PixelEncoder(nn.Module):
#   __init__(obs_shape, feature_dim, num_layers, num_filters, output_logits=False)
#   - Build conv stack:
#       conv1: stride=2
#       conv_i: stride=1
#   - Compute flatten_dim dynamically via dummy forward:
#       with torch.no_grad(): run conv on zeros -> compute flattened size
#   - Define:
#       self.fc = nn.Linear(flatten_dim, feature_dim)
#       self.ln = nn.LayerNorm(feature_dim)
#   - Store:
#       obs_shape, feature_dim, num_layers, output_logits
#
# TODO: forward_conv(obs)
# - Normalize obs to [0,1] (assume uint8 input in [0,255])
# - Apply conv + ReLU
# - Flatten -> (B, flatten_dim)
#
# TODO: forward(obs, detach=False)
# - h = forward_conv(obs)
# - if detach: h = h.detach()
# - z = ln(fc(h))
# - if output_logits: return z
#   else: return torch.tanh(z)
#
# TODO: copy_conv_weights_from(source_encoder)
# - Tie/copy ONLY conv weights

# =============================================================================
# TODO 3) IdentityEncoder (vector obs)
# =============================================================================
# TODO:
# class IdentityEncoder(nn.Module):
# - feature_dim = obs_shape[0]
# - forward returns obs (cast to float if needed)
# - copy_conv_weights_from: pass

# =============================================================================
# TODO 4) Factory: make_encoder
# =============================================================================
# TODO:
# _AVAILABLE_ENCODERS = {"pixel": PixelEncoder, "identity": IdentityEncoder}
#
# def make_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False):
# - validate encoder_type
# - return instance

# =============================================================================
# TODO 5) Notes for CCFDM usage
# =============================================================================
# - ccfdm_agent.py will instantiate:
#     qe = make_encoder(...)
#     ke = make_encoder(...)
# - ke starts as a copy of qe weights (one-time copy)
# - ke is updated ONLY via EMA in ccfdm_agent.py