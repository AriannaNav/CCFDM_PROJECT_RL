# losses.py
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearSim(nn.Module):
    """sim(q,k) = q^T W k (learnable W)."""
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.W)

    def matrix(self, q, k):
        """(B,D),(B,D)->(B,B)  S_ij = q_i^T W k_j"""
        return (q @ self.W) @ k.t()

    def pair(self, q, k):
        """(B,D),(B,D)->(B,)  s_i = q_i^T W k_i"""
        return (q @ self.W * k).sum(dim=1)


# ---------------------------------------------------------------------
# 1) InfoNCE (paper Eq.8) — in-batch negatives
# ---------------------------------------------------------------------

def infonce_logits_inbatch(
    q_pred,
    k_pos,
    bilinear,
    temperature=1.0,
    normalize=True,
    eps=1e-8,
):
    """
    Paper Eq.(8):
      - logits: (B,B) with positives on diagonal
      - labels: arange(B)
    """
    assert q_pred.ndim == 2 and k_pos.ndim == 2
    assert q_pred.shape == k_pos.shape

    if normalize:
        q_pred = F.normalize(q_pred, dim=1, eps=eps)
        k_pos = F.normalize(k_pos, dim=1, eps=eps)

    temperature = max(float(temperature), eps)

    logits = bilinear.matrix(q_pred, k_pos) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return logits, labels


def infonce_loss(
    q_pred,
    k_pos,
    bilinear,
    temperature=1.0,
    normalize=True,
    eps=1e-8,
):
    logits, labels = infonce_logits_inbatch(
        q_pred=q_pred,
        k_pos=k_pos,
        bilinear=bilinear,
        temperature=temperature,
        normalize=normalize,
        eps=eps,
    )
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------
# 2) Other contrastive losses (same bilinear similarity)
# ---------------------------------------------------------------------

def triplet_bilinear_loss(
    anchor,
    positive,
    negative,
    bilinear,
    margin=0.2,
    normalize=True,
    eps=1e-8,
):
    """
    L = max(0, margin + sim(a,n) - sim(a,p))
    """
    if normalize:
        anchor = F.normalize(anchor, dim=1, eps=eps)
        positive = F.normalize(positive, dim=1, eps=eps)
        negative = F.normalize(negative, dim=1, eps=eps)

    s_ap = bilinear.pair(anchor, positive)
    s_an = bilinear.pair(anchor, negative)
    return F.relu(margin + s_an - s_ap).mean()


def byol_regression_loss(
    q_pred,
    k_target,
    normalize=True,
    eps=1e-8,
):
    """
    BYOL-style cosine regression:
      L = 2 - 2*cos(q,k)
    """
    if normalize:
        qn = F.normalize(q_pred, dim=1, eps=eps)
        kn = F.normalize(k_target, dim=1, eps=eps)
    else:
        qn, kn = q_pred, k_target

    return (2.0 - 2.0 * (qn * kn).sum(dim=1)).mean()


def contrastive_loss(
    method,
    q_pred,
    k_pos,
    bilinear,
    *,
    normalize=True,
    temperature=1.0,
    eps=1e-8,
    triplet_margin=0.2,
    k_neg=None,
):
    """
    method: "infonce" | "triplet" | "byol"
    """
    method = method.lower().strip()

    if method == "infonce":
        return infonce_loss(
            q_pred=q_pred,
            k_pos=k_pos,
            bilinear=bilinear,
            temperature=temperature,
            normalize=normalize,
            eps=eps,
        )

    if method == "byol":
        return byol_regression_loss(
            q_pred=q_pred,
            k_target=k_pos,
            normalize=normalize,
            eps=eps,
        )

    if method == "triplet":
        if k_neg is None:
            # hardest in-batch negative (exclude diagonal)
            with torch.no_grad():
                a = F.normalize(q_pred, dim=1, eps=eps) if normalize else q_pred
                p = F.normalize(k_pos, dim=1, eps=eps) if normalize else k_pos
                S = bilinear.matrix(a, p)
                B = S.shape[0]
                mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
                S_neg = S.masked_fill(~mask, float("-inf"))
                j_hard = S_neg.argmax(dim=1)
                k_neg = p[j_hard]

        return triplet_bilinear_loss(
            anchor=q_pred,
            positive=k_pos,
            negative=k_neg,
            bilinear=bilinear,
            margin=triplet_margin,
            normalize=normalize,
            eps=eps,
        )

    raise ValueError("method must be one of: infonce | triplet | byol")


# ---------------------------------------------------------------------
# 3) Curiosity module (Eq.9-style)
# ---------------------------------------------------------------------

class CuriosityModule:
    """
    r_i = C * exp(-gamma*t) * (d / rmax_i) * rmax_e

    d = dissimilarity derived from bilinear similarity:
      d = -(q^T W k) shifted to be >= 0
    """

    def __init__(
        self,
        device,
        bilinear,
        C=0.1,
        gamma=1e-6,
        normalize_inputs=True,
        eps=1e-8,
    ):
        self.device = device
        self.bilinear = bilinear
        self.C = float(C)
        self.gamma = float(gamma)
        self.normalize_inputs = bool(normalize_inputs)
        self.eps = float(eps)

        self.rmax_e = torch.tensor(0.0, device=self.device)
        self.rmax_i = torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def update_rmax_e(self, r_ext):
        r_ext = r_ext.detach().to(self.device).float().view(-1)
        self.rmax_e = torch.maximum(self.rmax_e, r_ext.max())

    @torch.no_grad()
    def dissimilarity(self, q, k):
        q = q.to(self.device)
        k = k.to(self.device)

        q = F.normalize(q, dim=1, eps=self.eps)
        k = F.normalize(k, dim=1, eps=self.eps)

        sim = (q * k).sum(dim=1)   # cosine similarity in [-1, 1]
        d = 1.0 - sim              # dissimilarity in [0, 2]
        return d

    @torch.no_grad()
    def intrinsic_reward(self, q, k, t):
        d = self.dissimilarity(q, k)

        self.rmax_i = torch.maximum(self.rmax_i, d.max())

        denom_i = torch.clamp(self.rmax_i, min=self.eps)
        scale_e = torch.clamp(self.rmax_e, min=self.eps)

        decay = math.exp(-self.gamma * float(t))
        return self.C * decay * (d / denom_i) * scale_e


# ---------------------------------------------------------------------
# 4) InfoNCE metrics (optional)
# ---------------------------------------------------------------------

@torch.no_grad()
def infonce_metrics_from_logits(logits, labels):
    B = logits.shape[0]
    pred = logits.argmax(dim=1)
    acc = (pred == labels).float().mean().item()

    pos = torch.diag(logits)
    mask = ~torch.eye(B, dtype=torch.bool, device=logits.device)
    neg = logits[mask].view(B, B - 1)

    return {
        "top1_acc": acc,
        "pos_mean": pos.mean().item(),
        "neg_mean": neg.mean().item(),
    }