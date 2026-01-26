# ccfdm_modules.py
import torch
import torch.nn as nn

from losses import BilinearSim, CuriosityModule, contrastive_loss


# =============================================================================
# Action Embedding (continuous actions only)
# =============================================================================

class ActionEmbedding(nn.Module):
    def __init__(self, action_dim, embed_dim):
        super().__init__()
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)

        self.net = nn.Sequential(
            nn.Linear(self.action_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, self.embed_dim),
        )

    def forward(self, a):
        return self.net(a.float())


# =============================================================================
# Forward Dynamics Model (latent space)
# =============================================================================

class ForwardDynamicsModel(nn.Module):
    def __init__(self, z_dim, a_emb_dim):
        super().__init__()
        self.z_dim = int(z_dim)
        self.a_emb_dim = int(a_emb_dim)
        in_dim = self.z_dim + self.a_emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, self.z_dim),
        )

    def forward(self, z_t, a_e):
        x = torch.cat([z_t.float(), a_e.float()], dim=1)
        return self.net(x)


# =============================================================================
# Convenience wrapper: ActionEmbedding + FDM
# =============================================================================

class LatentDynamics(nn.Module):
    def __init__(self, action_dim, action_embed_dim, z_dim):
        super().__init__()
        self.action_embedding = ActionEmbedding(action_dim=action_dim, embed_dim=action_embed_dim)
        self.fdm = ForwardDynamicsModel(z_dim=z_dim, a_emb_dim=action_embed_dim)

    def forward(self, z_t, a_t):
        a_e = self.action_embedding(a_t)
        return self.fdm(z_t, a_e)


# =============================================================================
# CURL / CCFDM Core Module
# =============================================================================

class CURL(nn.Module):


    def __init__(
        self,
        obs_shape,
        action_shape,
        z_dim,
        critic,
        critic_target,
        device,
        action_embed_dim=50,
        # contrastive config
        contrastive_method="infonce",
        temperature=1.0,
        normalize=True,
        triplet_margin=0.2,
        # curiosity config
        curiosity_C=0.1,
        curiosity_gamma=1e-6,
        eps=1e-8,
    ):
        super().__init__()

        self.device = device
        self.z_dim = int(z_dim)

        # contrastive knobs
        self.contrastive_method = str(contrastive_method).lower().strip()
        self.temperature = float(temperature)
        self.normalize = bool(normalize)
        self.triplet_margin = float(triplet_margin)
        self.eps = float(eps)

        # encoders from critic (online + target)
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder

        # latent dynamics (action embedding + forward model)
        self.dynamics = LatentDynamics(
            action_dim=action_shape[0],
            action_embed_dim=action_embed_dim,
            z_dim=z_dim,
        ).to(self.device)

        # single shared bilinear similarity
        self.bilinear = BilinearSim(dim=z_dim).to(self.device)

        # curiosity (same bilinear)
        self.curiosity = CuriosityModule(
            device=self.device,
            C=curiosity_C,
            gamma=curiosity_gamma,
            eps=self.eps,
        )

    def encode(self, obs, detach=False):
        obs = obs.to(self.device)
        z = self.encoder(obs, detach=detach)
        return z

    @torch.no_grad()
    def encode_target(self, obs):
        obs = obs.to(self.device)
        z = self.encoder_target(obs, detach=True)
        return z

    def predict_next(self, z_t, a_t):
        z_t = z_t.to(self.device)
        a_t = a_t.to(self.device)
        return self.dynamics(z_t, a_t)

    def compute_contrastive_loss(self, q_pred, k_pos):
        # dispatcher in losses.py (infonce/triplet/byol)
        return contrastive_loss(
            method=self.contrastive_method,
            q_pred=q_pred,
            k_pos=k_pos,
            bilinear=self.bilinear,
            normalize=self.normalize,
            temperature=self.temperature,
            eps=self.eps,
            triplet_margin=self.triplet_margin,
            k_neg=None,
        )

    @torch.no_grad()
    def intrinsic_reward(self, q_pred, k_target, r_ext, t):
        # update rmax_e from extrinsic batch, then compute intrinsic reward
        self.curiosity.update_rmax_e(r_ext)
        return self.curiosity.intrinsic_reward(q_pred, k_target, t)

    def forward_ccfdm(self, obs, action, next_obs, obs_pos=None):
        """
        Paper-faithful CCFDM:   
        - positive key is the next observation (same transition)
        - obs_pos is accepted for compatibility but not used

        Returns:
        z_t (online), z_next_target (target), z_pred_next, loss_contrastive
        """
        obs = obs.to(self.device)
        action = action.to(self.device)
        next_obs = next_obs.to(self.device)

        # z_t online (grad ok)
        z_t = self.encode(obs, detach=False)

        # z_{t+1} target (no grad)
        with torch.no_grad():
            z_next_target = self.encode_target(next_obs)

        # prediction in latent (grad ok)
        z_pred_next = self.predict_next(z_t, action)

        # contrastive loss: q_pred vs k_next (positive = next_obs)
        loss_c = self.compute_contrastive_loss(z_pred_next, z_next_target)

        return z_t, z_next_target, z_pred_next, loss_c
    #ok