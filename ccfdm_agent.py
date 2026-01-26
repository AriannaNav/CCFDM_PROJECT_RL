# ccfdm_agent.py
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from sac import SACActor, SACCritic
from ccfdm_modules import CURL
from utils import soft_update


class CCFDMAgent(object):
    """
    SAC + CCFDM (Algorithm 1)
    """

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.1,
        alpha_lr=1e-4,
        alpha_beta=0.5,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.01,
        critic_target_update_freq=2,
        encoder_type="pixel",
        encoder_feature_dim=50,
        num_layers=4,
        num_filters=32,
        encoder_tau=0.01,
        ccfmd_update_freq=1,
        # contrastive config
        contrastive_method="infonce",
        temperature=1.0,
        normalize=True,
        triplet_margin=0.2,
        # curiosity config
        curiosity_C=0.2,
        curiosity_gamma=2e-5,
        intrinsic_weight=1.0,
        intrinsic_decay=0.0,  # kept for backward compatibility; not used
        action_embed_dim=50,
        # NEW: keep actor encoder synced with critic encoder
        actor_encoder_sync=True,
    ):
        self.device = device

        self.discount = float(discount)
        self.critic_tau = float(critic_tau)
        self.encoder_tau = float(encoder_tau)
        self.actor_update_freq = int(actor_update_freq)
        self.critic_target_update_freq = int(critic_target_update_freq)
        self.ccfmd_update_freq = int(ccfmd_update_freq)

        self.intrinsic_weight = float(intrinsic_weight)
        self.intrinsic_decay = float(intrinsic_decay)  # not used

        self.actor_encoder_sync = bool(actor_encoder_sync)

        # --- networks
        self.actor = SACActor(
            obs_dim=obs_shape,
            action_dim=action_shape,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            encoder_feature_dim=encoder_feature_dim,
            log_std_min=actor_log_std_min,
            log_std_max=actor_log_std_max,
            num_layers=num_layers,
            num_filters=num_filters,
        ).to(device)

        self.critic = SACCritic(
            obs_dim=obs_shape,
            action_dim=action_shape,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            encoder_feature_dim=encoder_feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
        ).to(device)

        self.critic_target = SACCritic(
            obs_dim=obs_shape,
            action_dim=action_shape,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            encoder_feature_dim=encoder_feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # tie conv weights actor<-critic (CURL-style init)
        self._sync_actor_encoder_from_critic()

        # --- temperature (alpha)
        self.log_alpha = torch.tensor(math.log(init_temperature), device=device)
        self.log_alpha.requires_grad_(True)
        self.target_entropy = -float(action_shape[0])

        # --- CCFDM module (uses critic encoder + critic_target encoder)
        self.ccfdm = CURL(
            obs_shape=obs_shape,
            action_shape=action_shape,
            z_dim=encoder_feature_dim,
            critic=self.critic,
            critic_target=self.critic_target,
            device=device,
            action_embed_dim=action_embed_dim,
            contrastive_method=contrastive_method,
            temperature=temperature,
            normalize=normalize,
            triplet_margin=triplet_margin,
            curiosity_C=curiosity_C,
            curiosity_gamma=curiosity_gamma,
        ).to(device)

        # --- optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # NO-OVERLAP optimizers:
        # (1) Q heads only
        q_params = list(self.critic.Q1.parameters()) + list(self.critic.Q2.parameters())
        self.critic_q_optimizer = torch.optim.Adam(
            q_params, lr=critic_lr, betas=(critic_beta, 0.999)
        )

        # (2) encoder only (updated by critic TD loss AND contrastive loss)
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        # (3) ccfdm-only (dynamics + bilinear) updated only by Eq.(8)
        ccfdm_only_params = (
            list(self.ccfdm.dynamics.parameters())
            + list(self.ccfdm.bilinear.parameters())
        )
        self.ccfdm_optimizer = torch.optim.Adam(
            ccfdm_only_params, lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.train(True)

    # ---------------- small helpers ----------------

    def _sync_actor_encoder_from_critic(self):
        """Keep actor encoder aligned with critic encoder (important to avoid representation drift)."""
        if not self.actor_encoder_sync:
            return
        if hasattr(self.actor, "encoder") and hasattr(self.critic, "encoder"):
            if hasattr(self.actor.encoder, "copy_conv_weights_from"):
                self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = bool(training)

        self.actor.train(self.training)
        self.critic.train(self.training)
        self.ccfdm.train(self.training)

        # target always eval (EMA updates only)
        self.critic_target.eval()

    # ---------------- acting ----------------

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            _, pi, _, _ = self.actor(obs, compute_pi=True, compute_log_pi=False)
            return pi.cpu().numpy().flatten()

    # ---------------- SAC updates ----------------

    def update_critic(self, obs, action, reward, next_obs, not_done, logger=None, step=0):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, compute_pi=True, compute_log_pi=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + not_done * self.discount * target_V

        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_q_optimizer.step()
        self.encoder_optimizer.step()

        if logger is not None:
            logger.log("train/critic_loss", float(critic_loss.item()), step)

    def update_actor_and_alpha(self, obs, logger=None, step=0):
        # detach encoder: actor update shouldn't backprop into encoder (CURL-style)
        _, pi, log_pi, _ = self.actor(obs, detach_encoder=True, compute_pi=True, compute_log_pi=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if logger is not None:
            logger.log("train/actor_loss", float(actor_loss.item()), step)
            logger.log("train/alpha_loss", float(alpha_loss.item()), step)
            logger.log("train/alpha", float(self.alpha.item()), step)

    # ---------------- CCFDM (Eq.8 + Eq.9) ----------------

    def update_ccfdm(self, obs, action, next_obs, logger=None, step=0):
        _, _, _, loss_c = self.ccfdm.forward_ccfdm(obs, action, next_obs, obs_pos=None)

        self.encoder_optimizer.zero_grad()
        self.ccfdm_optimizer.zero_grad()
        loss_c.backward()
        self.encoder_optimizer.step()
        self.ccfdm_optimizer.step()

        if logger is not None:
            logger.log("train/contrastive_loss", float(loss_c.item()), step)

        return loss_c

    @torch.no_grad()
    def compute_intrinsic_reward(self, obs, action, next_obs, r_ext, t):
        z_t = self.ccfdm.encode(obs, detach=True)
        z_next_target = self.ccfdm.encode_target(next_obs)
        z_pred_next = self.ccfdm.predict_next(z_t, action)

        # NOTE: what intrinsic_reward computes depends on CuriosityModule in losses.py
        ri = self.ccfdm.intrinsic_reward(z_pred_next, z_next_target, r_ext, t)
        return ri

    # ---------------- main update ----------------

    def update(self, replay_buffer, logger=None, step=0):
        batch_rl = replay_buffer.sample()

        obs = batch_rl.obs.to(self.device)
        action = batch_rl.action.to(self.device)
        reward = batch_rl.reward.to(self.device)
        next_obs = batch_rl.next_obs.to(self.device)
        not_done = batch_rl.not_done.to(self.device)

        if self.training:
            ri = self.compute_intrinsic_reward(obs, action, next_obs, reward, step)
            reward_total = reward + self.intrinsic_weight * ri.view(-1, 1)
        else:
            ri = None
            reward_total = reward

        if logger is not None:
            logger.log("train/reward_ext_mean", float(reward.mean().item()), step)
            logger.log("train/reward_total_mean", float(reward_total.mean().item()), step)
            if self.training and ri is not None:
                logger.log("train/reward_int_mean", float(ri.mean().item()), step)

        # SAC updates
        self.update_critic(obs, action, reward_total, next_obs, not_done, logger, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, logger, step)

        # CCFDM contrastive update (Eq.8)
        if step % self.ccfmd_update_freq == 0:
            batch_cpc = replay_buffer.sample_cpc()
            obs_anchor = batch_cpc.obs.to(self.device)
            action_cpc = batch_cpc.action.to(self.device)
            next_obs_cpc = batch_cpc.next_obs.to(self.device)

            self.update_ccfdm(obs_anchor, action_cpc, next_obs_cpc, logger, step)

        # target / momentum update (critic_target + encoder_target)
        if step % self.critic_target_update_freq == 0:
            soft_update(self.critic_target.Q1, self.critic.Q1, self.critic_tau)
            soft_update(self.critic_target.Q2, self.critic.Q2, self.critic_tau)
            soft_update(self.critic_target.encoder, self.critic.encoder, self.encoder_tau)

            # IMPORTANT: keep actor encoder aligned with critic encoder
            self._sync_actor_encoder_from_critic()

    # ---------------- io ----------------

    def save(self, path):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "ccfdm": self.ccfdm.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.ccfdm.load_state_dict(payload["ccfdm"])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))

        # after load, re-sync actor encoder (safe)
        self._sync_actor_encoder_from_critic()