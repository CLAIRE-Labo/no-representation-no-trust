# Original source code licensed under the MIT license found in https://github.com/pytorch/rl/blob/main/LICENSE

import math
from dataclasses import dataclass

import torch
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictModuleBase,
)
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torchrl.objectives import distance_loss


class BaseLoss(TensorDictModuleBase):
    @property
    def tensor_keys(self):
        return self._tensor_keys

    def set_keys(self, **kwargs) -> None:
        """Set tensordict key names.
        E.g.,
        loss.set_keys(psi="value_target_actor")
        """
        for key, value in kwargs.items():
            if key not in self._AcceptedKeys.__dict__:
                raise ValueError(f"{key} is not an accepted tensordict key")
            if value is not None:
                setattr(self._tensor_keys, key, value)
            else:
                setattr(self._tensor_keys, key, getattr(self.default_keys, key))
        self._refresh_in_keys()


class PPOLoss(BaseLoss):
    """PPO loss (https://arxiv.org/abs/1707.06347) than can:
    - use episodic mask to discard steps not belonging to complete episodes (e.g., because their returns are not complete).
    - clip the policy loss.
    - add adaptive KL penalty.
    - add entropy bonus.
    - add feature trust region.
        - can either implement a regularization or a clipping mechanism.
        - clipping cuts the gradients of the sample with loss(phi_old(x), phi_new(x)) > coef
    """

    @dataclass
    class _AcceptedKeys:
        action: NestedKey = "action"
        complete_episode_mask: NestedKey = "complete_episode_mask"
        sample_log_prob: NestedKey = "sample_log_prob"
        advantage: NestedKey = "advantage_actor"  # The reinforcement psi \\in {advantage_actor, value_target_actor}

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        actor: ProbabilisticTensorDictSequential,
        *,
        use_episodic_mask: bool = False,
        normalize_advantage: bool = True,
        entropy_coef: float = 0.01,  # Set to 0 to disable entropy bonus.
        samples_mc_entropy: int = 1,
        use_clipped_loss: bool = True,  # Set to False to disable clipping.
        clip_epsilon: float = 0.2,  # Epsilon for clipping. If disabled will still be used to compute clip fraction.
        beta_kl: float = 0.0,  # Set to 0 to disable KL penalty, in that case the 4 following args are not used.
        beta_increment: float = 2,
        beta_decrement: float = 0.5,
        kl_target: float = 0.01,  # dtarg in the PPO paper.
        samples_mc_kl: int = 1,
        feature_trust_region_regularize_or_clip: str = "regularize",  # \in {"regularize", "clip"}
        feature_trust_region_coef: float = 0.0,  # For use with regularization. Set to 0 to disable trust region.
        feature_trust_region_limit: float = 0.1,  # For use with clip.
        feature_trust_region_type: str = "l2",  # \in {"l2", "cosine"}
        feature_trust_use_preactivation: bool = True,  # Use preactivation for feature trust region.
        feature_trust_all_layers: bool = False,  # Else only on the last feature layer.
        safe_logratio: bool = False,  # Use clamp logratio to avoid NaNs in the gradients.
    ):
        super().__init__()
        self.actor = actor
        self._tensor_keys = self._AcceptedKeys()
        self._refresh_in_keys()
        self.out_keys = [
            "loss_policy",
            "loss_kl",
            "loss_entropy",
            "clipped_fraction",
            "kl",
            "entropy",
            "beta_kl",
        ]
        self.use_episodic_mask = use_episodic_mask
        self.normalize_psi = normalize_advantage
        self.entropy_coef = entropy_coef
        self.samples_mc_entropy = samples_mc_entropy
        self.use_clipped_loss = use_clipped_loss
        self.clip_epsilon = clip_epsilon
        self.init_beta_kl = beta_kl
        self.register_buffer("beta_kl", torch.tensor(beta_kl))
        if beta_increment < 1.0:
            raise ValueError(f"Beta increment should be >= 1.0 in KLPENPPOLoss, got {beta_increment:4.4f}")
        self.beta_increment = beta_increment
        if beta_decrement > 1.0:
            raise ValueError(f"Beta decrement should be <= 1.0 in KLPENPPOLoss, got {beta_decrement:4.4f}")
        self.beta_decrement = beta_decrement
        self.kl_target = kl_target
        self.samples_mc_kl = samples_mc_kl
        self._beta_init = beta_kl
        self.feature_trust_region_limit = feature_trust_region_limit
        self.feature_trust_region_coef = feature_trust_region_coef
        self.feature_trust_region_regularize_or_clip = feature_trust_region_regularize_or_clip
        self.feature_trust_region_type = feature_trust_region_type
        self.feature_trust_use_preactivation = feature_trust_use_preactivation
        self.feature_trust_all_layers = feature_trust_all_layers
        self.safe_logratio = safe_logratio

    def _refresh_in_keys(self):
        self.in_keys = [
            self.tensor_keys.action,
            self.tensor_keys.complete_episode_mask,
            self.tensor_keys.advantage,
            self.tensor_keys.sample_log_prob,
            *self.actor.in_keys,
        ]

    @property
    def _clip_bounds(self):
        return (
            math.log1p(-self.clip_epsilon),
            math.log1p(self.clip_epsilon),
        )

    def forward(self, prev_tensordict: TensorDictBase) -> TensorDict:
        new_tensordict = prev_tensordict.select(*self.actor.in_keys)
        psi = prev_tensordict.get(self.tensor_keys.advantage).clone()  # [B, 1]
        if self.use_episodic_mask:
            complete_episode_mask = prev_tensordict.get(self.tensor_keys.complete_episode_mask)  # [B, 1]
            n_mask_valid = complete_episode_mask.sum()
            # non-urgent-fix: can this be done without the if to avoid CUDA syncing?
            if n_mask_valid > 0:
                correction_factor = complete_episode_mask.numel() / n_mask_valid
            else:
                correction_factor = 0.0
        else:
            complete_episode_mask = 1.0
            correction_factor = 1.0
        if self.normalize_psi and psi.numel() > 1:
            loc = (psi * complete_episode_mask).mean() * correction_factor
            scale = (((psi - loc) ** 2) * complete_episode_mask).mean() * correction_factor
            # scale is the biased variance estimator as in torch.var(correction=0)
            psi = ((psi - loc) * complete_episode_mask) / scale.sqrt().clamp_min(1e-6)

        # Doesn't do forward pass on the actor, just gets the distribution.
        prev_dist = self.actor.build_dist_from_params(prev_tensordict)
        prev_log_prob = prev_tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError(f"tensordict prev_log_prob={self.tensor_keys.sample_log_prob} requires grad.")

        # Does forward pass on the actor and gets the distribution.
        new_dist = self.actor.get_dist(new_tensordict)
        new_log_prob = new_dist.log_prob(prev_tensordict.get(self.tensor_keys.action))

        # Log prob.
        # Log weigh is clamped to avoid inf and nan in gradients.
        # This happens in the other direction of the advantage despite PPO clipping with bad function approximation.
        if self.safe_logratio:
            log_weight = (new_log_prob - prev_log_prob).unsqueeze(-1).clamp(min=math.log(10e-8), max=math.log(10e8))
        else:
            log_weight = (new_log_prob - prev_log_prob).unsqueeze(-1)
        log_weight_clip = log_weight.clamp(*self._clip_bounds)

        # Policy loss.
        gain_no_clip = log_weight.exp() * psi  # Termed "policy advantage" in Kakade and Langford (2002)
        gain_clipped = log_weight_clip.exp() * psi
        gain_ppo, clipped_fraction = torch.stack([gain_no_clip, gain_clipped], -1).min(dim=-1)
        if self.use_clipped_loss:
            gain = gain_ppo
        else:
            gain = gain_no_clip

        # KL.
        try:
            kl = torch.distributions.kl.kl_divergence(prev_dist, new_dist)
        except NotImplementedError:
            x = prev_dist.sample((self.samples_mc_kl,))  # non-urgent-fix: sample or take the mode sample?
            # x has shape [samples_mc_entropy, B, ...]
            kl = (prev_dist.log_prob(x) - new_dist.log_prob(x)).mean(0)
            # so take mean over samples_mc_entropy and keep batch dimension
        kl = kl.unsqueeze(-1)  # [B, 1]

        # Entropy.
        try:
            entropy = new_dist.entropy()
        except NotImplementedError:
            x = new_dist.sample((self.samples_mc_entropy,))  # non-urgent-fix TorchRL uses rsample why?
            # x has shape [samples_mc_entropy, B, ...]
            entropy = -new_dist.log_prob(x).mean(0)
            # so take mean over samples_mc_entropy and keep batch dimension
        entropy = entropy.unsqueeze(-1)  # [B, 1]

        # Feature trust-region.
        # loss on higher than margin.
        if self.feature_trust_use_preactivation:
            if self.feature_trust_all_layers:
                feats_key = "all_preactivations_policy"
            else:
                feats_key = "features_preactivation_policy"
        else:
            if self.feature_trust_all_layers:
                raise NotImplementedError("Feature trust region on all layers post-activation not implemented.")
            else:
                feats_key = "features_policy"

        features_l2 = (new_tensordict[feats_key] - prev_tensordict[feats_key]).pow(2).mean(-1).unsqueeze(-1)
        features_cosine = torch.nn.functional.cosine_similarity(
            new_tensordict[feats_key], prev_tensordict[feats_key], dim=-1
        ).unsqueeze(-1)

        if self.feature_trust_region_type == "cosine":
            features_outoftrust = features_cosine.detach() < (1 - self.feature_trust_region_limit)
            features_loss = -features_cosine
        elif self.feature_trust_region_type == "l2":
            features_outoftrust = features_l2.detach() > self.feature_trust_region_limit
            features_loss = features_l2
        else:
            raise ValueError(f"Unknown feature trust region type {self.feature_trust_region_type}")

        if self.feature_trust_region_regularize_or_clip == "clip":
            # Block the gain gradients if the features are out of trust.
            gain = (gain * features_outoftrust.float()).detach() + gain * (1 - features_outoftrust.float())
            # No regularization.
            features_loss = features_loss.detach()

        # Mask and correctly average.
        gain = (gain * complete_episode_mask).mean() * correction_factor
        gain_no_clip = (gain_no_clip.detach() * complete_episode_mask).mean() * correction_factor
        prob_ratio = (log_weight.detach().exp() * complete_episode_mask).mean() * correction_factor
        prob_ratio_clip = (log_weight_clip.detach().exp() * complete_episode_mask).mean() * correction_factor
        clipped_fraction = (clipped_fraction.detach().float() * complete_episode_mask).mean() * correction_factor
        kl = (kl * complete_episode_mask).mean() * correction_factor
        entropy = (entropy * complete_episode_mask).mean() * correction_factor
        features_loss = (features_loss * complete_episode_mask).mean() * correction_factor
        features_outoftrust = (features_outoftrust.detach().float() * complete_episode_mask).mean() * correction_factor

        return TensorDict(
            {
                "loss_policy": -gain,
                "loss_kl": self.beta_kl * kl,
                "loss_entropy": -self.entropy_coef * entropy,
                "loss_features_policy": self.feature_trust_region_coef * features_loss,
                "clipped_fraction": clipped_fraction.detach(),
                "neg_policy_advantage": -gain_no_clip.detach(),
                "policy_prob_ratio": prob_ratio.detach(),
                "policy_prob_ratio_clip": prob_ratio_clip.detach(),
                "kl": kl.detach(),
                "entropy": entropy.detach(),
                "beta_kl": self.beta_kl.clone(),
                "features_policy_loss_no_coef": features_loss.detach(),
                "features_policy_outoftrust_fraction": features_outoftrust.detach(),
            },
            [],
        )

    def reset(self) -> None:
        """Reset the dynamic hyperparameter for the KL penalty."""
        self.beta_kl = torch.tensor(self._beta_init)

    def update_adaptive_parameters(self, kl) -> None:
        """Update the dynamic hyperparameter (here the KL penalty)."""
        if kl > (self.kl_target * 1.5):
            self.beta_kl.data = self.beta_kl.data * self.beta_increment
        elif kl < (self.kl_target / 1.5):
            self.beta_kl.data = self.beta_kl.data * self.beta_decrement


class ValueLoss(BaseLoss):
    """Computes the loss of a value network for a given targets."""

    @dataclass
    class _AcceptedKeys:
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        complete_episode_mask: NestedKey = "complete_episode_mask"
        value_target: NestedKey = "value_target"  # GAE, etc.

    default_keys = _AcceptedKeys()

    def __init__(
        self,
        critic: TensorDictModule,
        *,
        coef: float = 1.0,
        use_episodic_mask: bool = False,
        loss_type: str = "l2",  # \in {"l2", "l1" or "smooth_l1" (kind of Huber loss)}
        feature_trust_region_regularize_or_clip: str = "regularize",  # \in {"regularize", "clip"}
        feature_trust_region_coef: float = 0.0,  # For use with regularization. Set to 0 to disable trust region.
        feature_trust_region_limit: float = 0.1,  # For use with clip.
        feature_trust_region_type: str = "l2",  # \in {"l2", "cosine"}
        feature_trust_use_preactivation: bool = True,  # Use preactivation for feature trust region.
        feature_trust_all_layers: bool = False,  # Else only on the last feature layer
    ):
        super().__init__()
        self.critic = critic
        self._tensor_keys = self._AcceptedKeys()
        self._refresh_in_keys()
        self.out_keys = [
            "loss_critic",
        ]
        self.coef = coef
        self.use_episodic_mask = use_episodic_mask
        self.loss_type = loss_type
        self.feature_trust_region_coef = feature_trust_region_coef
        self.feature_trust_region_limit = feature_trust_region_limit
        self.feature_trust_region_regularize_or_clip = feature_trust_region_regularize_or_clip
        self.feature_trust_region_type = feature_trust_region_type
        self.feature_trust_use_preactivation = feature_trust_use_preactivation
        self.feature_trust_all_layers = feature_trust_all_layers

    def _refresh_in_keys(self):
        self.in_keys = [
            self.tensor_keys.value_target,
            self.tensor_keys.complete_episode_mask,
            *self.critic.in_keys,
        ]

    def forward(self, prev_tensordict: TensorDictBase) -> TensorDict:
        target = prev_tensordict.get(self.tensor_keys.value_target)
        if self.use_episodic_mask:
            complete_episode_mask = prev_tensordict.get(self.tensor_keys.complete_episode_mask)
            n_mask_valid = complete_episode_mask.sum()
            if n_mask_valid > 0:
                correction_factor = complete_episode_mask.numel() / n_mask_valid
            else:
                correction_factor = 0.0
        else:
            complete_episode_mask = 1.0
            correction_factor = 1.0

        new_tensordict = prev_tensordict.select(*self.critic.in_keys)
        self.critic(new_tensordict)
        value = new_tensordict.get("state_value")
        loss = distance_loss(
            value,
            target,
            loss_function=self.loss_type,
        )

        # Feature trust-region.
        # loss on higher than margin.
        if self.feature_trust_use_preactivation:
            if self.feature_trust_all_layers:
                feats_key = "all_preactivations_value"
            else:
                feats_key = "features_preactivation_value"
        else:
            if self.feature_trust_all_layers:
                raise NotImplementedError("Feature trust region on all layers post-activation not implemented.")
            else:
                feats_key = "features_value"

        features_l2 = (new_tensordict[feats_key] - prev_tensordict[feats_key]).pow(2).mean(-1).unsqueeze(-1)
        features_cosine = torch.nn.functional.cosine_similarity(
            new_tensordict[feats_key], prev_tensordict[feats_key], dim=-1
        ).unsqueeze(-1)

        if self.feature_trust_region_type == "cosine":
            features_outoftrust = features_cosine.detach() < (1 - self.feature_trust_region_limit)
            features_loss = -features_cosine
        elif self.feature_trust_region_type == "l2":
            features_outoftrust = features_l2.detach() > self.feature_trust_region_limit
            features_loss = features_l2
        else:
            raise ValueError(f"Unknown feature trust region type {self.feature_trust_region_type}")

        if self.feature_trust_region_regularize_or_clip == "clip":
            # Block the loss gradients if the features are out of trust.
            loss = (loss * features_outoftrust.float()).detach() + loss * (1 - features_outoftrust.float())
            # No regularization.
            features_loss = features_loss.detach()

        loss = (loss * complete_episode_mask).mean() * correction_factor
        features_loss = (features_loss * complete_episode_mask).mean() * correction_factor
        features_outoftrust = (features_outoftrust.float() * complete_episode_mask).mean() * correction_factor

        return TensorDict(
            {
                "loss_critic": self.coef * loss,
                "loss_features_value": self.feature_trust_region_coef * features_loss,
                "features_value_loss_no_coef": features_loss.detach(),
                "features_value_outoftrust_fraction": features_outoftrust,
            },
            [],
        )
