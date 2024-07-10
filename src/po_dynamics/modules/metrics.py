import numpy as np
import torch
import wandb
from torch.nn import functional as F

from po_dynamics.utils.logging import dict_with_prefix


def compute_ranks_from_features(feature_matrices):
    """Computes different approximations of the rank of the feature matrices.

    Args:
        feature_matrices (torch.Tensor): A tensor of shape (B_matrices, N_obs, D_dims).

    (1) Effective rank.
    A continuous approximation of the rank of a matrix.
    Definition 2.1. in Roy & Vetterli, (2007) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7098875
    Also used in Huh et al. (2023) https://arxiv.org/pdf/2103.10427.pdf


    (2) Approximate rank.
    Threshold at the dimensions explaining 99% of the variance in a PCA analysis.
    Section 2 in Yang et al. (2020) https://arxiv.org/pdf/1909.12255.pdf

    (3) srank.
    Another (incorrect?) version of (2).
    Section 3 in Kumar et al. https://arxiv.org/pdf/2010.14498.pdf

    (4) Feature rank.
    A threshold rank: normalize by dim size and discard dimensions with singular values below 0.01.
    Equations (4) and (5). Lyle et al. (2022) https://arxiv.org/pdf/2204.09560.pdf

    (5) PyTorch/NumPy rank.
    Rank defined in PyTorch and NumPy (https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html)
    (https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html)
    Quoting Numpy:
        This is the algorithm MATLAB uses [1].
        It also appears in Numerical recipes in the discussion of SVD solutions for linear least squares [2].
        [1] MATLAB reference documentation, “Rank” https://www.mathworks.com/help/techdoc/ref/rank.html
        [2] W. H. Press, S. A. Teukolsky, W. T. Vetterling and B. P. Flannery, “Numerical Recipes (3rd edition)”,
        Cambridge University Press, 2007, page 795.

    """
    cutoff = 0.01  # not used in (1), 1 - 99% in (2), delta in (3), epsilon in (4).
    threshold = 1 - cutoff

    if feature_matrices.shape[1] < feature_matrices.shape[2]:
        return {}  # N < D.

    svals = torch.linalg.svdvals(feature_matrices)

    # (1) Effective rank. Roy & Vetterli (2007)
    sval_sum = torch.sum(svals, dim=1)
    sval_dist = svals / sval_sum.unsqueeze(-1)
    # Replace 0 with 1. This is a safe trick to avoid log(0) = -inf
    # as Roy & Vetterli assume 0*log(0) = 0 = 1*log(1).
    sval_dist_fixed = torch.where(sval_dist == 0, torch.ones_like(sval_dist), sval_dist)
    effective_ranks = torch.exp(-torch.sum(sval_dist_fixed * torch.log(sval_dist_fixed), dim=1))

    # (2) Approximate rank. PCA variance. Yang et al. (2020)
    sval_squares = svals**2
    sval_squares_sum = torch.sum(sval_squares, dim=1)
    cumsum_squares = torch.cumsum(sval_squares, dim=1)
    threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum.unsqueeze(-1))
    approximate_ranks = (~threshold_crossed).sum(dim=-1) + 1

    # (3) srank. Weird. Kumar et al. (2020)
    cumsum = torch.cumsum(svals, dim=1)
    threshold_crossed = cumsum >= threshold * sval_sum.unsqueeze(-1)
    sranks = (~threshold_crossed).sum(dim=-1) + 1

    # (4) Feature rank. Most basic. Lyle et al. (2022)
    n_obs = torch.tensor(feature_matrices.shape[1], device=feature_matrices.device)
    svals_of_normalized = svals / torch.sqrt(n_obs)
    over_cutoff = svals_of_normalized > cutoff
    feature_ranks = over_cutoff.sum(dim=-1)

    # (5) PyTorch/NumPy rank.
    pytorch_ranks = torch.linalg.matrix_rank(feature_matrices)

    # Some singular values.
    singular_values = dict(
        lambda_1=svals_of_normalized[:, 0],
        lambda_N=svals_of_normalized[:, -1],
    )
    if svals_of_normalized.shape[1] > 1:
        singular_values.update(lambda_2=svals_of_normalized[:, 1])

    ranks = dict(
        effective_rank_vetterli=effective_ranks,
        approximate_rank_pca=approximate_ranks,
        srank_kumar=sranks,
        feature_rank_lyle=feature_ranks,
        pytorch_rank=pytorch_ranks,
    )

    out = {**singular_values, **ranks}

    return out


def compute_eval_stats(batches):
    """Compute performance stats from a batch of rollouts."""
    out = {
        # These are stats on rollouts (parts of trajectories) not on episodes.
        f"perf/max_timestep": batches["next", "step_count"].max(),
        f"perf/min_timestep": batches["step_count"].min(),
        f"perf/max_reward": batches["next", "reward"].max(),
        f"perf/min_reward": batches["next", "reward"].min(),
        f"perf/avg_reward": batches["next", "reward"].mean(),
    }
    # non-urgent-fix: can skip the syncing .any() if it's okay to log max as -inf and avg as nan.
    if batches["next", "done"].any():
        # These are stats on episodes.
        n_done = batches["next", "done"].sum()
        # Tricks to get the max and avg out of the valid returns only without mask_select to avoid CUDA syncing.
        mask = 1 / batches["next", "done"] - 1  # 0 if done, inf otherwise.
        returns_masked = batches["next", "return"] - mask  # return[i] if done[i], -inf otherwise.
        max_return = returns_masked.max()
        avg_return = (batches["next", "return"] * batches["next", "done"]).sum() / n_done
        max_end_timestep = (batches["next", "step_count"] * batches["next", "done"]).max()
        avg_end_timestep = (batches["next", "step_count"] * batches["next", "done"]).sum() / n_done
        out.update(
            {
                f"perf/max_return": max_return,
                f"perf/avg_return": avg_return,
                f"perf/max_episode_timestep": max_end_timestep,
                f"perf/avg_episode_timestep": avg_end_timestep,
            }
        )

        if ("next", "return_raw") in batches.keys(include_nested=True):
            returns_raw_masked = batches["next", "return_raw"] - mask
            max_return_raw = returns_raw_masked.max()
            avg_return_raw = (batches["next", "return_raw"] * batches["next", "done"]).sum() / n_done
            out.update({f"perf/max_return_raw": max_return_raw, f"perf/avg_return_raw": avg_return_raw})
    return out


def compute_effective_ranks(data_list, data_groups, data_features):
    """
    Computes the effective ranks of each of the data_features[i][j] in data_list[i] in a batched way.
    Expects flat (no time dimension) tensordicts containing a feature matrix per data_features[i][j].

    The method Does not support different feature matrix shapes.
    """
    stack = [data[data_feature] for i, data in enumerate(data_list) for data_feature in data_features[i]]
    groups = [f"{data_feature}_{group}" for i, group in enumerate(data_groups) for data_feature in data_features[i]]
    features = torch.stack(stack, dim=0)
    try:
        ranks = compute_ranks_from_features(features)
    except torch._C._LinAlgError:
        return {}
    out = {}
    for rank_group, ranks_values in ranks.items():
        for i, data_feature_group in enumerate(groups):
            out[f"SVD/{rank_group}/{data_feature_group}"] = ranks_values[i]
    return out


def compute_feature_norms(data_list, data_groups, data_features):
    """
    Computes the norms of each of the data_features[i][j] in data_list[i] in a batched way.
    """
    stack = [data[data_feature] for i, data in enumerate(data_list) for data_feature in data_features[i]]
    groups = [f"{data_feature}_{group}" for i, group in enumerate(data_groups) for data_feature in data_features[i]]
    features = torch.stack(stack, dim=0)  # (B, N, D)
    norms = torch.linalg.norm(features, dim=-1).mean(dim=-1)
    means = features.mean(dim=-1).mean(dim=-1)
    stds = features.std(dim=-1).mean(dim=-1)
    out = {}
    for i, data_feature_group in enumerate(groups):
        out[f"feature_stats/norm_{data_feature_group}"] = norms[i]
        out[f"feature_stats/avg_{data_feature_group}"] = means[i]
        out[f"feature_stats/std_{data_feature_group}"] = stds[i]
    return out


def compute_dead_neurons(data_list, data_groups, data_feature, activation):
    stack = [data[data_feature] for data in data_list]
    groups = [f"{data_feature}_{group}" for group in data_groups]
    features = torch.stack(stack, dim=0)
    dead_neurons = compute_dead_neurons_from_features(features, activation)
    out = {}
    for i, data_feature_group in enumerate(groups):
        out[f"dead_neurons/{data_feature_group}"] = dead_neurons[i]
    return out


def compute_dead_neurons_from_features(features, activation):
    TANH_STD_THRESHOLD = 0.001
    match activation:
        case "ReLU" | "GELU":
            return (features == 0).all(dim=1).sum(dim=-1)
        case "Tanh":
            return (features.std(dim=1) < TANH_STD_THRESHOLD).sum(dim=-1)
        case "LeakyReLU":
            return (features < 0).all(dim=1).sum(dim=-1)
        case other:
            raise NotImplementedError(f"Activation {activation} not implemented.")


def compute_value_diversity(data):
    state_values = data["state_value"].squeeze(-1)
    return dict_with_prefix(
        {
            "avg_state_value": state_values.mean(),
            "std_state_value": state_values.std(),
        },
        "state_value_diversity/",
    )


def compute_policy_diversity(data, policy_module, compute_histograms=False):
    """Computes the variance across action distributions."""
    dist = policy_module.build_dist_from_params(data)
    match dist.__class__.__name__:
        case "OneHotCategorical":
            policies = dist.probs  # (B, A)
        case "Normal":
            policies = torch.stack([dist.loc, dist.scale], dim=1)  # (B, 2=A)
        case "TanhNormal":
            policies = torch.stack([dist.loc, dist.scale], dim=1)  # (B, 2=A)
        case other:
            raise NotImplementedError(f"Policy diversity not implemented for {dist.__class__.__name__}.")

    mean_policy = policies.mean(dim=0)  # (A,) Mean across states for each action/policy_param.
    policy_vars = policies.var(dim=0)  # (A,) Variance across states for each action/policy_param.
    policy_var = policy_vars.mean()  # Average variance of an action/parameter across states; policy variance.
    logs = {"policy_variance": policy_var}
    if compute_histograms:
        logs.update(
            {
                "means_hist_wandb": wandb.Histogram(
                    np_histogram=(mean_policy.cpu().numpy(), np.arange(len(mean_policy) + 1))
                ),
                "vars_hist_wandb": wandb.Histogram(
                    np_histogram=(policy_vars.cpu().numpy(), np.arange(len(policy_vars) + 1))
                ),
            }
        )
    return dict_with_prefix(logs, "action_diversity/")


def compute_applied_grad_norm(params, prev_params):
    """Computes the norm of the difference between the current and previous policy parameters."""
    # A hack to use clip grad norm to compute the norm of the gradient.
    # Store the difference between the current and previous policy parameters in the grad field.
    for p, p_prev in zip(params, prev_params):
        p_prev.grad = p.data.detach() - p_prev
    return torch.nn.utils.clip_grad_norm_(prev_params, 1e10)


def compute_weights_norm(params, copy_params):
    """Computes the norm of the weights, using same trick as above."""
    for p, p_copy in zip(params, copy_params):
        p_copy.grad = p.data.detach()
    return torch.nn.utils.clip_grad_norm_(copy_params, 1e10)


def compute_kl_divergence_from_dists(old_dist, new_dist):
    return {
        "kl_dist": torch.distributions.kl.kl_divergence(old_dist, new_dist).mean(),
    }


def compute_kl_divergence_from_samples(old_log_probs, new_log_probs):
    """Computes the empirical KL divergence between two sample probabilities.

    That is, KL(p, q) where p is the old policy and q is the new policy.
    old_log_probs and new_log_probs should come from actions sampled from the old policy.
    """
    log_ratio = new_log_probs - old_log_probs  # log(q/p)
    ratio = log_ratio.exp()
    kl = (-log_ratio).mean()  # E_p[-log(q/p)] = k1 in http://joschu.net/blog/kl-approx.html.
    kl_schulman = ((ratio - 1) - log_ratio).mean()  # k3 in http://joschu.net/blog/kl-approx.html.
    return {"kl_sample_naive": kl, "kl_sample_schulman": kl_schulman}


def compute_models_diff(data_outputs, data_targets, policy_module, reverse_kl=False, ratio_epsilon=0.0):
    dist_outputs = policy_module.build_dist_from_params(data_outputs)
    dist_targets = policy_module.build_dist_from_params(data_targets)
    # Compute KL.
    p, q = dist_targets, dist_outputs
    if reverse_kl:
        p, q = dist_outputs, dist_targets
    out = {"kl_dist_batch": torch.distributions.kl.kl_divergence(p, q).mean()}
    # Compute prob ratios.
    actions = data_targets["action"]
    output_logprobs = dist_outputs.log_prob(actions)
    target_logprobs = dist_targets.log_prob(actions)
    ratio = (output_logprobs - target_logprobs).exp()
    out.update(
        {
            "max_prob_ratio": ratio.max(),
            "min_prob_ratio": ratio.min(),
            "avg_prob_ratio": ratio.mean(),
            "std_prob_ratio": ratio.std(),
        }
    )
    # The above but only for the clipped elements.
    if ratio_epsilon > 0:
        above_epsilon = ratio > 1 + ratio_epsilon
        below_epsilon = ratio < 1 - ratio_epsilon
        for prefix in ["above", "below"]:
            mask = above_epsilon if prefix == "above" else below_epsilon
            if mask.any():
                ratio_clipped = ratio[mask]
                out.update(
                    {
                        f"max_prob_ratio_{prefix}_epsilon": ratio_clipped.max(),
                        f"min_prob_ratio_{prefix}_epsilon": ratio_clipped.min(),
                        f"avg_prob_ratio_{prefix}_epsilon": ratio_clipped.mean(),
                        f"std_prob_ratio_{prefix}_epsilon": ratio_clipped.std(),
                    }
                )

    for prefix in ["features", "features_preactivation", "all_preactivations"]:
        for model in ["policy", "value"]:
            key = f"{prefix}_{model}"
            out[f"{key}_l2_batch"] = F.mse_loss(data_outputs[key], data_targets[key])
            out[f"{key}_cosine"] = F.cosine_similarity(data_outputs[key], data_targets[key], dim=-1).mean()
    return out
