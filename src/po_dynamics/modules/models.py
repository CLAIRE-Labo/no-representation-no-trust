import torch
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.envs import ExplorationType
from torchrl.modules import (
    MLP,
    ConvNet,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.modules.models.utils import SquashDims

from po_dynamics.modules.env import OBS_KEY


def forward_record_preactivations(module, x, activation_class):
    """Forward pass that records all preactivations recursively (if sequential inside sequential)
    Assumes all activation functions are the same.
    Limited to use cases in this codebase with Sequential networks
    """
    all_preactivations = []
    if isinstance(module, ConvNet):
        *batch, C, L, W = x.shape
        if len(batch) > 1:
            x = x.flatten(0, len(batch) - 1)
        for m in module:
            x, preactivations = forward_record_preactivations(m, x, activation_class)
            if len(batch) > 1:
                all_preactivations.extend([p.unflatten(0, batch) for p in preactivations])
            else:
                all_preactivations.extend(preactivations)
        if len(batch) > 1:
            x = x.unflatten(0, batch)

    elif isinstance(module, nn.Sequential):
        for m in module:
            x, preactivations = forward_record_preactivations(m, x, activation_class)
            all_preactivations.extend(preactivations)
    else:
        x = module(x)
        if not (isinstance(module, activation_class) or isinstance(module, SquashDims)):
            all_preactivations.append(x)
    return x, all_preactivations


class SequentialRecordPreactivationsRecursive(nn.Sequential):
    def __init__(self, *args, activation_class=None):
        self.activation_class = getattr(nn, activation_class)
        super().__init__(*args)

    def forward(self, x):
        """Forward pass that records all preactivations recursively (if sequential inside sequential)
        Assumes all activation functions are the same.
        Assumes the last layer had a single output dimension (e.g., a linear layer)
        """
        out, preactivations = forward_record_preactivations(self, x, self.activation_class)
        batch_dims = len(preactivations[-1].shape) - 1
        for i, p in enumerate(preactivations):
            preactivations[i] = torch.flatten(p, start_dim=batch_dims)
        return out, torch.cat(preactivations, dim=-1)


def build_features_module(
    model_config, device, suffix="", pre_feature_layers=None
) -> tuple[TensorDictModule, list[nn.Module]]:
    """Returns a lazy sequential module representing the feature layers"""
    if pre_feature_layers is None:
        pre_feature_layers = []
        if model_config.conv_layers.num_layers > 0:
            pre_feature_layers.append(
                ConvNet(
                    kernel_sizes=list(model_config.conv_layers.kernel_sizes),
                    strides=list(model_config.conv_layers.strides),
                    num_cells=list(model_config.conv_layers.num_filters),
                    activation_class=getattr(nn, model_config.activation),
                    device=device,
                )
            )
        pre_feature_layers.append(
            MLP(
                num_cells=model_config.linear_layers.layer_size,
                depth=model_config.linear_layers.num_layers - 1,
                out_features=model_config.linear_layers.layer_size,
                activation_class=getattr(nn, model_config.activation),
                activate_last_layer=False,
                device=device,
            )
        )
    feature_preactivation = TensorDictModule(
        SequentialRecordPreactivationsRecursive(*pre_feature_layers, activation_class=model_config.activation),
        in_keys=[OBS_KEY],
        out_keys=["features_preactivation" + suffix, "all_preactivations" + suffix],
    )
    features = TensorDictModule(
        getattr(nn, model_config.activation)(),
        in_keys=["features_preactivation" + suffix],
        out_keys=["features" + suffix],
    )
    return TensorDictSequential(feature_preactivation, features), pre_feature_layers


def build_value_module(model_config, device, pre_feature_layers=None) -> tuple[TensorDictModule, TensorDictModule]:
    """Builds the value module (critic network) with lazy tensors that need to be initialized.

    The features_module is an intermediate tensordict module that can be used
    to obtain the features computed by the feature layers.
    """
    features_module, _ = build_features_module(
        model_config, device, suffix="_value", pre_feature_layers=pre_feature_layers
    )
    output_module = TensorDictModule(
        nn.LazyLinear(1, device=device),
        in_keys=["features_value"],
        out_keys=["state_value"],
    )
    value_module = TensorDictSequential(
        features_module,
        output_module,
    )

    return value_module, features_module


def build_policy_module(
    model_config: DictConfig, action_space, action_spec, device: str, pre_feature_layers=None
) -> tuple[ProbabilisticActor, TensorDictModule]:
    """Builds the policy module (actor network + distribution) with lazy tensors that need to be initialized.

    The feature_module is an intermediate tensordict module that can be used
    to obtain the features computed by the feature layers.
    The policy_module is a ProbabilisticActor in TorchRL terms, it's the network + the distribution.
    """
    # Feature layers.
    features_module, _ = build_features_module(
        model_config, device, suffix="_policy", pre_feature_layers=pre_feature_layers
    )

    # Action layers.
    if action_space == "discrete":
        num_outputs = action_spec.space.n
        out_keys = ["logits"]
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    elif action_space == "continuous":
        # One dimension for the mean and one for the standard deviation per action dimension.
        # Indexing action_spec.shape at -1 as the first dimensions will be env batch dimensions.
        num_outputs = 2 * action_spec.shape[-1]
        out_keys = ["loc", "scale"]
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": action_spec.space.low[-1],
            "max": action_spec.space.high[-1],
        }
    else:
        raise NotImplementedError(f"Policy for action_space={action_space} not implemented.")

    action_layers = [nn.LazyLinear(num_outputs, device=device)]
    if action_space == "continuous":
        action_layers.append(NormalParamExtractor())

    output_module = TensorDictModule(
        nn.Sequential(*action_layers),
        in_keys=["features_policy"],
        out_keys=out_keys,
    )

    # The probabilistic actor module.
    # policy network + distribution
    policy_module = ProbabilisticActor(
        module=TensorDictSequential(
            features_module,
            output_module,
        ),  # The policy network.
        spec=action_spec,
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        in_keys=out_keys,  # in_keys of the distribution.
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    return policy_module, features_module


def model_snapshots(
    policy_module,
    value_module,
    optimizer_policy,
    optimizer_value,
    scheduler_policy,
    scheduler_value,
    counters,
    config,
    action_spec,
):
    """Returns a dictionary to save models."""
    return {
        "policy_module": policy_module.state_dict(),
        "value_module": value_module.state_dict(),
        "optimizer_policy": optimizer_policy.state_dict(),
        "optimizer_value": optimizer_value.state_dict(),
        "scheduler_policy": scheduler_policy.state_dict(),
        "scheduler_value": scheduler_value.state_dict(),
        "counters/env_steps": counters["env_steps"],
        "config": config,
        "action_spec": action_spec,
        "counters": counters,
    }


def load_models(
    filename, device, policy_modules=None, value_modules=None
) -> tuple[TensorDictModule, TensorDictModule, TensorDictModule, TensorDictModule, dict, DictConfig]:
    """Loads the policy and value modules from disk.
    Shared models are loaded independently and do not share the same trunk anymore.
    They have to be linked manually if needed.
    """
    checkpoint = torch.load(filename, device)
    load_config = checkpoint["config"]

    # Build or skip.
    if policy_modules is None:
        policy_module, policy_feats_module = build_policy_module(
            load_config.models.actor, load_config.env.action_space, checkpoint["action_spec"], device
        )
    else:
        policy_module, policy_feats_module = policy_modules

    # Build or skip.
    if value_modules is None:
        value_module, value_feats_module = build_value_module(load_config.models.critic, device)
    else:
        value_module, value_feats_module = value_modules

    # Feature modules are loaded with the modules.
    policy_module.load_state_dict(checkpoint["policy_module"])
    value_module.load_state_dict(checkpoint["value_module"])

    return (
        policy_module,
        policy_feats_module,
        value_module,
        value_feats_module,
        checkpoint["counters"],
        load_config,
    )
