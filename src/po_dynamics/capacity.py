"""
Fit policy to random policy.
Fit critic to random critic.

Analyse rank on initial batch.

Action diversity histogram.

"""
import copy
import logging
import time
from collections import defaultdict
from pathlib import Path

import hydra
import tensordict
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, omegaconf
from torch.nn import functional as F
from torchrl.collectors import SyncDataCollector
from torchrl.data import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType
from tqdm import tqdm

from po_dynamics import modules, utils
from po_dynamics.modules import metrics
from po_dynamics.modules.env import OBS_KEY
from po_dynamics.modules.metrics import compute_models_diff
from po_dynamics.utils.logging import (
    DictWithPrefix,
    LogLevel,
    LogTracker,
    dict_with_prefix,
    filter_out_underscore,
    filter_out_wandb,
)

logger = logging.getLogger(__name__)

utils.config.register_resolvers()


@hydra.main(version_base=None, config_path="configs", config_name="capacity")
def main(capacity_config: DictConfig) -> None:
    Path("config/").mkdir()
    OmegaConf.save(capacity_config, "config/capacity_config_unresolved.yaml")
    OmegaConf.resolve(capacity_config)
    OmegaConf.save(capacity_config, "config/capacity_config_resolved.yaml")

    with omegaconf.open_dict(capacity_config):
        config = load_and_merge_solve_config(capacity_config)
    OmegaConf.save(config, "config/config_resolved.yaml")

    config_anonymized = OmegaConf.to_container(config)
    config_anonymized = utils.config.anonymize_config(config_anonymized)
    wandb.init(
        config=config_anonymized,
        project=config.wandb.project,
        tags=config.wandb.tags,
        anonymous=config.wandb.anonymous,
        mode=config.wandb.mode,
        dir=Path(config.wandb.dir).absolute(),
    )
    logger.info(f"Running in directory: {Path.cwd()}")
    logger.info(f"Running with config:\n{OmegaConf.to_yaml(config)}")

    utils.seeding.seed_everything(config)

    if config.env.normalize_obs:
        normalization_stats = torch.load(f"{config.solve_dir}/env_normalization_stats.tar")
    else:
        normalization_stats = {}

    env = modules.env.make_env(config, is_eval=True, normalization_stats=normalization_stats, seed=config.seed)

    # Build models.
    policy_module, policy_feats_module = modules.models.build_policy_module(
        config.models.actor, config.env.action_space, env.action_spec, config.device.training
    )
    value_module, value_feats_module = modules.models.build_value_module(config.models.critic, config.device.training)
    dummy_t = env.reset().to(config.device.training)
    policy_module(dummy_t)
    value_module(dummy_t)

    # Build rollout collector.
    # Collects the dataset for the fit.
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=config.capacity.agent_steps_per_batch,
        total_frames=config.capacity.total_agent_steps,
        env_device=config.device.rollout,
        policy_device=config.device.rollout,
        storing_device=config.device.collector_storage,  # This can be very large.
        exploration_type=ExplorationType.RANDOM,
        reset_at_each_iter=True,
    )
    logger.debug(f"Evaluation collector: {collector}")

    # Sort the checkpoints.
    solve_checkpoints = sorted(
        (Path(config.solve_dir) / "logs/models").iterdir(),
        key=lambda x: int(x.stem),
    )

    target_policy_module = None
    target_value_module = None

    pbar = tqdm(total=len(solve_checkpoints))
    counters = defaultdict(int)
    timers = {}

    checkpoint_logger = LogTracker(
        name="checkpoint",
        counter_key="capacity-counters/env_steps",
        trigger_level=LogLevel.BATCH,
        logging_level=config.capacity.log_level,
        final_idx=len(solve_checkpoints),
        add_first=True,
        add_last=True,
    )
    model_logger = LogTracker(
        name="model",
        counter_key="counters/env_steps",
        trigger_level=LogLevel.BATCH,
        logging_level=config.capacity.log_level,
        final_idx=len(solve_checkpoints),
        add_first=True,
        add_last=True,
    )

    data = None
    data_targets = None
    data_outputs = None
    all_losses_policy = []
    all_losses_value = []
    for checkpoint_num, solve_checkpoint in enumerate(solve_checkpoints):
        checkpoint_logger.register_progress(checkpoint_num)
        model_logger.register_progress(checkpoint_num)
        checkpoint_logs = DictWithPrefix("capacity-checkpoint/", {})
        # Load the models:
        (
            policy_module,
            policy_feats_module,
            value_module,
            value_feats_module,
            checkpoint_counters,
            checkpoint_config,
        ) = modules.models.load_models(
            solve_checkpoint,
            config.device.training,
            (policy_module, policy_feats_module),
            (value_module, value_feats_module),
        )

        # Collect data.
        if config.capacity.fit_checkpoint_data or checkpoint_num == 0:  # For the init data too, the first time.
            collector.update_policy_weights_()
            timers["collector"] = time.time()
            data = collector.next().reshape(-1)
            del data["next"]
            timers["collector"] = time.time() - timers["collector"]
            timers["fps_collector"] = (data.numel() * config.env.frame_skip) / timers["collector"]
            pbar.set_description(f"{int(timers['fps_collector'])} FPS")

        # Record the initial models.
        if checkpoint_num == 0:
            target_policy_module = copy.deepcopy(policy_module)
            target_value_module = copy.deepcopy(value_module)

            if config.capacity.fit_random:
                target_policy_module.reset_parameters_recursive()
                target_value_module.reset_parameters_recursive()

        if not config.capacity.fit_checkpoint_data:
            # Data comes from init data.
            # Compute the targets on the init data only once.
            if checkpoint_num == 0:
                data_targets = data
                forward_policy_and_value(
                    config,
                    data_targets,
                    target_policy_module,
                    target_value_module,
                    skip_policy=not config.capacity.fit_random,  # Can skip as it's the first batch unless new target..
                )
            # Data comes from the init data.
            # Compute the outputs on the init data from scratch.
            data_outputs = data.select(OBS_KEY, "action")
            forward_policy_and_value(config, data_outputs, policy_module, value_module)
        else:
            # Data comes from the checkpoint data.
            # Compute the targets on the checkpoint data from scratch.
            data_targets = data.select(OBS_KEY, "action")
            forward_policy_and_value(config, data_targets, target_policy_module, target_value_module)
            # Compute the outputs on the checkpoint, but skip the policy as already there.
            data_outputs = data
            forward_policy_and_value(config, data_outputs, policy_module, value_module, skip_policy=True)

        # Diff at the start:
        diff = compute_models_diff(data_outputs, data_targets, policy_module)
        checkpoint_logs.update_with_prefix(
            dict_with_prefix(
                {
                    **diff,
                    **metrics.compute_effective_ranks(
                        [data_outputs], ["outputs"], [["features_policy", "features_value"]]
                    ),
                    **metrics.compute_dead_neurons(
                        [data_outputs], ["outputs"], "features_policy", config.models.actor.activation
                    ),
                    **metrics.compute_dead_neurons(
                        [data_outputs], ["outputs"], "features_value", config.models.critic.activation
                    ),
                    **metrics.compute_feature_norms(
                        [data_outputs],
                        ["outputs"],
                        [
                            [
                                "features_policy",
                                "features_preactivation_policy",
                                "features_value",
                                "features_preactivation_value",
                            ]
                        ],
                    ),
                    **metrics.compute_feature_norms(
                        [data_outputs],
                        ["outputs"],
                        [
                            [
                                "all_preactivations_policy",
                                "all_preactivations_value",
                            ]
                        ],
                    ),
                    **metrics.compute_policy_diversity(data_outputs, policy_module),
                    **metrics.compute_value_diversity(data_outputs),
                },
                "start/",
            )
        )

        # Fit.
        optimizer_policy = getattr(torch.optim, config.optim.policy.algo)(
            policy_module.parameters(), **config.optim.policy.kwargs
        )
        scheduler_policy = torch.optim.lr_scheduler.LambdaLR(
            optimizer_policy, lambda step: 1 - step / config.capacity.num_optim_steps_per_checkpoint
        )
        optimizer_value = getattr(torch.optim, config.optim.value.algo)(
            value_module.parameters(), **config.optim.value.kwargs
        )
        scheduler_value = torch.optim.lr_scheduler.LambdaLR(
            optimizer_value, lambda step: 1 - step / config.capacity.num_optim_steps_per_checkpoint
        )

        # Epoch loop.
        epoch_logger = LogTracker(
            name="epoch",
            counter_key="capacity-counters/epoch_idx_capacity",
            trigger_level=LogLevel.EPOCH,
            logging_level=config.capacity.log_level,
            final_idx=config.capacity.num_epochs,
            depends_on=checkpoint_logger,
            add_first=True,
            add_last=True,
        )
        first_epoch_minibatch_logs = {}  # Used in the batch logger to log the first minibatch of the first epoch.
        last_epoch_minibatch_logs = {}  # Used in the batch logger to log the last minibatch of the last epoch.
        losses_policy_batch = torch.zeros(config.capacity.num_optim_steps_per_checkpoint)
        # Track all the epoch & minibatch losses and log them at the checkpoint level.
        losses_value_batch = torch.zeros(config.capacity.num_optim_steps_per_checkpoint)
        # Track all the epoch & minibatch losses and log them at the checkpoint level.
        for epoch_idx in range(config.capacity.num_epochs):
            epoch_logger.register_progress(epoch_idx)
            sampler = SamplerWithoutReplacement(drop_last=True)
            # Minibatch loop.
            minibatch_logger = LogTracker(
                name="minibatch",
                counter_key="capacity-counters/minibatch_idx_capacity",
                trigger_level=LogLevel.MINIBATCH,
                logging_level=config.capacity.log_level,
                final_idx=config.capacity.num_minibatches_per_epoch,
                depends_on=epoch_logger,
                add_first=(config.logging.log_level <= LogLevel.EPOCH)
                or (config.logging.log_level > LogLevel.EPOCH and epoch_logger.is_first),
                # Always add the first minibatch of when logging at epoch level.
                # Only add the first_epoch_first_minibatch when logging at higher level.
                add_last=epoch_logger.is_last,
                # Only add the last minibatch of the last epoch,
                # because end_of_minibatch loss is recorded at the start of the next epoch, unless it's the last epoch.
            )
            first_minibatch_logs = {}  # Used in the epoch logger to log its first minibatch.
            last_minibatch_logs = {}  # Used in the epoch logger to log its last minibatch.
            local_minibatch_idx = 0
            while not sampler.ran_out:
                local_update_index = epoch_idx * config.capacity.num_minibatches_per_epoch + local_minibatch_idx
                minibatch_logger.register_progress(local_minibatch_idx)
                minibatch_logs = DictWithPrefix("capacity-minibatch/", {})
                minibatch_indices = sampler.sample(data_targets, config.optim.minibatch_size)[0]
                minibatch = data_targets[minibatch_indices].to(config.device.training)

                out = minibatch.select(OBS_KEY, "action")
                policy_module.get_dist(out)
                value_module(out)

                # losses.
                dist_targets = policy_module.build_dist_from_params(minibatch)
                dist_outputs = policy_module.build_dist_from_params(out)
                p, q = dist_targets, dist_outputs
                loss_policy = torch.distributions.kl.kl_divergence(p, q).mean()
                loss_value = F.mse_loss(out["state_value"], minibatch["state_value"], reduction="mean")
                losses_policy_batch[local_update_index] = loss_policy.detach()
                losses_value_batch[local_update_index] = loss_value.detach()
                # Backprop.
                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                (loss_value + loss_policy).backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_module.parameters(), config.optim.policy.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    value_module.parameters(), config.optim.value.max_grad_norm
                )
                optimizer_policy.step()
                optimizer_value.step()

                if minibatch_logger.log_this_round:
                    minibatch_logs.update_with_prefix(
                        {
                            "loss/policy": loss_policy.detach(),
                            "loss/value": loss_value.detach(),
                            "grad_norm/actor": actor_grad_norm,
                            "grad_norm/critic": critic_grad_norm,
                            "grad_norm/actor_clipped": min(config.optim.policy.max_grad_norm, actor_grad_norm),
                            "grad_norm/critic_clipped": min(config.optim.value.max_grad_norm, critic_grad_norm),
                            "learning_rate/policy": optimizer_policy.param_groups[0]["lr"],
                            "learning_rate/value": optimizer_value.param_groups[0]["lr"],
                        }
                    )
                    to_log = {
                        "capacity-counters/minibatch_idx_capacity": counters["minibatch_idx_capacity"],
                        **minibatch_logs,
                    }
                    minibatch_logger.log_to_file(filter_out_wandb(to_log))
                    wandb.log(filter_out_underscore(to_log))

                if minibatch_logger.is_first:
                    first_minibatch_logs = dict_with_prefix(minibatch_logs, "capacity-epoch/first_")
                if minibatch_logger.is_last:
                    last_minibatch_logs = dict_with_prefix(minibatch_logs, "capacity-epoch/last_")

                local_minibatch_idx += 1
                counters["minibatch_idx_capacity"] += 1
                # End of minibatch loop.
                if config.capacity.anneal_policy_lr:
                    scheduler_policy.step()
                if config.capacity.anneal_value_lr:
                    scheduler_value.step()

            # Back to epoch loop.
            if epoch_logger.log_this_round:
                to_log = {
                    "capacity-counters/epoch_idx_capacity": counters["epoch_idx_capacity"],
                    "capacity-counters/last_minibatch_idx_capacity": counters["minibatch_idx_capacity"] - 1,
                    **first_minibatch_logs,
                    # the last minibatch of intermediate epochs is expressed in the first minibatch of the next epoch.
                    # the last minibatch of last epoch is logged in checkpoint/last_epoch/last_minibatch.
                }
                epoch_logger.log_to_file(filter_out_wandb(to_log))
                wandb.log(filter_out_underscore(to_log))
            if epoch_logger.is_first:
                first_epoch_minibatch_logs = dict_with_prefix(first_minibatch_logs, "capacity-checkpoint/first_")
            if epoch_logger.is_last:
                last_epoch_minibatch_logs = dict_with_prefix(last_minibatch_logs, "capacity-checkpoint/last_")

            counters["epoch_idx_capacity"] += 1
            # End of epoch loop.

        all_losses_policy.append(losses_policy_batch.to("cpu"))
        all_losses_value.append(losses_value_batch.to("cpu"))

        # Diff at end.
        # Recompute the outputs.
        forward_policy_and_value(config, data_outputs, policy_module, value_module)
        diff = compute_models_diff(data_outputs, data_targets, policy_module)

        checkpoint_logs.update_with_prefix(
            dict_with_prefix(
                {
                    **diff,
                    **metrics.compute_effective_ranks(
                        [data_outputs], ["outputs"], [["features_policy", "features_value"]]
                    ),
                    **metrics.compute_dead_neurons(
                        [data_outputs], ["outputs"], "features_policy", config.models.actor.activation
                    ),
                    **metrics.compute_dead_neurons(
                        [data_outputs], ["outputs"], "features_value", config.models.critic.activation
                    ),
                    **metrics.compute_policy_diversity(data_outputs, policy_module),
                    **metrics.compute_value_diversity(data_outputs),
                    "losses_policy_": losses_policy_batch,
                    "losses_value_": losses_value_batch,
                },
                "end/",
            )
        )
        checkpoint_logs.update_with_prefix(dict_with_prefix(get_all_diffs(checkpoint_logs), "diff/"))
        to_log = {
            "capacity-counters/collector_steps": checkpoint_counters[
                "collector_steps"
            ],  # How many collector steps were done.
            "capacity-counters/agent_steps": checkpoint_counters["agent_steps"],  # How many agent steps were performed.
            "capacity-counters/env_steps": checkpoint_counters["env_steps"],  # How many env steps were performed.
            "capacity-counters/batch_steps": checkpoint_counters["batch_idx"],  # The global batch index.
            "capacity-counters/grad_steps": checkpoint_counters["minibatch_idx"],  # num of updates done in solve.
            "capacity-counters/checkpoint_idx": checkpoint_num,
            "capacity-timers/fps_collector": timers["fps_collector"],
            **checkpoint_logs,
            **first_epoch_minibatch_logs,
            **last_epoch_minibatch_logs,
        }
        checkpoint_logger.log_to_file(filter_out_wandb(to_log))
        wandb.log(filter_out_underscore(to_log))

        # Save models.
        model_logger.log_to_file(
            modules.models.model_snapshots(
                policy_module,
                value_module,
                optimizer_policy,
                optimizer_value,
                scheduler_policy,
                scheduler_value,
                checkpoint_counters,
                config,
                env.action_spec,
            )
        )

        # End of checkpoint loop.
        pbar.update(1)

    all_losses_policy = torch.cat(all_losses_policy)
    all_losses_value = torch.cat(all_losses_value)
    wandb.log(
        {
            "losses_policy_wandb": wandb.plot.line_series(
                xs=range(len(all_losses_policy)),
                ys=[all_losses_policy],
                keys=["loss_policy"],
                title="Losses Policy",
                xname="minibatch",
            ),
            "losses_value_wandb": wandb.plot.line_series(
                xs=range(len(all_losses_value)),
                ys=[all_losses_value],
                keys=["loss_value"],
                title="Losses Value",
                xname="minibatch",
            ),
        }
    )
    # End of checkpoint loop.
    collector.shutdown()


@torch.no_grad()
def forward_policy_and_value(config, data, policy_module, value_module, skip_policy=False):
    if config.device.training == config.device.collector_storage:
        if not skip_policy:
            policy_module.get_dist(data)
        value_module(data)
    else:
        for i in range(0, data.shape[0], config.optim.minibatch_size):
            sub = data[i : i + config.optim.minibatch_size].to(config.device.training)
            if not skip_policy:
                policy_module.get_dist(sub)
            value_module(sub)
            data[i : i + config.optim.minibatch_size] = sub.to(config.device.collector_storage)


def get_all_diffs(checkpoint_logs):
    res = {}
    for key in checkpoint_logs.keys():
        if key.startswith("capacity-checkpoint/start/"):
            metric_name = key[len("capacity-checkpoint/start/") :]
            if f"capacity-checkpoint/end/{metric_name}" in checkpoint_logs:
                res[metric_name] = checkpoint_logs[f"capacity-checkpoint/end/{metric_name}"] - checkpoint_logs[key]
    return res


def load_and_merge_solve_config(capacity_config):
    # Get solve config.
    solve_dir = Path(hydra.utils.get_original_cwd()) / capacity_config.solve_dir_from_cwd  # for anonymity in sweeps.
    solve_config = OmegaConf.load(solve_dir / "config/config_resolved.yaml")
    n_checkpoints = len(list((solve_dir / "logs/models").iterdir()))
    # Merge configs
    # Override the solve config with the new config.
    config = OmegaConf.merge(solve_config, capacity_config)
    config.env.eval.num_envs = config.capacity.num_envs
    config.eval.record_video = False
    # And add new keys.
    config.working_dir = f"{Path.cwd()}"
    config.solve_dir = f"{solve_dir}"
    # Steps to collector for each fit.
    config.capacity.agent_steps_per_batch = config.env.eval.agent_steps_per_eval_env * config.capacity.num_envs
    # So is the dataset size to fit the target.
    config.capacity.agent_steps_fit = config.capacity.agent_steps_per_batch
    if not config.capacity.fit_checkpoint_data:
        # Only the initial batch will be collected.
        config.capacity.total_agent_steps = config.capacity.agent_steps_per_batch
    else:
        config.capacity.total_agent_steps = config.capacity.agent_steps_per_batch * n_checkpoints
    config.capacity.total_env_steps = config.capacity.total_agent_steps * config.env.frame_skip
    config.capacity.num_minibatches_per_epoch = config.capacity.agent_steps_per_batch // config.optim.minibatch_size
    config.capacity.num_optim_steps_per_checkpoint = (
        config.capacity.num_epochs * config.capacity.num_minibatches_per_epoch
    )
    return config


if __name__ == "__main__":
    main()
