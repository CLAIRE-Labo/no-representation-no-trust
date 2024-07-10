import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf, omegaconf
from tensordict import tensordict
from torchrl import objectives
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType
from torchrl.objectives.value import GAE
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


@hydra.main(version_base=None, config_path="configs", config_name="solve")
def main(config: DictConfig) -> None:
    Path("config/").mkdir()
    OmegaConf.save(config, "config/config_before_fixes_unresolved.yaml")
    with omegaconf.open_dict(config):
        fix_and_augment_config(config)
    OmegaConf.resolve(config)
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

    ####################################################################################################################
    # Setup of RL modules.
    utils.seeding.seed_everything(config)

    # Create env with transforms.
    normalization_stats = {}
    train_env = modules.env.make_env(config, is_eval=False, normalization_stats=normalization_stats, seed=config.seed)
    if config.env.normalize_obs:
        torch.save(normalization_stats, Path("env_normalization_stats.tar"))
    logger.debug(f"Training env: {train_env}")
    eval_env = None
    if config.eval.log_level <= LogLevel.MAX_LEVEL:
        eval_env = modules.env.make_env(
            config, is_eval=True, normalization_stats=normalization_stats, seed=(config.seed**2) % (2**32 - 1)
        )
        logger.debug(f"eval env: {eval_env}")

    # Build models.
    if config.models.share_features:
        _, pre_feature_layers = modules.models.build_features_module(config.models, config.device.training)
    else:
        pre_feature_layers = None
    policy_module, _ = modules.models.build_policy_module(
        config.models.actor, config.env.action_space, train_env.action_spec, config.device.training, pre_feature_layers
    )
    value_module, _ = modules.models.build_value_module(
        config.models.critic, config.device.training, pre_feature_layers
    )
    # And initialize lazy tensors of model parameters.
    dummy_t = train_env.reset().to(config.device.training)
    with torch.no_grad():
        policy_module(dummy_t)
        value_module(dummy_t)
    logger.debug(f"Policy module: {policy_module}")
    logger.debug(f"Value module: {value_module}")

    # Build rollout collectors.
    training_collector = SyncDataCollector(
        train_env,
        policy_module,
        frames_per_batch=config.collector.agent_steps_per_batch,  # Nb steps before a policy update.
        total_frames=config.collector.total_agent_steps,  # Total agent steps budget.
        env_device=config.device.rollout,
        policy_device=config.device.rollout,  # Where the model will be copied for sampling policy_device
        storing_device=config.device.collector_storage,  # Where the collected data will be stored.
        exploration_type=ExplorationType.RANDOM,
        reset_at_each_iter=config.collector.is_episodic,
    )
    logger.debug(f"Training collector: {training_collector}")
    eval_collector = None
    if config.eval.log_level <= LogLevel.MAX_LEVEL:
        # non-urgent-fix: a collector has a pre-allocated output tensordict that is alive while the collector is in scope
        # Although it's only needed during evaluation.
        #  To save memory one can delete and re-create the collector at each evaluation.
        eval_collector = SyncDataCollector(
            eval_env,
            policy_module,
            frames_per_batch=config.eval.agent_steps_per_eval,
            total_frames=config.eval.total_agent_steps,
            env_device=config.device.rollout,
            policy_device=config.device.rollout,
            storing_device="cpu",  # To store the pixels for eval video etc.
            reset_at_each_iter=True
            # Exploration type is set explicitly during eval.
        )
        logger.debug(f"eval collector: {eval_collector}")

    # Build target estimators.
    # One estimator is used in the loss of the value function.
    target_estimator_critic = GAE(**config.target_estimator_kwargs.critic, value_network=None)
    target_estimator_critic.set_keys(value_target="value_target_critic", advantage="advantage_critic")
    logger.debug(f"Critic target module: {target_estimator_critic}")
    # And one estimator is used in the loss of the policy.
    target_estimator_actor = GAE(**config.target_estimator_kwargs.actor, value_network=None)
    target_estimator_actor.set_keys(value_target="value_target_actor", advantage="advantage_actor")
    logger.debug(f"Actor target module: {target_estimator_actor}")

    # Losses
    # Policy Loss
    # Project-defined losses.
    if hasattr(modules.losses, config.loss.policy.module):
        loss_class = getattr(modules.losses, config.loss.policy.module)
    # For torchrl losses.
    else:
        loss_class = getattr(objectives, config.loss.policy.module)
    loss_module_policy = loss_class(actor=policy_module, **config.loss.policy.kwargs)
    loss_module_policy.set_keys(advantage=config.loss.policy.advantage)
    logger.debug(f"Loss module: {loss_module_policy}")
    # Value Loss
    loss_module_value = modules.losses.ValueLoss(critic=value_module, **config.loss.value.kwargs)
    loss_module_value.set_keys(value_target="value_target_critic")

    # Optimizer and scheduler.
    optimizer_policy = getattr(torch.optim, config.optim.policy.algo)(
        policy_module.parameters(), **config.optim.policy.kwargs
    )
    scheduler_policy = torch.optim.lr_scheduler.LambdaLR(
        optimizer_policy, lambda batch: 1 - batch / config.collector.num_batches
    )
    optimizer_value = getattr(torch.optim, config.optim.value.algo)(
        value_module.parameters(), **config.optim.value.kwargs
    )
    scheduler_value = torch.optim.lr_scheduler.LambdaLR(
        optimizer_value, lambda batch: 1 - batch / config.collector.num_batches
    )
    logger.debug(f"Optimizer policy: {optimizer_policy}\nOptimizer value: {optimizer_value}")

    ####################################################################################################################
    # Training

    # Logs. Progress bar, etc.
    pbar = tqdm(total=config.collector.total_env_steps)
    pbar_desc = dict(train="", eval="")
    progress_acc = 0.0
    counters = defaultdict(int)
    timers = {}  # Note: timers on GPU async operations only make sense if followed by a CPU-GPU syncing operation.
    batch_perf_stats = {}
    batch_perf_stats_no_reset = {}

    eval_logger = LogTracker(
        name="eval",
        counter_key="counters/env_steps",
        trigger_level=LogLevel.BATCH,  # BATCH is every policy iteration (collect data + update).
        logging_level=config.eval.log_level,
        final_idx=config.collector.num_batches,
        add_first=False,  # Will be done before training starts.
        add_last=True,
    )
    save_model_logger = LogTracker(
        name="models",
        counter_key="counters/env_steps",
        trigger_level=LogLevel.BATCH,
        logging_level=config.logging.save_model_level,
        final_idx=config.collector.num_batches,
        add_first=False,  # Will be done before training starts.
        add_last=True,
    )
    batch_logger = LogTracker(
        name="batch",
        counter_key="counters/env_steps",
        trigger_level=LogLevel.BATCH,
        logging_level=config.logging.log_level,
        final_idx=config.collector.num_batches,
        add_first=True,
        add_last=True,
    )

    # Saving initial models.
    if config.logging.save_model_level <= LogLevel.MAX_LEVEL:
        to_log = modules.models.model_snapshots(
            policy_module,
            value_module,
            optimizer_policy,
            optimizer_value,
            scheduler_policy,
            scheduler_value,
            counters,
            config,
            train_env.action_spec,
        )
        save_model_logger.log_to_file(filter_out_wandb(to_log))

    # Evaluation before training.
    if config.eval.log_level <= LogLevel.MAX_LEVEL:
        evaluate(config, counters, eval_collector, eval_logger, pbar, pbar_desc, progress_acc, timers, value_module)

    logger.debug("Starting training.")
    timers["collector"] = time.time()
    for batch_idx, batches in enumerate(training_collector):  # batches are on config.device.collector_storage.
        timers["collector"] = time.time() - timers["collector"]
        timers["fps_collector"] = (batches.numel() * config.env.frame_skip) / timers["collector"]
        progress = batches.numel() / config.collector.total_agent_steps
        progress_acc += progress
        counters["collector_steps"] += 1
        counters["agent_steps"] += batches.numel()
        counters["env_steps"] += batches.numel() * config.env.frame_skip

        # Update loggers.
        save_model_logger.register_progress(batch_idx, progress)
        eval_logger.register_progress(batch_idx, progress)
        batch_logger.register_progress(batch_idx, progress)

        batch_logs = DictWithPrefix("batch/", {})

        # Create mask over samples of incomplete episodes for episodic algorithms.
        # To be used with GAE lambda=1 to obtain unbiased Monte-Carlo returns, keeping only \sum_T R_t.
        if hasattr(loss_module_policy.tensor_keys, "complete_episode_mask"):
            complete_episode_mask = compute_complete_episode_mask(batches)
            batches.set(loss_module_policy.tensor_keys.complete_episode_mask, complete_episode_mask)
            if batch_logger.log_this_round:
                batch_logs.update_with_prefix(
                    {"perf/in_finished_episode_fraction": complete_episode_mask.float().mean()}
                )

        # This update can be confusing.
        # Rollout metrics are always updated while episode metrics may not.
        # Creating a discrepancy in the timestep to which the metrics are referring.
        # This is to log the latest statistics available for each key as the batch is not always logged.
        batch_perf_stats.update(metrics.compute_eval_stats(batches))
        batch_perf_stats_no_reset.update(batch_perf_stats)
        if "perf/avg_return" in batch_perf_stats_no_reset:
            perf_string = lambda stat: f"{batch_perf_stats_no_reset[stat]:.2f}"
        else:
            perf_string = lambda stat: "no-data"
        pbar_desc["train"] = (
            f"Train "
            f"{int(timers['fps_collector'])} FPS "
            f"({progress_acc:.0%}): {perf_string('perf/avg_return')} "
            f"({perf_string('perf/max_return')})"
        )
        update_pbar_desc(pbar, pbar_desc)
        if batch_logger.log_this_round:
            # Log the latest return not logged yet.
            batch_logs.update_with_prefix(batch_perf_stats)
            # Reset if the latest return has been logged to not log twice.
            if "perf/max_return_raw" in batch_perf_stats:
                # batch_perf_stats = {}
                # Reset commented out for reproducibility.
                # Experiments in the paper were run before the reset was added.
                # We make sure not to account for the same return twice in the computation of the paper metrics.
                pass

        # Target estimation.
        with torch.no_grad():
            timers["state_value"] = time.time()
            if config.device.training == config.device.collector_storage:
                # non-urgent-fix: vectorize this.
                value_module(batches)
                value_module(batches.get("next"))
            else:
                # non-urgent-fix: this does not give the exact same numbers as above. Why?
                step = config.optim.minibatch_size
                for i in range(batches.shape[0]):  # Env dim.
                    for j in range(0, batches.shape[1], step):  # Time dim.
                        sub = batches.select(*value_module.in_keys)[i, j : j + step]
                        sub = value_module(sub.to(config.device.training)).select(*value_module.out_keys)
                        batches[i, j : j + step] = sub.to(config.device.collector_storage)
                        next_sub = batches.get("next").select(*value_module.in_keys)[i, j : j + step]
                        next_sub = value_module(next_sub.to(config.device.training)).select(*value_module.out_keys)
                        batches.get("next")[i, j : j + step] = next_sub.to(config.device.collector_storage)

            timers["state_value"] = time.time() - timers["state_value"]
            timers["targets_batch"] = time.time()
            target_estimator_critic(batches)
            target_estimator_actor(batches)
            timers["targets_batch"] = time.time() - timers["targets_batch"]

        # Flattens the batch dimensions (num_envs, num_frames) -> (num_envs * num_frames).
        data = batches.reshape(-1)
        if batch_logger.log_this_round:
            timers["eff_rnk_batch"] = time.time()
            batch_logs.update_with_prefix(
                dict_with_prefix(
                    {
                        **metrics.compute_effective_ranks(
                            [data],
                            ["batch"],
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
                            [data],
                            ["batch"],
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
                            [data],
                            ["batch"],
                            [
                                [
                                    "all_preactivations_policy",
                                    "all_preactivations_value",
                                ]
                            ],
                        ),
                        **metrics.compute_dead_neurons(
                            [data], ["batch"], "features_policy", config.models.actor.activation
                        ),
                        **metrics.compute_dead_neurons(
                            [data], ["batch"], "features_value", config.models.critic.activation
                        ),
                        **metrics.compute_policy_diversity(data, policy_module),
                        **metrics.compute_value_diversity(data),
                    },
                    "start/",
                )
            )
            timers["eff_rnk_batch"] = time.time() - timers["eff_rnk_batch"]

        # Epoch loop.
        epoch_logger = LogTracker(
            name="epoch",
            counter_key="counters/epoch_idx",
            trigger_level=LogLevel.EPOCH,
            logging_level=config.logging.log_level,
            final_idx=config.optim.num_epochs,
            depends_on=batch_logger,
            add_first=True,
            add_last=True,
        )
        first_epoch_minibatch_logs = {}  # Used in the batch logger to log the first minibatch of the first epoch.
        last_epoch_minibatch_logs = {}  # Used in the batch logger to log the last minibatch of the last epoch.
        kl_early_stop = False
        losses_policy_batch = tensordict.TensorDict(
            {}, batch_size=config.optim.max_gradient_steps_per_rollout
        )  # Track all the epoch & minibatch losses and log them at the batch level.
        losses_value_batch = tensordict.TensorDict(
            {}, batch_size=config.optim.max_gradient_steps_per_rollout
        )  # Track all the epoch & minibatch losses and log them at the batch level.
        for epoch_idx in range(config.optim.num_epochs):
            timers["epoch"] = time.time()
            epoch_logger.register_progress(epoch_idx)
            # Minibatch loop.
            minibatch_logger = LogTracker(
                name="minibatch",
                counter_key="counters/minibatch_idx",
                trigger_level=LogLevel.MINIBATCH,
                logging_level=config.logging.log_level,
                final_idx=config.optim.num_minibatches_per_epoch,
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
            # Assumes sampling the whole replay buffer which only contains on-policy data.
            kl_epoch = torch.zeros(1, device=config.device.training)
            local_minibatch_idx = 0
            sampler = SamplerWithoutReplacement(drop_last=True)
            while not sampler.ran_out:
                local_update_index = epoch_idx * config.optim.num_minibatches_per_epoch + local_minibatch_idx
                timers["minibatch"] = time.time()
                minibatch_logger.register_progress(local_minibatch_idx)
                minibatch_logs = DictWithPrefix("minibatch/", {})
                minibatch_indices = sampler.sample(data, config.optim.minibatch_size)[0]
                minibatch = data[minibatch_indices].to(config.device.training)
                timers["loss_minibatch"] = time.time()  # Should be async!
                losses_policy = loss_module_policy(minibatch)
                losses_value = loss_module_value(minibatch)
                timers["loss_minibatch"] = time.time() - timers["loss_minibatch"]
                losses_policy_batch[local_update_index] = losses_policy.detach()
                losses_value_batch[local_update_index] = losses_value.detach()
                total_loss = sum(loss for key, loss in losses_policy.items() if key.startswith("loss_"))
                total_loss += sum(loss for key, loss in losses_value.items() if key.startswith("loss_"))
                optimizer_policy.zero_grad()
                optimizer_value.zero_grad()
                timers["backward_minibatch"] = time.time()
                with torch.autograd.set_detect_anomaly(True):
                    total_loss.backward()
                timers["backward_minibatch"] = time.time() - timers["backward_minibatch"]

                if "kl" in losses_policy.keys():
                    kl_early_stop = losses_policy["kl"] > config.loss.policy.kl_stop_limit
                    kl_epoch += losses_policy["kl"]
                if kl_early_stop and config.loss.policy.kl_early_stop:
                    # Mark as if this was the last epoch and minibatch.
                    epoch_logger.register_progress(config.optim.num_epochs - 1)
                    minibatch_logger.add_last = epoch_logger.is_last  # Updated.
                    minibatch_logger.register_progress(config.optim.num_minibatches_per_epoch - 1)

                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_module.parameters(), config.optim.policy.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    value_module.parameters(), config.optim.value.max_grad_norm
                )
                prev_policy_params = None
                prev_value_params = None
                if minibatch_logger.log_this_round:
                    minibatch_logs.update_with_prefix({f"loss/{k}": v.detach() for k, v in losses_policy.items()})
                    minibatch_logs.update_with_prefix({f"loss/{k}": v.detach() for k, v in losses_value.items()})
                    minibatch_logs.update_with_prefix({"loss/kl_early_stop": float(kl_early_stop)})
                    minibatch_logs.update_with_prefix({"loss/updates_done": local_update_index})
                    # Record previous parameters.
                    prev_policy_params = [p.detach().clone() for p in policy_module.parameters()]
                    prev_value_params = [p.detach().clone() for p in value_module.parameters()]

                # Update parameters.
                optimizer_policy.step()
                optimizer_value.step()

                if minibatch_logger.log_this_round:
                    minibatch_logs.update_with_prefix(
                        {
                            "grad_norm/actor": actor_grad_norm,
                            "grad_norm/critic": critic_grad_norm,
                            "grad_norm/actor_clipped": min(config.optim.policy.max_grad_norm, actor_grad_norm),
                            "grad_norm/critic_clipped": min(config.optim.value.max_grad_norm, critic_grad_norm),
                            "grad_norm/actor_applied": metrics.compute_applied_grad_norm(
                                policy_module.parameters(), prev_policy_params
                            ),
                            "grad_norm/critic_applied": metrics.compute_applied_grad_norm(
                                value_module.parameters(), prev_value_params
                            ),
                            "weight_norm/actor": metrics.compute_weights_norm(
                                policy_module.parameters(), prev_policy_params
                            ),
                            "weight_norm/critic": metrics.compute_weights_norm(
                                value_module.parameters(), prev_value_params
                            ),
                            "learning_rate/policy": optimizer_policy.param_groups[0]["lr"],
                            "learning_rate/value": optimizer_value.param_groups[0]["lr"],
                        }
                    )
                    to_log = {
                        # The global minibatch index.
                        # Is also the num of optim updates done to the models.
                        "counters/minibatch_idx": counters["minibatch_idx"],
                        "timers/minibatch": time.time() - timers["minibatch"],
                        "timers/minibatch_backward": timers["backward_minibatch"],
                        "timers/minibatch_loss": timers["loss_minibatch"],
                        **minibatch_logs,
                    }
                    minibatch_logger.log_to_file(filter_out_wandb(to_log))
                    wandb.log(filter_out_underscore(to_log))

                if minibatch_logger.is_first:
                    first_minibatch_logs = dict_with_prefix(minibatch_logs, "epoch/first_")
                if minibatch_logger.is_last:
                    last_minibatch_logs = dict_with_prefix(minibatch_logs, "epoch/last_")
                counters["minibatch_idx"] += 1
                local_minibatch_idx += 1
                if kl_early_stop and config.loss.policy.kl_early_stop:
                    break
                # End of minibatch loop.

            # Back to epoch loop.
            kl_epoch /= local_minibatch_idx + 1
            # Log the first minibatch stats with at the epoch frequency.
            if epoch_logger.log_this_round:
                to_log = {
                    "counters/epoch_idx": counters["epoch_idx"],  # The global epoch index.
                    "counters/last_minibatch_idx": counters["minibatch_idx"] - 1,  # The global minibatch index.
                    "timers/epoch": time.time() - timers["epoch"],
                    "epoch/kl_mean": kl_epoch,
                    **first_minibatch_logs,
                    # the last minibatch of intermediate epochs is expressed in the first minibatch of the next epoch.
                    # the last minibatch of last epoch is logged in batch/last_epoch/last_minibatch.
                }
                epoch_logger.log_to_file(filter_out_wandb(to_log))
                wandb.log(filter_out_underscore(to_log))

            if epoch_logger.is_first:
                first_epoch_minibatch_logs = dict_with_prefix(first_minibatch_logs, "batch/first_")
            if epoch_logger.is_last:
                last_epoch_minibatch_logs = dict_with_prefix(last_minibatch_logs, "batch/last_")
            counters["epoch_idx"] += 1
            if kl_early_stop and config.loss.policy.kl_early_stop:
                break
            # End of epoch loop.

        # Back to collector/batch loop.

        # Update the KL beta parameter.
        # Faithful to the original implementation in
        # https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L208C19-L208C31
        # This is done at the end of the batch based on the average KL in the last epoch.
        if hasattr(loss_module_policy, "update_adaptive_parameters"):
            loss_module_policy.update_adaptive_parameters(kl_epoch)

        if batch_logger.log_this_round:
            with torch.no_grad():
                # Recompute features, state values and policy to see the effects at the end of the batch.
                end_data = data.select(OBS_KEY)
                if config.device.training == config.device.collector_storage:
                    value_module(end_data)
                    policy_module.get_dist(end_data)
                else:
                    for i in range(0, end_data.shape[0], config.optim.minibatch_size):
                        sub = end_data[i : i + config.optim.minibatch_size].to(config.device.training)
                        value_module(sub)
                        policy_module.get_dist(sub)
                        end_data[i : i + config.optim.minibatch_size] = sub.to(config.device.collector_storage)

                batch_logs.update_with_prefix(
                    dict_with_prefix(
                        {
                            **metrics.compute_effective_ranks(
                                [end_data],
                                ["batch"],
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
                                [end_data],
                                ["batch"],
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
                                [end_data],
                                ["batch"],
                                [
                                    [
                                        "all_preactivations_policy",
                                        "all_preactivations_value",
                                    ]
                                ],
                            ),
                            **metrics.compute_dead_neurons(
                                [end_data], ["batch"], "features_policy", config.models.actor.activation
                            ),
                            **metrics.compute_dead_neurons(
                                [end_data], ["batch"], "features_value", config.models.critic.activation
                            ),
                            **metrics.compute_policy_diversity(end_data, policy_module),
                            **metrics.compute_value_diversity(end_data),
                            **compute_kl_batch(data, end_data, policy_module),
                            "losses_policy_batch_": losses_policy_batch,
                            "losses_value_batch_": losses_value_batch,
                            "epoch_kl_mean": kl_epoch,
                        },
                        "end/",
                    )
                )
                batch_logs.update_with_prefix(
                    dict_with_prefix(
                        {
                            **compute_models_diff(
                                end_data,
                                data,
                                policy_module,
                                ratio_epsilon=config.loss.policy.kwargs.clip_epsilon
                                if config.loss.policy.kwargs.clip_epsilon
                                else 0.0,
                            ),
                            **get_rank_diff(batch_logs),
                        },
                        "diff/",
                    )
                )
            to_log = {
                "counters/collector_steps": counters["collector_steps"],  # How many collector steps were done.
                "counters/agent_steps": counters["agent_steps"],  # How many agent steps were performed.
                "counters/env_steps": counters["env_steps"],  # How many env steps were performed.
                "global_step": counters["env_steps"],
                "counters/batch_idx": counters["batch_idx"],  # The global batch index.
                "counters/last_epoch_idx": counters["epoch_idx"] - 1,  # The global epoch index.
                "counters/last_minibatch_idx": counters["minibatch_idx"] - 1,
                "timers/collector_rollout": timers["collector"],
                "timers/fps_collector": timers["fps_collector"],
                "timers/effective_ranks_batch": timers["eff_rnk_batch"],
                **batch_logs,
                **first_epoch_minibatch_logs,
                **last_epoch_minibatch_logs,
            }
            batch_logger.log_to_file(filter_out_wandb(to_log))
            wandb.log(filter_out_underscore(to_log))

        # Save models.
        if save_model_logger.log_this_round:
            to_log = modules.models.model_snapshots(
                policy_module,
                value_module,
                optimizer_policy,
                optimizer_value,
                scheduler_policy,
                scheduler_value,
                counters,
                config,
                train_env.action_spec,
            )
            save_model_logger.log_to_file(filter_out_wandb(to_log))

        # Reset the optimizer
        # The lr will be updated correctly by the scheduler if needed.
        if config.optim.policy.reset_state:
            optimizer_policy = getattr(torch.optim, config.optim.policy.algo)(
                policy_module.parameters(), **config.optim.policy.kwargs
            )
            scheduler_policy.optimizer = optimizer_policy
        if config.optim.value.reset_state:
            optimizer_value = getattr(torch.optim, config.optim.value.algo)(
                value_module.parameters(), **config.optim.value.kwargs
            )
            scheduler_value.optimizer = optimizer_value

        if config.optim.policy.anneal_linearly:
            scheduler_policy.step()
        if config.optim.value.anneal_linearly:
            scheduler_value.step()

        # Evaluation phase.
        if eval_logger.log_this_round:
            evaluate(config, counters, eval_collector, eval_logger, pbar, pbar_desc, progress_acc, timers, value_module)

        pbar.update(data.numel() * config.env.frame_skip)
        training_collector.update_policy_weights_()
        counters["batch_idx"] += 1
        timers["collector"] = time.time()

    # End of training loop.
    training_collector.shutdown()
    if config.eval.log_level <= LogLevel.MAX_LEVEL:
        eval_collector.shutdown()


def evaluate(config, counters, eval_collector, eval_logger, pbar, pbar_desc, progress_acc, timers, value_module):
    logger.info(f"Evaluation at {counters['env_steps'] / config.collector.total_env_steps:.0%} of training.")
    eval_logs = DictWithPrefix("eval/", {})
    timers["eval"] = time.time()
    eval_logs.update_with_prefix(evaluate_(eval_collector, value_module, config))
    to_log = {
        **eval_logs,
        "counters/collector_steps": counters["collector_steps"],
        "counters/agent_steps": counters["agent_steps"],
        "counters/env_steps": counters["env_steps"],
        "counters/batch_idx": counters["batch_idx"],
        "counters/last_epoch_idx": counters["epoch_idx"] - 1,
        "counters/last_minibatch_idx": counters["minibatch_idx"] - 1,
        "timers/eval": time.time() - timers["eval"],
        "timers/fps_eval_mode": eval_logs["eval/mode/perf/fps"],
        "timers/eval_rollout": eval_logs["eval/mode/perf/rollout_time"],
    }
    if config.eval.record_video:
        to_log.update(
            {
                "timers/video": eval_logs["eval/mode/perf/video_time"],
            }
        )
    eval_logger.log_to_file(filter_out_wandb(to_log))
    wandb.log(filter_out_underscore(to_log))
    plt.close()
    if "eval/mode/perf/avg_return" in eval_logs:
        perf_string = lambda stat: f"{eval_logs[stat]:.2f}"
    else:
        perf_string = lambda stat: "no-data"
    pbar_desc["eval"] = (
        f"Eval "
        f"{int(eval_logs['eval/mode/perf/fps'])} FPS "
        f"({progress_acc:.0%}): {perf_string('eval/mode/perf/avg_return')} "
        f"({perf_string('eval/mode/perf/max_return')})"
    )
    update_pbar_desc(pbar, pbar_desc)


def update_pbar_desc(pbar, pbar_desc):
    pbar.set_description(f"[avg_return (max_return)] | {pbar_desc['train']} | {pbar_desc['eval']}")


def compute_kl_batch(old_data, new_data, policy_module):
    """"""
    res = {}
    old_dist = policy_module.build_dist_from_params(old_data)
    new_dist = policy_module.build_dist_from_params(new_data)
    new_sample_log_prob = new_dist.log_prob(old_data["action"])
    res.update(metrics.compute_kl_divergence_from_dists(old_dist, new_dist))
    res.update(metrics.compute_kl_divergence_from_samples(old_data["sample_log_prob"], new_sample_log_prob))
    return dict_with_prefix(res, "kl_batch/")


def compute_complete_episode_mask(batches):
    """Computes a mask over the samples of incomplete episodes for episodic algorithms."""
    last_traj_not_done = ~batches["next", "done"][:, -1]
    is_last_traj = batches["collector", "traj_ids"] == batches["collector", "traj_ids"][:, -1].unsqueeze(-1)
    to_mask = is_last_traj * last_traj_not_done
    return ~to_mask.unsqueeze(-1)


@torch.no_grad()
def evaluate_(eval_collector, value_module, config):
    eval_collector.update_policy_weights_()
    logs = {}
    # Evaluate with two sampling strategies: random-sample and best-sample.
    for exploration_type, exploration_name in zip([ExplorationType.RANDOM, ExplorationType.MODE], ["random", "mode"]):
        eval_collector.exploration_type = exploration_type
        eval_rollout_time = time.time()
        batches = eval_collector.next()  # Batches are in CPU.
        eval_rollout_time = time.time() - eval_rollout_time
        logs[f"{exploration_name}/perf/rollout_time"] = eval_rollout_time
        logs[f"{exploration_name}/perf/fps"] = (batches.numel() * config.env.frame_skip) / eval_rollout_time
        # Eval stats.
        logs.update(dict_with_prefix(metrics.compute_eval_stats(batches), f"{exploration_name}/"))
        # Evolution of the state value in the trajectory.
        # Take only one traj per env.
        mask = batches["collector", "traj_ids"] == batches["collector", "traj_ids"][:, :1]
        max_traj_len = mask.sum(dim=-1).max()
        batches = batches[:, :max_traj_len]
        trajs = batches.select(*value_module.in_keys)
        # Compute state values.
        step = config.optim.minibatch_size
        for i in range(trajs.shape[0]):  # Env dim.
            for j in range(0, trajs.shape[1], step):  # Time dim.
                sub = trajs[i, j : j + step]
                sub = value_module(sub.to(config.device.training)).select(*value_module.out_keys)
                trajs[i, j : j + step] = sub.to("cpu")
        # Mask out the extra steps.
        trajs["state_value"] *= mask[:, :max_traj_len].unsqueeze(-1)
        # Plot it.
        fig, ax = plt.subplots(figsize=(10, 5))
        x_values = np.arange(max_traj_len)
        for i in range(trajs.shape[0]):
            ax.plot(x_values, trajs["state_value"][i].squeeze(-1).numpy())
        ax.set_title("State value evolution for different trajectories")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("State value")
        ax.legend([f"Traj {i}" for i in range(trajs.shape[0])])
        logs[f"{exploration_name}/perf/state_value_evolution_"] = trajs["state_value"]
        logs[f"{exploration_name}/perf/state_value_evolution_plot_wandb"] = fig

        # Video
        if config.eval.record_video:
            pixels = batches.get("pixels")
            pixels = pixels.permute(0, 1, -1, -3, -2).cpu().numpy()
            video_time = time.time()
            video = wandb.Video(pixels, fps=config.eval.video_agent_fps, format="mp4")
            video_time = time.time() - video_time
            logs[f"{exploration_name}/perf/video_wandb"] = video
            logs[f"{exploration_name}/perf/video_time"] = video_time

        del batches
    return logs


def get_rank_diff(batch_logs):
    res = {}
    for key in batch_logs.keys():
        if key.startswith("batch/start/"):
            metric_name = key[len("batch/start/") :]
            if f"batch/end/{metric_name}" in batch_logs:
                res[metric_name] = batch_logs[f"batch/end/{metric_name}"] - batch_logs[key]
    return res


def fix_and_augment_config(config):
    """
    Fix inconsistent elements (e.g., non-divisible batch size, etc.), add read-only elements,
    and issue warnings for config elements that were corrected.

    The read-only variables are directly computed and determined from user-set variable.
    They are meant to be used in the code as helpers or in the experiment logs
    when navigating the experiments to filter, sort, etc.
    """
    # Env.
    if config.env.truncate_env_steps_at % config.env.frame_skip != 0:
        warnings.warn(
            f"The frame_skip ({config.env.frame_skip}) does not divide"
            f"The max number of env steps per episode truncate_env_steps_at ({config.env.truncate_env_steps_at}).\n"
            f"The truncation limit will be adjusted to the floor."
        )
    config.env.truncate_env_steps_at = (
        config.env.truncate_env_steps_at // config.env.frame_skip
    ) * config.env.frame_skip
    # The maximum number of agent steps per episode.
    # Gives truncation limit.
    config.env.truncate_agent_steps_at = config.env.truncate_env_steps_at // config.env.frame_skip

    # Collector.
    # The number of steps collected per batch.
    config.collector.agent_steps_per_batch = config.env.num_envs * config.collector.agent_steps_per_env

    if config.collector.total_env_steps % config.env.frame_skip != 0:
        warnings.warn(
            f"The frame_skip ({config.env.frame_skip}) does not divide the"
            f"total number of env steps to perform during training ({config.collector.total_env_steps}).\n"
            f"The extra env steps will be discarded."
        )
    if (config.collector.total_env_steps // config.env.frame_skip) % config.collector.agent_steps_per_batch != 0:
        warnings.warn(
            f"The total number of env steps to perform during training ({config.collector.total_env_steps})\n"
            f"divided by the frame_skip ({config.env.frame_skip}) and divided by the number of agent steps per batch "
            f"({config.collector.agent_steps_per_batch}) is not an integer.\n"
            f"The total number will be adjusted to be a multiple of both."
        )

    # The total number of environment steps collected during training.
    config.collector.total_env_steps = (
        ((config.collector.total_env_steps // config.env.frame_skip) // config.collector.agent_steps_per_batch)
        * config.collector.agent_steps_per_batch
        * config.env.frame_skip
    )
    # The total agent steps collected during training.
    config.collector.total_agent_steps = config.collector.total_env_steps // config.env.frame_skip
    config.collector.num_batches = config.collector.total_agent_steps // config.collector.agent_steps_per_batch
    config.collector.num_training_phases = config.collector.num_batches

    # Eval.
    if config.env.eval.env_steps_per_eval_env % config.env.frame_skip != 0:
        warnings.warn(
            f"The frame_skip ({config.env.frame_skip}) does not divide"
            f"the number of env steps per evaluation env ({config.env.eval.env_steps_per_eval_env}).\n"
            f"The extra env steps will be discarded."
        )
    # Number of agent steps in each eval.
    config.env.eval.env_steps_per_eval_env = (
        config.env.eval.env_steps_per_eval_env // config.env.frame_skip
    ) * config.env.frame_skip
    config.env.eval.agent_steps_per_eval_env = config.env.eval.env_steps_per_eval_env // config.env.frame_skip
    config.eval.agent_steps_per_eval = config.env.eval.agent_steps_per_eval_env * config.env.eval.num_envs
    # The total number of agent steps collected during all evals (lazy upper bound as may not eval every batch).
    config.eval.total_agent_steps = config.eval.agent_steps_per_eval * config.collector.num_batches

    if config.env.eval.video_env_fps % config.env.frame_skip != 0:
        warnings.warn(
            f"The video env fps ({config.env.eval.video_env_fps}) is not divisible by the frame_skip "
            f"({config.env.frame_skip}).\n"
            f"The video env fps will be adjusted to the floor."
        )
    config.env.eval.video_env_fps = (config.env.eval.video_env_fps // config.env.frame_skip) * config.env.frame_skip
    config.eval.video_agent_fps = config.env.eval.video_env_fps // config.env.frame_skip

    # Optim.
    # Assumes on-policy and sampling the whole batch from the replay buffer.
    config.optim.num_minibatches_per_epoch = config.collector.agent_steps_per_batch // config.optim.minibatch_size
    config.optim.max_gradient_steps_per_rollout = config.optim.num_minibatches_per_epoch * config.optim.num_epochs
    config.optim.max_total_gradient_steps = config.optim.max_gradient_steps_per_rollout * config.collector.num_batches

    # Utilities
    config.working_dir = f"{Path.cwd()}"


if __name__ == "__main__":
    main()
