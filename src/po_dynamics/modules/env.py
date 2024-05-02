import logging
import time
from typing import Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tensordict.utils import NestedKey

logger = logging.getLogger(__name__)

import torchrl
from torchrl.collectors import RandomPolicy, SyncDataCollector
from torchrl.envs import (
    CatFrames,
    CatTensors,
    DoubleToFloat,
    EnvBase,
    FlattenObservation,
    GrayScale,
    NoopResetEnv,
    ObservationNorm,
    RenameTransform,
    Resize,
    RewardSum,
    SignTransform,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs

OBS_KEY = "obs"


def build_single_env(config: DictConfig, is_eval: bool) -> TransformedEnv:
    """Build a single environment with local transforms.

    Returns env has step counter and noop reset if set.
    """
    native_kwargs = config.env.kwargs if hasattr(config.env, "kwargs") else {}
    eval_needs_pixels = is_eval and config.eval.record_video
    match config.env.lib:
        case "gym":
            env = GymEnv(
                config.env.name,
                frame_skip=config.env.frame_skip,
                from_pixels=config.env.from_pixels or eval_needs_pixels,
                pixels_only=config.env.from_pixels or not eval_needs_pixels,
                device=config.env.device,
                **native_kwargs,
            )
        case "dm_control":
            env = DMControlEnv(
                config.env.name,
                config.env.task,
                frame_skip=config.env.frame_skip,
                from_pixels=config.env.from_pixels or eval_needs_pixels,
                pixels_only=config.env.from_pixels or not eval_needs_pixels,
                device=config.env.device,
                **native_kwargs,
            )
        case _:
            raise NotImplementedError

    env = get_env_with_local_transforms(env, config, is_eval)
    return env


def build_batched_single_env(config: DictConfig, is_eval: bool) -> TransformedEnv:
    if config.debug.check_env_specs:
        logger.info("Checking a single env specs after local transforms.")
        check_env_specs(build_single_env(config, is_eval))

    num_envs = config.env.num_envs if not is_eval else config.env.eval.num_envs
    batched_env = getattr(torchrl.envs, config.env.parallel_class)(
        num_envs,
        lambda: build_single_env(config, is_eval),
        device=config.device.rollout,
    )
    return TransformedEnv(batched_env)


def build_native_batched_env(config: DictConfig, is_eval: bool) -> TransformedEnv:
    """For envs that have native parallelism, e.g. IsaacGym, Jax environments, etc."""
    num_envs = config.env.num_envs if not is_eval else config.env.eval.num_envs

    match config.env.lib:
        case _:
            raise NotImplementedError

    batched_env = get_env_with_local_transforms(batched_env, config, is_eval)
    return batched_env


def get_env_with_local_transforms(env: EnvBase, config: DictConfig, is_eval: bool) -> TransformedEnv:
    """Add step counter and noop reset if set.

    No-op reset only works on single envs (non-batched).
    And to keep the termination limit Markovian, we need to track the steps before no-op reset.
    So StepCounter is also applied to single-envs.

    For consistency batched envs also have this function apply only step counter.
    """

    env = TransformedEnv(env)
    # Add step counter for correct truncation.
    if config.env.use_truncation_as_termination:
        truncated_key = "terminated"
    else:
        truncated_key = "truncated"
    if is_eval:
        # Truncate at eval budget.
        env.append_transform(
            CustomStepCounter(
                max_steps=config.env.eval.agent_steps_per_eval_env,
                truncated_key=truncated_key,
                set_terminated=config.env.use_truncation_as_termination,
            )
        )
    else:
        env.append_transform(
            CustomStepCounter(
                max_steps=config.env.truncate_agent_steps_at,
                truncated_key=truncated_key,
                set_terminated=config.env.use_truncation_as_termination,
            )
        )

    # No-op reset transform to introduce stochasticity at initial state.
    # Only works with single envs.
    if config.env.noop_reset_steps > 0:
        env.append_transform(NoopResetEnv(noops=config.env.noop_reset_steps, random=config.env.noop_is_random))
    return env


def add_env_specific_transforms(batched_env: TransformedEnv, config: DictConfig, is_eval: bool):
    """batched_env will contain the original keys from the env which should not be modified.
    the OBS_KEY which should not be modified if config.env.from_pixels, as it would contain pixels.
    """
    match config.env.lib:
        case "gym":
            if config.env.sublib == "mujoco" and not config.env.from_pixels:
                batched_env.append_transform(DoubleToFloat(in_keys=[OBS_KEY]))
        case "dm_control":
            in_keys = ["reward"]
            if not config.env.from_pixels:
                in_keys.append(OBS_KEY)
            batched_env.append_transform(
                DoubleToFloat(
                    in_keys=in_keys,
                    in_keys_inv=["action"],  # Action should be given back as a double to the env.
                ),
            )


def add_pixel_transforms(env, config, is_eval):
    """Add transforms for pixel environments.

    The output of these transforms will be in the OBS_KEY with the shape (S, C, H, W),
    or (N, T, S, C, H, W) in case of a batched env in a collector where
    N is the number of parallel envs, T the timestamps, and S is the number of stacked frames,
    regardless of the config choices.
    """
    # Move pixels to OBS_KEY and apply transforms.
    # Leave the raw pixels if this is an eval env to record videos.
    eval_needs_pixels = is_eval and config.eval.record_video
    if eval_needs_pixels:
        env.append_transform(CatTensors(in_keys=["pixels"], out_key=OBS_KEY, del_keys=False))  # This performs a copy.
    else:
        env.append_transform(RenameTransform(in_keys=["pixels"], out_keys=[OBS_KEY]))

    env.append_transform(ToTensorImage(in_keys=[OBS_KEY]))  # (..., C, H, W).
    if config.env.image_transforms.grayscale:  # (..., C=1, H, W).
        env.append_transform(GrayScale(in_keys=[OBS_KEY]))
    if config.env.image_transforms.resize:  # (..., C, H, W).
        env.append_transform(
            Resize(
                config.env.image_transforms.resize_w,
                config.env.image_transforms.resize_h,
                in_keys=[OBS_KEY],
            )
        )
    env.append_transform(UnsqueezeTransform(-4, in_keys=[OBS_KEY]))  # (..., S=1, C, H, W) where S=Stacked frames.
    if config.env.image_transforms.frame_stack > 1:
        env.append_transform(
            CatFrames(
                N=config.env.image_transforms.frame_stack,
                dim=-4,
                in_keys=[OBS_KEY],
            )
        )  # (..., S, C, H, W)


def make_env(config: DictConfig, is_eval: bool, normalization_stats: dict, seed: int):
    if config.env.batchmode == "native":
        # For IsaacGym, Jax environments, etc.
        batched_env = build_native_batched_env(config, is_eval)
    elif config.env.batchmode == "torchrl":
        # For ParallelEnv, and SerialEnv of torchrl.
        batched_env = build_batched_single_env(config, is_eval)
    else:
        raise NotImplementedError
    # At this stage, the batched env has a step counter and optionally a noop reset
    # only applied locally to single envs.

    batched_env.set_seed(seed)

    if config.debug.check_env_specs:
        logger.info("Checking parallel environment specs after (local) transforms.")
        check_env_specs(batched_env)
    if config.debug.benchmark_env and not is_eval:
        logger.info("Benchmarking parallel environment before observation transforms.")
        benchmark_env(batched_env, config)

    # Observation transforms.
    if config.env.from_pixels:
        add_pixel_transforms(batched_env, config, is_eval)
    else:  # Flatten the observation.
        batched_env.append_transform(
            CatTensors(
                in_keys=list(
                    key
                    for key in batched_env.observation_spec.keys(True, True)
                    if key not in ["pixels", "step_count"]
                    # Pixels can be there to record videos.
                ),
                out_key=OBS_KEY,
            )
        )
    # At this stage, the observation (pixel or not) will be in the OBS_KEY.

    # Add env-specific transforms (type conversion of OBS_key and actions to be read by models and fed back to env).
    add_env_specific_transforms(batched_env, config, is_eval)

    # Observation normalization.
    if config.env.normalize_obs:
        if "loc" in normalization_stats:
            batched_env.append_transform(
                ObservationNorm(
                    in_keys=[OBS_KEY],
                    loc=normalization_stats["loc"],
                    scale=normalization_stats["scale"],
                )
            )
        else:
            if config.env.from_pixels:
                cat_dim = 1  # list[(N, T, S, C, H, W)] -> (N, \sum T, S, C, H, W).
                reduce_dim = (0, 1, 2, -2, -1)  # (N, \sum T, S, C, H, W) -> (C,).
                # I.e also reduce over stacked frames and all the pixels per channel.
                keep_dims = (-2, -1)  # loc and scale will thus be (C, 1, 1).
            else:
                cat_dim = 1  # list[(N, T_rollout, Obs)] -> (N, \sum T, Obs).
                reduce_dim = (0, 1)  # (N, \sum T, Obs) -> (Obs,).
                keep_dims = None  # loc and scale will thus be (Obs,).

            batched_env.append_transform(CustomizableObservationNorm(in_keys=[OBS_KEY]))
            logger.info("Initializing observation normalization. Collecting random trajectories...")
            num_steps = config.env.eval.agent_steps_per_eval_env * batched_env.batch_size[0]
            if config.device.rollout == "cpu":
                num_steps_per_env_per_batch = num_steps
            elif config.device.rollout == config.device.collector_storage:
                num_steps_per_env_per_batch = config.collector.agent_steps_per_env
            elif config.device.rollout == config.device.training:
                num_steps_per_env_per_batch = config.optim.minibatch_size // batched_env.batch_size[0]
            else:
                num_steps_per_env_per_batch = 1
            fps_init_norm_obs = time.time()
            batched_env.transform[-1].custom_init_stats(
                num_steps=num_steps,
                storing_device="cpu",
                num_steps_per_env_per_batch=num_steps_per_env_per_batch,
                cat_dim=cat_dim,
                reduce_dim=reduce_dim,
                keep_dims=keep_dims,
            )
            fps_init_norm_obs = int(num_steps / (time.time() - fps_init_norm_obs))
            normalization_stats.update(
                {
                    "loc": batched_env.transform[-1].loc.clone(),
                    "scale": batched_env.transform[-1].scale.clone(),
                }
            )
            logger.info(f"Normalization stats initialized at {fps_init_norm_obs} FPS.")
            logger.info(f"Loc: {normalization_stats['loc']}")
            logger.info(f"Scale: {normalization_stats['scale']}")

    # Stack stacked frames with the channel dimension (it just becomes more channels).
    # (N, T, S, C, H, W) -> (N, T, S*C, H, W).
    if config.env.from_pixels:
        batched_env.append_transform(FlattenObservation(-4, -3, in_keys=[OBS_KEY]))

    # Time-aware observation and observation normalization.
    # Only for non-pixel envs so far.
    # non-urgent-fix: If env is from pixels, this needs to be a new channel.
    #   For now, we assume a notion of time is present through the pixels.
    if config.env.time_aware:
        if config.env.from_pixels:
            raise NotImplementedError("Time aware not implemented for pixel envs.")
        # Copy to a separate key.
        batched_env.append_transform(CatTensors(in_keys=["step_count"], out_key="obs_step_count", del_keys=False))
        # And normalize to [-1, 1].
        batched_env.append_transform(
            ObservationNorm(
                in_keys=["obs_step_count"],
                loc=-1,
                scale=2 / config.env.truncate_agent_steps_at,
            )
        )
        # Merge with observation.
        batched_env.append_transform(CatTensors(in_keys=[OBS_KEY, "obs_step_count"], out_key=OBS_KEY))

    # Reward transforms
    # Only applied to the train env.
    # Eval env keeps the original reward.
    if not is_eval:
        # Return before transform.
        batched_env.append_transform(RewardSum(in_keys=["reward"], out_keys=["return_raw"]))

        if config.env.reward_transforms.sign:
            batched_env.append_transform(SignTransform(in_keys=["reward"], out_keys=["reward"]))

    # return after transforms.
    # non-urgent-fix: return spec is wrong when SignTransform is applied.
    batched_env.append_transform(RewardSum(in_keys=["reward"], out_keys=["return"]))

    if config.debug.check_env_specs:
        logger.info("Checking batched environment specs after transforms.")
        # Will crash as return spec has bugs for now.
        check_env_specs(batched_env)
    if config.debug.benchmark_env and not is_eval:
        logger.info("Benchmarking batched environment after transforms.")
        benchmark_env(batched_env, config)

    return batched_env


def benchmark_env(env, config):
    total_agent_steps = config.collector.agent_steps_per_batch * 10
    collector = SyncDataCollector(
        env,
        policy=RandomPolicy(env.action_spec),
        frames_per_batch=config.collector.agent_steps_per_batch,
        total_frames=total_agent_steps,
        device=config.device.rollout,
        storing_device=config.device.collector_storage,
    )
    times = []
    start = time.time()
    for _ in collector:
        times.append(time.time() - start)
        start = time.time()
    collector.shutdown()
    logger.info(
        f"{int((total_agent_steps * config.env.frame_skip) / sum(times))} FPS with {config.env.parallel_class}\n"
        f"Base env on {config.env.device}\n"
        f"Rolling on {config.device.rollout} and stored in {config.device.collector_storage}."
    )


class CustomStepCounter(StepCounter):
    def __init__(self, set_terminated=False, **kwargs):
        super().__init__(**kwargs)
        self.set_terminated = set_terminated

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        for step_count_key, truncated_key, done_key, terminated_key in zip(
            self.step_count_keys,
            self.truncated_keys,
            self.done_keys,
            self.terminated_keys,
        ):
            step_count = tensordict.get(step_count_key)
            next_step_count = step_count + 1
            next_tensordict.set(step_count_key, next_step_count)

            if self.max_steps is not None:
                truncated = next_step_count >= self.max_steps
                truncated = truncated | next_tensordict.get(truncated_key, False)
                if self.update_done:
                    done = next_tensordict.get(done_key, None)
                    terminated = next_tensordict.get(terminated_key, None)
                    if terminated is not None and not self.set_terminated:
                        truncated = truncated & ~terminated
                    done = truncated | done  # we assume no done after reset
                    next_tensordict.set(done_key, done)
                next_tensordict.set(truncated_key, truncated)
        return next_tensordict


class CustomizableObservationNorm(ObservationNorm):
    """Allows to store the rollout data on a different device than the parent env and rollout in batches."""

    def custom_init_stats(
        self,
        num_steps: int,
        storing_device: str,
        num_steps_per_env_per_batch: int,
        reduce_dim: Union[int, Tuple[int]] = 0,
        cat_dim: Optional[int] = None,
        key: Optional[NestedKey] = None,
        keep_dims: Optional[Tuple[int]] = None,
    ) -> None:
        """Initializes the loc and scale stats of the parent environment.

        Normalization constant should ideally make the observation statistics approach
        those of a standard Gaussian distribution. This method computes a location
        and scale tensor that will empirically compute the mean and standard
        deviation of a Gaussian distribution fitted on data generated randomly with
        the parent environment for a given number of steps.

        Args:
            num_steps (int): number of random iterations to run in the environment.
            storing_device (str): device to store the collected data on.
            num_steps_per_env_per_batch (int): number steps per env in a rollout batch (performed on the env device)
                before moving the data to the storing device.
            reduce_dim (int or tuple of int, optional): dimension to compute the mean and std over.
                Defaults to 0.
            cat_dim (int, optional): dimension along which the batches collected will be concatenated.
                It must be part equal to reduce_dim (if integer) or part of the reduce_dim tuple.
                Defaults to the same value as reduce_dim.
            key (NestedKey, optional): if provided, the summary statistics will be
                retrieved from that key in the resulting tensordicts.
                Otherwise, the first key in :obj:`ObservationNorm.in_keys` will be used.
            keep_dims (tuple of int, optional): the dimensions to keep in the loc and scale.
                For instance, one may want the location and scale to have shape [C, 1, 1]
                when normalizing a 3D tensor over the last two dimensions, but not the
                third. Defaults to None.

        """
        if cat_dim is None:
            cat_dim = reduce_dim
            if not isinstance(cat_dim, int):
                raise ValueError("cat_dim must be specified if reduce_dim is not an integer.")
        if (isinstance(reduce_dim, tuple) and cat_dim not in reduce_dim) or (
            isinstance(reduce_dim, int) and cat_dim != reduce_dim
        ):
            raise ValueError("cat_dim must be part of or equal to reduce_dim.")
        if self.initialized:
            raise RuntimeError(f"Loc/Scale are already initialized: ({self.loc}, {self.scale})")

        if len(self.in_keys) > 1 and key is None:
            raise RuntimeError("Transform has multiple in_keys but no specific key was passed as an argument")
        key = self.in_keys[0] if key is None else key

        def raise_initialization_exception(module):
            if isinstance(module, ObservationNorm) and not module.initialized:
                raise RuntimeError(
                    "ObservationNorms need to be initialized in the right order."
                    "Trying to initialize an ObservationNorm "
                    "while a parent ObservationNorm transform is still uninitialized"
                )

        parent = self.parent
        if parent is None:
            raise RuntimeError("Cannot initialize the transform if parent env is not defined.")
        parent.apply(raise_initialization_exception)

        collected_frames = 0
        data = []
        last_step_td = parent.reset()
        while collected_frames < num_steps:
            tensordict = parent.rollout(
                max_steps=num_steps_per_env_per_batch,
                auto_reset=False,
                break_when_any_done=False,
                tensordict=last_step_td,
            )
            # tensordict on the parent device, store to the storing device.
            data.append(tensordict.get(key).to(storing_device))
            collected_frames += tensordict.numel()
            last_step_td = tensordict[:, -1]

        data = torch.cat(data, cat_dim)
        if isinstance(reduce_dim, int):
            reduce_dim = [reduce_dim]
        # make all reduce_dim and keep_dims negative
        reduce_dim = sorted(dim if dim < 0 else dim - data.ndim for dim in reduce_dim)

        if keep_dims is not None:
            keep_dims = sorted(dim if dim < 0 else dim - data.ndim for dim in keep_dims)
            if not all(k in reduce_dim for k in keep_dims):
                raise ValueError("keep_dim elements must be part of reduce_dim list.")
        else:
            keep_dims = []
        loc = data.mean(reduce_dim, keepdim=True).to(parent.device)
        scale = data.std(reduce_dim, keepdim=True).to(parent.device)
        for r in reduce_dim:
            if r not in keep_dims:
                loc = loc.squeeze(r)
                scale = scale.squeeze(r)

        if not self.standard_normal:
            scale = 1 / scale.clamp_min(self.eps)
            loc = -loc * scale

        if not torch.isfinite(loc).all():
            raise RuntimeError("Non-finite values found in loc")
        if not torch.isfinite(scale).all():
            raise RuntimeError("Non-finite values found in scale")
        self.loc.materialize(shape=loc.shape, dtype=loc.dtype)
        self.loc.copy_(loc)
        self.scale.materialize(shape=scale.shape, dtype=scale.dtype)
        self.scale.copy_(scale.clamp_min(self.eps))
