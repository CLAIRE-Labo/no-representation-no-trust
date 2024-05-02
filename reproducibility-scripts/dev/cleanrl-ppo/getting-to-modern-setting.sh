# Baseline at 100M
python -m cleanrl.ppo_atari_1model --exp_name baseline
python -m cleanrl.ppo_atari_2models --exp_name baseline_2models

# Change the env
python -m cleanrl.ppo_atari_1model --env_id ALE/Phoenix-v5 --frameskip 3 --repeat_action_probability 0.25 --no-noop_reset --no-episodic_life --no-fire_reset --exp_name change-env
python -m cleanrl.ppo_atari_2models --env_id ALE/Phoenix-v5 --frameskip 3 --repeat_action_probability 0.25 --no-noop_reset --no-episodic_life --no-fire_reset --exp_name change-env-2models

# Change the optim
python -m cleanrl.ppo_atari_1model --no-layer_init --no-anneal_lr --no-clip_vloss --adam_eps 1e-8 --vf_coef 1 --exp_name change-optim
python -m cleanrl.ppo_atari_2models -no-layer_init --no-anneal_lr --no-clip_vloss --adam_eps 1e-8 --vf_coef 1 --exp_name change-optim-2models


# Change the env and the optim
# Same setting as ours
python -m cleanrl.ppo_atari_1model --env_id ALE/Phoenix-v5 --frameskip 3 --repeat_action_probability 0.25 --no-noop_reset --no-episodic_life --no-fire_reset --no-layer_init --no-anneal_lr --no-clip_vloss --adam_eps 1e-8 --vf_coef 1 --exp_name change-env-optim
python -m cleanrl.ppo_atari_2models --env_id ALE/Phoenix-v5 --frameskip 3 --repeat_action_probability 0.25 --no-noop_reset --no-episodic_life --no-fire_reset --no-layer_init --no-anneal_lr --no-clip_vloss --adam_eps 1e-8 --vf_coef 1 --exp_name change-env-optim-2models

# Reproduce collapse
python -m cleanrl.ppo_atari_2models --env_id ALE/Phoenix-v5 --frameskip 3 --update_epochs 8 --repeat_action_probability 0.25 --no-noop_reset --no-episodic_life --no-fire_reset --no-layer_init --no-anneal_lr --no-clip_vloss --adam_eps 1e-8 --vf_coef 1 --exp_name change-env-optim-2models
