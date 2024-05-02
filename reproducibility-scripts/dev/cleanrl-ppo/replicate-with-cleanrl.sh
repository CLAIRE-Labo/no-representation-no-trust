for epochs in 4 6 8; do
  for seed in 1 2 3 4 5; do
     python -m cleanrl.ppo_atari_2models --seed $seed --update_epochs $epochs --env_id ALE/Phoenix-v5 --frameskip 3 --repeat_action_probability 0.25 --no-noop_reset --no-episodic_life --no-fire_reset --no-layer_init --no-anneal_lr --no-clip_vloss --adam_eps 1e-8 --vf_coef 1 --wandb_project_name no-representation-no-trust-replicate-with-cleanrl
  done
done


for epochs in 4 6 8; do
  for seed in 25 7 64 27 4; do
    python -m po_dynamics.solve env=gym-atari env.name=ALE/Phoenix-v5 optim.num_epochs=$epochs seed=$seed wandb.mode=online
  done
done
