# new sweeps:

# What the team uses:
# Atari - training
runai submit \
  --name no-representation-no-trust-atari \
  --image registry.rcp.epfl.ch/claire/moalla/no-representation-no-trust:run-latest-moalla \
  --pvc runai-claire-moalla-scratch:/claire-rcp-scratch \
  -e PROJECT_ROOT_AT=/claire-rcp-scratch/home/moalla/no-representation-no-trust/run \
  -e WANDB_API_KEY_FILE_AT=/claire-rcp-scratch/home/moalla/.wandb-api-key-anonymous \
  -e WANDB_CONSOLE=off \
  -e WANDB_DISABLE_GIT=1 \
  -e OMP_NUM_THREADS=1 \
  --cpu 10 -g 1 --large-shm --node-type G9 \
  -- something

# Atari - capacity
runai submit \
  --name no-representation-no-trust-atari-capacity \
  --image registry.rcp.epfl.ch/claire/moalla/no-representation-no-trust:run-latest-moalla \
  --pvc runai-claire-moalla-scratch:/claire-rcp-scratch \
  -e PROJECT_ROOT_AT=/claire-rcp-scratch/home/moalla/no-representation-no-trust/run \
  -e WANDB_API_KEY_FILE_AT=/claire-rcp-scratch/home/moalla/.wandb-api-key-anonymous \
  -e WANDB_CONSOLE=off \
  -e WANDB_DISABLE_GIT=1 \
  -e OMP_NUM_THREADS=4 \
  --cpu 6 -g 1 --large-shm \
  -- something

# For MuJoCo rendering use -e DISPLAY=:99
# MuJoCo - training
runai submit \
  --name no-representation-no-trust-mujoco \
  --image registry.rcp.epfl.ch/claire/moalla/no-representation-no-trust:run-latest-moalla \
  --pvc runai-claire-moalla-scratch:/claire-rcp-scratch \
  -e PROJECT_ROOT_AT=/claire-rcp-scratch/home/moalla/no-representation-no-trust/run \
  -e WANDB_API_KEY_FILE_AT=/claire-rcp-scratch/home/moalla/.wandb-api-key-anonymous \
  -e WANDB_CONSOLE=off \
  -e WANDB_DISABLE_GIT=1 \
  -e OMP_NUM_THREADS=4 \
  --cpu 48 --node-type S8 --large-shm \
  -- zsh reproducibility-scripts/utils/run-in-parallel.sh 8 wandb agent ...


# MuJoCo - capacity
runai submit \
  --name no-representation-no-trust-mujoco-capacity \
  --image registry.rcp.epfl.ch/claire/moalla/no-representation-no-trust:run-latest-moalla \
  --pvc runai-claire-moalla-scratch:/claire-rcp-scratch \
  -e PROJECT_ROOT_AT=/claire-rcp-scratch/home/moalla/no-representation-no-trust/run \
  -e WANDB_API_KEY_FILE_AT=/claire-rcp-scratch/home/moalla/.wandb-api-key-anonymous \
  -e WANDB_CONSOLE=off \
  -e WANDB_DISABLE_GIT=1 \
  -e OMP_NUM_THREADS=4 \
  --cpu 20 -g 1 --large-shm \
  -- something

# Simple examples.
runai submit \
  --name example-unattended \
  --image registry.rcp.epfl.ch/claire/moalla/no-representation-no-trust:run-latest-moalla \
  --pvc runai-claire-moalla-scratch:/claire-rcp-scratch \
  -e PROJECT_ROOT_AT=/claire-rcp-scratch/home/moalla/no-representation-no-trust/run \
  -- python -m po_dynamics.template_experiment some_arg=2

# template_experiment is an actual script that you can run.
# or -- zsh po_dynamics/reproducibility-scripts/template-experiment.sh

# To separate the dev state of the project from frozen checkouts to be used in unattended jobs you can observe that
# we're pointing to the .../run instance of the repository on the PVC.
# That would be a copy of the no-representation-no-trust repo frozen in a commit at a working state to be used in unattended jobs.
# Otherwise while developing we would change the code that would be picked by newly scheduled jobs.

# Useful commands.
# runai describe job example-unattended
# runai logs example-unattended
