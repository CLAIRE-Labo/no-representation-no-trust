# Reproducibility Scripts

##  Limitations

Runs are reproducible on the same platform and device where they were run.
We ran all our experiments with the `docker-amd64-cuda` platform but with two different devices (V100s and A100s).
So to obtain the same curve you have to check on WandB which device was used to run the experiment and use the same device to reproduce the results.
In any case, our work should be replicable, giving the same trend with enough seeds, so getting the same exact curves is not necessary.

On macOS, the TorchRL version we use does not give reproducible runs with ParallelEnv.
To get reproducible runs, you have to use SerialEnv.
We only ran on macOS during development.

## Experiments

Contents of this directory:

```text
.
├── release   # All runs in in this project are wandb sweeps. Names are self-explanatory.
│   ├── atari-ppo
│   │   ├── baselines.yaml
│   │   ├── capacity-all.yaml
│   │   ├── control-optimizer.yaml
│   │   ├── control-regularize-all-layers.yaml
│   │   ├── control-regularize.yaml
│   │   ├── control-shared-trunk.yaml
│   │   └── get-all-runs-for-capacity.sh
│   └── mujoco-ppo
│       ├── baselines.yaml
│       ├── capacity-all.yaml
│       ├── control-optimizer.yaml
│       ├── control-regularize-relu.yaml
│       ├── control-regularize-tanh.yaml
│       ├── control-shared-trunk.yaml
│       └── get-all-runs-for-capacity.sh
├── dev
│   └── cleanrl-ppo     # Scripts to compare our implementation with CleanRL's
│        ├── getting-to-modern-setting.sh    # Scripts to run cleanrl on a modern research setting (as used in our work) to have an idea of the performance difference.
│        └── replicate-with-cleanrl.sh       # Scripts to replicate collapse with cleanrl.
└── plotting.ipynb  # Jupyter notebook to plot the figures.
```
