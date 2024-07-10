# Installation with conda

## Cloning the repository

Clone the git repository.

```bash
# Clone with SSH or HTTPS.
git clone git@github.com:CLAIRE-Labo/no-representation-no-trust.git no-representation-no-trust
cd no-representation-no-trust
```

We will refer the absolute path to the root of the repository as `PROJECT_ROOT`.

## Creating the environment

**Prerequisites**

- `brew`: [Homebrew](https://brew.sh/).
- `mamba` (or equivalently `conda`): we recommend [Mambaforge](https://github.com/conda-forge/miniforge).

**Installation**

System dependencies:

We list below the important system dependencies that are not available in conda,
but it is hard to list all the system dependencies needed to run the code.
We let you install the missing ones when you encounter errors.

- None.

The conda environment:

Create the environment with

```bash
# When in the PROJECT_ROOT directory.
# Tested on macOS 14.4.1.
mamba env create --file installation/conda-osx-arm64-mps/environment.yml
```

Build and install TorchRL and Tensordict from source.

```bash
mamba activate no-representation-no-trust

cd third-party
git clone https://github.com/pytorch/tensordict.git
cd tensordict
git checkout 46eef3c9a9ebd9d983820f51434f9c189b338af0
python setup.py bdist_wheel
pip install dist/tensordict*.whl

cd ..
git clone https://github.com/pytorch/rl.git
cd rl
git checkout 2cfd9b6c8d831043949ee6d2a5122791542d8723
python setup.py bdist_wheel
pip install dist/torchrl*.whl

cd ../..
```

Install the project with

```bash
# Activate the environment
mamba activate no-representation-no-trust
# When in the PROJECT_ROOT directory.
pip install -e .
```

## Running code in the environment

```bash
mamba activate no-representation-no-trust
```

Run scripts from the `PROJECT_ROOT` directory.
Here are some examples.

```bash
# When in the PROJECT_ROOT directory.
# template_experiment is an actual script that you can run.
python -m po_dynamics.template_experiment some_arg=some_value
zsh reproducibility-scripts/template-experiment.sh

# To test a training script.
python -m po_dynamics.solve
```

The environment is set up.
Return to the root README for the rest of the instructions to run our experiments.

## Maintaining the environment

System dependencies are managed by conda, otherwise when not available, by brew.
(We try to keep everything self-container as much as possible.)
Python dependencies are managed by both conda and pip.

- Use `conda` for system and non-Python dependencies needed to run the project code (e.g., image libraries, etc.).
  If not available on conda use `brew`.
- Use `conda` for Python dependencies packaged with more that just Python code (e.g. `pytorch`, `numpy`).
  These will typically be your main dependencies and will likely not change as your project grows.
- Use `pip` for the rest of the Python dependencies (e.g. `tqdm`).
- For more complex dependencies that may require a custom installation or build,
  manually follow their installation steps.

Here are references and reasons to follow the above claims:

* [A guide for managing conda + `pip` environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).
* [Reasons to use conda for not-Python-only dependencies](https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries).
* [Ways of combining conda and `pip`](https://towardsdatascience.com/conda-essential-concepts-and-tricks-e478ed53b5b#42cb).

There are two ways to add dependencies to the environment:

1. **Manually edit the `environment.yml` file.**
   This will be needed the first time you set up the environment.
   It will also be useful if you run into conflicts and have to restart from scratch.
2. **Add/upgrade dependencies interactively** while running a shell with the environment activated
   to experiment with which dependency is needed.
   This is probably what you'll be doing after creating the environment for the first time.

In both cases, after any change, a snapshot of the full environment specification should be saved.
We describe how to do so in the freeze the environment section.
Remember to commit the changes every time you freeze the environment.

### Manual editing (before/while building)

- To edit the conda and pip dependencies, edit the `environment.yml` file.
- For the `brew` and the more complex dependencies, describe the installation steps in the
  [Creating the environment](#creating-the-environment) section.

When manually editing the `environment.yml` file,
you do not need to specify the version of all the dependencies,
these will be written to the file when you freeze the environment.
You should just specify the major versions of specific dependencies you need.

After manually editing the `environment.yml` file, you need to recreate the environment.

```bash
# When in the PROJECT_ROOT directory.
mamba deactivate
mamba env remove --name no-representation-no-trust
mamba env create --file installation/conda-osx-arm64-mps/environment.yml
mamba activate no-representation-no-trust
```

### Interactively (while developing)

Conda dependencies should all be installed before any `pip` dependency.
This will cause conflicts otherwise as conda doesn't track the `pip` dependencies.
So if you need to add a conda dependency after you already installed some `pip` dependencies, you need to
manually add the dependency to the `environment.yml` file then recreate the environment.

* To add conda/pip dependencies run `(mamba | pip) install <package>`
* To add a `brew`  dependency run `brew install <package>`

### Freeze the environment

After any change to the dependencies, a snapshot of the full environment specification should be written to the
`environment.yml` file.
This includes manual changes to the file and changes made interactively.
This is to ensure that the environment is reproducible and that the dependencies are tracked at any point in time.

To do so, run the following command.
The script overwrites the `environment.yml` file with the current environment specification,
so it's a good idea to commit the changes to the environment file before and after running it.

```bash
# When in the PROJECT_ROOT directory.
zsh installation/conda-osx-arm64-mps/update-env-file.sh
```

There are some caveats (e.g., packages installed from GitHub with pip), so have a look at
the output file to make sure it does what you want.
The `update-env-file.sh` gives some hints for what to do, and in any case you can always patch the file manually.

For `brew` and more complex dependencies describe how to install them in the system dependencies section of
the [instructions to install the environment](#creating-the-environment).

If one of the complex dependencies shows in the `environment.yml` after the freeze,
you have to remove it, so that conda does not pick it up, and it is installed later by the user.

## Troubleshooting
