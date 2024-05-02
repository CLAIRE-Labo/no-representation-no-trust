# Installation with Docker (OCI container)

## The environment

We provide the following guides for obtaining/building and running the environment:

- To run the image locally with Docker & Docker Compose, follow the instructions
  to [obtain/build the environment](#obtainingbuilding-the-environment) then
  the instructions [run locally with Docker Compose](#running-locally-with-docker-compose).
- To run on the EPFL Run:ai clusters, follow the instructions
  to [obtain/build the environment](#obtainingbuilding-the-environment)
  (perform them on your local machine)
  then refer to the `./EPFL-runai-setup/README.md`.

  The guide also provides instructions to do remote development on the Run:ai cluster.
  Other managed cluster users can get inspiration from it too.
- We also provide an image with the dependencies needed to run the environment
  that you can use with your favorite OCI-compatible container runtime.
  Follow the instructions
  in [Running with your favorite container runtime](#running-with-your-favorite-container-runtime) for the details.

## Obtaining/building the environment

### Prerequisites

* `docker` (`docker version` >= v23). [Install here.](https://docs.docker.com/engine/)
* `docker compose` (`docker compose version` >= v2). [Install here.](https://docs.docker.com/compose/install/)

### Clone the repository

Clone the git repository.

```bash
# Clone with SSH or HTTPS.
git clone git@github.com:CLAIRE-Labo/no-representation-no-trust.git no-representation-no-trust
cd no-representation-no-trust
```

### Obtain/build the images

All commands should be run from the `installation/docker-amd64-cuda/` directory.

```bash
cd installation/docker-amd64-cuda
```

1. Create an environment file for your personal configuration with
   ```bash
   ./template.sh env
   ```
   This creates a `.env` file with pre-filled values.
    - The `USRID` and `GRPID` are used to give the container user read/write access to the volumes that will be mounted
      when the container is run, containing the code of the project, the data, and where you'll write your outputs.
      Edit them so that they match the user permissions on the mounted volumes.
      (If you're deploying locally, i.e., where you're building, these values should be filled correctly by default.)

      (**EPFL Note:** _These should match the permissions on your lab's shared storage when mounting from there
      and running on some shared infrastructure, like HaaS setup with LDAP login or Run:ai.
      They will typically be your GASPAR credentials.
      CLAIRE members should use the `claire-storage` group._)
    - `LAB_NAME` will be the first element in name of the local images you get.

      (**EPFL Note:** _If pushing to the IC or RCP registries this should be the name of your lab's project
      in the registry.
      CLAIRE members should use `claire`._)
    - You can ignore the rest of the variables after `## For running locally`.
      These don't influence the build, they will be used later to run your image.

2. Pull or build the generic images.
   These are the runtime (`run`) and development (`dev`) images with root as user.
   The runtime images will be used to run the code in an unattended way.
   The dev image has additional utilities that facilitate development in the container.
   They will be named according to the image name in your `.env`.
   They will be tagged with `run-latest-root` and `dev-latest-root` and if you're building them,
   they will also be tagged with the latest git commit hash `run-<sha>-root` and `dev-<sha>-root`.
    - Pull the generic images if they're available.
      ```bash
      # Pull the generic image if available.
      # For EPFL
      ./template.sh pull_generic registry.rcp.epfl.ch/claire/moalla/no-representation-no-trust
      # Outside EPFL
      ./template.sh pull_generic docker.io/skandermoalla/no-representation-no-trust
      ````
    - Otherwise, build them.
      ```bash
      ./template.sh build_generic
      ```
3. You can run quick checks on the image to check it that it has what you expect it to have:
   ```bash
   # Check all your dependencies are there.
   ./template.sh list_env

    # Get a shell and check manually other things.
    # This will only contain the environment and not the project code.
    # Project code can be debugged on the cluster directly.
    ./template.sh empty_interactive
   ```

4. Build the images configured for your user.
   ```bash
   ./template.sh build_user
   ```
   This will build a user layer on top of each generic image
   and tag them `*-*-${USR}` instead of `*-*-root`.
   These will be the images that you actually run and deploy to match the permissions on your mounted storage.

For the local deployment option with Docker Compose, follow the instructions below,
otherwise get back to the instructions of the deployment option you're following.

## Running locally with Docker Compose

**Prerequisites**

Steps prefixed with [CUDA] are only required to use NVIDIA GPUs.

* `docker` (`docker version` >= v23). [Install here.](https://docs.docker.com/engine/)
* `docker compose` (`docker compose version` >= v2). [Install here.](https://docs.docker.com/compose/install/)
* [CUDA] [Nvidia CUDA Driver](https://www.nvidia.com/download/index.aspx) (Only the driver. No CUDA toolkit, etc.)
* [CUDA] `nvidia-docker` (the NVIDIA Container
  Toolkit). [Install here.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

**Run**

Edit the `.env` file to specify which hardware acceleration to use with the `ACCELERATION` variable.
Supported values are `cpu` and `cuda`.

Then you can:

- Start the development container with
    ```bash
    ./template.sh up
    ```
  This will start a container running the development image in the background.
  It has an entrypoint that installs the project, checking that the code directory has correctly been mounted.
  The Docker Compose run and dev services are already setup to mount the project code and specify its location
  to the entrypoint.

  You can check its logs with
    ```bash
    ./template.sh logs
    ```
  and open a shell in this background container with
    ```bash
    ./template.sh shell
    ```
  You can stop the container or delete it with
    ```bash
    # To stop.
    ./template.sh stop
    # Which can then be restarted with
    ./template.sh start
    # Or, to delete.
    ./template.sh down
    ```

- Run jobs in independent containers running the runtime image with
    ```bash
    # You can for example open tmux shells and run your experiments in them.
    # template_experiment is an actual script that you can run.
    ./template.sh run your_command
    ./template.sh run python --version
    ./template.sh run python -m po_dynamics.template_experiment some_arg=some_value
    ```
  These containers start with the entrypoint and then run the command you specified.
  By default, they are automatically removed after they exit.

  You should not need to override the entrypoint of the container, it performs important setups.
  It installs the project from its mounted location when specified to avoid hacky imports,
  runs the original entrypoint of your base image if it exists,
  and execs your command with PID 1.
  Only do so if you need to debug the entrypoint itself or if you have a custom use case.

  You can read the following section for quick tips on development with containers,
  then return to the root README for the rest of the instructions to run our experiments.


### Development

For remote development with this Docker Compose setup, you should have your IDE
running on the machine where you run the Docker Compose services (not inside the container),
E.g. Pycharm Remote Development or VSCode Remote Development.

Then you would use the remote development features of this IDE to connect to the container
through Docker Compose with `dev-local-${ACCELERATION}` service, if the IDE allows,
which has the mount set up to the code directory.
Otherwise, through the image directly and you'll have to add the mount yourself
(look at how this is done in `compose.yaml`).

## Running with your favorite container runtime

An image with the runtime environment and an image with the development environment (includes shell utilities)
both running as root (but with a configured zshell for users specified at runtime as well)
is available at `docker.io/skandermoalla/no-representation-no-trust`

The tags are `run-latest-root` and `dev-latest-root` for the runtime and development images respectively.
You can use your favorite container runtime to run these images.

They have an entrypoint which installs the project with pip
and expects it to be mounted in the container and its location specified with the
environment variable `PROJECT_ROOT_AT`.
E.g., you can mount it at `/project/no-representation-no-trust` and specify `PROJECT_ROOT_AT=/project/no-representation-no-trust`.
The entrypoint can then take any command to run in the container and will run it with PID 1.
(If you don't specify the `PROJECT_ROOT_AT`, the entrypoint will skip the project installation and warn you about it.)

You can refer to the `EPFL-runai-setup/README.md` for an idea of how this would work on a Kubernetes cluster
interfaced with Run:ai.

Return to the root README for the rest of the instructions to run our experiments.

## Instructions to maintain the environment

The environment is based on an image which already contains system and Python dependencies.
Extra dependencies are managed as follows:

System dependencies are managed by `apt`.
Python dependencies are managed by `pip`.

Complex dependencies that may require a custom installation
should have their instructions performed in the `Dockerfile` directly.

There are two ways to add dependencies to the environment:

1. **Manually edit the dependency files.**
   This will be needed the first time you set up the environment.
   It will also be useful if you run into conflicts and have to restart from scratch.
2. **Add/upgrade dependencies interactively** while running a shell in the container to experiment with which
   dependency is needed.
   This is probably what you'll be doing after building the image for the first time.

In both cases, after any change, a snapshot of the full environment
specification should be written to the dependency files.
We describe how to do so in the Freeze the Environment section.

### Manual editing (before/while building)

- To add `apt` dependencies, edit the `dependencies/apt-*.txt` files.
  `apt` dependencies are separated into three files to help with multi-stage builds and keep final images small.
    - In `apt-build.txt` put the dependencies needed to build the environment, e.g., compilers, build tools, etc.
      We provide a set of minimal dependencies as an example.
    - In `apt-runtime.txt` put the dependencies needed to run the environment, e.g., image processing libraries.
    - In `apt-dev.txt` put the utilities that will help you develop in the container, e.g. `htop`, `vim`, etc.

  If you're not familiar with which dependencies are needed for each stage, you can start with the minimal set we
  give, and when you encounter errors during the image build, add the missing dependencies to the stage where the error
  occurred.
- To edit `pip` dependencies, edit the `dependencies/requirements.txt` file.
- To edit the more complex dependencies, edit the `Dockerfile`.

When manually editing the dependency files,
you do not need to specify the specific version of all the dependencies,
these will be written to the file when you freeze the environment.
You should just specify the major versions of specific dependencies you need.

### Interactively (while developing)

* To add `apt`  dependencies run `sudo apt install <package>`
* To add `pip` dependencies run `pip install <package>`

### Freeze the environment

After any change to the dependencies, a snapshot of the full environment specification should be written to the
dependency files.
This includes changes during a build and changes made interactively.
This is to ensure that the environment is reproducible and that the dependencies are tracked at any point in time.

To do so, run the following from a login shell in the container.
The script overwrites the `dependencies/requirements.txt` file with the current environment specification,
so it's a good idea to commit the changes to the environment file before/after running it.

The script isn't just a `pip freeze` and the file it generates isn't made to recreate the environment from scratch,
it is tightly coupled to the Dockerfile and the base image it uses.
In this sense, packages that are already installed in the base image or installed by the Dockerfile
may not be listed in the file or may be listed without a version
(this is because that may have been installed from wheels not present anymore in the final image).

The purpose of the generated `requirements.txt` is to be used always at the same stage of the Dockerfile
to install the same set of missing dependencies between its previous stage and its next stage.
(so not reinstall the dependencies already installed in the base image, for example).
In any case,
the Dockerfile also records the snapshots of the dependency files used to generate each stage for debugging that can be
found in the `/opt/template-dependencies/` directory.

```bash
update-env-file
```

The script isn't perfect, and there are some caveats (e.g., packages installed from GitHub with pip),
so have a look at the output file to make sure it does what you want.
The `dependencies/update-env-file.sh` gives some hints for what to do,
and in any case you can always patch the file manually.

For dependencies that require a custom installation or build, edit the `Dockerfile`.
If one of these complex dependencies shows in the `requirements.txt` after the freeze,
you have to remove it, so that pip does not pick it up, and it is installed independently in the `Dockerfile`.
(Something similar is done in the `update-env-file`.)

For `apt` dependencies add them manually to the `apt-*.txt` files.

## Troubleshooting

### Debugging the Docker build

If your build fails at some point, the build will print the message with the line in the Dockerfile
that caused the error.
Identify the stage at in which the line is: it's the earliest FROM X as Y before the line.
Then add a new stage right before the failing line starting from the stage you identified.
Something like:

```dockerfile
FROM X as Y

RUN something-that-works

# Add this line.
FROM Y as debug

RUN something-that-breaks
```

Then in the `compose.yaml` file, change the `target: runtime-generic` to `target: Y`
(replacing Y with its correct stage name).
Your build will then stop at the line before the failing line.

```bash
# Say you're building the generic images.
./template.sh build_generic
```

You can open a shell in that layer and debug the issue.

```bash
# IMAGE_NAME can be found in the .env file.
docker run --rm -it --entrypoint /bin/bash ${IMAGE_NAME}:run-latest-root
```

### My image doesn't build with my initial dependencies.

Try removing the dependencies causing the issue, rebuilding, and then installing them interactively when running the
container.
The error messages will possibly be more informative, and you will be able to dig into the issue.

Alternatively, you can open a container at the layer before the installation of the dependencies,
like described above, and try to install the environment manually.

## Acknowledgements

This Docker setup is based on the [Cresset template](https://github.com/cresset-template/cresset).
