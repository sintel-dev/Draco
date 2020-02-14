# Docker Usage

**GreenGuard** comes configured and ready to be distributed and run as a docker image which starts
a jupyter notebook already configured to use greenguard, with all the required dependencies already
installed.

## Requirements

The only requirement in order to run the GreenGuard Docker image is to have Docker installed and
that the user has enough permissions to run it.

Installation instructions for any possible system compatible can be found [here](https://docs.docker.com/install/)

Additionally, the system that builds the GreenGuard Docker image will also need to have a working
internet connection that allows downloading the base image and the additional python depenedencies.

## Building the GreenGuard Docker Image

After having cloned the **GreenGuard** repository, all you have to do in order to build the GreenGuard Docker
Image is running this command:

```bash
make docker-jupyter-build
```

After a few minutes, the new image, called `greenguard-jupyter`, will have been built into the system
and will be ready to be used or distributed.

## Distributing the GreenGuard Docker Image

Once the `greenguard-jupyter` image is built, it can be distributed in several ways.

### Distributing using a Docker registry

The simplest way to distribute the recently created image is [using a registry](https://docs.docker.com/registry/).

In order to do so, we will need to have write access to a public or private registry (remember to
[login](https://docs.docker.com/engine/reference/commandline/login/)!) and execute these commands:

```bash
docker tag greenguard-jupyter:latest your-registry-name:some-tag
docker push your-registry-name:some-tag
```

Afterwards, in the receiving machine:

```bash
docker pull your-registry-name:some-tag
docker tag your-registry-name:some-tag greenguard-jupyter:latest
```

### Distributing as a file

If the distribution of the image has to be done offline for any reason, it can be achieved
using the following command.

In the system that already has the image:

```bash
docker save --output greenguard-jupyter.tar greenguard-jupyter
```

Then copy over the file `greenguard-jupyter.tar` to the new system and there, run:

```bash
docker load --input greenguard-jupyter.tar
```

After these commands, the `greenguard-jupyter` image should be available and ready to be used in the
new system.


## Running the greenguard-jupyter image

Once the `greenguard-jupyter` image has been built, pulled or loaded, it is ready to be run.

This can be done in two ways:

### Running greenguard-jupyter with the code

If the GreenGuard source code is available in the system, running the image is as simple as running
this command from within the root of the project:

```bash
make docker-jupyter-run
```

This will start a jupyter notebook using the docker image, which you can access by pointing your
browser at http://127.0.0.1:8888

In this case, the local version of the project will also mounted within the Docker container,
which means that any changes that you do in your local code will immediately be available
within your notebooks, and that any notebook that you create within jupyter will also show
up in your `notebooks` folder!

### Running greenguard-jupyter without the greenguard code

If the GreenGuard source code is not available in the system and only the Docker Image is, you can
still run the image by using this command:

```bash
docker run -ti -p8888:8888 greenguard-jupyter
```

In this case, the code changes and the notebooks that you create within jupyter will stay
inside the container and you will only be able to access and download them through the
jupyter interface.
