# Docker Usage

**GreenGuard** comes configured and ready to be distributed and run as a docker image which starts
a jupyter notebook already configured to use greenguard, with all the required dependencies already
installed.

## Docker Requirements

The only requirement in order to run the GreenGuard Docker image is to have Docker installed and
that the user has enough permissions to run it.

Installation instructions for any possible system compatible can be found [here](
https://docs.docker.com/install/).

Additionally, the system that builds the GreenGuard Docker image will also need to have a working
internet connection that allows downloading the base image and the additional python depenedencies.

## Building the GreenGuard Docker Image

After having cloned the **GreenGuard** repository, all you have to do in order to build the
GreenGuard Docker Image is running this command:

```bash
make docker-build
```

After a few minutes, the new image, called `greenguard`, will have been built into the system
and will be ready to be used or distributed.

## Distributing the GreenGuard Docker Image

Once the `greenguard` image is built, it can be distributed in several ways.

### Distributing using a Docker registry

The simplest way to distribute the recently created image is [using a registry](
https://docs.docker.com/registry/).

In order to do so, we will need to have write access to a public or private registry (remember to
[login](https://docs.docker.com/engine/reference/commandline/login/)!) and execute these commands:

```bash
docker tag greenguard:latest your-registry-name:some-tag
docker push your-registry-name:some-tag
```

Afterwards, in the receiving machine:

```bash
docker pull your-registry-name:some-tag
docker tag your-registry-name:some-tag greenguard:latest
```

### Distributing as a file

If the distribution of the image has to be done offline for any reason, it can be achieved
using the following command.

In the system that already has the image:

```bash
docker save --output greenguard.tar greenguard
```

Then copy over the file `greenguard.tar` to the new system and there, run:

```bash
docker load --input greenguard.tar
```

After these commands, the `greenguard` image should be available and ready to be used in the
new system.


## Running the greenguard image

Once the `greenguard` image has been built, pulled or loaded, it is ready to be run.

This can be done in two ways:

### Running greenguard with the code

If the GreenGuard source code is available in the system, running the image is as simple as running
this command from within the root of the project:

```bash
make docker-run
```

This will start a jupyter notebook using the docker image, which you can access by pointing your
browser at http://127.0.0.1:8888

In this case, the local version of the project will also mounted within the Docker container,
which means that any changes that you do in your local code will immediately be available
within your notebooks, and that any notebook that you create within jupyter will also show
up in your `notebooks` folder!

### Running greenguard without the greenguard code

If the GreenGuard source code is not available in the system and only the Docker Image is, you can
still run the image by using this command:

```bash
docker run -ti -p 8888:8888 greenguard
```

In this case, the code changes and the notebooks that you create within jupyter will stay
inside the container and you will only be able to access and download them through the
jupyter interface.

## Running the greenguard image on kubernetes

### Running as pod

There is a possiblity to run GreenGuard's docker image on a local kubernetes cluster. Once you have
created the docker image (locally or remotely) and you have [kubernetes](
https://kubernetes.io/docs/home/) properly setup at your local environment, copy and paste the
following pod configuration into a `yml` file:

```yml
apiVersion: v1
kind: Pod
metadata:
  name: greenguard
spec:
  containers:
  - name: greenguard
    image: signals-dev/greenguard-jupyter:0.2.2.dev0
    ports:
    - containerPort: 8888
```

**Note** If you would like to use your local image that you created previously, or an image
from another repository that's not the official one, change the `image` value to the one that
corresponds to yours.

Once you have created the `yml` file, you can run the following command to launch the pod:

```bash
kubectl apply -f file.yml
```

This will create a pod named `greenguard` and in order to access it, we will have to forward
the port 8888 from the pod to our localhost. To do so, just run the following command:

```bash
kubectl port-forward greeguard 8888
```

Finally we can point our browser to http://localhost:8888 and use the GreenGuard software.

### Running GreenGuard a service

Kubernetes allows the posibility to run a docker image as a services, inside this folder you
will find a `greenguard-deployment.yml` file, ready to use as an deployment service, which has
the port forwarded to the `30088`. You can use this template to adapt it to your needs.

## What's next?

For more details about **GreenGuard** and all its possibilities and features, please check the
[project documentation site](https://signals-dev.github.io/GreenGuard/)!
