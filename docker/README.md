# Run GreenGuard using Docker

GreenGuard is prepared to be run using [Docker](https://docker.com/).

These are the commands needed to start a Docker container locally that runs a [Jupyter Notebook](
https://jupyter.org/) already configured to run GreenGuard.

```bash
docker run -ti -p8888:8888 signalsdev/greenguard:latest
```

This will start a Jupyter Notebook instance on your computer already configured to use GreenGuard.
You can access it by pointing your browser at http://127.0.0.1:8888

Further details about the usage of this image can be found [here](
https://hub.docker.com/repository/docker/signalsdev/greenguard).

## Run GreenGuard on Kubernetes

GreenGuard can also be started using [Kubernetes](https://kubernetes.io/).

Here are the minimum steps required to create a POD in a local Kubernetes cluster:

1. Create a yaml file with these contents:

For this example, we are assuming that the yaml file is named `greegunard-pod.yml`.

```yml
apiVersion: v1
kind: Pod
metadata:
  name: greenguard
spec:
  containers:
  - name: greenguard
    image: signalsdev/greenguard:latest
    ports:
    - containerPort: 8888
```

2. Create a POD:

After creating the yaml file, you can create a POD in your Kubernetes cluster using the `kubectl`
command:

```bash
kubectl apply -f greenguard-pod.yml
```

3. Forward the port 8888

After the POD is started, you still need to forward a local port to it in order to access the
Jupyter instance.

```bash
kubectl port-forward greenguard 8888
```

4. Point your browser at http://localhost:8888

> **NOTE**: If GreenGuard is run in a production environment we recommend you to use a service and
a deployment instead of just a simple POD. You can find a template of this setup [here](
greenguard-deployment.yml)

## Building the Docker image from scratch

If you want to build the Docker image from scratch instead of using the dockerhub image
you will need to:

1. Clone the repository

```bash
git clone git@github.com:sintel-dev/GreenGuard.git
cd GreenGuard
```

2. Build the docker image using the GreenGuard make command.

```bash
make docker-build
```

## What's next?

For more details about **GreenGuard** and all its possibilities and features, please check the
[project documentation site](https://sintel-dev.github.io/GreenGuard/)!
