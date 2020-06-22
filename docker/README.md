# Run GreenGuard using Docker

GreenGuard is prepared to be run using [Docker](https://docker.com/).

This is the command needed to start a Docker container locally that runs a [Jupyter Notebook](
https://jupyter.org/) already configured to run GreenGuard.

```bash
docker run -ti -p8888:8888 signals-dev/greenguard:latest
```

Further details about the usage of this image can be found [here](
https://hub.docker.com/repository/docker/signalsdev/greenguard).

## Run GreenGuard on Kubernetes

GreenGuard can also be started using [Kubernetes](https://kubernetes.io/).

Here are the minimum steps to do so:

1. Create a POD yaml file with the these contents:

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

2. Start the POD locally

```bash
kubectl apply -f pod-file.yml
```

3. Forward the port 8888

```bash
kubectl port-forward greenguard 8888
```

4. Point your browser at http://localhost:8888

On the other hand, if you are planing to run GreenGuard on a distributed service, we provided a
[template file](
https://github.com/signals-dev/GreenGuard/blob/master/docker/greenguard-deployment.yml)
that you can use to achieve so.

## Building the Docker image from scratch

In order to build the Docker image from scratch you will need to:

1. Clone the repository

```bash
git clone git@github.com:signals-dev/GreenGuard.git
cd GreenGuard
```

2. Build the docker image

```bash
make docker-build
```

3. If you are generating a new release, you can push to Docker hub using:

```bash
make docker-push
```

## What's next?

For more details about **GreenGuard** and all its possibilities and features, please check the
[project documentation site](https://signals-dev.github.io/GreenGuard/)!
