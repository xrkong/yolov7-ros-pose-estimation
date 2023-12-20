### Pull image from DockerHub
``` docker pull dustynv/ros:foxy-pytorch-l4t-r34.1.1 ```
This image is for amd arch only which is for orin

### Build a image
``` DOCKER_BUILDKIT=1 docker build -t foxy-torch:test_entry . --ssh default ```

```DOCKER_BUILDKIT=1``` allows you pip install

```foxy-torch:test_entry``` is the target image tag. if you change the tag, you should also change the ```.env```

```---ssh default``` is for pull repo from GitHub repo.

### Start te container