
import sys
import time
import subprocess
from datetime import datetime

import docker


def print_container_log_tail(
    container_name: str,
    tail_length: int
):
    container = docker.from_env().containers.list(
        all=True, filters={"name": container_name})[0]
    print('##### local docker container log tail BEGIN :',
          file=sys.stderr)
    print(container.logs(timestamps=True, tail=tail_length
                        ).decode('utf-8'),
          file=sys.stderr)
    print('##### local docker container log tail END.',
          file=sys.stderr)


def build_and_run_docker(
    image_name: str,
    image_tag: str,
    build_path: str = ".",
    dockerfile: str = "Dockerfile",
    ports_publish_dict: dict = {},
    env_vars_dict = {},
    volumes_dict = {}
) -> bool:
    """
    Spins a container from build path.

    Parmas:
        - image_name (str):
            name to give to the image built
        - image_tag (str):
            name to assign to the image built
        - build_path (str):
            defaults to "."
        - dockerfile (str):
            defaults to "Dockerfile"
        - ports_publish_dict (dict):
            Docker run port publish param value.
            Used to publish container's port(s)
            to the host machine.
            can for instance be {'8080/tcp': 8080}}
        - env_vars_dict (dict):
            Environment variables to set inside
            the container.
        - volumes_dict (dict):
            Used to configure volumes mounted
            inside the container.
            The key is either the host path or a
            volume name, and the value is
            a dictionary with the keys:
                - ``bind`` The path to mount the
                  volume inside the container.
                - ``mode`` Either ``rw`` to mount
                  the volume read/write, or
                  ``ro`` to mount it read-only.
            Example :
                {'/home/user1/': {'bind': '/mnt/vol2',
                                  'mode': 'rw'},
                 '/var/www': {'bind': '/mnt/vol1',
                              'mode': 'ro'}}

    Results:
        - (bool):
            success/failure
    """

    docker_client = docker.from_env()
    full_image_name = f"{image_name}:{image_tag}"

    # Build the Docker image
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{timestamp} Building Docker image...")
    build_success = True
    try:
        build_response = docker_client.api.build(
            path=build_path,
            dockerfile=dockerfile,
            tag=full_image_name,
            rm=True,
            forcerm=True,
            # squash =True, # experimental ; unstable, connection reset
            decode=True,
            network_mode="host"
        )

        for chunk in build_response:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if 'stream' in chunk:
                print(f"{timestamp} {chunk['stream'].strip()}")
            elif 'error' in chunk:
                build_success = False
                print(f"{timestamp} Error: {chunk['error']}")
            elif 'aux' in chunk:
                print(f"{timestamp} Auxiliary: {chunk['aux']}")

    except docker.errors.BuildError as e:
        build_success = False
        print(f"BuildError: {e}")
    except docker.errors.APIError as e:
        build_success = False
        print(f"APIError: {e}")
    except Exception as e:
        build_success = False
        print(f"Unexpected error: {e}")

    if not build_success:
            return False

    # Run the Docker container
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{timestamp} Running Docker container...")
    try:
        container = docker_client.containers.run(
            full_image_name,
            remove=True,
            detach=True,
            ports=ports_publish_dict,
            environment=env_vars_dict,
            volumes=volumes_dict,
            name=image_name,
            runtime='nvidia',
            # gpus='all'
        )
    except Exception as ex:
        print(ex)
        return False

    # await
    done_created = False
    while not done_created:
        # refresh "container"
        # (it is a snapshot, not a pointer to the living thing)
        container = docker_client.containers.list(
            all=True, filters={"name": image_name})[0]
        done_created = "created" != container.status.lower()
    print(f"{timestamp} Docker container {image_name}" +
          " started successfully " +
          f"({container.id[:12]}, {container.status}).")

    return True


def cleanup_docker(
    container_name: str,
    image_name: str = None
):
    """
    Params:
        - container (str):
        - image_name (str):
            Name of the image to remove from Docker
            (if specified).
    """

    docker_client = docker.from_env()

    container = docker_client.containers.list(
        all=True, filters={"name": container_name})[0]
    if container is None:
        print(f"no {container_name} container found.")
    else:
        # Stop the Docker container
        print(f"Stopping Docker container {container.name}...")
        container.stop()
        stopped = False
        while not stopped:
            # refresh "container"
            # (it is a snapshot, not a pointer to the living thing)
            container = docker_client.containers.list(
                all=True, filters={"name": container_name})[0]
            stopped = "running" != container.status.lower()

        print(f"Docker container {container.name} stopped.")
        print(container)

    # Remove the Docker image
    print(f"Removing Docker image {container_name}...")
    if image_name is not None:
        subprocess.run(['docker', 'rmi', image_name], check=True)
        print(f"Docker image {container_name} removed.")

