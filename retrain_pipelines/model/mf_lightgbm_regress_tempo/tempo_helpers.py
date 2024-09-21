
import os
import sys
import time
import requests
import traceback

import docker

import tempo


def tempo_wait_ready(
    tempo_model: tempo.Model,
    timeout: int = 30
) -> bool:
    """
    Wait until the Docker container is up and running
    and the Seldon Uvicorn/MLServer service is available
    (exits silently if the container exited).

    Efforts were mostly put in catching loggs
    from within the docker container running the service
    when soemthing goes wrong.

    params:
        - tempo_model (tempo.Model)
        - timeout (int):
            timeout in seconds. defaults to 30s.
    """

    local_docker_client = docker.from_env(
        environment={k: v for k, v in dict(os.environ).items()
                     if k != 'REQUESTS_CA_BUNDLE'}
    )

    runtime_port = tempo_model.runtime._get_host_ip_port(
                        tempo_model.model_spec)[1]
    model_name = tempo_model.model_spec.model_details.name

    tempo_local_container = local_docker_client.containers.list(
                                all=True, filters={"name": model_name})[0]
    # print(tempo_local_container)
    # print(f"Binds : {tempo_local_container.attrs['HostConfig']['Binds']}")
    # print(f"LogPath : {tempo_local_container.attrs['LogPath']}")
    # print(f"Cmd : {tempo_local_container.attrs['Config']['Cmd']}")
    # print(f"Env : {tempo_local_container.attrs['Config']['Env']}")
    print(f"Ports : {tempo_local_container.attrs['NetworkSettings']['Ports']}")

    rdEx = None
    loops_count = 0
    start_time = time.time()
    tempo_is_ready = False

    while (
        ('running' == local_docker_client.containers.list(
                all=True, filters={"name": model_name}
            )[0].status) and
        (not tempo_is_ready)
    ):
        try:
            response = requests.get(
                f"http://localhost:{runtime_port}/v2/health/live")
            print('\n'+str(response.status_code))
            rdEx = None
            tempo_is_ready = (200 == response.status_code)
        except requests.exceptions.ConnectionError as ex:
            if not rdEx: print('waiting', end='')
            rdEx = ex
            loops_count += 1
            print('.', end='')
            time.sleep(.5)
            if (timeout < time.time() - start_time):
                # cascade breaking reason (here, print "timeout")
                print("BREAKING, TIMEOUT REACHED")
                break
    print('')

    if not tempo_is_ready:
        print(f"http://localhost:{runtime_port}/v2/health/live")
        if rdEx:
            traceback.print_tb(rdEx.__traceback__, file=sys.stdout)
        print('##### local docker container log tail BEGIN :',
              file=sys.stderr)
        print(tempo_local_container.logs(timestamps=True, tail=20
                                        ).decode('utf-8'),
              file=sys.stderr)
        print('##### local docker container log tail END.',
              file=sys.stderr)
        try:
            # try to cleanup after failure
            tempo_model.undeploy()
        except any_ex:
            pass

    return tempo_is_ready


def tempo_predict(
    tempo_model: tempo.Model,
    inference_request: dict
) -> str:
    """
    Validates request handling from Tempo endpoint.
    Simply succeeds or fails, nothing more.
    """

    runtime_port = tempo_model.runtime._get_host_ip_port(
                        tempo_model.model_spec)[1]
    print("runtime_port : " + str(runtime_port))
    model_name = tempo_model.model_spec.model_details.name
    endpoint = f"http://localhost:{runtime_port}/v2/models/{model_name}/infer"
    print("inference request being sent at : " + endpoint)
    response = requests.post(endpoint, json=inference_request)

    response_payload = None
    if 200 == response.status_code:
        response_payload = response.json()
    else:
        local_docker_client = docker.from_env(
            environment={k: v for k, v in dict(os.environ).items()
                         if k != 'REQUESTS_CA_BUNDLE'}
        )
        tempo_local_container = local_docker_client.containers.list(
                                    all=True,
                                    filters={"name": model_name})[0]
        print('##### local docker container log tail BEGIN :',
              file=sys.stderr)
        print(tempo_local_container.logs(timestamps=True, tail=20
                                        ).decode('utf-8'),
              file=sys.stderr)
        print('##### local docker container log tail END.',
              file=sys.stderr)
        raise requests.exceptions.ConnectionError(
            f"tempo response : {response.status_code}\n" +
            response.content.decode())

    return response_payload

