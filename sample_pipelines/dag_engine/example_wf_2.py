import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, taskgroup, execute, render_networkx, \
    render_plotly, render_svg


# ---- Example: Group of tasks ----


@task
def start():
    # Root task
    return "titi"


@task
def snake_head_1(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " snake_head_1"


@task
def snake_head_2(payload):

    # Do whatever you want

    return payload["start"] + " snake_head_2"


@task
def snake_head_3(payload):

    # Do whatever you want

    return payload["start"] + " snake_head_3"


@taskgroup
def snake_heads():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_2, snake_head_1, snake_head_3


@task()
def join_snake_heads(payload):
    print(type(payload))
    # Concat results (e.g. put them into a list)
    this_task_result = [
        payload["snake_head_1"],
        payload["snake_head_2"],
        payload["snake_head_3"]
    ]
    return this_task_result


@task
def end(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["join_snake_heads"] \
            == payload.get("join_snake_heads") \
            == payload

    assert payload == ['titi snake_head_1', 'titi snake_head_2', \
                       'titi snake_head_3']
    return None


# Compose the DAG using operator overloading (>>)
final = start >> snake_heads >> join_snake_heads >> end


if __name__ == "__main__":
    os.environ["RP_ARTIFACTS_STORE"] = os.path.dirname(__file__)
    # Run the DAG
    print("Final result:", execute(final))
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{final.exec_id}]")
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))

