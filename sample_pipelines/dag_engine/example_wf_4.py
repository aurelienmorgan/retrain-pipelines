import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, taskgroup, execute, render_networkx, \
    render_plotly, render_svg


# ---- Example: Nested groups of tasks ----


@task
def start():
    """Root task."""
    return "titi"


# ----


@task
def snake_head_A(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " A"


# ----


@task
def snake_head_AA1(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA1"


@task
def snake_head_AA2(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA2"


@task
def snake_head_AA3(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA3"


@task
def snake_head_AA4(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA4"


@taskgroup
def snake_heads_AA():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_AA1, snake_head_AA2, snake_head_AA3, snake_head_AA4


# ----


@taskgroup
def snake_heads_A():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_heads_AA, snake_head_A


# ----


@task
def join_snake_heads(snake_heads_A_results):
    """Task that returns a flattened raw results
    from prior nested groups of tasks."""
    this_task_result = list(snake_heads_A_results.values())
    return this_task_result


@task
def end(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["join_snake_heads"] \
            == payload.get("join_snake_heads") \
            == payload

    assert payload == ['titi AA1', 'titi AA2', 'titi AA3', 'titi AA4', \
                       'titi A']
    return None


# ----


# Compose the DAG using operator overloading (>>)
final = start >> snake_heads_A >> join_snake_heads >> end


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

