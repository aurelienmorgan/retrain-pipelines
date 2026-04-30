import os

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, \
    dag, UiCss
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg


# ---- Example: Nested groups of tasks ----


@task
def start():
    """Root task."""
    import time ; time.sleep(3)                             ### DEBUG - DELETE ###
    return "titi"


# ----


@task
def snake_head_A(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " A"


# ----


@task
def snake_head_AA1(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA1"


@task
def snake_head_AA2(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA2"


@task
def snake_head_AA3(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA3"


@task
def snake_head_AA4(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " AA4"


@taskgroup
def snake_heads_AA():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_AA1, snake_head_AA2, snake_head_AA3, snake_head_AA4


# ----


@taskgroup
def snake_heads_A():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_heads_AA, snake_head_A


# ----


@task
def join_snake_heads(snake_heads_A_results: TaskPayload):
    """Task that returns a flattened raw results
    from prior nested groups of tasks."""
    this_task_result = list(snake_heads_A_results.values())
    return this_task_result


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["join_snake_heads"] \
            == payload.get("join_snake_heads") \
            == payload

    assert payload == ['titi AA1', 'titi AA2', 'titi AA3', 'titi AA4', \
                       'titi A']
    return None


# ----


@dag
def retrain_pipeline():
    """1-level deep nested taskgroups.
    """
    # Compose the DAG using operator overloading (>>)
    return start >> snake_heads_A >> join_snake_heads >> end


if __name__ == "__main__":
    # print(f"to_tasktypes_list : {retrain_pipeline.to_tasktypes_list(serializable=True)}")

    # Render the DAG
    svg_fullname = os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "dag.html"
    ))
    render_svg(retrain_pipeline, svg_fullname)

    # Run the DAG
    final_result, context_dump = execute(retrain_pipeline, params=None)
    print(
        f"execution {context_dump['exec_id']} - " +
        f"{context_dump['pipeline_name']} - final result : {final_result}"
    )
    import json
    print("context_dump : " +
          json.dumps(context_dump, indent=4))
    print(f"DAG SVG written to {svg_fullname}")

