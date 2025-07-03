import os

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg, render_networkx, render_plotly


# ---- Example: tripple chaining of tasks-groups (mostly to validate DAG rendering) ----


@task
def start():
    """Root task."""
    return


# ----


@task
def snake_head_A1(_):
    return "A1"


@task
def snake_head_A2(_):
    return "A2"


@taskgroup
def snake_heads_A():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_A1, snake_head_A2


# ----


@task
def snake_head_B1(snake_heads_A_results: TaskPayload):

    # Do whatever you want

    return "B1"


@task
def snake_head_B2(snake_heads_A_results: TaskPayload):

    # Do whatever you want

    return "B2"


@taskgroup
def snake_heads_B():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_B1, snake_head_B2


# ----


@task
def snake_head_C1(snake_heads_B_results: TaskPayload):

    # Do whatever you want

    return "C1"


@task
def snake_head_C2(snake_heads_B_results: TaskPayload):

    # Do whatever you want

    return "C2"


@taskgroup
def snake_heads_C():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_C1, snake_head_C2


# ----


@task
def join_snake_heads(snake_heads_C_results: TaskPayload):
    """Task that returns a flattened raw results
    from prior nested groups of tasks."""

    # Do whatever you want

    this_task_result = list(snake_heads_C_results.values())
    return this_task_result


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["join_snake_heads"] \
            == payload.get("join_snake_heads") \
            == payload

    assert payload == ['C1', 'C2']
    return None


# ----


# Compose the DAG using operator overloading (>>)
final = start >> snake_heads_A >> snake_heads_B >> snake_heads_C >> join_snake_heads >> end


if __name__ == "__main__":
    os.environ["RP_ARTIFACTS_STORE"] = os.path.dirname(__file__)
    # Run the DAG
    print("Final result:", execute(final, dag_params=None))
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{final.exec_id}]")
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))

