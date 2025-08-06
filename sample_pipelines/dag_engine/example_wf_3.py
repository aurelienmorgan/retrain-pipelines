import os

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, \
    dag, UiCss
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg


# ---- Example: Several groups of tasks, chained ----


@task
def start():
    """Root task."""
    return


@task
def snake_head_A1(_):

    # Do whatever you want

    return "A1"


@task
def snake_head_A2(_):

    # Do whatever you want

    return "A2"


@task
def snake_head_A3(_):

    # Do whatever you want

    return "A3"


@taskgroup
def snake_heads_A():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_A1, snake_head_A2, snake_head_A3


# ----


@task
def snake_head_B1(snake_heads_A_results: TaskPayload):
    """input is a list of tasks-results from prior group

    Direct parent is a task-group, which means that
    the herein task gets all the inputs from its priors
    from that parent group.

    As a treatment of that input,
    we could for instance concat in a list
    """
    this_task_result = [
        ("B1_" + snake_heads_A_results["snake_head_A1"]),
        ("B1_" + snake_heads_A_results["snake_head_A2"]),
        ("B1_" + snake_heads_A_results["snake_head_A3"])
    ]
    return this_task_result


@task
def snake_head_B2(snake_heads_A_results: TaskPayload):
    """input is a list of tasks-results from prior group

    Direct parent is a task-group, which means that
    the herein task gets all the inputs from its priors
    from that parent group.

    As a treatment of that input,
    we could for instance concat in a list
    """
    this_task_result = [
        ("B2_" + snake_heads_A_results["snake_head_A1"]),
        ("B2_" + snake_heads_A_results["snake_head_A2"]),
        ("B2_" + snake_heads_A_results["snake_head_A3"])
    ]
    return this_task_result


@task
def snake_head_B3(snake_heads_A_results: TaskPayload):
    """input is a list of tasks-results from prior group.

    Direct parent is a task-group, which means that
    the herein task gets all the inputs from its priors
    from that parent group.

    As a treatment of that input,
    we could for instance concat in a list
    """
    this_task_result = [
        ("B3_" + snake_heads_A_results["snake_head_A1"]),
        ("B3_" + snake_heads_A_results["snake_head_A2"]),
        ("B3_" + snake_heads_A_results["snake_head_A3"])
    ]
    return this_task_result


@taskgroup
def snake_heads_B():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""

    # Direct parent is a task-group itself,
    # which means each task in the herein group
    # gets all the inputs from its priors from that other group

    return snake_head_B1, snake_head_B2, snake_head_B3


# ----


@task()
def join_snake_heads(snake_heads_B_results: TaskPayload):
    """input is a list of tasks-results from prior group
    we could for instance concat in a flattened list
    just for fun (and because "why not?")
    """
    this_task_result = (
        snake_heads_B_results["snake_head_B1"] +
        snake_heads_B_results["snake_head_B2"] +
        snake_heads_B_results["snake_head_B3"]
    )
    return this_task_result


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["join_snake_heads"] \
            == payload.get("join_snake_heads") \
            == payload

    assert payload == ['B1_A1', 'B1_A2', 'B1_A3', \
                       'B2_A1', 'B2_A2', 'B2_A3', \
                       'B3_A1', 'B3_A2', 'B3_A3']
    return None


@dag(ui_css=UiCss(background="#00ffff"))
def retrain_pipeline():
    # Compose the DAG using operator overloading (>>)
    return start >> snake_heads_A >> snake_heads_B >> join_snake_heads >> end


if __name__ == "__main__":
    # Render the DAG
    svg_fullname = os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html")
    render_svg(retrain_pipeline, svg_fullname)
    # Run the DAG
    print("Final result:", execute(retrain_pipeline, dag_params=None))
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{retrain_pipeline.exec_id}]")
    print(f"DAG SVG written to {svg_fullname}")

