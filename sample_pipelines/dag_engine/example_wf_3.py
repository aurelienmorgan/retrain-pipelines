
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
    import time ; time.sleep(3)                             ### DEBUG - DELETE ###
    return


@task
def snake_head_A1(_):

    # Do whatever you want
    print("Do whatever you want - A1")

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
    in parallel, on the same set of inputs.
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

    print(this_task_result)

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

    print(this_task_result)

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

    print(this_task_result)

    return this_task_result


@taskgroup
def snake_heads_B():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parallel, on the same set of inputs.
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
    """2 taskgroups in series.

    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida.
    """
    # Compose the DAG using operator overloading (>>)
    return start >> snake_heads_A >> snake_heads_B >> join_snake_heads >> end


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

