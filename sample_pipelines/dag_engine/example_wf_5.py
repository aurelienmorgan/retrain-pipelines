import os
import logging

from retrain_pipelines.dag_engine.task import \
    TaskPayload, task, taskgroup, execute, \
    render_networkx, render_plotly, render_svg


# ---- Example: 3-levels nesting of groups of tasks ----


@task
def start():
    # Root task
    return


@task
def inline1(payload: TaskPayload):
    """A simple task, gets executed inline."""
    input = payload["start"]

    # Do whatever you want

    return None


# ----


@task
def snake_head_A(_):
    return "A"


# ----


@task
def snake_head_AAA1(_):
    return "AAA1"


@task
def snake_head_AAA2(_):
    return "AAA2"


@taskgroup
def snake_heads_AAA():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_AAA1, snake_head_AAA2


# ----



@task
def snake_head_AA1(_):
    return "AA1"


@task
def snake_head_AA2(_):
    return "AA2"


@task
def snake_head_AA3(_):
    return "AA3"


@task
def snake_head_AA4(_):
    return "AA4"


@taskgroup
def snake_heads_AA():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_heads_AAA, snake_head_AA1, snake_head_AA2, \
           snake_head_AA3, snake_head_AA4


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
def join_snake_heads(snake_heads_A_results: TaskPayload):
    """Task that returns a flattened raw results
    from prior nested groups of tasks."""

    # You can access individual parent results by name :
    logging.getLogger().info([
        snake_heads_A_results["snake_head_AAA1"],
        snake_heads_A_results["snake_head_A"]
    ])

    # Do whatever you want

    this_task_result = list(snake_heads_A_results.values())

    return this_task_result


@task
def inline2(payload: TaskPayload):
    """A simple task, gets executed inline."""
    input = payload["join_snake_heads"]

    # Do whatever you want

    return payload


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["inline2"] \
            == payload.get("inline2") \
            == payload

    assert payload == ['AAA1', 'AAA2', \
                       'AA1', 'AA2', 'AA3', 'AA4', \
                       'A']
    return None


# ----


# Compose the DAG using operator overloading (>>)
final = start >> inline1 >> snake_heads_A >> join_snake_heads >> inline2 >> end


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

