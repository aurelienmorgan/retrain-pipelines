import os
import logging

from typing import List, Union

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, parallel_task
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg, render_networkx, render_plotly


# ---- Example: TaskGroup in Parallelism and Merging ----


@task
def start():
    """Root task: produces a list of numbers."""

    # Do whatever you want
    # e.g. you could handle pipeline parameters here

    #################################
    # Return must be an enumerator, #
    # for following parallel task   #
    # to split/distribute handling. #
    #################################
    return [1, 2, 3, 4]


@parallel_task
def parallel(payload: TaskPayload):
    """For each input of the incoming iterator,

    produce a collection of task instances
    for upcoming task."""
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    return [payload * 10 + i for i in range(2)]


# ----


@task
def snake_head_A1(payload: TaskPayload) -> List[int]:
    # Receives a 1D list.

    # Since the herein task only has 1 direct parent =>
    assert payload["parallel"] == payload.get("parallel") == payload

    # Do whatever you want

    result = payload * 1  # force value

    return result


@task
def snake_head_A2(payload: TaskPayload) -> List[int]:
    # Receives a 1D list.

    # Since the herein task only has 1 direct parent =>
    assert payload["parallel"] == payload.get("parallel") == payload

    # Do whatever you want

    result = [x * 2 for x in payload]

    return result


@taskgroup
def snake_heads_A():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_A1, snake_head_A2


@task
def join_snake_heads(
    snake_heads_A_results: TaskPayload
) -> List[int]:
    """Task that returns a combo of results
    from prior group of tasks."""

    # You can access individual parent results by name :
    logging.getLogger().info(
        "input from 'snake_head_A1': " +
        str(snake_heads_A_results['snake_head_A1'])
    )

    # Do whatever you want

    this_task_result = matrix_sum_cols(
        list(snake_heads_A_results.values())
    )

    return this_task_result

# ----


def matrix_sum_cols(matrix: List[List[Union[int, float]]]):
    """Computes the sum of each column in a 2D matrix
    returning a 1D list of numerics."""
    return [sum(col) for col in zip(*matrix)]

@task(merge_func=matrix_sum_cols)
def merge(payload: TaskPayload) -> List[int]:
    """Input is the merged results of
    parallel prior tasks from taskgroup."""
    # Since the herein task only has 1 direct parent =>
    assert payload["join_snake_heads"] == \
           payload.get("join_snake_heads") == \
           payload

    # Do whatever you want for further custom processing
    # on the aggregated result here, e.g.:
    result = list(map(lambda x: x * 2, payload))

    return result


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["merge"] == payload.get("merge") == payload

    assert payload == [600, 624]
    return None


# Compose the DAG using operator overloading (>>)
final = start >> parallel >> snake_heads_A >> join_snake_heads >> merge >> end


if __name__ == "__main__":
    # Run the DAG
    print("Final result:", execute(final, dag_params=None))
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{final.exec_id}]")
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))

