
import os
import logging

from typing import List, Union

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, parallel_task, \
    dag, UiCss
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg


# ---- Example: TaskGroup in Parallelism and Merging ----


@task(ui_css=UiCss(background="#ff7b00", color="#ff7b00", border="#ff7b00"))
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


@parallel_task(ui_css=UiCss(background="#00ff37"))
def parallel(payload: TaskPayload):
    """For each input of the incoming iterator,

    produce a collection of task instances
    for following task."""
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


@task(ui_css=UiCss(background="#ffee00"))
def snake_head_A2(payload: TaskPayload) -> List[int]:
    # Receives a 1D list.

    # Since the herein task only has 1 direct parent =>
    assert payload["parallel"] == payload.get("parallel") == payload

    # Do whatever you want

    result = [x * 2 for x in payload]

    return result


@taskgroup(ui_css=UiCss(background="#000000", color="#e00000", border="#00fff7"))
def snake_heads_A():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_A1, snake_head_A2


@task(ui_css=UiCss(background="#752500", border="#964a29"))
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

@task(merge_func=matrix_sum_cols,
      ui_css=UiCss(background="#ff0000"))
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


@task(ui_css=UiCss(background="#ff00f7"))
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["merge"] == payload.get("merge") == payload

    assert payload == [600, 624]
    return None


@dag(ui_css=UiCss(background="#ff7b00"))
def retrain_pipeline():
    """TaskGroup in parallel line."""
    # Compose the DAG using operator overloading (>>)
    return start >> parallel >> snake_heads_A >> join_snake_heads >> merge >> end


if __name__ == "__main__":
    # Render the DAG
    svg_fullname = os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html")
    render_svg(retrain_pipeline, svg_fullname)
    # Run the DAG
    print("Final result:", execute(retrain_pipeline, dag_params=None))
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{retrain_pipeline.exec_id}]")
    print(f"DAG SVG written to {svg_fullname}")

