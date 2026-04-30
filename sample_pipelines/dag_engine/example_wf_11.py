
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


# ---- Example: Nested Parallelism and Merging with intermediary inline tasks ----


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
    return [1, 3]


@parallel_task
def outer_parallel(payload: TaskPayload):
    """For each input x, produce a list for inner parallelism.
    Given payload["start"] = x, returns [x * 10, x * 10 + 1]."""

    # Do whatever you want

    #################################
    # Return must be an enumerator, #
    # for following parallel task   #
    # to split/distribute handling. #
    #################################
    return [payload["start"] * 10 + i for i in range(2)]


@task
def inline1(payload: TaskPayload):
    """A simple task, gets executed inline in parallel branches."""
    input = payload["outer_parallel"]

    # Do whatever you want

    return input


@parallel_task
def inner_parallel(payload: TaskPayload):
    """For each input, returns a list containing
    the result of doubling that value,
    repeated in a 2D list."""

    # Do whatever you want

    #################################
    # Return must be an enumerator, #
    # for following parallel task   #
    # to split/distribute handling. #
    #################################
    return [payload["inline1"] * 2 for j in range(2)]


@task
def inline2a(payload: TaskPayload):
    """A simple task, gets executed inline in parallel branches."""
    input = payload["inner_parallel"]

    # Do whatever you want

    return input


@task
def inline2b(payload: TaskPayload):
    """A simple task, gets executed inline in parallel branches."""
    input = payload["inner_parallel"]

    # Do whatever you want

    return input


@taskgroup
def inline2():
    return inline2a, inline2b


@task
def inline3(payload: TaskPayload):
    """A simple task, gets executed inline in parallel branches."""
    input = payload["inline2a"]

    # Do whatever you want

    return input


# Preserve hierarchy in merge functions
def matrix_sum_cols(matrix: List[List[Union[int, float]]]):
    """Computes the sum of each column in a 2D matrix
    returning a 1D list of numerics."""
    return [sum(col) for col in zip(*matrix)]

@task(merge_func=matrix_sum_cols)
def merge_inner(payload: TaskPayload):
    """Merge inner parallel results per outer group"""
    return payload["inline3"]


@task
def inline4(payload: TaskPayload):
    """A simple task, gets executed inline in parallel branches."""
    input = payload["merge_inner"]

    # Do whatever you want

    return input


@task(merge_func=matrix_sum_cols)
def merge_outer(payload: TaskPayload):
    """Merge outer parallel results"""
    return payload["inline4"]


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["merge_outer"] == payload.get("merge_outer") == payload

    assert payload == [164, 164]
    return None


@dag(ui_css=UiCss(background="#e100ff"))
def retrain_pipeline():
    """1-level deep nested sub-DAGing with an inner taskgroup & inner inline tasks.
    """
    # Compose the DAG using operator overloading (>>)
    return start >> outer_parallel >> inline1 \
           >> inner_parallel >> inline2>> inline3  >> merge_inner \
           >> inline4 >> merge_outer >> end


if __name__ == "__main__":
    # print(f"to_tasktypes_list : {retrain_pipeline.to_tasktypes_list(serializable=True)}")

    # Render the DAG
    svg_fullname = os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "dag.html"
    ))
    logging.getLogger().info(f"svg_fullname : {svg_fullname}")
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

