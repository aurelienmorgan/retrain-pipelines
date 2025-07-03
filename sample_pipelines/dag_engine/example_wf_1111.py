import os

from typing import List, Union

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, parallel_task
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg, render_networkx, render_plotly


# ---- Example: Sequence of Nested Parallelism and Merging ----


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


@parallel_task
def inner_parallel1(payload: TaskPayload):
    """For each input, returns a list containing
    the result of doubling that value,
    repeated in a 2D list."""
    # Since the herein task only has 1 direct parent =>
    assert payload["outer_parallel"] == payload.get("outer_parallel") == payload

    # Do whatever you want

    #################################
    # Return must be an enumerator, #
    # for following parallel task   #
    # to split/distribute handling. #
    #################################
    return [payload["outer_parallel"] * 2 for j in range(2)]


# Preserve hierarchy in merge functions
def matrix_sum_cols(matrix: List[List[Union[int, float]]]):
    """Computes the sum of each column in a 2D matrix
    returning a 1D list of numerics."""
    print(f"matrix_sum_cols - {matrix}")
    return [sum(col) for col in zip(*matrix)]

@task(merge_func=matrix_sum_cols)
def merge_inner1(payload: TaskPayload):
    """Merge inner parallel results per outer group"""

    # Do whatever you want

    return payload["inner_parallel1"]


@parallel_task
def inner_parallel2(payload: TaskPayload):
    """For each input, returns a list containing
    the result of doubling that value,
    repeated in a 2D list."""

    # Do whatever you want

    #################################
    # Return must be an enumerator, #
    # for following parallel task   #
    # to split/distribute handling. #
    #################################
    return [payload["merge_inner1"] * 2 for j in range(2)]


@task(merge_func=matrix_sum_cols)
def merge_inner2(payload: TaskPayload):
    """Merge inner parallel results per outer group"""

    # Do whatever you want

    return payload["inner_parallel2"]


@task(merge_func=matrix_sum_cols)
def merge_outer(payload: TaskPayload):
    """Merge outer parallel results"""

    # Do whatever you want

    return payload["merge_inner2"]


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["merge_outer"] == payload.get("merge_outer") == payload

    assert payload == [656, 656]
    return None


# Compose the DAG using operator overloading (>>)
final = start >> outer_parallel >> inner_parallel1 >> merge_inner1 >> inner_parallel2 >> merge_inner2 >> merge_outer >> end

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

