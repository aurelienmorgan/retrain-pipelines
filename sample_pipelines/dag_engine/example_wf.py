import os

from typing import List, Union

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, parallel_task, \
    dag, UiCss
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg


# ---- Example: Parallelism and Merging ----


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


def matrix_sum_cols(matrix: List[List[Union[int, float]]]):
    """Computes the sum of each column in a 2D matrix
    returning a 1D list of numerics."""

    return [sum(col) for col in zip(*matrix)]

@task(merge_func=matrix_sum_cols)
def merge(payload: TaskPayload):
    """Input is the merged results of parallel prior task."""
    # Since the herein task only has 1 direct parent =>
    assert payload["parallel"] == payload.get("parallel") == payload

    # Do whatever you want for further custom processing
    # on the aggregated result here, e.g.:
    result = list(map(lambda x: x * 2, payload))

    return result


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["merge"] == payload.get("merge") == payload

    assert payload == [200, 208]

    return None


@dag(ui_css=UiCss(background="#000", color="#ffd700", border="#ffd700"))
def retrain_pipeline():
    # Compose the DAG using operator overloading (>>)
    return start >> parallel >> merge >> end


if __name__ == "__main__":
    # Render the DAG
    svg_fullname = os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html")
    render_svg(retrain_pipeline, svg_fullname)
    # Run the DAG
    print("Final result:", execute(retrain_pipeline, dag_params=None))
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{retrain_pipeline.exec_id}]")
    print(f"DAG SVG written to {svg_fullname}")

