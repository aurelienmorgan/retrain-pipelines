
import os

from typing import List, Union

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, parallel_task, \
    dag, UiCss, DagParam, ctx
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg


# ---- Example: Parallelism and Merging ----


@task
def start():
    """Root task: produces a list of numbers."""

    # Do whatever you want

    ################################
    # Access DAG execution-context #
    #      (parameters, etc.)      #
    ################################
    print(f"execution param 1: {ctx.dummy_param_1}")
    print(f"execution param 2: {ctx.dummy_param_2}")

    from datetime import datetime, timezone
    ctx.added_entry = \
        datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') + \
        " UTC - execution-context, task dynamically added entry"
    print(f"ctx addon: {ctx.added_entry}")
    ################################

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
    for following task."""
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

    # Declare DAG parameters (will be used in tasks via ctx)
    dummy_param_1 = DagParam(
        description="a dummy param for that dag execution",
        default="dummy param default value"
    )
    dummy_param_2 = DagParam(
        description="another dummy param for that dag execution",
        default="dummy param default value 2"
    )

    # Compose the DAG using operator overloading (>>)
    return start >> parallel >> merge >> end


if __name__ == "__main__":
    # print(f"to_tasktypes_list : {retrain_pipeline.to_tasktypes_list(serializable=True)}")
    # Render the DAG
    svg_fullname = os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html")
    render_svg(retrain_pipeline, svg_fullname)

    print(retrain_pipeline.help()) # help string (parameters definitions and defaults)

    # Execute with parameter overrides
    import random
    print("Final result:", execute(retrain_pipeline, params={
        "dummy_param_1": f"{random.randint(1, 10)} - override default for that execution"
    }))

    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{retrain_pipeline.exec_id}]")
    print(f"DAG SVG written to {svg_fullname}")

