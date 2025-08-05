import os

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, dag
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg, render_networkx, render_plotly


# ---- Example: Group of tasks ----


@task
def start():
    # Root task
    return "titi"


@task
def snake_head_1(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    # Do whatever you want

    return payload + " snake_head_1"


@task
def snake_head_2(payload: TaskPayload):

    # Do whatever you want

    return payload["start"] + " snake_head_2"


@task
def snake_head_3(payload: TaskPayload):

    # Do whatever you want

    return payload["start"] + " snake_head_3"


@taskgroup
def snake_heads():
    """Group of tasks with different processing logics
    that are to be run independently,
    in parrallel, on the same set of inputs.
    Note that the downward task(s) will have to
    await for all of those to complete before they can start."""
    return snake_head_2, snake_head_1, snake_head_3


@task()
def join_snake_heads(payload: TaskPayload):
    # Concat results (e.g. put them into a list)
    this_task_result = [
        payload["snake_head_1"],
        payload["snake_head_2"],
        payload["snake_head_3"]
    ]
    return this_task_result


@task
def end(payload: TaskPayload):
    # Since the herein task only has 1 direct parent =>
    assert payload["join_snake_heads"] \
            == payload.get("join_snake_heads") \
            == payload

    assert payload == ['titi snake_head_1', 'titi snake_head_2', \
                       'titi snake_head_3']
    return None


@dag(ui_css={"background": "#ffffff"})
def retrain_pipeline():
    # Compose the DAG using operator overloading (>>)
    return start >> snake_heads >> join_snake_heads >> end


if __name__ == "__main__":
    # Run the DAG
    print("Final result:", execute(retrain_pipeline, dag_params=None))
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}[{retrain_pipeline.exec_id}]")
    # Render the DAG
    render_svg(retrain_pipeline, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(retrain_pipeline, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(retrain_pipeline, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))

