
import os

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, taskgroup, dag
from retrain_pipelines.dag_engine.runtime import \
    execute
from retrain_pipelines.dag_engine.renderer import \
    render_svg


# ---- Example: Group of tasks ----


@task
def start():
    # Root task
    import time ; time.sleep(3)                             ### DEBUG - DELETE ###
    return "start"


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
    in parallel, on the same set of inputs.
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

    assert payload == ['start snake_head_1', 'start snake_head_2', \
                       'start snake_head_3']
    return None


@dag(ui_css={"background": "#ffffff"})
def retrain_pipeline():
    """Simple taskgroup.
    """
    # Compose the DAG using operator overloading (>>)
    return start >> snake_heads >> join_snake_heads >> end


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

