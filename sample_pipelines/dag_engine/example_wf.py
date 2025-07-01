import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, execute, render_networkx, \
    render_plotly, render_svg


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
def parallel(payload):
    """For each input of the incoming iterator,

    produce a collection of task instances
    for upcoming task."""
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["start"] == payload.get("start") == payload

    return [payload * 10 + i for i in range(2)]


def elementwise_sum(m):
    print(f"m : {m}")
    return list(map(lambda *args: sum(args), *m))

@task(merge_func=elementwise_sum)
def merge(payload):
    """Input is the merged results of parallel prior task."""
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["parallel"] == payload.get("parallel") == payload

    # Do whatever you want for further custom processing
    # on the aggregated result here, e.g.:
    result = list(map(lambda x: x * 2, payload))

    return result


@task
def end(payload):
    print(type(payload))
    # Since the herein task only has 1 direct parent =>
    assert payload["merge"] == payload.get("merge") == payload

    assert payload == [200, 208]
    return None


# Compose the DAG using operator overloading (>>)
final = start >> parallel >> merge >> end


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

