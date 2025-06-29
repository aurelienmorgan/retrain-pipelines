import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: Parallelism and Merging ----


@task
def start():
    """Root task: produces a list of numbers."""
    return [1, 2, 3, 4]  ########################## Must be an enumerator, for following parallel task to split/distribute handling


@parallel_task
def parallel(x):
    """For each input x, produce a list for inner parallelism."""
    return [x["start"] * 10 + i for i in range(2)]  ######## Must be an enumerator, for following parallel-merging task to aggregate right


def elementwise_sum(m):
    print(f"m : {m}")
    return list(map(lambda *args: sum(args), *m))

@task(merge_func=elementwise_sum)
def merge(results):
    """Input is the merged results of parallel prior task."""
    input = results["parallel"]
    # Whole room for further custom processing
    # on the aggregated result here, e.g.:
    result = list(map(lambda x: x * 2, input))

    return result


@task
def end(results):
    return None


# Compose the DAG using operator overloading (>>)
final = start >> parallel >> merge >> end

if __name__ == "__main__":
    os.environ["RP_ARTIFACTS_STORE"] = os.path.dirname(__file__)
    # Run the DAG
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}")
    print("start docstring:", start.func.__doc__)
    print("Final result:", execute(final))
    print(f"exec_id : {final.exec_id}")
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))
