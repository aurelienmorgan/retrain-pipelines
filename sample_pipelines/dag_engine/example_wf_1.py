import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: Nested Parallelism and Merging ----


@task
def start():
    """Root task: produces a list of numbers."""
    return [1, 2, 3, 4]  ########################## Must be an enumerator, for following parallel task to split/distribute handling


@parallel_task
def outer_parallel(x):
    """For each input x, produce a list for inner parallelism."""
    return [x["start"] * 10 + i for i in range(2)]  ######## Must be an enumerator, for following parallel task to split/distribute handling


@parallel_task
def inner_parallel(y):
    """For each y, double it."""
    return [y["outer_parallel"] * 2 for j in range(2)]  ######## Must be an enumerator of dimension equivalent to the nesting level (here, an iterator of iterator, for following parallel task to aggregate right


# lists of element-wise sum of list of lists of numerics
# i.e. lists of elementwise_2D_sum
elementwise_3D_sum = lambda m: [
    [sum(values) for values in zip(*rows)]
    for rows in zip(*m)
]

@task(merge_func=elementwise_3D_sum)
def merge_inner(results):
    """Merge results of inner_parallel (element-wise sum them)."""
    return results


# element-wise sum of list of lists of numerics
elementwise_2D_sum = lambda m: list(map(lambda *args: sum(args), *m))

@task(merge_func=elementwise_2D_sum)
def merge_outer(results):
    """Merge results of merge_inner (sum them)."""
    return results


@task
def end(results):
    return None


# Compose the DAG using operator overloading (>>)
final = start >> outer_parallel >> inner_parallel >> merge_inner >> merge_outer >> end

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
