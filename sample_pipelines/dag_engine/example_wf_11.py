import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: Nested Parallelism and Merging with intermediary inline tasks ----


@task
def start():
    """Root task: produces a list of numbers."""
    return [1, 3]  ########################## Must be an enumerator, for following parallel task to split/distribute handling


@parallel_task
def outer_parallel(x):
    """For each input x, produce a list for inner parallelism."""
    return [x["start"] * 10 + i for i in range(2)]  ######## Must be an enumerator, for following parallel task to split/distribute handling


@task
def inline1(results):
    """A simple task, gets executed inline in parrallel branches."""
    input = results["outer_parallel"]
    return input


@parallel_task
def inner_parallel(y):
    """For each y, double it."""
    return [y["inline1"] * 2 for j in range(2)]  ######## Must be an enumerator, for following parallel-merging task to aggregate right


@task
def inline2(results):
    """A simple task, gets executed inline in parrallel branches."""
    input = results["inner_parallel"]
    return input



# Preserve hierarchy in merge functions
def elementwise_2D_sum(matrix):
    """Sum elements in 2D matrix while preserving structure"""
    print(__name__)
    print(matrix)
    return [sum(col) for col in zip(*matrix)]

@task(merge_func=elementwise_2D_sum)
def merge_inner(results):
    """Merge inner parallel results per outer group"""
    return results["inline2"]  ######## Must be an enumerator, for following parallel-merging task to aggregate right


@task
def inline3(results):
    """A simple task, gets executed inline in parrallel branches."""
    input = results["merge_inner"]
    return input


@task(merge_func=elementwise_2D_sum)
def merge_outer(results):
    """Merge outer parallel results"""
    return results["inline3"]


@task
def end(results):
    return None


# Compose the DAG using operator overloading (>>)
final = start >> outer_parallel >> inline1 >> inner_parallel >> inline2 >> merge_inner >> inline3 >> merge_outer >> end

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
