import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, taskgroup, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: Nested groups of tasks ----


@task
def start():
    """Root task."""
    return "titi"


# ----


@task
def snake_head_A(x):
    return x["start"] + " A"


# ----


@task
def snake_head_AA1(x):
    return x["start"] + " AA1"


@task
def snake_head_AA2(x):
    return x["start"] + " AA2"


@task
def snake_head_AA3(x):
    return x["start"] + " AA3"


@task
def snake_head_AA4(x):
    return x["start"] + " AA4"


@taskgroup
def snake_heads_AA():
    return snake_head_AA1, snake_head_AA2, snake_head_AA3, snake_head_AA4


# ----


@taskgroup
def snake_heads_A():
    return snake_heads_AA, snake_head_A


# ----


@task
def concat_snake_heads(snake_heads_A_results):
    """Task that returns a flattened raw results
    from prior nested groups of tasks."""
    result = list(snake_heads_A_results.values())
    print(result)
    return result


@task
def end(results):
    return None


# ----


# Compose the DAG using operator overloading (>>)
final = start >> snake_heads_A >> concat_snake_heads >> end


if __name__ == "__main__":
    # Run the DAG
    print("Final result:", execute(final))
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))

