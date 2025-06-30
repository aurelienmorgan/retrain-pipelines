import os
import logging

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, taskgroup, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: tripple chaining of tasks-groups (mostly to validate DAG rendering) ----


@task
def start():
    """Root task."""
    return


# ----


@task
def snake_head_A1(_):
    return "A1"


@task
def snake_head_A2(_):
    return "A2"


@taskgroup
def snake_heads_A():
    return snake_head_A1, snake_head_A2


# ----


@task
def snake_head_B1(snake_heads_A_results):
    return "B1"


@task
def snake_head_B2(snake_heads_A_results):
    return "B2"


@taskgroup
def snake_heads_B():
    return snake_head_B1, snake_head_B2


# ----


@task
def snake_head_C1(snake_heads_B_results):
    return "C1"


@task
def snake_head_C2(snake_heads_B_results):
    return "C2"


@taskgroup
def snake_heads_C():
    return snake_head_C1, snake_head_C2


# ----


@task
def aggreg_snake_heads(snake_heads_C_results):
    """Task that returns a flattened raw results
    from prior nested groups of tasks."""
    result = list(snake_heads_C_results.values())
    return result


@task
def end(results):
    return None


# ----


# Compose the DAG using operator overloading (>>)
final = start >> snake_heads_A >> snake_heads_B >> snake_heads_C >> aggreg_snake_heads >> end


if __name__ == "__main__":
    os.environ["RP_ARTIFACTS_STORE"] = os.path.dirname(__file__)
    # Run the DAG
    print("Final result:", execute(final))
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))

