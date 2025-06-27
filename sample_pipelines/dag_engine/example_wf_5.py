import os
import logging

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, TaskGroup, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: 3-levels nesting of groups of tasks ----


@task
def start():
    # Root task: produces a list of numbers
    return


@task
def snake_head_A(_):
    return "A"


@task
def snake_head_AAA1(_):
    return "AAA1"


@task
def snake_head_AAA2(_):
    return "AAA2"


@task
def snake_head_AA1(_):
    return "AA1"


@task
def snake_head_AA2(_):
    return "AA2"


@task
def snake_head_AA3(_):
    return "AA3"


@task
def snake_head_AA4(_):
    return "AA4"


@task
def aggreg_snake_heads(snake_heads_A_results):
    """Task that returns a flattened raw results
    from prior nested groups of tasks."""

    logging.getLogger().info([
        snake_heads_A_results["snake_head_AAA1"],
        snake_heads_A_results["snake_head_A"]
    ])

    result = list(snake_heads_A_results.values())

    return result


@task
def end(results):
    return None


# Compose the DAG using operator overloading (>>)
snake_heads_AAA = TaskGroup(snake_head_AAA1, snake_head_AAA2)
snake_heads_AA = TaskGroup(snake_heads_AAA, snake_head_AA1, snake_head_AA2, snake_head_AA3, snake_head_AA4)
snake_heads_A = TaskGroup(snake_heads_AA, snake_head_A)
final = start >> snake_heads_A >> aggreg_snake_heads >> end

if __name__ == "__main__":
    os.environ["RP_ARTIFACTS_STORE"] = os.path.dirname(__file__)
    # Run the DAG
    print("Final result:", execute(final))
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))
