import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, TaskGroup, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: Group of tasks ----


@task
def start():
    # Root task
    return None


@task
def snake_head_1(_):
    return 1


@task
def snake_head_2(_):
    return 2


@task
def snake_head_3(_):
    return 3


@task()
def concat_snake_heads(parent_group_results):
    # Concat results (e.g. put them into a list)
    result = [
        parent_group_results["snake_head_1"],
        parent_group_results["snake_head_2"],
        parent_group_results["snake_head_3"]
    ]
    print(result)
    return result


@task
def end(results):
    return None


# Compose the DAG using operator overloading (>>)
snake_heads = TaskGroup(snake_head_2, snake_head_1, snake_head_3)
final = start >> snake_heads >> concat_snake_heads >> end

if __name__ == "__main__":
    # Run the DAG
    print("Final result:", execute(final))
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))
