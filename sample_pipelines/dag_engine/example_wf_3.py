import os

from retrain_pipelines.dag_engine.task import \
    task, parallel_task, TaskGroup, execute, render_networkx, \
    render_plotly, render_svg

# ---- Example: Several groups of tasks ----


@task
def start():
    # Root task: produces a list of numbers
    return


@task
def snake_head_A1(_):
    return "A1"


@task
def snake_head_A2(_):
    return "A2"


@task
def snake_head_A3(_):
    return "A3"


@task
def snake_head_B1(snake_heads_A_results):
    """input is a list of tasks-results from prior group
    as a treatment of that input,
    we could for instance concat in a list
    """
    result = [
        "B1_" + snake_heads_A_results["snake_head_A1"],
        "B1_" + snake_heads_A_results["snake_head_A2"],
        "B1_" + snake_heads_A_results["snake_head_A3"]
    ]
    print(result)
    return result


@task
def snake_head_B2(snake_heads_A_results):
    """input is a list of tasks-results from prior group
    as a treatment of that input,
    we could for instance concat in a list
    """
    result = [
        "B2_" + snake_heads_A_results["snake_head_A1"],
        "B2_" + snake_heads_A_results["snake_head_A2"],
        "B2_" + snake_heads_A_results["snake_head_A3"]
    ]
    print(result)
    return result


@task
def snake_head_B3(snake_heads_A_results):
    """input is a list of tasks-results from prior group
    as a treatment of that input,
    we could for instance concat in a list
    """
    result = [
        "B3_" + snake_heads_A_results["snake_head_A1"],
        "B3_" + snake_heads_A_results["snake_head_A2"],
        "B3_" + snake_heads_A_results["snake_head_A3"]
    ]
    print(result)
    return result


@task()
def concat_snake_heads_B(snake_heads_B_results):
    """input is a list of tasks-results from prior group
    we could for instance concat in a flattened list
    just for fun (and because "when not?")
    """
    import itertools
    result = itertools.chain.from_iterable([
        snake_heads_B_results["snake_head_B1"],
        snake_heads_B_results["snake_head_B2"],
        snake_heads_B_results["snake_head_B3"]
    ])
    print(result)
    return result


@task
def end(results):
    return None


# Compose the DAG using operator overloading (>>)
snake_heads_A = TaskGroup(snake_head_A1, snake_head_A2, snake_head_A3)
snake_heads_B = TaskGroup(snake_head_B1, snake_head_B2, snake_head_B3)
final = start >> snake_heads_A >> snake_heads_B >> concat_snake_heads_B >> end

if __name__ == "__main__":
    # Run the DAG
    print(f"execution {os.path.splitext(os.path.basename(__file__))[0]}")
    print("start docstring:", start.func.__doc__)
    print("Final result:", execute(final))
    # Render the DAG
    render_svg(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.svg"))
    print("DAG SVG written to dag.svg")
    render_networkx(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.png"))
    render_plotly(final, os.path.join(os.environ["RP_ARTIFACTS_STORE"], "dag.html"))

