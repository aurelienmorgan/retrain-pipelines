
import os
import copy
import logging
import concurrent.futures

from collections import defaultdict, deque
from typing import List, Optional, Union, \
    Dict, Any

from .core import TaskType, TaskGroup, \
    TaskPayload, TaskFuncException, \
    DAG, DagExecutionContext, _dag_execution_context_var


def _topological_sort(
    tasks: List[TaskType]
) -> List[Union[TaskType, TaskGroup]]:
    """Topological sort of first-level elements of DAG.

    i.e. elements of task-groups are not referenced,
    just the top-level task-groups themselves

    Params:
        - tasks (List[TaskType]):
            roots of the DAG to be considered.

    Results:
        - List of first level elements of the DAG.
    """
    in_degree = defaultdict(int)
    children_map = defaultdict(list)
    all_tasks = set()

    # Stack-based traversal to gather tasks
    # and build in-degree graph
    stack = list(tasks)
    while stack:
        node = stack.pop()
        if not isinstance(node, TaskType):
            continue
        if node in all_tasks:
            continue
        all_tasks.add(node)
        for child in node.children:
            if isinstance(child, TaskGroup):
                for t in child.tasks:
                    in_degree[t] += 1
                    children_map[node].append(t)
                    stack.append(t)
            else:
                in_degree[child] += 1
                children_map[node].append(child)
                stack.append(child)

    queue = deque(t for t in all_tasks if in_degree[t] == 0)
    seen_groups = set()
    order = []

    while queue:
        t = queue.popleft()

        group = t.task_group
        if group is None:
            order.append(t)
        elif group.task_group is None and group not in seen_groups:
            seen_groups.add(group)
            order.append(group)

        for child in children_map[t]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return order


def _rich_log_execution_id_with_timestamp(execution_id: int):
    """Force the timestamp to show on execution start log with rich logger."""
    log = logging.getLogger()
    for handler in log.handlers:
        # Check if handler is RichHandler (or subclass)
        if hasattr(handler, "_log_render"):
            handler._log_render.omit_repeated_times = False
            handler._log_render.last_log_time = None  # Force timestamp on next log
            log.info(f"Execution ID: {execution_id}")
            handler._log_render.omit_repeated_times = True


def _find_subdag_end(task_order, start_idx):
    """Find where the parallel subdag ends, e.g.
    at the next merge task, accounting for potential nesting."""
    parallel_depth = 0
    for i in range(start_idx, len(task_order)):
        if isinstance(task_order[i], TaskGroup):
            continue
        elif task_order[i].is_parallel:
            parallel_depth += 1
        elif task_order[i].merge_func:
            parallel_depth -= 1
            if parallel_depth == 0:
                return i

    return len(task_order)


def _collect_parent_results(
    elmt: Union[TaskType, TaskGroup],
    results: TaskPayload
) -> TaskPayload:
    """
    Collects results from parent tasks
    for a given task or taskgroup.

    Params:
        - t (Union[TaskType, TaskGroup]):
            The current task or taskgroup.
        - results (TaskPayload):
            The dictionary of previously computed results.

    Resuts:
        - TaskPayload:
            A payload containing
            the results from all available parent tasks.
    """
    parent_results = TaskPayload({})
    if isinstance(elmt, TaskType):
        for p in elmt.parents:
            if p.name in results:
                parent_results[p.name] = results[p.name]
    elif isinstance(elmt, TaskGroup):
        # Since all elements in a TaskGroup share relatives
        # (both parents and children), we look into the first only.
        parent_results = _collect_parent_results(elmt.elements[0], results)

    return parent_results


def _parallel_input_count(parent_results):
    """
    Determines how many parallel executions are needed
    based on the parent results.

    Params:
        - parent_results (TaskPayload):
            The payload of inputs from parent tasks.

    Results:
        - int:
            The number of parallel branches to execute.
    """
    if not parent_results:
        return 1
    first = list(parent_results.values())[0]
    return len(first) if isinstance(first, list) else 1


def _execute_parallel_branches(
    subdag_elements: List[Union[TaskType, TaskGroup]],
    parent_results: TaskPayload,
    count: int,
    exec_id: int,
    rank: Optional[List[int]] = None
) -> List[TaskPayload]:
    """
    Executes a subdag in parallel across multiple input branches.

    Params:
        - subdag_elements (List[Union[TaskType, TaskGroup]]):
            The first-level tasks and taskgroups
            forming the parallel subdag.
        - parent_results (TaskPayload):
            Input values for the branches.
        - count (int):
            Number of parallel executions to perform.
        - exec_id (int):
            The execution ID for this DAG run.
        - rank (List[int], optional):
            Index path for nested branches.
            Indices of branches if branch inside a parallel lane.

    Results:
        - List[TaskPayload]:
            Results from each parallel branch.
    """
    # CAPTURE THE CONTEXT BEFORE THREADS
    context = _dag_execution_context_var.get()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx in range(count):
            branch_input = TaskPayload({
                k: (v[idx] if isinstance(v, list) else v)
                for k, v in parent_results.items()
            })
            futures.append(executor.submit(
                _execute_branch_with_context,
                context,
                subdag_elements,
                branch_input,
                exec_id,
                (rank or []) + [idx]
            ))
        return [f.result() for f in futures]


def _execute_branch(
    branch_elements: List[Union[TaskType, TaskGroup]],
    branch_input: TaskPayload,
    exec_id: int,
    rank: List[int]
) -> TaskPayload:
    """Execute a single parallel branch with recursive subdag handling.

    Params:
        - branch_elements
        - branch_input: (TaskPayload):
            outputs from task parents
        - exec_id (int):
            The execution ID for this DAG run.
        - rank: List[int]:
            Index path for nested branches.
            Indices of branches if inside a parallel lane.

    Results:
        - (TaskPayload)
            results of the last task in the branch.
    """
    branch_results = branch_input.copy()
    i = 0

    while i < len(branch_elements):
        elmt = branch_elements[i]
        parent_results = _collect_parent_results(elmt, branch_results)

        if isinstance(elmt, TaskType):
            if elmt.is_parallel and i > 0:
                # Nested Parallelism
                elmt.log.info(f"nested sub-DAG parent_results: {parent_results}")

                # Find nested subdag end
                nested_subdag_end = _find_subdag_end(branch_elements, i)
                nested_subdag_tasks = branch_elements[i:nested_subdag_end]
                elmt.log.info("nested_subdag_tasks [red on yellow]{}{}[/]".format(
                    rank, [elmt.name for elmt in nested_subdag_tasks]))

                # Execute nested parallel subdag
                parent_value = list(parent_results.values())[0]
                if isinstance(parent_value, list):
                    nested_results = _execute_parallel_branches(
                        nested_subdag_tasks, parent_results,
                        len(parent_value), exec_id, rank)
                else:
                    nested_results = _execute_branch(
                        nested_subdag_tasks, parent_results, exec_id, rank)

                branch_results[nested_subdag_tasks[-1].name] = nested_results
                i = nested_subdag_end
                continue
            else:
                task_type = "merge task" if elmt.merge_func else "task"
                elmt.log.info(f"Executing {task_type}: " +
                            f"[rgb(0,255,255) on #af00ff]{elmt.name}{rank}[/]")
                result = _execute_regular_task(elmt, parent_results, exec_id, rank)
                branch_results[elmt.name] = result
        else:
            # TaskGroup
            elmt.log.info("Executing taskgroup: " +
                        f"[rgb(0,255,255) on #af00ff]{elmt.name}{rank}[/]")
            result = _execute_regular_taskgroup(elmt, parent_results, exec_id, rank)
            for task_name, task_result in result.items():
                branch_results[task_name] = task_result

        i += 1

    return branch_results[branch_elements[-1].name]

def _execute_branch_with_context(
    context: DagExecutionContext,
    branch_elements: List[Union[TaskType, TaskGroup]],
    branch_input: TaskPayload,
    exec_id: int,
    rank: List[int]
) -> TaskPayload:
    """Wrapper that sets context before executing branch."""
    token = _dag_execution_context_var.set(context)
    try:
        return _execute_branch(
            branch_elements, branch_input, exec_id, rank)
    finally:
        _dag_execution_context_var.reset(token)


def _execute_regular_task(
    t: TaskType,
    parent_results: TaskPayload,
    exec_id: int,
    rank: Optional[List[int]] = None
) -> TaskPayload:
    """
    Executes a single non-parallel task,
    optionally applying a merge function.

    Params:
        - t (TaskType):
            The task to execute.
        - parent_results (TaskPayload):
            Input values from parent tasks.
        - exec_id (int):
            The execution ID for this DAG run.
        - rank (List[int], optional):
            Index path for nested branches.
            Indices of branches if inside a parallel lane.

    Results:
        - TaskPayload:
            The result produced by the task.
    """
    task_id = None
    if t.merge_func:
        task_id, merged = t.merge_func(list(parent_results.values())[0], rank=rank, exec_id=exec_id)
        parent_results = TaskPayload({t.parents[0].name: merged})

        t.log.info(
            f"[#FFFFE0]`{t.name}{rank if rank else ''} merged "
            f" {t.merge_func.__name__}(parent_results) :\n"
            f"Inputs :\n"
            f"  \N{BULLET} {str(parent_results)}[/]"
        )

    if parent_results._data:
        _, result = t.func(parent_results, rank=rank, exec_id=exec_id, task_id=task_id) \
               if rank \
               else t.func(parent_results, exec_id=exec_id, task_id=task_id)
    else:
        _, result = t.func(rank=rank, exec_id=exec_id, task_id=task_id) if rank else t.func(exec_id=exec_id, task_id=task_id)

    return result


def _execute_regular_taskgroup(
    tg: TaskGroup,
    parent_results: TaskPayload,
    exec_id: int,
    rank: Optional[List[int]] = None
) -> TaskPayload:
    """Executes tasks (and potential nested taskgroups) asynchronously.

    All individual (potentially nested) tasks
    get the same input payload.

    The top-level taskgroup completes
    when all (potentially nested) task do.

    The following task gets the TaskPayload objects
    with all (potentially nested) tasks' outputs

    Params:
        - tg (TaskGroup):
            The taskgroup to execute.
        - parent_results (TaskPayload):
            Input values from parent tasks.
        - exec_id (int):
            The execution ID for this DAG run.
        - rank (List[int], optional):
            Index path for nested branches.
            Indices of branches
            if inside a parallel lane.

    Results:
        - TaskPayload:
            The result produced by the tasks
            (potentially nested) of the taskgroup.
    """
    # CAPTURE THE CONTEXT BEFORE THREADS
    context = _dag_execution_context_var.get()

    result = TaskPayload({})
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all elements (both tasks and nested taskgroups) to executor
        futures = {}
        for elmt in tg.elements:
            result[elmt.name] = None  # to preserve keys ordering
            if isinstance(elmt, TaskType):
                future = executor.submit(
                    _execute_task_with_context,
                    context,
                    elmt,
                    parent_results,
                    exec_id,
                    (rank or [])
                )
                futures[future] = elmt.name
            else:
                # Submit nested taskgroups to executor as well
                future = executor.submit(
                    _execute_taskgroup_with_context,
                    context,
                    elmt,
                    parent_results,
                    exec_id,
                    rank
                )
                futures[future] = elmt.name

        # Wait for all submitted tasks and taskgroups to complete
        for future in concurrent.futures.as_completed(futures):
            element_name = futures[future]
            try:
                element_result = future.result()
                if isinstance(element_result, TaskPayload):
                    # For taskgroups, merge all individual task results
                    for task_name, task_result in element_result.items():
                        result[task_name] = task_result
                else:
                    # For regular tasks, store the result directly
                    result[element_name] = element_result
            except Exception as exc:
                # Log the exception and re-raise to maintain error behavior
                logging.getLogger().error(
                    f"Element {element_name} generated an exception: {exc}"
                )
                raise

    return result

def _execute_task_with_context(
    context: DagExecutionContext,
    t: TaskType,
    parent_results: TaskPayload,
    exec_id: int,
    rank: Optional[List[int]] = None
) -> TaskPayload:
    """Wrapper that sets context before executing task."""
    token = _dag_execution_context_var.set(context)
    try:
        return _execute_regular_task(t, parent_results, exec_id, rank)
    finally:
        _dag_execution_context_var.reset(token)

def _execute_taskgroup_with_context(
    context: DagExecutionContext,
    tg: TaskGroup,
    parent_results: TaskPayload,
    exec_id: int,
    rank: Optional[List[int]] = None
) -> TaskPayload:
    """Wrapper that sets context before executing taskgroup."""
    token = _dag_execution_context_var.set(context)
    try:
        return _execute_regular_taskgroup(tg, parent_results, exec_id, rank)
    finally:
        _dag_execution_context_var.reset(token)


def execute(
    dag: DAG,
    params: Optional[Dict[str, Any]] = None
) -> (TaskPayload, dict):
    """From the start, executes the DAG that contains the given task.

    Handles task-groups, parallel and nested parallel tasks,
    and merges results as needed.

    Params:
        - task (TaskType):
            A task from the DAG to be executed.
        - params (Optional[Dict[str, Any]])
            Parameters for this DAG execution.

    Results:
        - (TaskPayload)
            results of the last task in the DAG.
        - (dict)
            DAG context dump.
    """
    # Build execution context with defaults and overrides
    params = params or {}
    resolved_params = {}
    for param_name, param_def in dag.params.items():
        if param_name in params:
            resolved_params[param_name] = params[param_name]
        elif param_def.default is not None:
            resolved_params[param_name] = param_def.default
        else:
            resolved_params[param_name] = None

    # threadsafe - set the context for this execution
    context = DagExecutionContext(resolved_params)
    token = _dag_execution_context_var.set(context)

    # actual DAG execution
    try:
        dag.init()
        exec_id = _dag_execution_context_var.get()._params.get("exec_id")
        _rich_log_execution_id_with_timestamp(exec_id)

        log = logging.getLogger()
        log.debug(f"Root tasks: {[t.name for t in dag.roots]}")

        order = _topological_sort(dag.roots)
        log.debug(f"Topological order: {[e.name for e in order]}")

        results = TaskPayload({})
        i = 0
        while i < len(order):
            elmt = order[i]
            parent_results = _collect_parent_results(elmt, results)

            if isinstance(elmt, TaskType):
                task_type = "merge task" if elmt.merge_func else "task"
                elmt.log.info(f"Executing {task_type}: " +
                           f"[rgb(0,255,255) on #af00ff]{elmt.name}[/]")

                if elmt.is_parallel:
                    elmt.log.info(f"sub-DAG parent_results: {parent_results}")

                    # Find the end of this parallel subdag
                    subdag_end = _find_subdag_end(order, i)
                    subdag_tasks = order[i:subdag_end]
                    elmt.log.info("subdag_tasks [red on yellow]{}[/]".format(
                        [t.name for t in subdag_tasks]))

                    # Execute parallel subdag for each input
                    input_count = _parallel_input_count(parent_results)
                    sub_results = _execute_parallel_branches(
                        subdag_tasks, parent_results, input_count, exec_id)

                    # Store results for the last task in subdag
                    results[subdag_tasks[-1].name] = sub_results

                    # Skip to after the subdag
                    i = subdag_end
                    continue
                else:
                    # This includes merge tasks at the top level
                    result = _execute_regular_task(elmt, parent_results, exec_id)
                    results[elmt.name] = result
            else:
                elmt.log.info("Executing taskgroup: " +
                           f"[rgb(0,255,255) on #af00ff]{elmt.name}[/]")
                tg_results = _execute_regular_taskgroup(elmt, parent_results, exec_id)
                for task_name, task_result in tg_results.items():
                    results[task_name] = task_result

            i += 1

        return (
            results[order[-1].name],
            copy.copy(_dag_execution_context_var.get()._params)
        )
    finally:
        # Reset context after execution
        _dag_execution_context_var.reset(token)

