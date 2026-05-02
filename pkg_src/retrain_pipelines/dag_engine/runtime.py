
import os
import copy
import time
import logging
import cloudpickle

from collections import defaultdict, deque
from typing import List, Optional, Union, \
    Tuple, Dict, Any

from concurrent.futures import \
    ProcessPoolExecutor, Executor, as_completed

from .core import TaskType, TaskGroup, \
    TaskPayload, TaskFuncException, \
    DAG, DagExecutionContext, \
    _dag_execution_context_var, get_trace_buffer

from .rp_logging import RichLoggingController

from .hybrid_pool_executor import HybridPoolExecutor as RetrainPipelinesExecutor
# from concurrent.futures import ThreadPoolExecutor as RetrainPipelinesExecutor
# from .hybrid_pool_executor import CloudpickleProcessPoolExecutor as RetrainPipelinesExecutor
# from concurrent.futures import ProcessPoolExecutor as RetrainPipelinesExecutor

from .grpc_client import GrpcClient


logger = logging.getLogger(__name__)


################################################################


# ---- Keyboard Interrupt Handling Infrastructure ----

import sys
import signal
import threading

from datetime import datetime, timezone

from .db.dao import DAO


class _TaskRegistry:
    """Thread-safe registry to track all running tasks for interrupt handling."""

    def __init__(self):
        self._lock = threading.Lock()
        # Map task_id -> pid
        self._running_tasks: Dict[int, Tuple[Union[int, None], str]] = {}
        self._interrupted = False

    def register_task(
        self, task_id: int, pid: int
    ):
        """Register a running task with its PID."""
        with self._lock:
            self._running_tasks[task_id] = pid

    def unregister_task(self, task_id: int):
        """Unregister a completed task."""
        with self._lock:
            self._running_tasks.pop(task_id, None)

    def get_running_tasks(self) -> Dict[int, Tuple[Union[int, None], str]]:
        """Get a snapshot of all currently running tasks."""
        with self._lock:
            return dict(self._running_tasks)

    def mark_interrupted(self):
        """Mark that an interrupt has been requested."""
        with self._lock:
            self._interrupted = True

    def is_interrupted(self) -> bool:
        """Check if an interrupt has been requested."""
        with self._lock:
            return self._interrupted

# Global registry instance
_task_registry = _TaskRegistry()


def _interrupt_thread(thread_ident: int):
    """Interrupt a thread by raising KeyboardInterrupt in it using ctypes."""
    try:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_ident),
            ctypes.py_object(KeyboardInterrupt)
        )
        if res == 0:
            logger.warning(
                f"Failed to interrupt thread {thread_ident}: invalid thread ID"
            )
        elif res > 1:
            # Revert if multiple threads affected
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_ident), None
            )
            logger.error(
                f"Failed to interrupt thread {thread_ident}: multiple threads affected"
            )
    except Exception as ex:
        logger.error(
            f"Exception while interrupting thread {thread_ident}: {ex}"
        )


def _kill_process(pid: int):
    """Kill a process using SIGKILL."""
    try:
        import psutil
        try:
            process = psutil.Process(pid)
            # Kill all children recursively first
            for child in process.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            # Then kill the parent
            process.kill()
        except psutil.NoSuchProcess:
            # Process already terminated
            pass
    except ImportError:
        # Fallback if psutil not available
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            # Process already terminated
            pass


def _update_interrupted_tasks_in_db(task_ids: List[int], exec_id: int):
    """Update database records for interrupted tasks.

    Defensive: instantiate DAO in try/except, update tasks one-by-one catching
    exceptions, and flush logger handlers to maximize the chance logs are visible
    if the process is tearing down.
    """
    logger.info(f"[bold white]_update_interrupted_tasks_in_db {task_ids}[/]")

    try:
        dao = DAO(os.environ["RP_METADATASTORE_URL"])
    except Exception as ex:
        logger.exception(
            "Failed to instantiate DAO ; aborting DB updates.")
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
        return

    end_timestamp = datetime.now(timezone.utc)

    logger.info(f"{os.getpid()} - tata [bold white]  -  {task_ids}[/]")

    for task_id in task_ids:
        logger.info(f"{os.getpid()} - titi [bold white]  -  {task_id}[/]")
        try:
            dao.update_task(
                id=task_id,
                end_timestamp=end_timestamp,
                failed=True
            )
            logger.info(f"{os.getpid()} - toto [bold white]  -  {task_id}[/]")
        except Exception as ex:
            logger.exception(
            f"Failed to update task {task_id} after interrupt: {ex}")
        finally:
            dao.dispose()

    # flush logs to increase visibility during teardown
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass


def _sigint_handler(signum, frame):
    logger.warning(f"{os.getpid()} - KEYBOARD INTERRUPT DETECTED")

    _task_registry.mark_interrupted()
    running_tasks = _task_registry.get_running_tasks()

    if not running_tasks:
        sys.exit(1)

    logger.warning(f"Cancelling {len(running_tasks)} tasks...")

    for task_id, (pid, executor_type, future) in running_tasks.items():
        # CANCEL FUTURE FIRST
        if future and not future.done():
            cancelled = future.cancel()
            logger.warning(f"Cancelled future for task {task_id}: {cancelled}")

        # THEN kill process
        if pid:
            logger.warning(f"Killing process for task {task_id} (PID: {pid})")
            _kill_process(pid)

    # Update DB
    try:
        context = _dag_execution_context_var.get()
        if context:
            exec_id = context._params.get("exec_id")
            if exec_id:
                _update_interrupted_tasks_in_db(list(running_tasks.keys()), exec_id)
    except Exception as ex:
        logger.error(f"Failed to update DB: {ex}")

    sys.exit(1)


def _install_interrupt_handler():
    """Install the SIGINT signal handler for keyboard interrupt handling."""
    signal.signal(signal.SIGINT, _sigint_handler)


################################################################


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


################################################################


def _execute_parallel_branches_with_context(
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
    # CAPTURE THE CONTEXT BEFORE THREADS/PROCESSES
    context = _dag_execution_context_var.get()
    all_updates = {}

    executor = RetrainPipelinesExecutor()

    try:
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

        # Collect results and context updates
        results = []
        for f in futures:
            result, updates = f.result()
            results.append(result)
            # Merge updates (later branches win)
            DagExecutionContext._deep_update(all_updates, updates)

        # Merge all updates back
        context.merge_updates(all_updates)

        return results
    finally:
        # Explicitly shutdown and wait for ALL threads
        # to complete their finally blocks
        executor.shutdown(wait=True, cancel_futures=False)


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
                    nested_results = _execute_parallel_branches_with_context(
                        nested_subdag_tasks, parent_results,
                        len(parent_value), exec_id, rank)
                else:
                    nested_results = _execute_branch(
                        nested_subdag_tasks, parent_results, exec_id, rank)

                branch_results[nested_subdag_tasks[-1].name] = nested_results
                i = nested_subdag_end
                continue
            else:
                # inlione task in parallel branch
                task_type = "merge task" if elmt.merge_func else "task"
                elmt.log.info(f"Executing {task_type}: " +
                            f"[rgb(0,255,255) on #af00ff]{elmt.name}{rank}[/]")
                result = _execute_task(elmt, parent_results, exec_id, rank)
                branch_results[elmt.name] = result
        else:
            # TaskGroup
            elmt.log.info("Executing taskgroup: " +
                        f"[rgb(0,255,255) on #af00ff]{elmt.name}{rank}[/]")
            result = _execute_taskgroup(elmt, parent_results, exec_id, rank)
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
) -> Tuple[TaskPayload, Dict[str, Any]]:
    """Wrapper that sets context before executing branch."""
    token = _dag_execution_context_var.set(context)
    try:
        result = _execute_branch(branch_elements, branch_input, exec_id, rank)
        return result, context.get_updates()
    finally:
        _dag_execution_context_var.reset(token)


def _execute_task(
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


    # import psutil, os, time ; cpu_count = os.cpu_count()
    # parent = psutil.Process(os.getpid())
    # children = parent.children(recursive=True)
    # while True:
        # children = parent.children(recursive=True)
        # num_subprocesses = len(children)
        # print(f"Number of subprocesses: {num_subprocesses}")

        # if num_subprocesses <= 2: # cpu_count:
            # break

        # time.sleep(1)  # Wait before checking again
    # print("Subprocess count is now under or equal to CPU cores.")


    # within the task process, init gRPC connection
    # (for task-traces streaming to WebConsole server)
    GrpcClient.init()
    try:
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
    finally:
        GrpcClient.shutdown()

    return result


def _execute_task_with_context(
    context: DagExecutionContext,
    t: TaskType,
    parent_results: TaskPayload,
    exec_id: int,
    rank: Optional[List[int]] = None
) -> Tuple[TaskPayload, Dict[str, Any]]:
    """Wrapper that sets context before executing task."""
    # We don't copy, but use shared context, and track updates
    token = _dag_execution_context_var.set(context)
    try:
        result = _execute_task(t, parent_results, exec_id, rank)
        # Return result and updates made during execution
        return result, context.get_updates()
    finally:
        _dag_execution_context_var.reset(token)


def _execute_taskgroup(
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
    # CAPTURE THE CONTEXT BEFORE THREADS/PROCESSES
    context = _dag_execution_context_var.get()
    all_updates = {}  # Collect all context updates

    executor = RetrainPipelinesExecutor()

    result = TaskPayload({})
    try:
        # Submit all elements
        futures = {}
        for elmt in tg.elements:
            result[elmt.name] = None
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
                future = executor.submit(
                    _execute_taskgroup_with_context,
                    context,
                    elmt,
                    parent_results,
                    exec_id,
                    rank
                )
                futures[future] = elmt.name

        # Wait for all to complete and merge context updates
        for future in as_completed(futures):
            element_name = futures[future]
            try:
                element_result, updates = future.result()

                # Merge context updates (child wins on conflicts)
                DagExecutionContext._deep_update(all_updates, updates)

                if isinstance(element_result, TaskPayload):
                    for task_name, task_result in element_result.items():
                        result[task_name] = task_result
                else:
                    result[element_name] = element_result
            except Exception as exc:
                logger.error(
                    f"Element {element_name} generated an exception: {exc}"
                )
                raise

        # Merge all updates back into parent context
        context.merge_updates(all_updates)

        return result
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def _execute_taskgroup_with_context(
    context: DagExecutionContext,
    tg: TaskGroup,
    parent_results: TaskPayload,
    exec_id: int,
    rank: Optional[List[int]] = None
) -> Tuple[TaskPayload, Dict[str, Any]]:
    """Wrapper that sets context before executing taskgroup."""
    token = _dag_execution_context_var.set(context)
    try:
        result = _execute_taskgroup(tg, parent_results, exec_id, rank)
        return result, context.get_updates()
    finally:
        _dag_execution_context_var.reset(token)


def _execute(
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
    # Install keyboard interrupt handler
    _install_interrupt_handler()

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

        logger.info(f"Execution ID: {exec_id}")
        logger.debug(f"Root tasks: {[t.name for t in dag.roots]}")

        order = _topological_sort(dag.roots)
        logger.debug(f"Topological order: {[e.name for e in order]}")

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
                    sub_results = _execute_parallel_branches_with_context(
                        subdag_tasks, parent_results, input_count, exec_id)

                    # Store results for the last task in subdag
                    results[subdag_tasks[-1].name] = sub_results

                    # Skip to after the subdag
                    i = subdag_end
                    continue
                else:
                    # This includes merge tasks at the top level
                    result = _execute_task(elmt, parent_results, exec_id)
                    results[elmt.name] = result
            else:
                elmt.log.info("Executing taskgroup: " +
                           f"[rgb(0,255,255) on #af00ff]{elmt.name}[/]")

                tg_results = _execute_taskgroup(elmt, parent_results, exec_id)
                for task_name, task_result in tg_results.items():
                    results[task_name] = task_result

            i += 1

        # Determine final result to return
        if isinstance(order[-1], TaskGroup):
            # Last element is a taskgroup =>
            # collect its task results into TaskPayload
            final_result = TaskPayload({})
            def collect_taskgroup_results(tg):
                for elmt in tg.elements:
                    if isinstance(elmt, TaskType):
                        final_result[elmt.name] = results[elmt.name]
                    elif isinstance(elmt, TaskGroup):
                        collect_taskgroup_results(elmt)
            collect_taskgroup_results(order[-1])
        else:
            final_result = results[order[-1].name]

        return (
            final_result,
            copy.copy(_dag_execution_context_var.get()._params)
        )
    finally:
        while len(_task_registry.get_running_tasks()) != 0:
            logger.info(
                f"[bold white]execute - EXIT - {_task_registry.get_running_tasks()}[/]")
            time.sleep(0.1)

        # Flush all pending traces before marking complete
        get_trace_buffer().stop()

        DAG.mark_complete(exec_id)
        # Reset context after execution
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
    """
    Establish rich custom logging.
    """
    rich_logging_controller = RichLoggingController()
    rich_logging_controller.activate()
    try:
        return _execute(dag, params)
    finally:
        rich_logging_controller.deactivate()

