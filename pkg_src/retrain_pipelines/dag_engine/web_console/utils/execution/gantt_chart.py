
import logging

from enum import Enum, auto
from datetime import datetime
from collections import defaultdict
from pydantic import BaseModel, Field, \
    ConfigDict
from typing import List, Tuple, Union, \
    Optional

from fasthtml.common import Div, Style, Script, \
    to_xml

from ....db.model import TaskExt, TaskGroup
from .....utils import hex_to_rgba


class ParallelLines:
    rank: Optional[List[int]]
    lines: dict
    merging_task: Optional[TaskExt]

    def __init__(self, rank, lines, merging_task=None):
        self.rank = rank
        self.lines = lines
        self.merging_task = merging_task

    def __repr__(self):
        return "PP" + str([self.lines] + \
               ([self.merging_task] if self.merging_task else []))


class GroupTypes(Enum):
    NONE = auto()
    TASKGROUP = auto()
    PARALLEL_LINES = auto()
    PARALLEL_LINE = auto()


class Style(dict):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        return instance

    def __init__(self, *args, **kwargs):
        if args:
            if len(args) == 1 and (isinstance(args[0], dict) or not args[0]):
                data = args[0] or {}
            elif len(args) in [3, 4]:
                data = {"color": args[0], "background": args[1], "border": args[2]}
                if len(args) == 4 and args[3]:
                    data["labelUnderlay"] = args[3]
            else:
                raise TypeError(f"Expected 3 or 4 positional arguments, got {len(args)}")
            if kwargs:
                data.update(kwargs)
        else:
            data = kwargs

        # Pydantic validation
        validated = StyleValidator(**data)
        super().__init__(validated.model_dump())

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Style' object has no attribute '{key}'")

    def __setitem__(self, key, value):
        # Validate on set
        temp = dict(self)
        temp[key] = value
        validated = StyleValidator(**temp)
        super().__setitem__(key, value)

class StyleValidator(BaseModel):
    model_config = ConfigDict(extra='forbid')

    color: Optional[str] = None
    background: Optional[str] = None
    border: Optional[str] = None
    labelUnderlay: Optional[str] = None


DEFAULT_GROUP_STYLES = {
    GroupTypes.NONE:
        Style("#FFFFFF", "#4d0066", "#FFD700"),
    GroupTypes.TASKGROUP:
        Style("#FFD700", "#4d0066", "#000000"),
    GroupTypes.PARALLEL_LINES:
        Style("#FFFFFF", "#4d0066", "#FEAFFF"),
    GroupTypes.PARALLEL_LINE:
        Style("#FFFFFF", "#4d0066", "#FFD700")
}


def fill_defaults(
    group_style: Style,
    group_type: GroupTypes
) -> None:
    """
    in place
    """
    group_style.color = hex_to_rgba(
        (
            group_style.color or
            DEFAULT_GROUP_STYLES[group_type].color
        ),
        0.75
    )
    group_style.background = hex_to_rgba(
        (
            group_style.background or
            DEFAULT_GROUP_STYLES[group_type].background
        ),
        0.65
    )
    group_style.border = hex_to_rgba(
        (
            group_style.border or
            DEFAULT_GROUP_STYLES[group_type].border
        ),
        # no transparency, so border color DOES NOT
        # vary depending on the row it is hovering
        1
    )

    return None


class GroupedRows:
    """collapsible grouped table rows serializer."""
    def __init__(
        self,
        id: str,
        name: str,
        start_timestamp: Optional[datetime],
        end_timestamp: Optional[datetime],
        callbacks: Optional[str],
        extraClasses: Optional[List[str]],
        children: Optional[List["GroupedRows"]],
        style: Optional[dict]
    ):
        self.id = id
        self.name = name
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.callbacks = callbacks
        self.extraClasses = extraClasses
        self.children = children if children is not None else []
        self.style = style

    def to_js_literal(self):
        """
        Convert a nested Python structure
        to a JavaScript literal string.

        Recursively serializes Python
        dicts, lists, primitives, None, and booleans
        to a valid JavaScript object literal (not JSON):
        dictionary keys are unquoted, None becomes null,
        True/False map to true/false,
        and strings are single-quoted.
        """
        def recursive_js(obj):
            # If object is class with to_js_literal, call it
            if hasattr(obj, "to_js_literal") and callable(obj.to_js_literal):
                return obj.to_js_literal()
            # For dict, recurse over key-values
            elif isinstance(obj, dict):
                items = ", ".join(f"{k}: {recursive_js(v)}" for k, v in obj.items())
                return "{" + items + "}"
            # For list (or tuple), recurse over items
            elif isinstance(obj, (list, tuple)):
                items = ", ".join(recursive_js(i) for i in obj)
                return "[" + items + "]"
            # For strings, escape and quote
            elif isinstance(obj, str):
                return repr(obj)
            elif obj is None:
                return "null"
            elif isinstance(obj, bool):
                return "true" if obj else "false"
            elif isinstance(obj, datetime):
                return int(obj.timestamp() * 1000)
            # For other primitives, just str
            else:
                return repr(str(obj))

        js_literal = (
            "{" +
                f"id: {recursive_js(self.id)}, " +
                "cells: {" +
                    f"name: {{ value: {recursive_js(self.name)}, attributes: {{}} }}, " +
                    "timeline: {" +
                        "value: null, " +
                        "attributes: {" +
                            (
                                f"start_timestamp: {recursive_js(self.start_timestamp)}, "
                                if self.start_timestamp else ""
                            ) +
                            (
                                f"end_timestamp: {recursive_js(self.end_timestamp)}"
                                if self.end_timestamp else ""
                            ) +
                        "}" +
                    "}" +
                "}, " +
                (f"callbacks: {recursive_js(self.callbacks)}, " if self.callbacks else "") +
                (f"extraClasses: {recursive_js(self.extraClasses)}, " if self.extraClasses else "") +
                (f"children: {recursive_js(self.children)}, " if self.children else "") +
                f"style: {recursive_js(self.style)}" +
            "}"
        )
        return js_literal


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"children={len(self.children)})"
        )


def _organize_tasks(
    tasks_list: List[TaskExt],
    taskgroups_list: List[TaskGroup]
) -> List[Union[TaskExt, Tuple]]:
    """Returns a topologically-organized structure

    of executed task instances (incl. taskgroups
    and parallel sub-DAG lines).

    e.g.:
    ```python
    tasks_list = [
        Task("task1", ""),
        Task("task2", "tg1"),
        Task("task3", "tg1"),
        Task("task4", "tg2"),
        Task("task5", "tg2"),
        Task("task6"),
        Task("task7", "tg3"),
        Task("task8", "tg3"),
        Task("task9")
    ]
    taskgroups_list = [
        TaskGroup("tg1", ["task2", "task3", "tg2"]),
        TaskGroup("tg2", ["task4", "task5"]),
        TaskGroup("tg3", ["task7", "task8"])
    ]
    organize_tasks(tasks_list, taskgroups_list)
    # returns
    # ['task1', ('tg1', ['task2', 'task3',
    #                    ('tg2', ['task4', 'task5'])]),
    #  'task6', ('tg3', ['task7', 'task8']), 'task9']
    ```

    Params:
        - tasks_list (List[TaskExt]):
            the exhaustive (topologically sorted)
            list of tasks for a given execution.
        - taskgroups_list (List[TaskGroup]):
            the exhaustive (topologically sorted)
            list of taskgroups for that execution.

    Results:
        - (List[Union[TaskExt, Tuple]]):
            a nested list of
            "task" and "tuple(taskgroup, tasks)".
    """
    task_by_id = {task_ext.id: task_ext for task_ext in tasks_list}
    taskgroup_by_uuid = {str(tg.uuid): tg for tg in taskgroups_list} \
                        if taskgroups_list else {}

    children_map = {}
    parent_of_taskgroup = {}
    if taskgroups_list:
        # Map each taskgroup to its direct children (tasks and taskgroups)
        for tg in taskgroups_list:
            for tg_element_uuid in tg.elements:
                # check if element is a task and, if so, add its id
                for task_ext in tasks_list:
                    if str(task_ext.tasktype_uuid) == tg_element_uuid:
                        children_map[str(tg.uuid)] = \
                            ([] if not str(tg.uuid) in children_map
                             else children_map[str(tg.uuid)]) + [task_ext.id]
                        # DO NOT BREAK, parallel tasks have different ids
                        # but common tasktype uuid

                children_map[str(tg.uuid)] = \
                    ([] if not str(tg.uuid) in children_map
                     else children_map[str(tg.uuid)]) + tg.elements

        # Determine which taskgroup (if any) contains each taskgroup
        for tg in taskgroups_list:
            for element in tg.elements:
                if element in taskgroup_by_uuid:
                    parent_of_taskgroup[element] = str(tg.uuid)

    emitted = set()
    result = []

    def emit_taskgroup(tg_uuid, rank: Optional[List[str]]):
        takgroup_rank = None

        children = []
        for element in children_map[tg_uuid]:
            if element in task_by_id:
                if (
                    element not in emitted and (
                        rank is None or rank == task_by_id[element].rank
                    )
                ):
                    emitted.add(element)
                    children.append(task_by_id[element])
            elif element in taskgroup_by_uuid:
                if f"{rank or []}-{element}" not in emitted:
                    children.append(emit_taskgroup(element, rank))

        emitted.add(f"{rank or []}-{tg_uuid}")

        return (taskgroup_by_uuid[tg_uuid], children)


    branching_number_in_series = {(): 0} # counter on occurence,
                                         # in case several are chained in series
                                         # (open-and-close, then again open-and-close)
    parallel_lines = {}                  # we identify sub-DAGs with "top/number-depth/rank"

    for task_ext in tasks_list:
        # print(f"{task_ext.id}-{task_ext.tasktype_uuid}, {task_ext.rank}, {branching_number_in_series}, {list(parallel_lines.keys())}")

        if task_ext.taskgroup_uuid is None or task_ext.taskgroup_uuid == "":
            if task_ext.is_parallel:
                if tuple(task_ext.rank[:-1]) not in branching_number_in_series:
                    branching_number_in_series[tuple(task_ext.rank[:-1])] = 0

                branching_number = branching_number_in_series[tuple(task_ext.rank[:-1])] \
                                   if task_ext.rank is not None else None

                if f"{branching_number}-{task_ext.rank[:-1]}" not in parallel_lines:
                    # first branch on newly encountered parallel sub-DAG
                    # create and object that will keep on being constructed
                    # as we further loop over tasks_list
                    parallel_lines[f"{branching_number}-{task_ext.rank[:-1]}"] = \
                        ParallelLines(task_ext.rank[:-1], {tuple(task_ext.rank): [task_ext]})
                    if len(task_ext.rank) == 1:
                        # top-level new sub-DAG
                        result.append(parallel_lines[f"{branching_number}-{task_ext.rank[:-1]}"])
                    else:
                        # nested branching
                        parallel_lines[
                            f"""{
                                    branching_number_in_series[tuple(task_ext.rank[:-2])]
                                }-{
                                    task_ext.rank[:-2]
                                }"""] \
                            .lines[tuple(task_ext.rank[:-1])].append(
                                parallel_lines[f"{branching_number}-{task_ext.rank[:-1]}"]
                            )
                else:
                    # new branch opens on existing parallel sub-DAG (any depth)
                    parallel_lines[f"{branching_number}-{task_ext.rank[:-1]}"] \
                        .lines[tuple(task_ext.rank)] = [task_ext]

            elif task_ext.rank is not None and task_ext.merge_func is None:
                # standard task inside a parallel line (any depth)
                parallel_lines[f"{branching_number}-{task_ext.rank[:-1]}"] \
                    .lines[tuple(task_ext.rank)].append(task_ext)

            elif task_ext.merge_func is not None:
                try:
                    parallel_lines[
                            f"""{branching_number_in_series[
                                    tuple(task_ext.rank) if task_ext.rank is not None else ()
                              ]}-{task_ext.rank if task_ext.rank is not None else []}"""
                        ].merging_task = task_ext
                except Exception as ex:
                    # TODO - bug in the DAG engine where merge tasks
                    # sometimes (TBC, but probably a mishandeled async "future" issue)
                    # get a wrong rank
                    logging.getLogger().error(f"{task_ext} rank={task_ext.rank}?")
                    logging.getLogger().error(ex)

                branching_number_in_series[
                        tuple(task_ext.rank) if task_ext.rank is not None else ()
                    ] += 1

            else:
                result.append(task_ext)
            emitted.add(task_ext.id)
        else:
            tg_uuid = str(task_ext.taskgroup_uuid)
            # Find the top-most taskgroup that contains this task and isn't emitted
            current = tg_uuid
            while current in parent_of_taskgroup:
                parent_uuid = parent_of_taskgroup[current]
                if f"{task_ext.rank or []}-{parent_uuid}" in emitted:
                    break
                current = parent_uuid

            if f"{task_ext.rank or []}-{current}" not in emitted:
                tg_struct = emit_taskgroup(current, rank=task_ext.rank)

                if task_ext.rank is not None and task_ext.merge_func is None:
                    # standard task inside a parallel line (any depth)
                    parallel_lines[f"{branching_number}-{task_ext.rank[:-1]}"] \
                        .lines[tuple(task_ext.rank)].append(tg_struct)
                else:
                    result.append(tg_struct)

    # print("organize_tasks result :\n", result)
    return result


def draw_chart(
    execution_id: int,
    tasks_list: List[TaskExt],
    taskgroups_list: List[TaskGroup]
) -> Script:
    """The js list of row literals.

    for the Gantt diagramm of a given execution.
    Format is the one expected by
    the collapsible-grouped-table init function.

    Params:
        - execution_id (int):
            The id of the execution
            to draw a Gantt diagramm for.
        - tasks_list (List[TaskExt]):
            the exhaustive (topologically sorted)
            list of tasks for a given execution.
        - taskgroups_list (List[TaskGroup]):
            the exhaustive (topologically sorted)
            list of taskgroups for that execution.

    Results:
        - (Script)
    """

    current_organize_tasks = _organize_tasks(tasks_list, taskgroups_list)

    rows = []
    for element in current_organize_tasks:
        if isinstance(element, Tuple):
            # taskgroup
            rows.append(taskgroup_grouped_rows(element))
        elif isinstance(element, ParallelLines):
            # parallel line
            rows.append(parallel_grouped_rows(element))
        else:
            # standalone inline task
            rows.append(task_row(element))

    js_rows = "[" + ", ".join(r.to_js_literal() for r in rows) + "]"
    return (
        Script(f"""
            const tableData = {js_rows};

            function countAllDepthsItems(data) {{
                let count = 0;

                function traverse(items) {{
                    for (const item of items) {{
                        count++; // Count the current item

                        // If item has children, traverse them recursively
                        if (item.children && Array.isArray(item.children)) {{
                            traverse(item.children);
                        }}
                    }}
                }}

                traverse(data);
                return count;
            }}
            console.log(`${{countAllDepthsItems(tableData)}} total rows ` +
                        'in tableData array.');

            /* load gantt-chart data */
            const interBarsSpacing = 2;     /* in px */
            init('gantt-{execution_id}', tableData, interBarsSpacing);

            /* instantiate timelines renderer */
            window.execGanttTimelineObj =
                new GanttTimeline('gantt-{execution_id}', 'timeline');

            /* handle collapsed group-header rows *
            *  for summary timeline timestamps    */
            initFormat('execGanttTimelineObj');
        """)
    )


def task_row(
    task_ext: TaskExt,
    labelUnderlay: Optional[str]=None
) -> GroupedRows:
    """
    Params:
        - task_ext (TaskExt):
        - labelUnderlay (str):
            overwrite default. Used for tasks within taskgroups.
    """

    # fill_defaults, complementing ui_css if need be
    row_style = Style(
        task_ext.ui_css,
        labelUnderlay=labelUnderlay or "#4d0066"
    )
    fill_defaults(row_style, GroupTypes.NONE)

    result = GroupedRows(
        id=task_ext.id,
        name=task_ext.name,
        start_timestamp=task_ext.start_timestamp,
        end_timestamp=task_ext.end_timestamp,
        callbacks=None,
        extraClasses=None if not task_ext.failed else ["failed"],
        children=None,
        style=row_style
    )

    return result


def parallel_grouped_rows(
    parallel_lines: ParallelLines
) -> GroupedRows:
    # get one of the split instances
    # of the opening parallel task
    parralel_task_ext = \
        parallel_lines.lines[next(iter(parallel_lines.lines))][0]
    # print(f"sub-DAG {parralel_task_ext.name} : {parallel_lines}")

    parallel_lines_list = []
    for parallel_line_rank, elements_list in parallel_lines.lines.items():
        line_rows = []
        for element in elements_list:
            if isinstance(element, ParallelLines):
                line_rows.append(parallel_grouped_rows(element))
            elif isinstance(element, Tuple):
                line_rows.append(taskgroup_grouped_rows(element))
            else:
                line_rows.append(task_row(element))

        # fill_defaults, complementing ui_css if need be
        group_rows_style = Style(parralel_task_ext.ui_css, labelUnderlay="#4d0066")
        fill_defaults(group_rows_style, GroupTypes.PARALLEL_LINE)

        parallel_lines_list.append(
            GroupedRows(
                id=f"{parralel_task_ext.name}.{parallel_line_rank}",
                name=f"{parralel_task_ext.name}.{parallel_line_rank}",
                start_timestamp=None,
                end_timestamp=None,
                callbacks=["toggleHeaderTimeline('execGanttTimelineObj', this);"],
                extraClasses=["parallel-line"],
                children=line_rows,
                style=group_rows_style
            )
        )

    if parallel_lines.merging_task is not None:
        # if distributed sub-DAG merging (last) task has started
        parallel_lines_list.append(task_row(parallel_lines.merging_task))

    # fill_defaults, complementing ui_css if need be
    group_rows_style = Style(parralel_task_ext.ui_css, labelUnderlay="#4d0066")
    fill_defaults(group_rows_style, GroupTypes.PARALLEL_LINES)

    return GroupedRows(
        id=parralel_task_ext.name + \
           str(parralel_task_ext.rank[:-1]) if len(parralel_task_ext.rank) > 1 else "",
        name="Distributed sub-pipeline",
        start_timestamp=None,
        end_timestamp=None,
        callbacks=["toggleHeaderTimeline('execGanttTimelineObj', this);"],
                extraClasses=["parallel-lines"],
        children=parallel_lines_list,
        style=group_rows_style
    )


def taskgroup_grouped_rows(
    taskgroup_tuple: Tuple[TaskGroup, List[Union[TaskExt, Tuple]]],
    rank: Optional[List[str]] = None
) -> GroupedRows:
    taskgroup, taskgroup_elements = taskgroup_tuple

    # fill_defaults, complementing ui_css if need be
    group_rows_style = Style(taskgroup.ui_css, labelUnderlay="#4d0066")
    fill_defaults(group_rows_style, GroupTypes.TASKGROUP)

    elements = []
    for element in taskgroup_elements:
        if isinstance(element, Tuple):
            elements.append(taskgroup_grouped_rows(element, rank))
        else:
            # standard task (for taskgroup in parallel branch)
            rank = element.rank
            elements.append(task_row(element, labelUnderlay=group_rows_style.background))

    return GroupedRows(
        id=str(taskgroup.uuid) + f".{rank}" if rank else "",
        name=taskgroup.name,
        start_timestamp=None,
        end_timestamp=None,
        callbacks=["toggleHeaderTimeline('execGanttTimelineObj', this);"],
        extraClasses=["taskgroup"],
        children=elements,
        style=group_rows_style
    )

