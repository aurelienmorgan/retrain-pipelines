
from typing import List, Tuple, Union

from fasthtml.common import Table, Tr, Td

from ....db.model import TaskExt, TaskGroup


def _organize_tasks(
    tasks_list: List[TaskExt],
    taskgroups_list: List[TaskGroup]
) -> List[Union[TaskExt, Tuple]]:
    """Returns a topologically-organized structure

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
                        children_map[str(tg.uuid)] = [task_ext.id] + \
                            ([] if not str(tg.uuid) in children_map
                             else children_map[str(tg.uuid)])
                        # DO NOT BREAK, parallel tasks have different ids
                        # but common tasktype uuid

                children_map[str(tg.uuid)] = tg.elements + \
                    ([] if not str(tg.uuid) in children_map
                     else children_map[str(tg.uuid)])

        # Determine which taskgroup (if any) contains each taskgroup
        for tg in taskgroups_list:
            for element in tg.elements:
                if element in taskgroup_by_uuid:
                    parent_of_taskgroup[element] = str(tg.uuid)

    emitted = set()
    result = []

    def emit_taskgroup(tg_uuid):
        emitted.add(tg_uuid)
        children = []
        for element in children_map[tg_uuid]:
            if element in task_by_id:
                if element not in emitted:
                    emitted.add(element)
                    children.append(task_by_id[element])
            elif element in taskgroup_by_uuid:
                if element not in emitted:
                    children.append(emit_taskgroup(element))
        return (taskgroup_by_uuid[tg_uuid], children)

    for task_ext in tasks_list:
        if task_ext.id in emitted:
            continue
        if task_ext.taskgroup_uuid is None or task_ext.taskgroup_uuid == "":
            result.append(task_ext)
            emitted.add(task_ext.id)
        else:
            tg_uuid = str(task_ext.taskgroup_uuid)
            # Find the top-most taskgroup that contains this task and isn't emitted
            current = tg_uuid
            while current in parent_of_taskgroup:
                parent_uuid = parent_of_taskgroup[current]
                if parent_uuid in emitted:
                    break
                current = parent_uuid
            if current not in emitted:
                tg_struct = emit_taskgroup(current)
                result.append(tg_struct)

    # print("organize_tasks result :\n", result)
    return result


def draw_chart(
    tasks_list: List[TaskExt],
    taskgroups_list: List[TaskGroup]
) -> Table:
    """The html DOM element

    for the Gantt diagramm of a given execution.

    Params:
        - tasks_list (List[TaskExt]):
            the exhaustive (topologically sorted)
            list of tasks for a given execution.
        - taskgroups_list (List[TaskGroup]):
            the exhaustive (topologically sorted)
            list of taskgroups for that execution.

    Results:
        - (Table)
    """

    current_organize_tasks = _organize_tasks(tasks_list, taskgroups_list)

    rows = []
    for element in current_organize_tasks:
        if isinstance(element, Tuple):
            rows.append(taskgroup_table(element))
        else:
            rows.append(task_row(element))

    return Table(*rows)


def task_row(task_ext: TaskExt) -> Tr:
    result = Tr(
        Td(
            task_ext.name
        ),
        id=task_ext.id
    )

    return result


def taskgroup_table(
    taskgroup_tuple: Tuple[TaskGroup, List[Union[TaskExt, Tuple]]]
) -> Tr:
    taskgroup, taskgroup_elements = taskgroup_tuple

    rows = []
    for element in taskgroup_elements:
        if isinstance(element, Tuple):
            rows.append(taskgroup_table(element))
        else:
            rows.append(task_row(element))

    return Tr(Td(Table(
            (taskgroup.name, *rows),
            id=taskgroup.uuid
        )))


















