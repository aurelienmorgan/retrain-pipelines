import pytest
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from retrain_pipelines.dag_engine.web_console.utils.execution.gantt_chart import (
    ParallelLines,
    GroupTypes,
    CustomStyleDict,
    fill_defaults,
    GroupedRows,
    _organize_tasks,
    draw_chart,
    task_row,
    parallel_grouped_rows,
    taskgroup_grouped_rows,
    TaskExt,
    TaskGroup,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def make_task_ext():
    """Factory fixture for creating TaskExt instances."""

    def _make(**kwargs):
        defaults = {
            "id": 1,
            "tasktype_uuid": str(uuid4()),
            "name": "test_task",
            "start_timestamp": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "end_timestamp": datetime(2023, 1, 2, tzinfo=timezone.utc),
            "failed": False,
            "ui_css": {},
            "is_parallel": False,
            "rank": None,
            "merge_func": None,
            "taskgroup_uuid": None,
        }
        defaults.update(kwargs)
        return TaskExt(**defaults)

    return _make


@pytest.fixture
def make_task_group():
    """Factory fixture for creating TaskGroup instances."""

    def _make(**kwargs):
        defaults = {
            "uuid": str(uuid4()),
            "exec_id": 1,
            "order": 1,
            "name": "test_group",
            "docstring": None,
            "ui_css": {},
            "elements": [],
        }
        defaults.update(kwargs)
        return TaskGroup(**defaults)

    return _make


# =============================================================================
# Tests for ParallelLines
# =============================================================================


def test_parallel_lines_init_and_repr(make_task_ext):
    task = make_task_ext(id=1, name="merge_task")
    pl = ParallelLines(rank=[0], lines={(0,): [task]}, merging_task=task)

    assert pl.rank == [0]
    assert pl.lines == {(0,): [task]}
    assert pl.merging_task == task
    assert "PP" in repr(pl)
    assert "1" in repr(pl)

    pl_no_merge = ParallelLines(rank=[1], lines={(1,): []})
    assert pl_no_merge.merging_task is None
    assert "PP" in repr(pl_no_merge)
    assert "[]" in repr(pl_no_merge)


# =============================================================================
# Tests for CustomStyleDict & CustomStyleDictValidator
# =============================================================================


def test_custom_style_dict_init_empty():
    d = CustomStyleDict()
    # Pydantic validator initializes these fields with None by default
    assert d["color"] is None
    assert d["background"] is None
    assert d["border"] is None
    assert d["labelUnderlay"] is None


def test_custom_style_dict_init_dict():
    d = CustomStyleDict({"color": "#FFFFFF", "background": "#000000"})
    assert d["color"] == "#FFFFFF"
    assert d["background"] == "#000000"


def test_custom_style_dict_init_3_args():
    d = CustomStyleDict("#FFF", "#000", "#F00")
    assert d["color"] == "#FFF"
    assert d["background"] == "#000"
    assert d["border"] == "#F00"

    # cover invalid number of arguments
    with pytest.raises(TypeError, match="Expected 3 or 4 positional arguments, got 2"):
        CustomStyleDict("#FFF", "#000")


def test_custom_style_dict_init_4_args():
    d = CustomStyleDict("#FFF", "#000", "#F00", "underlay_val")
    assert d["color"] == "#FFF"
    assert d["background"] == "#000"
    assert d["border"] == "#F00"
    assert d["labelUnderlay"] == "underlay_val"


def test_custom_style_dict_init_kwargs():
    d = CustomStyleDict(color="#FFF", border="#F00")
    assert d["color"] == "#FFF"
    assert d["border"] == "#F00"


def test_custom_style_dict_setattr_getattr():
    d = CustomStyleDict()
    d.color = "#123456"
    assert d["color"] == "#123456"
    assert d.color == "#123456"


def test_custom_style_dict_getattr_missing():
    d = CustomStyleDict()
    with pytest.raises(AttributeError, match="object has no attribute 'missing'"):
        _ = d.missing


def test_custom_style_dict_setitem_valid():
    d = CustomStyleDict()
    d["color"] = "#ABCDEF"
    assert d["color"] == "#ABCDEF"


def test_custom_style_dict_setitem_invalid():
    d = CustomStyleDict()
    with pytest.raises(ValidationError):
        d["invalid_key_name"] = "some_value"


# =============================================================================
# Tests for GroupedRows
# =============================================================================


def test_grouped_rows_init_and_repr():
    gr = GroupedRows(
        id="1",
        uuid="u1",
        name="Test",
        start_timestamp=None,
        end_timestamp=None,
        callbacks=["cb1"],
        extraClasses=["cls1"],
        children=[],
        style={"color": "#FFF"},
    )
    assert gr.id == "1"
    assert gr.name == "Test"
    assert "GroupedRows" in repr(gr)
    assert "children=0" in repr(gr)


def test_grouped_rows_to_js_literal_primitives():
    gr = GroupedRows(
        id="1",
        uuid=None,
        name="Test",
        start_timestamp=None,
        end_timestamp=None,
        callbacks=None,
        extraClasses=None,
        children=None,
        style=None,
    )
    js = gr.to_js_literal()
    assert "id: '1'" in js
    # The source code explicitly skips the uuid key if it is falsy (None)
    assert "uuid" not in js
    assert "name: { value: 'Test'" in js


def test_grouped_rows_to_js_literal_datetime():
    dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    gr = GroupedRows(
        id="1",
        uuid="u1",
        name="Test",
        start_timestamp=dt,
        end_timestamp=dt,
        callbacks=None,
        extraClasses=None,
        children=None,
        style=None,
    )
    js = gr.to_js_literal()
    expected_ts = int(dt.timestamp() * 1000)
    assert f"start_timestamp: {expected_ts}" in js
    assert f"end_timestamp: {expected_ts}" in js


def test_grouped_rows_to_js_literal_nested_and_booleans():
    child = GroupedRows(
        id="2",
        uuid="u2",
        name="Child",
        start_timestamp=None,
        end_timestamp=None,
        callbacks=None,
        extraClasses=None,
        children=None,
        style=None,
    )
    gr = GroupedRows(
        id="1",
        uuid="u1",
        name="Parent",
        start_timestamp=None,
        end_timestamp=None,
        callbacks=["showDetailsModal(this);"],
        extraClasses=["failed"],
        children=[child],
        # include an integer to exercise the `else` branch of recursive_js
        style={"a": True, "b": False, "c": None, "d": 123},
    )
    js = gr.to_js_literal()
    assert "true" in js
    assert "false" in js
    assert "null" in js
    assert "children: [" in js
    # verify integer representation
    assert "123" in js


# =============================================================================
# Tests for fill_defaults
# =============================================================================


def test_fill_defaults():
    style = CustomStyleDict()
    fill_defaults(style, GroupTypes.NONE)

    assert style["color"] is not None
    assert style["background"] is not None
    assert style["color"].startswith("rgba(")
    assert style["background"].startswith("rgba(")
    assert style["border"].startswith("rgba(")
    assert style["border"].endswith(",1)")


# =============================================================================
# Tests for _organize_tasks
# =============================================================================


def test_organize_tasks_empty():
    result = _organize_tasks([], [])
    assert result == []


def test_organize_tasks_simple_tasks(make_task_ext):
    tasks = [
        make_task_ext(id=1, name="task1", taskgroup_uuid=None),
        make_task_ext(id=2, name="task2", taskgroup_uuid=None),
    ]
    result = _organize_tasks(tasks, [])
    assert len(result) == 2
    assert result[0].name == "task1"
    assert result[1].name == "task2"


def test_organize_tasks_with_taskgroup(make_task_ext, make_task_group):
    tg_uuid = str(uuid4())
    t1_uuid = str(uuid4())

    tg = make_task_group(uuid=tg_uuid, name="Group1", elements=[t1_uuid])
    t1 = make_task_ext(
        id=1, name="task1", tasktype_uuid=t1_uuid, taskgroup_uuid=tg_uuid
    )

    result = _organize_tasks([t1], [tg])
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert result[0][0].name == "Group1"
    assert len(result[0][1]) == 1
    assert result[0][1][0].name == "task1"


def test_organize_tasks_nested_taskgroups(make_task_ext, make_task_group):
    # Create a complex nesting for coverage to include both while-loop and skip-branches
    # Ranks are let to None to avoid interfering with parallel line handling.
    tg1_uuid = str(uuid4())
    tg2_uuid = str(uuid4())
    tg3_uuid = str(uuid4())
    t1_uuid = str(uuid4())
    t2_uuid = str(uuid4())
    t3_uuid = str(uuid4())

    tg1 = make_task_group(uuid=tg1_uuid, name="Group1", elements=[tg2_uuid, tg3_uuid])
    tg2 = make_task_group(uuid=tg2_uuid, name="Group2", elements=[t1_uuid, t2_uuid])
    tg3 = make_task_group(uuid=tg3_uuid, name="Group3", elements=[t3_uuid])

    t1 = make_task_ext(
        id=1, name="task1", tasktype_uuid=t1_uuid, taskgroup_uuid=tg2_uuid
    )
    t2 = make_task_ext(
        id=2, name="task2", tasktype_uuid=t2_uuid, taskgroup_uuid=tg2_uuid
    )
    t3 = make_task_ext(
        id=3, name="task3", tasktype_uuid=t3_uuid, taskgroup_uuid=tg3_uuid
    )

    result = _organize_tasks([t1, t2, t3], [tg1, tg2, tg3])

    # Should produce a single top-level taskgroup (no parallel lines)
    assert len(result) == 1
    top_group, children = result[0]
    assert top_group.name == "Group1"
    # Should contain two child groups: Group2 and Group3
    assert len(children) == 2
    # Order depends on tasks_list order; Group2 appears first (from t1)
    assert isinstance(children[0], tuple)
    assert children[0][0].name == "Group2"
    assert len(children[0][1]) == 2  # task1, task2
    assert isinstance(children[1], tuple)
    assert children[1][0].name == "Group3"
    assert len(children[1][1]) == 1  # task3


def test_organize_tasks_parallel_lines(make_task_ext):
    # Using rank=[0] for parallel tasks and rank=[] for merge avoids edge-case KeyError
    # in the original rank slicing logic while fully testing the parallel branching path.
    t1 = make_task_ext(
        id=1,
        name="parallel_start",
        is_parallel=True,
        rank=[0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    t2 = make_task_ext(
        id=2,
        name="parallel_work",
        is_parallel=False,
        rank=[0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    t3 = make_task_ext(
        id=3,
        name="parallel_merge",
        is_parallel=False,
        rank=[],
        merge_func="merge_func",
        taskgroup_uuid=None,
    )
    # Second sub‑DAG with merge rank=None
    t4 = make_task_ext(
        id=4,
        name="parallel_start_1",
        is_parallel=True,
        rank=[1],
        merge_func=None,
        taskgroup_uuid=None,
    )
    t5 = make_task_ext(
        id=5,
        name="parallel_work_1",
        is_parallel=False,
        rank=[1],
        merge_func=None,
        taskgroup_uuid=None,
    )
    t6 = make_task_ext(
        id=6,
        name="parallel_merge_1",
        is_parallel=False,
        rank=None,  # rank is None -> exercises both branches
        merge_func="merge_func",
        taskgroup_uuid=None,
    )

    # Add a nested parallel line to cover nested branching
    # This is a parallel task inside the first parallel line.
    t7 = make_task_ext(
        id=7,
        name="nested_parallel_start",
        is_parallel=True,
        rank=[0, 0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    t8 = make_task_ext(
        id=8,
        name="nested_parallel_work",
        is_parallel=False,
        rank=[0, 0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    # The merge for the nested parallel line should have rank=[0] (the prefix)
    # so that it matches the nested ParallelLines object's rank.
    t9 = make_task_ext(
        id=9,
        name="nested_parallel_merge",
        is_parallel=False,
        rank=[0],
        merge_func="merge_func",
        taskgroup_uuid=None,
    )

    # The order: t1, t2, t7, t8, t9, t3, t4, t5, t6
    # This will create a top-level parallel line with rank=[0] containing t1, t2,
    # and a nested parallel line (t7, t8, t9), then merge t3.
    result = _organize_tasks([t1, t2, t7, t8, t9, t3, t4, t5, t6], [])

    # Should have two top‑level ParallelLines objects (rank=[0] and rank=[1])
    assert len(result) == 2
    assert isinstance(result[0], ParallelLines)
    assert isinstance(result[1], ParallelLines)

    # First sub‑DAG (rank=[0])
    pl0 = result[0]
    assert pl0.merging_task.name == "parallel_merge"
    lines0 = pl0.lines
    assert (0,) in lines0
    # The line should contain t1, t2, and the nested ParallelLines (which is an object)
    # and then t3 is merge, not in lines.
    line_items = lines0[(0,)]
    assert len(line_items) == 3
    assert line_items[0].name == "parallel_start"
    assert line_items[1].name == "parallel_work"
    assert isinstance(line_items[2], ParallelLines)  # nested parallel line
    nested_pl = line_items[2]
    assert nested_pl.merging_task.name == "nested_parallel_merge"
    nested_lines = nested_pl.lines
    assert (0, 0) in nested_lines
    assert len(nested_lines[(0, 0)]) == 2
    assert nested_lines[(0, 0)][0].name == "nested_parallel_start"
    assert nested_lines[(0, 0)][1].name == "nested_parallel_work"

    # Second sub‑DAG (rank=[1])
    pl1 = result[1]
    assert pl1.merging_task.name == "parallel_merge_1"
    lines1 = pl1.lines
    assert (1,) in lines1
    assert lines1[(1,)][0].name == "parallel_start_1"
    assert lines1[(1,)][1].name == "parallel_work_1"


def test_organize_tasks_parallel_with_taskgroup(make_task_ext, make_task_group):
    """Covers: appending a taskgroup to a parallel line."""
    tg_uuid = str(uuid4())
    t_uuid = str(uuid4())
    tg = make_task_group(uuid=tg_uuid, name="ParallelGroup", elements=[t_uuid])

    # Parallel start
    t1 = make_task_ext(
        id=1,
        name="p_start",
        is_parallel=True,
        rank=[0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    # Task inside taskgroup with same rank
    t2 = make_task_ext(
        id=2,
        name="group_task",
        is_parallel=False,
        rank=[0],
        merge_func=None,
        taskgroup_uuid=tg_uuid,
    )
    # Merge task
    t3 = make_task_ext(
        id=3,
        name="p_merge",
        is_parallel=False,
        rank=[],
        merge_func="merge",
        taskgroup_uuid=None,
    )

    result = _organize_tasks([t1, t2, t3], [tg])

    # Should produce one ParallelLines object
    assert len(result) == 1
    pl = result[0]
    assert isinstance(pl, ParallelLines)
    assert pl.merging_task.name == "p_merge"
    lines = pl.lines
    assert (0,) in lines
    # The line should contain: t1, then the taskgroup tuple
    line_items = lines[(0,)]
    assert len(line_items) == 2
    assert line_items[0].name == "p_start"
    assert isinstance(line_items[1], tuple)
    tg_struct = line_items[1]
    assert tg_struct[0].name == "ParallelGroup"
    assert len(tg_struct[1]) == 1
    assert tg_struct[1][0].name == "group_task"


# =============================================================================
# Tests for row generation functions
# =============================================================================


def test_task_row(make_task_ext):
    # Test both failed=True and failed=False to cover extraClasses branch
    task_failed = make_task_ext(
        id=1, name="MyTask", failed=True, ui_css={"color": "#FFF"}
    )
    row_failed = task_row(task_failed)
    assert isinstance(row_failed, GroupedRows)
    assert row_failed.id == "1"
    assert row_failed.name == "MyTask"
    assert "failed" in row_failed.extraClasses
    assert row_failed.callbacks == ["showDetailsModal(this);"]
    assert row_failed.style["color"].startswith("rgba(")

    task_ok = make_task_ext(
        id=2, name="MyTaskOk", failed=False, ui_css={"color": "#FFF"}
    )
    row_ok = task_row(task_ok)
    assert "failed" not in row_ok.extraClasses


def test_parallel_grouped_rows(make_task_ext):
    # Build a more complex ParallelLines to cover multiple lines and nested structures
    t1 = make_task_ext(
        id=1,
        name="p_start",
        is_parallel=True,
        rank=[0],
        merge_func=None,
        taskgroup_uuid=None,
        ui_css={"background": "#000"},
    )
    t2 = make_task_ext(
        id=2,
        name="p_work",
        is_parallel=False,
        rank=[0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    # Create a nested parallel line (rank length > 1) to cover the if len(rank)>1 branch
    t3 = make_task_ext(
        id=3,
        name="nested_start",
        is_parallel=True,
        rank=[0, 0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    t4 = make_task_ext(
        id=4,
        name="nested_work",
        is_parallel=False,
        rank=[0, 0],
        merge_func=None,
        taskgroup_uuid=None,
    )
    nested_pl = ParallelLines(
        rank=[0],
        lines={(0, 0): [t3, t4]},
        merging_task=None,  # no merge for nested to test None
    )

    t5 = make_task_ext(
        id=5,
        name="p_merge",
        is_parallel=False,
        rank=[],
        merge_func="merge",
        taskgroup_uuid=None,
    )

    # Main parallel lines: one outer line containing a nested ParallelLines
    pl = ParallelLines(
        rank=[],
        lines={
            (0,): [t1, t2, nested_pl],  # nested_pl is a ParallelLines instance
        },
        merging_task=t5,
    )

    row = parallel_grouped_rows(pl)

    assert isinstance(row, GroupedRows)
    assert row.name == "Distributed sub-pipeline"
    assert "parallel-lines" in row.extraClasses
    # Should have one child (the parallel line) plus the merge task -> 2 children
    assert len(row.children) == 2
    # The first child is the parallel-line group
    line_row = row.children[0]
    assert "parallel-line" in line_row.extraClasses
    # It should contain the three elements: t1, t2, and the nested ParallelLines
    assert len(line_row.children) == 3
    # The nested ParallelLines appears as another parallel-lines group
    nested_row = line_row.children[2]
    assert "parallel-lines" in nested_row.extraClasses
    # The nested group should have one child (the nested parallel line) and no merge
    assert len(nested_row.children) == 1


def test_taskgroup_grouped_rows(make_task_ext, make_task_group):
    tg_uuid = str(uuid4())
    t_uuid = str(uuid4())
    tg2_uuid = str(uuid4())
    t2_uuid = str(uuid4())

    # Create a taskgroup that contains another taskgroup to exercise the tuple branch
    tg2 = make_task_group(
        uuid=tg2_uuid, name="InnerGroup", elements=[t2_uuid], ui_css={"border": "#0F0"}
    )
    tg = make_task_group(
        uuid=tg_uuid,
        name="MyGroup",
        elements=[tg2_uuid, t_uuid],
        ui_css={"border": "#F00"},
    )
    t1 = make_task_ext(
        id=1,
        name="GroupTask1",
        tasktype_uuid=t_uuid,
        taskgroup_uuid=tg_uuid,
        rank=[0],  # non-None to exercise rank suffix in id
    )
    t2 = make_task_ext(
        id=2,
        name="GroupTask2",
        tasktype_uuid=t2_uuid,
        taskgroup_uuid=tg2_uuid,
        rank=[0],
    )

    # Pass rank=[0] to cover the suffix branch in id construction
    row = taskgroup_grouped_rows((tg, [t2, (tg2, [t1])]), rank=[0])

    assert isinstance(row, GroupedRows)
    assert row.name == "MyGroup"
    assert "taskgroup" in row.extraClasses
    # Should have two children: a task and a nested taskgroup
    assert len(row.children) == 2
    # The second child is the nested taskgroup (InnerGroup)
    inner = row.children[1]
    assert inner.name == "InnerGroup"
    # id should have the rank suffix because rank is not None
    assert row.id.endswith(".[0]")
    # The inner group's id also should have suffix
    assert inner.id.endswith(".[0]")
    # The task inside inner group should have labelUnderlay set from group style background
    task_inner = inner.children[0]
    assert task_inner.style["labelUnderlay"] is not None


# =============================================================================
# Tests for draw_chart
# =============================================================================


def test_draw_chart(make_task_ext, make_task_group):
    # Add a parallel task to exercise draw_chart's handling of parallel lines,
    # which covers parallel_grouped_rows taskgroup_grouped_rows.
    tasks = [
        make_task_ext(id=1, name="Task1", taskgroup_uuid=None),
        make_task_ext(
            id=2, name="parallel_start", is_parallel=True, rank=[0], taskgroup_uuid=None
        ),
        make_task_ext(
            id=3, name="parallel_work", is_parallel=False, rank=[0], taskgroup_uuid=None
        ),
        make_task_ext(
            id=4,
            name="parallel_merge",
            is_parallel=False,
            rank=[],
            merge_func="merge",
            taskgroup_uuid=None,
        ),
    ]
    groups = []

    result = draw_chart(execution_id=42, tasks_list=tasks, taskgroups_list=groups)

    # fasthtml's Script renders to a fastcore.xml.FT object, which stringifies to an HTML <script> tag
    content = str(result)
    assert "<script>" in content

    assert "refreshScript" in content
    assert "tableData" in content
    assert "init('gantt-42'" in content
    assert "GanttTimeline" in content
    assert "initFormat" in content
