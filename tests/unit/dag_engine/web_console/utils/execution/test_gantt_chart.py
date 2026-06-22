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
        style={"a": True, "b": False, "c": None},
    )
    js = gr.to_js_literal()
    assert "true" in js
    assert "false" in js
    assert "null" in js
    assert "children: [" in js


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
    tg1_uuid = str(uuid4())
    tg2_uuid = str(uuid4())
    t1_uuid = str(uuid4())
    t2_uuid = str(uuid4())

    tg1 = make_task_group(uuid=tg1_uuid, name="Group1", elements=[tg2_uuid])
    tg2 = make_task_group(uuid=tg2_uuid, name="Group2", elements=[t1_uuid, t2_uuid])

    t1 = make_task_ext(
        id=1, name="task1", tasktype_uuid=t1_uuid, taskgroup_uuid=tg2_uuid
    )
    t2 = make_task_ext(
        id=2, name="task2", tasktype_uuid=t2_uuid, taskgroup_uuid=tg2_uuid
    )

    result = _organize_tasks([t1, t2], [tg1, tg2])
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert result[0][0].name == "Group1"
    assert isinstance(result[0][1][0], tuple)
    assert result[0][1][0][0].name == "Group2"


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

    result = _organize_tasks([t1, t2, t3], [])

    assert len(result) == 1
    assert isinstance(result[0], ParallelLines)
    assert result[0].merging_task.name == "parallel_merge"

    lines_dict = result[0].lines
    assert (0,) in lines_dict
    assert lines_dict[(0,)][0].name == "parallel_start"
    assert lines_dict[(0,)][1].name == "parallel_work"


# =============================================================================
# Tests for row generation functions
# =============================================================================


def test_task_row(make_task_ext):
    task = make_task_ext(id=1, name="MyTask", failed=True, ui_css={"color": "#FFF"})
    row = task_row(task)

    assert isinstance(row, GroupedRows)
    assert row.id == "1"
    assert row.name == "MyTask"
    assert "failed" in row.extraClasses
    assert row.callbacks == ["showDetailsModal(this);"]
    assert row.style["color"].startswith("rgba(")


def test_parallel_grouped_rows(make_task_ext):
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
    t3 = make_task_ext(
        id=3,
        name="p_merge",
        is_parallel=False,
        rank=[],
        merge_func="merge",
        taskgroup_uuid=None,
    )

    pl = ParallelLines(rank=[], lines={(0,): [t1, t2]}, merging_task=t3)
    row = parallel_grouped_rows(pl)

    assert isinstance(row, GroupedRows)
    assert row.name == "Distributed sub-pipeline"
    assert "parallel-lines" in row.extraClasses
    assert len(row.children) == 2  # parallel line + merging task


def test_taskgroup_grouped_rows(make_task_ext, make_task_group):
    tg_uuid = str(uuid4())
    t_uuid = str(uuid4())

    tg = make_task_group(
        uuid=tg_uuid, name="MyGroup", elements=[t_uuid], ui_css={"border": "#F00"}
    )
    t = make_task_ext(
        id=1, name="GroupTask", tasktype_uuid=t_uuid, taskgroup_uuid=tg_uuid, rank=[0]
    )

    row = taskgroup_grouped_rows((tg, [t]), rank=[0])

    assert isinstance(row, GroupedRows)
    assert row.name == "MyGroup"
    assert "taskgroup" in row.extraClasses
    assert len(row.children) == 1
    assert row.children[0].name == "GroupTask"


# =============================================================================
# Tests for draw_chart
# =============================================================================


def test_draw_chart(make_task_ext, make_task_group):
    tasks = [make_task_ext(id=1, name="Task1", taskgroup_uuid=None)]
    groups = []

    # Relies on the actual fasthtml.common.Script available in the test env
    result = draw_chart(execution_id=42, tasks_list=tasks, taskgroups_list=groups)

    # fasthtml's Script renders to a fastcore.xml.FT object, which stringifies to an HTML <script> tag
    content = str(result)
    assert "<script>" in content

    assert "refreshScript" in content
    assert "tableData" in content
    assert "init('gantt-42'" in content
    assert "GanttTimeline" in content
    assert "initFormat" in content
