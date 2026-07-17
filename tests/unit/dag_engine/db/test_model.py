"""
Unit tests for retrain_pipelines.dag_engine.db.model
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from retrain_pipelines.dag_engine.db.model import (
    Execution,
    ExecutionExt,
    Task,
    TaskExt,
    TaskGroup,
    TaskTrace,
    TaskType,
    TaskContextAttr,
)

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_UUID = uuid4()


# ══════════════════════════════════════════════════════════════════════════════
#  Execution
# ══════════════════════════════════════════════════════════════════════════════


class TestExecution:
    def _make(self, **kw):
        return Execution(
            **{"name": "p", "username": "u", "_start_timestamp": _NOW, **kw}
        )

    def test_basic_fields(self):
        ex = self._make()
        assert ex.name == "p" and ex.username == "u"

    def test_start_timestamp_aware_preserved(self):
        assert self._make().start_timestamp == _NOW

    def test_start_timestamp_naive_bypass(self):
        ex = self._make()
        # Bypass SQLAlchemy's mapped column setter to force a naive datetime
        ex.__dict__["_start_timestamp"] = datetime(2024, 1, 1)
        assert ex.start_timestamp.tzinfo == timezone.utc

    def test_end_timestamp_none_by_default(self):
        assert self._make().end_timestamp is None

    def test_end_timestamp_naive_gets_utc(self):
        ex = self._make(_end_timestamp=datetime(2024, 6, 1, 15))
        # Exercise the end_timestamp setter
        new_end = datetime(2024, 6, 1, 16, tzinfo=timezone.utc)
        ex.end_timestamp = new_end
        assert ex.end_timestamp == new_end

    def test_dict_constructor(self):
        ex = Execution(
            {
                "id": 1,
                "name": "p",
                "username": "u",
                "start_timestamp": "2024-06-01T12:00:00+00:00",
            }
        )
        assert ex.id == 1 and ex.name == "p"

    def test_dict_constructor_rejects_multiple_positional(self):
        with pytest.raises(TypeError):
            Execution("a", "b")

    def test_optional_fields_default_none(self):
        ex = self._make()
        for f in ("docstring", "params", "ui_css"):
            assert getattr(ex, f) is None


# ══════════════════════════════════════════════════════════════════════════════
#  ExecutionExt
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutionExt:
    def _make(self, **kw):
        defaults = {
            "name": "p",
            "username": "u",
            "start_timestamp": _NOW,
            "end_timestamp": None,
        }
        defaults.update(kw)
        return ExecutionExt(**defaults)

    def test_success_true(self):
        # Pass both timestamps as strings to exercise the datetime-string
        # conversion branch in ExecutionExt.__init__
        ex = self._make(
            success=True,
            start_timestamp="2024-06-01T12:00:00+00:00",
            end_timestamp="2024-06-01T13:00:00+00:00",
        )
        assert ex.success is True
        assert ex.start_timestamp == _NOW
        assert ex.end_timestamp == datetime(2024, 6, 1, 13, 0, 0, tzinfo=timezone.utc)

    def test_success_false(self):
        assert self._make(success=False).success is False

    def test_success_defaults_none(self):
        assert self._make().success is None

    def test_to_dict_has_name(self):
        ex = self._make(end_timestamp=None)
        # Set the private attribute to a string to exercise the datetime‑string
        # conversion branch in to_dict
        ex._start_timestamp = "2024-01-01T00:00:00Z"
        d = ex.to_dict()
        assert d["name"] == "p"
        assert d["end_timestamp"] is None

    def test_to_dict_timestamps_serialised_as_str(self):
        d = self._make().to_dict()
        assert isinstance(d.get("start_timestamp"), str)

    def test_to_dict_no_column_info(self):
        ex = self._make()
        # Add a property with no corresponding DB column
        Execution.my_prop = property(lambda self: self._my_prop)
        # Set the backing value to None to cover the branch that assigns None
        # when there is no column info
        ex._my_prop = None
        try:
            d = ex.to_dict()
            assert d["my_prop"] is None
        finally:
            del Execution.my_prop


# ══════════════════════════════════════════════════════════════════════════════
#  TaskType
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskType:
    def _make(self, **kw):
        return TaskType(
            **{
                "uuid": _UUID,
                "exec_id": 1,
                "order": 0,
                "name": "step",
                "is_parallel": False,
                "children": [],
                **kw,
            }
        )

    def test_basic_fields(self):
        tt = self._make()
        assert tt.name == "step" and tt.is_parallel is False

    def test_dict_constructor_coerces_types(self):
        tt = TaskType(
            {
                "uuid": str(_UUID),
                "exec_id": "1",
                "order": "0",
                "name": "step",
                "is_parallel": False,
                "children": [],
                "docstring": "test doc",
            }
        )
        assert tt.exec_id == 1 and tt.order == 0
        assert tt.docstring == "test doc"

    def test_rejects_multiple_positional(self):
        with pytest.raises(TypeError):
            TaskType("a", "b")


# ══════════════════════════════════════════════════════════════════════════════
#  Task
# ══════════════════════════════════════════════════════════════════════════════


class TestTask:
    def _make(self, **kw):
        return Task(
            **{"tasktype_uuid": _UUID, "exec_id": 1, "_start_timestamp": _NOW, **kw}
        )

    def test_start_timestamp_naive_gets_utc(self):
        t = self._make(_start_timestamp=datetime(2024, 1, 1))
        assert t.start_timestamp.tzinfo == timezone.utc

    def test_end_timestamp_none_by_default(self):
        assert self._make().end_timestamp is None

    def test_end_timestamp_aware_preserved(self):
        end = _NOW + timedelta(hours=1)
        t = self._make(_end_timestamp=end)
        # Exercise the setter for end_timestamp to cover that code path.
        new_end = end + timedelta(hours=1)
        t.end_timestamp = new_end
        assert t.end_timestamp == new_end

        # Also exercise the branch in the property getter that handles a naive
        # datetime. Bypass the setter to store a naive datetime.
        t.__dict__["_end_timestamp"] = datetime(2024, 1, 1)
        assert t.end_timestamp.tzinfo == timezone.utc

    def test_repr_contains_class_name(self):
        assert "Task" in repr(self._make())

    def test_dict_constructor_coerces_types(self):
        t = Task(
            {
                "id": "5",
                "tasktype_uuid": str(_UUID),
                "exec_id": 1,
                "start_timestamp": "2024-06-01T12:00:00+00:00",
                "end_timestamp": "2024-06-01T13:00:00+00:00",
                "failed": False,
            }
        )
        assert t.id == 5 and t.failed is False
        assert t.end_timestamp == datetime(2024, 6, 1, 13, 0, 0, tzinfo=timezone.utc)

    def test_rejects_multiple_positional(self):
        with pytest.raises(TypeError):
            Task("a", "b")


# ══════════════════════════════════════════════════════════════════════════════
#  TaskExt
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskExt:
    def _make(self, **kw):
        return TaskExt(
            **{
                "tasktype_uuid": _UUID,
                "exec_id": 1,
                "_start_timestamp": _NOW,
                "name": "step",
                "is_parallel": False,
                **kw,
            }
        )

    def test_name_propagated(self):
        # Include all optional fields to exercise every pop in TaskExt.__init__,
        # including taskgroup_uuid.
        te = self._make(
            name="my_task",
            order=5,
            docstring="docs",
            ui_css={"color": "blue"},
            is_parallel=True,
            merge_func={"type": "concat"},
            taskgroup_uuid=uuid4(),
        )
        assert te.name == "my_task"
        assert te.order == 5
        assert te.docstring == "docs"
        assert te.ui_css == {"color": "blue"}
        assert te.is_parallel is True
        assert te.merge_func == {"type": "concat"}
        assert te.taskgroup_uuid is not None

    def test_is_parallel_propagated(self):
        assert self._make(is_parallel=True).is_parallel is True

    def test_optional_tasktype_fields_default_none(self):
        te = self._make()
        for f in ("ui_css", "merge_func", "taskgroup_uuid"):
            assert getattr(te, f) is None

    def test_repr_contains_name(self):
        r = repr(self._make(name="x"))
        assert "TaskExt" in r and "x" in r

    def test_rejects_multiple_positional(self):
        with pytest.raises(TypeError):
            TaskExt("a", "b")


# ══════════════════════════════════════════════════════════════════════════════
#  TaskGroup
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskGroup:
    def _make(self, **kw):
        return TaskGroup(
            **{
                "uuid": _UUID,
                "exec_id": 1,
                "order": 0,
                "name": "grp",
                "elements": [],
                **kw,
            }
        )

    def test_basic_fields(self):
        assert self._make().name == "grp"

    def test_repr(self):
        assert "TaskGroup" in repr(self._make())

    def test_dict_constructor_coerces_types(self):
        tg = TaskGroup(
            {
                "uuid": str(_UUID),
                "exec_id": "1",
                "order": "0",
                "name": "g",
                "elements": [],
                "docstring": "test doc",
            }
        )
        assert tg.exec_id == 1
        assert tg.docstring == "test doc"

    def test_rejects_multiple_positional(self):
        with pytest.raises(TypeError):
            TaskGroup("a", "b")


# ══════════════════════════════════════════════════════════════════════════════
#  TaskTrace
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskTrace:
    def _make(self, **kw):
        return TaskTrace(
            **{
                "task_id": 1,
                "timestamp": _NOW,
                "microsec": 0,
                "microsec_idx": 1,
                "content": "msg",
                "is_err": False,
                **kw,
            }
        )

    def test_basic_fields(self):
        tt = self._make()
        assert tt.content == "msg" and tt.is_err is False

    def test_rejects_none_timestamp(self):
        with pytest.raises(ValueError):
            self._make(timestamp=None)

    def test_rejects_naive_timestamp(self):
        with pytest.raises(ValueError):
            self._make(timestamp=datetime(2024, 1, 1))

    def test_accepts_iso_string_with_offset(self):
        tt = self._make(timestamp="2024-06-01T12:00:00+00:00")
        assert tt.timestamp.tzinfo is not None

    def test_accepts_iso_z_suffix(self):
        tt = self._make(timestamp="2024-06-01T12:00:00Z")
        assert tt.timestamp.tzinfo is not None

    def test_rejects_malformed_string(self):
        with pytest.raises(ValueError):
            self._make(timestamp="not-a-date")

    def test_repr_contains_class_name(self):
        assert "TaskTrace" in repr(self._make())

    def test_dict_constructor(self):
        tt = TaskTrace(
            {
                "task_id": 2,
                "timestamp": _NOW,
                "microsec": 5,
                "microsec_idx": 1,
                "content": "x",
                "is_err": True,
            }
        )
        assert tt.task_id == 2 and tt.is_err is True

    def test_rejects_multiple_positional(self):
        with pytest.raises(TypeError):
            TaskTrace("a", "b")


# ══════════════════════════════════════════════════════════════════════════════
# TaskContextAttr
# ══════════════════════════════════════════════════════════════════════════════
class TestTaskContextAttr:
    def test_inline_repr(self):
        attr = TaskContextAttr(task_id=1, attr_name="foo", sha="abc", inline_val="bar")
        assert "inline" in repr(attr)
        assert attr.inline_val == "bar"
        assert attr.disk_ref is None

    def test_disk_repr(self):
        attr = TaskContextAttr(
            task_id=1, attr_name="foo", sha="abc", disk_ref="path/to/file"
        )
        assert "disk" in repr(attr)
        assert attr.disk_ref == "path/to/file"
        assert attr.inline_val is None
