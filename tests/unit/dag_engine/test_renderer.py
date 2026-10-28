"""
Unit tests for retrain_pipelines.dag_engine.renderer
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from uuid import uuid4

from retrain_pipelines.dag_engine.renderer import dag_svg, render_svg

_OPEN = "builtins.open"
_ENV = "retrain_pipelines.dag_engine.renderer.Environment"
_IN_NB = "retrain_pipelines.dag_engine.renderer.in_notebook"
_ADAO = "retrain_pipelines.dag_engine.renderer.AsyncDAO"


def _dag(nodes=None, groups=None):
    d = MagicMock()
    d.to_elements_lists.return_value = (nodes or [], groups or [])
    return d


def _env(rendered=""):
    template = MagicMock()
    template.render.return_value = rendered
    env = MagicMock()
    env.get_template.return_value = template
    return env, template


def _dao(tasktypes, taskgroups):
    dao = AsyncMock()
    dao.get_execution_tasktypes_list = AsyncMock(return_value=tasktypes)
    dao.get_execution_taskgroups_list = AsyncMock(return_value=taskgroups)
    return dao


def _tasktype_stub(tg_uuid=None):
    tt = MagicMock()
    tt.__dict__ = {
        "uuid": uuid4(),
        "taskgroup_uuid": tg_uuid,
        "order": 0,
        "name": "step",
        "ui_css": None,
        "is_parallel": False,
        "merge_func": None,
    }
    return tt


def _taskgroup_stub():
    """Build a taskgroup-like stub whose __dict__ carries a UUID field."""
    tg = MagicMock()
    tg.__dict__ = {
        "uuid": uuid4(),
        "name": "group",
        "order": 0,
    }
    return tg


# ══════════════════════════════════════════════════════════════════════════════
#  render_svg
# ══════════════════════════════════════════════════════════════════════════════


class TestRenderSvg:
    def test_calls_to_elements_lists(self):
        dag = _dag()
        env, _ = _env()
        with patch(_OPEN, mock_open(read_data="")), patch(_ENV, return_value=env):
            render_svg(dag, filename="/dev/null")
        dag.to_elements_lists.assert_called_once_with(serializable=True)

    def test_output_file_written(self, tmp_path):
        out = str(tmp_path / "dag.html")
        dag = _dag()
        env, template = _env("<rendered/>")
        template.render.return_value = "<rendered/>"

        written = []

        def fake_open(path, *a, **kw):
            m = mock_open(read_data="css{}")()
            if "w" in str(a):
                m.write.side_effect = lambda s: written.append(s)
            return m

        with patch(_OPEN, side_effect=fake_open), patch(_ENV, return_value=env):
            render_svg(dag, filename=out)


# ══════════════════════════════════════════════════════════════════════════════
#  dag_svg – dag branch
# ══════════════════════════════════════════════════════════════════════════════


class TestDagSvgDagBranch:
    def _run(self, dag, css="", rendered=""):
        env, _ = _env(rendered)
        with patch(_OPEN, mock_open(read_data=css)), patch(_ENV, return_value=env):
            return dag_svg(dag=dag)

    def test_returns_str(self):
        assert isinstance(self._run(_dag()), str)

    def test_style_tag_present(self):
        assert "<style>" in self._run(_dag(), css=".x{}")

    def test_css_included(self):
        assert "custom-css" in self._run(_dag(), css="custom-css")

    def test_rendered_content_included(self):
        assert "<g/>" in self._run(_dag(), rendered="<g/>")

    def test_calls_to_elements_lists_serializable(self):
        dag = _dag()
        self._run(dag)
        dag.to_elements_lists.assert_called_once_with(serializable=True)

    def test_id_prefix_five_chars(self):
        dag = _dag()
        env, template = _env()
        with patch(_OPEN, mock_open(read_data="")), patch(_ENV, return_value=env):
            dag_svg(dag=dag)
        _, kw = template.render.call_args
        assert len(kw["id_prefix"]) == 5

    def test_empty_taskgroups_passed_as_empty_list(self):
        dag = _dag(nodes=[], groups=[])
        env, template = _env()
        with patch(_OPEN, mock_open(read_data="")), patch(_ENV, return_value=env):
            dag_svg(dag=dag)
        _, kw = template.render.call_args
        assert kw["taskgroups"] == []

    def test_multiple_calls_produce_distinct_id_prefixes(self):
        dag = _dag()
        prefixes = set()
        for _ in range(20):
            env, template = _env()
            with patch(_OPEN, mock_open(read_data="")), patch(_ENV, return_value=env):
                dag_svg(dag=dag)
            _, kw = template.render.call_args
            prefixes.add(kw["id_prefix"])
        assert len(prefixes) > 1


# ══════════════════════════════════════════════════════════════════════════════
#  dag_svg – execution_id branch
# ══════════════════════════════════════════════════════════════════════════════


class TestDagSvgExecutionIdBranch:
    def test_valid_id_non_notebook_returns_str(self):
        tt = _tasktype_stub()
        dao = _dao([tt], [])
        env, _ = _env("<rendered/>")
        with (
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            result = dag_svg(execution_id=1)
        assert isinstance(result, str)

    def test_none_tasktypes_returns_error_string(self):
        dao = _dao(None, None)
        with (
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
        ):
            result = dag_svg(execution_id=999)
        assert "999" in result

    def test_tasktype_with_taskgroup_uuid_serialised(self):
        tt = _tasktype_stub(tg_uuid=uuid4())
        dao = _dao([tt], [])
        env, template = _env("")
        with (
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            dag_svg(execution_id=1)
        _, kw = template.render.call_args
        assert isinstance(kw["nodes"][0]["taskgroup_uuid"], str)

    def test_tasktype_without_taskgroup_uuid_empty_str(self):
        tt = _tasktype_stub(tg_uuid=None)
        dao = _dao([tt], [])
        env, template = _env("")
        with (
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            dag_svg(execution_id=1)
        _, kw = template.render.call_args
        assert kw["nodes"][0]["taskgroup_uuid"] == ""

    def test_taskgroup_uuid_serialised(self):
        """Cover the branch where taskgroups_list is non-empty,
        ensuring each taskgroup dict's uuid is converted to a string
        before being passed to the template."""
        tt = _tasktype_stub()
        tg = _taskgroup_stub()
        dao = _dao([tt], [tg])
        env, template = _env("")
        with (
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            dag_svg(execution_id=1)
        _, kw = template.render.call_args
        taskgroups = kw["taskgroups"]
        assert len(taskgroups) == 1
        assert isinstance(taskgroups[0]["uuid"], str)

    def test_notebook_path_uses_thread_pool(self):
        """Cover the notebook branch where the fetch coroutine runs in
        a dedicated worker thread via ThreadPoolExecutor.

        in_notebook is patched because pytest runs outside any IPython
        kernel and there is no way to make the real in_notebook() return
        True in a standard test runner.
        """
        tt = _tasktype_stub()
        dao = _dao([tt], [])
        env, _ = _env("")
        with (
            patch(_IN_NB, return_value=True),
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            result = dag_svg(execution_id=1)
        assert isinstance(result, str)

    def test_no_current_event_loop_creates_new_one(self):
        """dag_svg recovers when no event loop is current.

        Select exception: ``_ADAO`` is patched because this test exercises
        the asyncio-error-handling path of the renderer, not DAO behaviour.
        ``policy.get_event_loop()`` is guarded with try/except: a previous
        test in the same xdist worker may have closed the loop via
        ``asyncio.run()``, leaving it None and causing ``RuntimeError`` here.
        """
        policy = asyncio.get_event_loop_policy()
        try:
            original_loop = policy.get_event_loop()
        except RuntimeError:
            original_loop = None
        policy.set_event_loop(None)
        tt = _tasktype_stub()
        dao = _dao([tt], [])
        env, _ = _env("<rendered/>")
        try:
            with (
                patch(_IN_NB, return_value=False),
                patch(_ADAO, return_value=dao),
                patch(_OPEN, mock_open(read_data="")),
                patch(_ENV, return_value=env),
            ):
                result = dag_svg(execution_id=1)
            assert isinstance(result, str)
        finally:
            policy.set_event_loop(original_loop)


# ══════════════════════════════════════════════════════════════════════════════
#  dag_svg – error branch
# ══════════════════════════════════════════════════════════════════════════════


class TestDagSvgErrors:
    def test_raises_when_neither_dag_nor_execution_id(self):
        with pytest.raises(ValueError, match="'dag' or 'execution_id'"):
            dag_svg()
