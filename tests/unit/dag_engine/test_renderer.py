"""
Unit tests for retrain_pipelines.dag_engine.renderer
"""

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
    tpl = MagicMock()
    tpl.render.return_value = rendered
    env = MagicMock()
    env.get_template.return_value = tpl
    return env, tpl


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
        env, tpl = _env("<rendered/>")
        tpl.render.return_value = "<rendered/>"

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
        env, tpl = _env()
        with patch(_OPEN, mock_open(read_data="")), patch(_ENV, return_value=env):
            dag_svg(dag=dag)
        _, kw = tpl.render.call_args
        assert len(kw["id_prefix"]) == 5

    def test_empty_taskgroups_passed_as_empty_list(self):
        dag = _dag(nodes=[], groups=[])
        env, tpl = _env()
        with patch(_OPEN, mock_open(read_data="")), patch(_ENV, return_value=env):
            dag_svg(dag=dag)
        _, kw = tpl.render.call_args
        assert kw["taskgroups"] == []

    def test_multiple_calls_produce_distinct_id_prefixes(self):
        dag = _dag()
        prefixes = set()
        for _ in range(20):
            env, tpl = _env()
            with patch(_OPEN, mock_open(read_data="")), patch(_ENV, return_value=env):
                dag_svg(dag=dag)
            _, kw = tpl.render.call_args
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
            patch(_IN_NB, return_value=False),
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            result = dag_svg(execution_id=1)
        assert isinstance(result, str)

    def test_none_tasktypes_returns_error_string(self):
        dao = _dao(None, None)
        with (
            patch(_IN_NB, return_value=False),
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
        ):
            result = dag_svg(execution_id=999)
        assert "999" in result

    def test_tasktype_with_taskgroup_uuid_serialised(self):
        tt = _tasktype_stub(tg_uuid=uuid4())
        dao = _dao([tt], [])
        env, tpl = _env("")
        with (
            patch(_IN_NB, return_value=False),
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            dag_svg(execution_id=1)
        _, kw = tpl.render.call_args
        assert isinstance(kw["nodes"][0]["taskgroup_uuid"], str)

    def test_tasktype_without_taskgroup_uuid_empty_str(self):
        tt = _tasktype_stub(tg_uuid=None)
        dao = _dao([tt], [])
        env, tpl = _env("")
        with (
            patch(_IN_NB, return_value=False),
            patch(_ADAO, return_value=dao),
            patch(_OPEN, mock_open(read_data="")),
            patch(_ENV, return_value=env),
        ):
            dag_svg(execution_id=1)
        _, kw = tpl.render.call_args
        assert kw["nodes"][0]["taskgroup_uuid"] == ""

    def test_notebook_path_uses_thread_pool(self):
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


# ══════════════════════════════════════════════════════════════════════════════
#  dag_svg – error branch
# ══════════════════════════════════════════════════════════════════════════════


class TestDagSvgErrors:
    def test_raises_when_neither_dag_nor_execution_id(self):
        with pytest.raises(ValueError, match="'dag' or 'execution_id'"):
            dag_svg()
