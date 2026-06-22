"""
Unit tests for retrain_pipelines.dag_engine.rp_logging
"""

import builtins
import io
import logging
import sys
from unittest.mock import MagicMock, patch
import retrain_pipelines.dag_engine.rp_logging as _rplog
from retrain_pipelines.dag_engine.rp_logging import (
    CustomRichHandler,
    RichLoggingController,
    _patched_print,
    _patched_stderr_write,
    _patched_stdout_write,
    parse_msg,
    rp_redirect_stdout,
)

# ══════════════════════════════════════════════════════════════════════════════
# Captured at import time ; used to ensure a clean state inside every test
# (the conftest autouse fixture already does this process-wide, but having a
# per-module reference is useful for the rp_logging lifecycle tests themselves)
# ══════════════════════════════════════════════════════════════════════════════
_REAL_PRINT = builtins.print
_REAL_GETLOGGER = logging.getLogger
_REAL_STDOUT_WRITE = sys.stdout.write
_REAL_STDERR_WRITE = sys.stderr.write


class MockFrame:
    """Simple mock for inspect.FrameType to avoid MagicMock infinite loop quirks."""

    def __init__(self, f_back, f_globals, co_name, co_filename="test.py", f_lineno=1):
        self.f_back = f_back
        self.f_globals = f_globals
        self.f_code = type(
            "Code", (object,), {"co_name": co_name, "co_filename": co_filename}
        )()
        self.f_lineno = f_lineno


# == helper: activate a controller and guarantee teardown ═════════════════════


class _CtrlGuard:
    """Context manager that activates a RichLoggingController and always deactivates it."""

    def __init__(self):
        self.ctrl = RichLoggingController()

    def __enter__(self):
        self.ctrl.activate()
        return self.ctrl

    def __exit__(self, *_):
        if self.ctrl._active:
            self.ctrl.deactivate()


# ══════════════════════════════════════════════════════════════════════════════
# parse_msg
# ══════════════════════════════════════════════════════════════════════════════
class TestParseMsg:
    def test_valid_traceback_line(self):
        msg = 'File "/some/path/foo.py", line 42, in some_func\nthe actual error'
        fname, lineno, body = parse_msg(msg)
        assert fname == "/some/path/foo.py"
        assert lineno == 42
        assert body == "the actual error"

    def test_non_traceback_returns_none(self):
        assert parse_msg("ordinary log message") is None

    def test_line_number_extracted_as_int(self):
        msg = 'File "/x.py", line 100, in fn\nerror'
        _, lineno, _ = parse_msg(msg)
        assert isinstance(lineno, int) and lineno == 100

    def test_multiline_body_starts_with_first_line(self):
        msg = 'File "/a/b.py", line 10, in f\nline1\nline2'
        _, _, body = parse_msg(msg)
        assert body.startswith("line1")


# ══════════════════════════════════════════════════════════════════════════════
# Module-level patched functions (fallback behaviour when no controller)
# ══════════════════════════════════════════════════════════════════════════════
class TestPatchedFunctionsFallback:
    def test_patched_print_calls_real_print_when_no_controller(self):
        _rplog._global_controller = None
        with patch.object(builtins, "print") as mock_print:
            _patched_print("hello")
            mock_print.assert_called_once_with("hello")

    def test_patched_stdout_write_falls_through_no_crash(self):
        _rplog._global_controller = None
        with patch.object(sys, "__stdout__", sys.stdout):
            _patched_stdout_write("test")

    def test_patched_stderr_write_falls_through_no_crash(self):
        _rplog._global_controller = None
        with patch.object(sys, "__stderr__", sys.stderr):
            _patched_stderr_write("err")

    def test_patched_print_dispatches_to_controller_when_active(self):
        with _CtrlGuard() as ctrl:
            ctrl._custom_print = MagicMock()
            _patched_print("dispatched")
            ctrl._custom_print.assert_called_once_with("dispatched")

    def test_patched_stdout_write_dispatches_to_controller_when_active(self):
        with _CtrlGuard() as ctrl:
            ctrl._custom_write = MagicMock(return_value=4)
            result = _patched_stdout_write("test")
            ctrl._custom_write.assert_called_once_with("test", False)
            assert result == 4

    def test_patched_stderr_write_dispatches_to_controller_when_active(self):
        with _CtrlGuard() as ctrl:
            ctrl._custom_write = MagicMock(return_value=3)
            result = _patched_stderr_write("err")
            ctrl._custom_write.assert_called_once_with("err", True)
            assert result == 3

    def test_patched_stdout_write_fallback_uses_global_controller_orig(self):
        with _CtrlGuard() as ctrl:
            ctrl._active = False
            ctrl._orig_stdout_write = MagicMock(return_value=1)
            _patched_stdout_write("x")
            ctrl._orig_stdout_write.assert_called_once_with("x")

    def test_patched_stderr_write_fallback_uses_global_controller_orig(self):
        with _CtrlGuard() as ctrl:
            ctrl._active = False
            ctrl._orig_stderr_write = MagicMock(return_value=1)
            _patched_stderr_write("x")
            ctrl._orig_stderr_write.assert_called_once_with("x")


# ══════════════════════════════════════════════════════════════════════════════
# Module import-time: in_notebook() True branch (nb_console_print imports)
# ══════════════════════════════════════════════════════════════════════════════
class TestNotebookImportBranch:
    def test_module_import_with_in_notebook_true_imports_nb_helpers(self, monkeypatch):
        import importlib
        import retrain_pipelines.utils as utils_mod

        monkeypatch.setattr(utils_mod, "in_notebook", lambda: True)

        spec = importlib.util.find_spec("retrain_pipelines.dag_engine.rp_logging")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            assert hasattr(module, "nb_activate")
            assert hasattr(module, "nb_console_print")
            assert hasattr(module, "nb_deactivate")
        finally:
            # never registered in sys.modules: nothing to clean up there ;
            # _rplog (the canonical module object) is left untouched.
            pass


# ══════════════════════════════════════════════════════════════════════════════
# CustomRichHandler.emit
# ══════════════════════════════════════════════════════════════════════════════
class TestCustomRichHandlerEmit:
    def _record(self, level, msg="test"):
        return logging.LogRecord("t", level, "/f.py", 1, msg, (), None)

    def test_info_level_no_exception(self):
        CustomRichHandler(markup=True).emit(
            self._record(logging.INFO, "hello [bold]world[/bold]")
        )

    def test_warning_level_no_exception(self):
        CustomRichHandler(markup=True).emit(self._record(logging.WARNING, "a warning"))

    def test_error_level_no_exception(self):
        CustomRichHandler(markup=True).emit(self._record(logging.ERROR, "an error"))

    def test_debug_level_no_exception(self):
        CustomRichHandler(markup=True).emit(self._record(logging.DEBUG, "debug"))

    def test_markup_false_no_exception(self):
        CustomRichHandler(markup=False).emit(self._record(logging.INFO, "plain text"))

    def test_emit_with_traceback_parsing(self):
        msg = 'File "/fake/traceback.py", line 99, in bad_func\nValueError: oops'
        record = logging.LogRecord(
            "test", logging.ERROR, "traceback.py", 1, msg, (), None
        )
        handler = CustomRichHandler(markup=True)
        with patch("rich.console.Console.print"):
            handler.emit(record)

    def test_emit_with_traceback_filename_but_unparsable_message(self):
        # filename is traceback.py but message doesn't match the
        # 'File "...", line N, in ...' regex -> parse_msg returns None,
        # falling back to record.lineno / record.getMessage()
        record = logging.LogRecord(
            "test", logging.ERROR, "traceback.py", 1, "not a traceback string", (), None
        )
        handler = CustomRichHandler(markup=True)
        with patch("rich.console.Console.print"):
            handler.emit(record)

    def test_emit_handles_internal_exception_via_handleError(self):
        record = self._record(logging.INFO, "boom")
        handler = CustomRichHandler(markup=True)
        with patch.object(_rplog, "Console", side_effect=RuntimeError("explode")):
            with patch.object(handler, "handleError") as mock_handle_error:
                handler.emit(record)
                mock_handle_error.assert_called_once_with(record)

    def test_emit_with_stream_to_db_in_notebook(self, monkeypatch):
        import retrain_pipelines.dag_engine.nb_console_print as nb_mod

        _rplog.__dict__["nb_activate"] = nb_mod.nb_activate
        _rplog.__dict__["nb_deactivate"] = nb_mod.nb_deactivate
        _rplog.__dict__["nb_console_print"] = nb_mod.nb_console_print

        monkeypatch.setattr(_rplog, "in_notebook", lambda: True)

        class MockStreamToDb:
            def __init__(self, is_err=False):
                self.is_err = is_err

            def write(self, s):
                pass

            def flush(self):
                pass

        import retrain_pipelines.dag_engine.core.core as core_mod

        orig = getattr(core_mod, "StreamToDb", None)
        core_mod.StreamToDb = MockStreamToDb
        try:
            old_stdout = sys.stdout
            sys.stdout = MockStreamToDb(is_err=False)
            try:
                record = logging.LogRecord(
                    "test", logging.INFO, "/f.py", 10, "msg", (), None
                )
                handler = CustomRichHandler(markup=True)
                with patch("rich.console.Console.print"):
                    handler.emit(record)
            finally:
                sys.stdout = old_stdout
        finally:
            if orig is None:
                if hasattr(core_mod, "StreamToDb"):
                    delattr(core_mod, "StreamToDb")
            else:
                core_mod.StreamToDb = orig


# ══════════════════════════════════════════════════════════════════════════════
# RichLoggingController lifecycle
# ══════════════════════════════════════════════════════════════════════════════
class TestRichLoggingControllerLifecycle:
    """Each test uses _CtrlGuard so deactivation is guaranteed even on failure."""

    def test_activate_sets_global_controller(self):
        with _CtrlGuard() as ctrl:
            assert _rplog._global_controller is ctrl
            assert ctrl._active is True

    def test_deactivate_clears_global_controller(self):
        ctrl = RichLoggingController()
        ctrl.activate()
        ctrl.deactivate()
        assert _rplog._global_controller is None
        assert ctrl._active is False

    def test_activate_is_idempotent(self):
        with _CtrlGuard() as ctrl:
            snap = builtins.print
            ctrl.activate()
            assert builtins.print is snap

    def test_deactivate_is_idempotent(self):
        ctrl = RichLoggingController()
        ctrl.deactivate()
        ctrl.deactivate()

    def test_activate_patches_builtins_print(self):
        with _CtrlGuard():
            assert builtins.print is _patched_print

    def test_activate_patches_stdout_write(self):
        with _CtrlGuard():
            assert sys.stdout.write is _patched_stdout_write

    def test_activate_patches_stderr_write(self):
        with _CtrlGuard():
            assert sys.stderr.write is _patched_stderr_write

    def test_deactivate_restores_builtins_print(self):
        ctrl = RichLoggingController()
        ctrl.activate()
        ctrl.deactivate()
        assert builtins.print is _REAL_PRINT

    def test_deactivate_restores_logging_getLogger(self):
        ctrl = RichLoggingController()
        ctrl.activate()
        ctrl.deactivate()
        assert logging.getLogger is _REAL_GETLOGGER

    def test_deactivate_removes_custom_rich_handlers(self):
        with _CtrlGuard():
            lg = logging.getLogger("test_cleanup_xyz_unique")
            lg.addHandler(CustomRichHandler())
        for h in lg.handlers:
            assert not isinstance(h, CustomRichHandler)

    def test_getLogger_wrapped_adds_custom_handler_to_package(self):
        with _CtrlGuard():
            lg = logging.getLogger("retrain_pipelines")
            assert any(isinstance(h, CustomRichHandler) for h in lg.handlers)

    def test_getLogger_wrapped_replaces_stream_handler_on_stdout(self):
        with _CtrlGuard():
            stream_handler = logging.StreamHandler(sys.stdout)
            # build the logger via the real (un-wrapped) getLogger so we can
            # pre-set its handlers before the wrapper ever sees this name
            real_lg = _REAL_GETLOGGER("some_other_logger_xyz")
            real_lg.handlers = [stream_handler]
            lg2 = logging.getLogger("some_other_logger_xyz")
            assert any(isinstance(h, CustomRichHandler) for h in lg2.handlers)
            assert stream_handler not in lg2.handlers

    def test_getLogger_wrapped_skips_uvicorn_loggers(self):
        with _CtrlGuard():
            stream_handler = logging.StreamHandler(sys.stdout)
            lg = logging.getLogger("uvicorn.access")
            lg.handlers = [stream_handler]
            lg2 = logging.getLogger("uvicorn.access")
            # uvicorn loggers are left untouched: original StreamHandler kept
            assert stream_handler in lg2.handlers

    def test_getLogger_wrapped_returns_cached_logger_unchanged(self):
        with _CtrlGuard():
            lg = logging.getLogger("cached_logger_xyz")
            handlers_snapshot = list(lg.handlers)
            # second call for the same name hits the
            # "name in self._logger_to_rich" early-return branch
            lg2 = logging.getLogger("cached_logger_xyz")
            assert lg2.handlers == handlers_snapshot

    def test_pickle_roundtrip_produces_inactive_controller(self):
        import pickle

        with _CtrlGuard() as ctrl:
            data = pickle.dumps(ctrl)
        ctrl2 = pickle.loads(data)
        assert ctrl2._active is False

    def test_reset_state_on_fresh_controller_clears_bookkeeping(self):
        ctrl = RichLoggingController()
        ctrl._logger_to_rich["x"] = logging.getLogger("x")
        ctrl._reset_state()
        assert ctrl._active is False
        assert ctrl._logger_to_rich == {}

    def test_activate_in_notebook_calls_nb_activate(self, monkeypatch):
        monkeypatch.setattr(_rplog, "in_notebook", lambda: True)
        mock_nb_activate = MagicMock()
        _rplog.__dict__["nb_activate"] = mock_nb_activate
        with _CtrlGuard() as ctrl:
            mock_nb_activate.assert_called_once_with(ctrl._orig_stdout_write)

    def test_deactivate_in_notebook_calls_nb_deactivate(self, monkeypatch):
        monkeypatch.setattr(_rplog, "in_notebook", lambda: True)
        _rplog.__dict__["nb_activate"] = MagicMock()
        mock_nb_deactivate = MagicMock()
        _rplog.__dict__["nb_deactivate"] = mock_nb_deactivate
        ctrl = RichLoggingController()
        ctrl.activate()
        ctrl.deactivate()
        mock_nb_deactivate.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# RichLoggingController private methods
# ══════════════════════════════════════════════════════════════════════════════
class TestRichLoggingControllerPrivateMethods:
    def test_custom_write_empty_string(self):
        with _CtrlGuard() as ctrl:
            ctrl._orig_stdout_write = MagicMock()
            result = ctrl._custom_write("", is_err=False)
            ctrl._orig_stdout_write.assert_called_once_with("")
            assert result == ctrl._orig_stdout_write.return_value

    def test_custom_write_from_code_module(self):
        with _CtrlGuard() as ctrl:
            ctrl._orig_stdout_write = MagicMock()
            with patch("inspect.currentframe") as mock_frame:
                f2 = MockFrame(
                    f_back=None,
                    f_globals={"__name__": "code something"},
                    co_name="interact",
                )
                f1 = MockFrame(f_back=f2, f_globals={}, co_name="test_func")
                mock_frame.return_value = f1
                result = ctrl._custom_write("error", is_err=False)
                ctrl._orig_stdout_write.assert_called_once_with("error")
                assert result == ctrl._orig_stdout_write.return_value

    def test_custom_write_from_excepthook(self):
        with _CtrlGuard() as ctrl:
            ctrl._orig_stderr_write = MagicMock()
            with patch("inspect.currentframe") as mock_frame:
                f2 = MockFrame(
                    f_back=None, f_globals={"__name__": "sys"}, co_name="excepthook"
                )
                f1 = MockFrame(f_back=f2, f_globals={}, co_name="test_func")
                mock_frame.return_value = f1
                result = ctrl._custom_write("error", is_err=True)
                ctrl._orig_stderr_write.assert_called_once_with("error")
                assert result == ctrl._orig_stderr_write.return_value

    def test_custom_write_from_rich_console(self, monkeypatch):
        import retrain_pipelines.dag_engine.nb_console_print as nb_mod

        _rplog.__dict__["nb_activate"] = nb_mod.nb_activate
        _rplog.__dict__["nb_deactivate"] = nb_mod.nb_deactivate
        _rplog.__dict__["nb_console_print"] = nb_mod.nb_console_print
        monkeypatch.setattr(_rplog, "in_notebook", lambda: True)

        with _CtrlGuard() as ctrl:
            ctrl._orig_stdout_write = MagicMock()
            with patch("inspect.currentframe") as mock_frame:
                f2 = MockFrame(
                    f_back=None, f_globals={"__name__": "rich.console"}, co_name="print"
                )
                f1 = MockFrame(f_back=f2, f_globals={}, co_name="test_func")
                mock_frame.return_value = f1
                result = ctrl._custom_write("text", is_err=False)
                assert result is None

    def test_custom_write_from_rich_console_not_in_notebook(self, monkeypatch):
        monkeypatch.setattr(_rplog, "in_notebook", lambda: False)
        with _CtrlGuard() as ctrl:
            ctrl._orig_stdout_write = MagicMock()
            with patch("inspect.currentframe") as mock_frame:
                f2 = MockFrame(
                    f_back=None, f_globals={"__name__": "rich.live"}, co_name="render"
                )
                f1 = MockFrame(f_back=f2, f_globals={}, co_name="test_func")
                mock_frame.return_value = f1
                result = ctrl._custom_write("text", is_err=False)
                ctrl._orig_stdout_write.assert_called_once_with("text")
                assert result == ctrl._orig_stdout_write.return_value

    def test_custom_write_frame_walk_exception_falls_back(self):
        with _CtrlGuard() as ctrl:
            ctrl._orig_stdout_write = MagicMock()
            with patch("inspect.currentframe") as mock_frame:
                # f_globals.get raises -> except Exception branch
                bad_frame = MagicMock()
                bad_frame.f_back = bad_frame
                type(bad_frame).f_globals = property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
                )
                top_frame = MockFrame(
                    f_back=bad_frame, f_globals={}, co_name="test_func"
                )
                mock_frame.return_value = top_frame
                result = ctrl._custom_write("oops", is_err=False)
                ctrl._orig_stdout_write.assert_called_once_with("oops")
                assert result == ctrl._orig_stdout_write.return_value

    def test_custom_write_no_current_frame(self):
        with _CtrlGuard() as ctrl:
            ctrl._orig_stdout_write = MagicMock()
            with patch("inspect.currentframe", return_value=None):
                with patch("rich.console.Console.print"):
                    result = ctrl._custom_write("text", is_err=False)
                    assert result == len("text")

    def test_custom_write_rich_emit_raises_falls_back(self):
        with _CtrlGuard() as ctrl:
            ctrl._orig_stdout_write = MagicMock()
            with patch.object(ctrl, "_rich_emit", side_effect=RuntimeError("boom")):
                result = ctrl._custom_write("text", is_err=False)
                ctrl._orig_stdout_write.assert_called_once_with("text")
                assert result == ctrl._orig_stdout_write.return_value

    def test_custom_write_normal_case_calls_rich_emit(self):
        with _CtrlGuard() as ctrl:
            with patch("rich.console.Console.print"):
                ctrl._custom_write("normal", is_err=False)

    def test_custom_print_normal_case(self):
        with _CtrlGuard() as ctrl:
            with patch("rich.console.Console.print"):
                ctrl._custom_print("hello")

    def test_custom_print_with_file_kwarg_not_streamtodb(self):
        with _CtrlGuard() as ctrl:
            ctrl._orig_print = MagicMock()
            fake_file = io.StringIO()
            ctrl._custom_print("hello", file=fake_file)
            ctrl._orig_print.assert_called_once_with("hello", file=fake_file)

    def test_custom_print_with_streamtodb_file(self, monkeypatch):
        class MockStreamToDb:
            def __init__(self, is_err=False):
                self.is_err = is_err

        monkeypatch.setattr(
            "retrain_pipelines.dag_engine.core.core.StreamToDb", MockStreamToDb
        )
        with _CtrlGuard() as ctrl:
            with patch("rich.console.Console.print"):
                stream = MockStreamToDb(is_err=True)
                ctrl._custom_print("msg", file=stream)

    def test_rich_emit_direct_call(self):
        with _CtrlGuard() as ctrl:
            with patch("rich.console.Console.print"):
                ctrl._rich_emit(
                    "test_source", "test message", is_err=False, frame_offset=2
                )

    def test_rich_emit_with_traceback(self):
        with _CtrlGuard() as ctrl:
            with patch("rich.console.Console.print"):
                ctrl._rich_emit(
                    "test",
                    'File "/fake/traceback.py", line 99, in bad_func\nValueError: oops',
                    is_err=False,
                    frame_offset=2,
                )

    def test_rich_emit_with_traceback_list_args_and_unparsable(self):
        # filename "traceback.py" but message doesn't match parse_msg regex,
        # and args_or_text is a non-str iterable -> " ".join branch
        with _CtrlGuard() as ctrl:
            with patch("inspect.currentframe") as mock_frame:
                f2 = MockFrame(
                    f_back=None,
                    f_globals={},
                    co_name="emit",
                    co_filename="/fake/traceback.py",
                    f_lineno=77,
                )
                f1 = MockFrame(f_back=f2, f_globals={}, co_name="caller")
                mock_frame.return_value = f1
                with patch("rich.console.Console.print"):
                    ctrl._rich_emit(
                        "test",
                        ["not", "a", "traceback"],
                        is_err=False,
                        frame_offset=1,
                    )

    def test_rich_emit_returns_when_currentframe_is_none(self):
        with _CtrlGuard() as ctrl:
            with patch("inspect.currentframe", return_value=None):
                result = ctrl._rich_emit("test", "msg", is_err=False, frame_offset=2)
                assert result is None

    def test_rich_emit_breaks_when_frame_back_is_none(self):
        with _CtrlGuard() as ctrl:
            with patch("inspect.currentframe") as mock_frame:
                only_frame = MockFrame(f_back=None, f_globals={}, co_name="caller")
                mock_frame.return_value = only_frame
                with patch("rich.console.Console.print"):
                    # frame_offset large, but f_back is None -> breaks out
                    ctrl._rich_emit("test", "msg", is_err=False, frame_offset=5)

    def test_rich_emit_in_notebook_calls_nb_console_print(self, monkeypatch):
        mock_nb_console_print = MagicMock()
        _rplog.__dict__["nb_console_print"] = mock_nb_console_print
        monkeypatch.setattr(_rplog, "in_notebook", lambda: True)

        class MockStreamToDb:
            is_err = False

            def write(self, s):
                pass

            def flush(self):
                pass

        import retrain_pipelines.dag_engine.core.core as core_mod

        orig = getattr(core_mod, "StreamToDb", None)
        core_mod.StreamToDb = MockStreamToDb
        try:
            with _CtrlGuard() as ctrl:
                with patch("rich.console.Console.print"):
                    ctrl._rich_emit(
                        "test", "notebook message", is_err=False, frame_offset=2
                    )
            mock_nb_console_print.assert_called_once()
        finally:
            if orig is None:
                if hasattr(core_mod, "StreamToDb"):
                    delattr(core_mod, "StreamToDb")
            else:
                core_mod.StreamToDb = orig


# ══════════════════════════════════════════════════════════════════════════════
# rp_redirect_stdout
# ══════════════════════════════════════════════════════════════════════════════
class TestRpRedirectStdout:
    def test_redirect_captures_print_when_no_controller(self):
        buf = io.StringIO()
        with rp_redirect_stdout(buf):
            print("goes to buf")
        buf.seek(0)
        assert "goes to buf" in buf.read()

    def test_deactivates_controller_inside_context(self):
        state_inside = []
        with _CtrlGuard() as ctrl:
            with rp_redirect_stdout(io.StringIO()):
                state_inside.append(ctrl._active)
        assert state_inside == [False]

    def test_reactivates_controller_after_context_exits(self):
        with _CtrlGuard() as ctrl:
            with rp_redirect_stdout(io.StringIO()):
                pass
            assert ctrl._active is True

    def test_reactivates_controller_even_on_inner_exception(self):
        with _CtrlGuard() as ctrl:
            try:
                with rp_redirect_stdout(io.StringIO()):
                    raise ValueError("inner")
            except ValueError:
                pass
            assert ctrl._active is True
