"""
Unit tests for retrain_pipelines/dag_engine/web_console/main_notebook.py.

Only mocked: real network calls (urlopen, ngrok tunnel), real display output
(IPython.display calls are verified via patch), and OS env for platform detection.
The lru_cache on _notebook_env is cleared between tests.
"""

import logging
import os
import sys
import threading
import types
import urllib.error
from unittest.mock import MagicMock, patch


import retrain_pipelines.dag_engine.web_console.main_notebook as nb
import retrain_pipelines.dag_engine.web_console.main as main_mod


# ---------------------------------------------------------------------------
# _notebook_env
# ---------------------------------------------------------------------------


class TestNotebookEnv:
    def setup_method(self):
        nb._notebook_env.cache_clear()

    def teardown_method(self):
        nb._notebook_env.cache_clear()

    def test_detects_kaggle(self):
        with patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
            assert nb._notebook_env() == "kaggle"

    def test_detects_colab(self):
        nb._notebook_env.cache_clear()
        google_mod = types.ModuleType("google")
        colab_mod = types.ModuleType("google.colab")
        google_mod.colab = colab_mod
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(sys.modules, {"google": google_mod, "google.colab": colab_mod}),
        ):
            nb._notebook_env.cache_clear()
            result = nb._notebook_env()
        assert result == "colab"

    def test_defaults_to_local(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch.dict(sys.modules, {"google.colab": None}),
        ):
            nb._notebook_env.cache_clear()
            result = nb._notebook_env()
        assert result == "local"


# ---------------------------------------------------------------------------
# _wait_for_server
# ---------------------------------------------------------------------------


class TestWaitForServer:
    def test_returns_true_on_successful_connection(self):
        with patch("urllib.request.urlopen", return_value=MagicMock()):
            assert nb._wait_for_server("http://localhost:8000/", timeout=1) is True

    def test_returns_true_on_http_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(None, 404, "Not Found", {}, None),
        ):
            assert nb._wait_for_server("http://localhost:8000/", timeout=1) is True

    def test_returns_false_on_timeout(self):
        with (
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            patch("time.sleep"),
        ):
            assert nb._wait_for_server("http://localhost:8000/", timeout=0.01) is False


# ---------------------------------------------------------------------------
# _ngrok_start
# ---------------------------------------------------------------------------


class TestNgrokStart:
    def setup_method(self):
        nb._notebook_env.cache_clear()

    def teardown_method(self):
        nb._notebook_env.cache_clear()

    def test_returns_none_when_pyngrok_not_installed(self):
        with patch.dict(
            sys.modules,
            {"pyngrok": None, "pyngrok.ngrok": None, "pyngrok.exception": None},
        ):
            result = nb._ngrok_start(8000)
        assert result is None

    def test_returns_public_url_on_success(self):
        tunnel = MagicMock()
        tunnel.public_url = "https://abc.ngrok.io"

        ngrok_mod = MagicMock()
        ngrok_mod.connect.return_value = tunnel

        pyngrok_pkg = types.ModuleType("pyngrok")
        exc_mod = types.ModuleType("pyngrok.exception")
        exc_mod.PyngrokNgrokError = Exception
        pyngrok_pkg.ngrok = ngrok_mod

        with (
            patch.dict(
                sys.modules,
                {
                    "pyngrok": pyngrok_pkg,
                    "pyngrok.ngrok": ngrok_mod,
                    "pyngrok.exception": exc_mod,
                },
            ),
            patch.dict("os.environ", {"NGROK_AUTHTOKEN": "tok"}),
            patch.object(nb, "_notebook_env", return_value="local"),
        ):
            result = nb._ngrok_start(8000)

        assert result == "https://abc.ngrok.io"

    def test_returns_none_on_pyngrok_auth_error(self):
        class PyngrokNgrokError(Exception):
            def __init__(self, msg="err"):
                super().__init__(msg)
                self.ngrok_error = msg

        ngrok_mod = MagicMock()
        ngrok_mod.connect.side_effect = PyngrokNgrokError("auth failure")

        pyngrok_pkg = types.ModuleType("pyngrok")
        exc_mod = types.ModuleType("pyngrok.exception")
        exc_mod.PyngrokNgrokError = PyngrokNgrokError
        pyngrok_pkg.ngrok = ngrok_mod

        with (
            patch.dict(
                sys.modules,
                {
                    "pyngrok": pyngrok_pkg,
                    "pyngrok.ngrok": ngrok_mod,
                    "pyngrok.exception": exc_mod,
                },
            ),
            patch.dict("os.environ", {"NGROK_AUTHTOKEN": "tok"}),
            patch.object(nb, "_notebook_env", return_value="local"),
        ):
            result = nb._ngrok_start(8000)

        assert result is None

    def test_reads_colab_secret_when_no_env_token(self):
        tunnel = MagicMock()
        tunnel.public_url = "https://colab.ngrok.io"
        ngrok_mod = MagicMock()
        ngrok_mod.connect.return_value = tunnel

        pyngrok_pkg = types.ModuleType("pyngrok")
        exc_mod = types.ModuleType("pyngrok.exception")
        exc_mod.PyngrokNgrokError = Exception
        pyngrok_pkg.ngrok = ngrok_mod

        userdata_mod = MagicMock()
        userdata_mod.get.return_value = "colab_token"
        userdata_mod.SecretNotFoundError = Exception

        colab_mod = types.ModuleType("google.colab")
        colab_mod.userdata = userdata_mod

        os.environ.pop("NGROK_AUTHTOKEN", None)
        with (
            patch.dict(
                sys.modules,
                {
                    "pyngrok": pyngrok_pkg,
                    "pyngrok.ngrok": ngrok_mod,
                    "pyngrok.exception": exc_mod,
                    "google.colab": colab_mod,
                    "google.colab.userdata": userdata_mod,
                },
            ),
            patch.object(nb, "_notebook_env", return_value="colab"),
        ):
            result = nb._ngrok_start(8000)

        assert result == "https://colab.ngrok.io"
        assert os.environ.get("NGROK_AUTHTOKEN") == "colab_token"

    def test_reads_kaggle_secret_when_no_env_token(self):
        tunnel = MagicMock()
        tunnel.public_url = "https://kaggle.ngrok.io"
        ngrok_mod = MagicMock()
        ngrok_mod.connect.return_value = tunnel

        pyngrok_pkg = types.ModuleType("pyngrok")
        exc_mod = types.ModuleType("pyngrok.exception")
        exc_mod.PyngrokNgrokError = Exception
        pyngrok_pkg.ngrok = ngrok_mod

        client = MagicMock()
        client.get_secret.return_value = "kaggle_tok"
        kaggle_mod = types.ModuleType("kaggle_secrets")
        kaggle_mod.UserSecretsClient = MagicMock(return_value=client)

        os.environ.pop("NGROK_AUTHTOKEN", None)
        with (
            patch.dict(
                sys.modules,
                {
                    "pyngrok": pyngrok_pkg,
                    "pyngrok.ngrok": ngrok_mod,
                    "pyngrok.exception": exc_mod,
                    "kaggle_secrets": kaggle_mod,
                },
            ),
            patch.object(nb, "_notebook_env", return_value="kaggle"),
        ):
            result = nb._ngrok_start(8000)

        assert result == "https://kaggle.ngrok.io"
        assert os.environ.get("NGROK_AUTHTOKEN") == "kaggle_tok"

    def test_colab_secret_not_found_falls_through_to_input(self):
        """SecretNotFoundError on colab => prompts user via input()."""
        tunnel = MagicMock()
        tunnel.public_url = "https://input.ngrok.io"
        ngrok_mod = MagicMock()
        ngrok_mod.connect.return_value = tunnel

        pyngrok_pkg = types.ModuleType("pyngrok")
        exc_mod = types.ModuleType("pyngrok.exception")
        exc_mod.PyngrokNgrokError = Exception
        pyngrok_pkg.ngrok = ngrok_mod

        class SecretNotFoundError(Exception):
            pass

        userdata_mod = MagicMock()
        userdata_mod.get.side_effect = SecretNotFoundError("missing")
        userdata_mod.SecretNotFoundError = SecretNotFoundError

        colab_mod = types.ModuleType("google.colab")
        colab_mod.userdata = userdata_mod

        os.environ.pop("NGROK_AUTHTOKEN", None)
        with (
            patch.dict(
                sys.modules,
                {
                    "pyngrok": pyngrok_pkg,
                    "pyngrok.ngrok": ngrok_mod,
                    "pyngrok.exception": exc_mod,
                    "google.colab": colab_mod,
                    "google.colab.userdata": userdata_mod,
                },
            ),
            patch.object(nb, "_notebook_env", return_value="colab"),
            patch("builtins.input", return_value="input_tok"),
        ):
            result = nb._ngrok_start(8000)

        assert result == "https://input.ngrok.io"
        assert os.environ.get("NGROK_AUTHTOKEN") == "input_tok"

    def test_kaggle_secret_exception_falls_through_to_input(self):
        """kaggle_secrets exception => prompts user via input()."""
        tunnel = MagicMock()
        tunnel.public_url = "https://input2.ngrok.io"
        ngrok_mod = MagicMock()
        ngrok_mod.connect.return_value = tunnel

        pyngrok_pkg = types.ModuleType("pyngrok")
        exc_mod = types.ModuleType("pyngrok.exception")
        exc_mod.PyngrokNgrokError = Exception
        pyngrok_pkg.ngrok = ngrok_mod

        kaggle_mod = types.ModuleType("kaggle_secrets")
        kaggle_mod.UserSecretsClient = MagicMock(side_effect=RuntimeError("no secrets"))

        os.environ.pop("NGROK_AUTHTOKEN", None)
        with (
            patch.dict(
                sys.modules,
                {
                    "pyngrok": pyngrok_pkg,
                    "pyngrok.ngrok": ngrok_mod,
                    "pyngrok.exception": exc_mod,
                    "kaggle_secrets": kaggle_mod,
                },
            ),
            patch.object(nb, "_notebook_env", return_value="kaggle"),
            patch("builtins.input", return_value=""),
        ):
            result = nb._ngrok_start(8000)

        # empty input => token stays None => no env var set, connect still tried
        assert result == "https://input2.ngrok.io"

    def test_no_platform_no_env_prompts_input_and_sets_token(self):
        """local env, no NGROK_AUTHTOKEN => user prompted, token from input."""
        tunnel = MagicMock()
        tunnel.public_url = "https://local.ngrok.io"
        ngrok_mod = MagicMock()
        ngrok_mod.connect.return_value = tunnel

        pyngrok_pkg = types.ModuleType("pyngrok")
        exc_mod = types.ModuleType("pyngrok.exception")
        exc_mod.PyngrokNgrokError = Exception
        pyngrok_pkg.ngrok = ngrok_mod

        os.environ.pop("NGROK_AUTHTOKEN", None)
        with (
            patch.dict(
                sys.modules,
                {
                    "pyngrok": pyngrok_pkg,
                    "pyngrok.ngrok": ngrok_mod,
                    "pyngrok.exception": exc_mod,
                },
            ),
            patch.object(nb, "_notebook_env", return_value="local"),
            patch("builtins.input", return_value="manual_tok"),
        ):
            result = nb._ngrok_start(8000)

        assert result == "https://local.ngrok.io"
        assert os.environ.get("NGROK_AUTHTOKEN") == "manual_tok"


# ---------------------------------------------------------------------------
# _ngrok_stop
# ---------------------------------------------------------------------------


class TestNgrokStop:
    def test_calls_ngrok_kill(self):
        ngrok_mod = MagicMock()
        pyngrok_pkg = types.ModuleType("pyngrok")
        pyngrok_pkg.ngrok = ngrok_mod
        with patch.dict(
            sys.modules, {"pyngrok": pyngrok_pkg, "pyngrok.ngrok": ngrok_mod}
        ):
            nb._ngrok_stop()
        ngrok_mod.kill.assert_called_once()

    def test_no_raise_when_pyngrok_missing(self):
        import types as _types

        ngrok_mod = MagicMock()
        pyngrok_pkg = _types.ModuleType("pyngrok")
        pyngrok_pkg.ngrok = ngrok_mod
        with patch.dict(
            sys.modules, {"pyngrok": pyngrok_pkg, "pyngrok.ngrok": ngrok_mod}
        ):
            nb._ngrok_stop()  # must not raise


# ---------------------------------------------------------------------------
# _webconsole_start_notebook
# ---------------------------------------------------------------------------


class TestWebconsoleStartNotebook:
    def setup_method(self):
        nb._notebook_env.cache_clear()

    def teardown_method(self):
        nb._notebook_env.cache_clear()

    def _base_patches(self, env="local", wait_results=None):
        """Return a dict of common patches. wait_results: list of bools for _wait_for_server."""
        if wait_results is None:
            wait_results = [True, True]
        return dict(
            wait=patch.object(nb, "_wait_for_server", side_effect=wait_results),
            env=patch.object(nb, "_notebook_env", return_value=env),
            logs_env=patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": ""}),
            webconsole=patch(
                "retrain_pipelines.dag_engine.web_console.main._webconsole_start"
            ),
            sleep=patch("time.sleep"),
            display=patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.display"
            ),
            clear=patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"
            ),
        )

    def test_returns_early_on_server_fail_no_iframe_displayed(self):
        with (
            patch.object(nb, "_wait_for_server", return_value=False),
            patch.object(nb, "_notebook_env", return_value="local"),
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": ""}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.display"
            ) as mock_display,
        ):
            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        displayed = " ".join(str(c) for c in mock_display.call_args_list)
        assert "iframe" not in displayed.lower()

    def test_displays_iframe_on_local(self):
        with (
            patch.object(nb, "_wait_for_server", return_value=True),
            patch.object(nb, "_notebook_env", return_value="local"),
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": ""}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch("time.sleep"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"
            ),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.display"
            ) as mock_display,
        ):
            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        displayed = " ".join(
            a.data if hasattr(a, "data") else str(a)
            for call in mock_display.call_args_list
            for a in call.args
        )
        assert "iframe" in displayed.lower()

    def test_calls_ngrok_start_on_colab(self):
        with (
            patch.object(nb, "_wait_for_server", return_value=True),
            patch.object(nb, "_notebook_env", return_value="colab"),
            patch.object(
                nb, "_ngrok_start", return_value="https://t.ngrok.io"
            ) as mock_ngrok,
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": ""}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch("time.sleep"),
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"
            ),
        ):
            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        mock_ngrok.assert_called_once_with(8000)

    def test_shuts_down_and_returns_when_ngrok_fails(self):
        with (
            patch.object(nb, "_wait_for_server", return_value=True),
            patch.object(nb, "_notebook_env", return_value="colab"),
            patch.object(nb, "_ngrok_start", return_value=None),
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": ""}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch("time.sleep"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.display"
            ) as mock_display,
        ):
            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        # webconsole_shutdown must have been called (via main._main)
        displayed = " ".join(str(c) for c in mock_display.call_args_list)
        assert "iframe" not in displayed.lower()

    def test_deactivates_logger_controller_after_iframe(self):
        with (
            patch.object(nb, "_wait_for_server", return_value=True),
            patch.object(nb, "_notebook_env", return_value="local"),
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": ""}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch("time.sleep"),
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"
            ),
            patch.object(main_mod._logger_controller, "deactivate") as mock_deact,
        ):
            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        mock_deact.assert_called()

    def test_returns_early_when_proxy_wait_fails_after_ngrok(self):
        """Second _wait_for_server (proxy check) fails => early return, no iframe."""
        with (
            patch.object(nb, "_wait_for_server", side_effect=[True, False]),
            patch.object(nb, "_notebook_env", return_value="colab"),
            patch.object(nb, "_ngrok_start", return_value="https://t.ngrok.io"),
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": ""}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch("time.sleep"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.display"
            ) as mock_display,
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"
            ),
        ):
            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        displayed = " ".join(
            a.data if hasattr(a, "data") else str(a)
            for c in mock_display.call_args_list
            for a in c.args
        )
        assert "iframe" not in displayed.lower()

    def test_live_log_path_installs_file_handler(self):
        """RP_WEB_SERVER_LOGS set => file handler added to root logger."""
        root_logger = logging.getLogger()
        handlers_before = list(root_logger.handlers)

        def _fake_handler_init(self, *args, **kwargs):
            self.stream = MagicMock()
            self.level = logging.DEBUG
            self.filters = []
            self.lock = threading.RLock()
            self.formatter = None
            self._name = None
            self.name = None

        with (
            patch.object(nb, "_wait_for_server", return_value=True),
            patch.object(nb, "_notebook_env", return_value="local"),
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": "/tmp/rp_test_logs"}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch("time.sleep"),
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"
            ),
            patch(
                "logging.handlers.TimedRotatingFileHandler.__init__", _fake_handler_init
            ),
            patch("logging.handlers.TimedRotatingFileHandler.emit"),
            patch("os.makedirs"),
        ):
            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        # restore handlers added during the call
        root_logger.handlers = handlers_before

    def test_live_log_path_spawns_cleanup_thread_and_registers_atexit(self):
        """With live_log_path set, a daemon cleanup thread is started and atexit registered."""

        def _fake_handler_init(self, *args, **kwargs):
            self.stream = MagicMock()
            self.level = logging.DEBUG
            self.filters = []
            self.lock = threading.RLock()
            self.formatter = None
            self._name = None
            self.name = None

        with (
            patch.object(nb, "_wait_for_server", return_value=True),
            patch.object(nb, "_notebook_env", return_value="local"),
            patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": "/tmp/rp_test_logs2"}),
            patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
            patch("time.sleep"),
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"
            ),
            patch("os.makedirs"),
            patch(
                "logging.handlers.TimedRotatingFileHandler.__init__", _fake_handler_init
            ),
            patch("logging.handlers.TimedRotatingFileHandler.emit"),
            patch("threading.Thread") as mock_thread_cls,
            patch("atexit.register") as mock_atexit,
        ):
            mock_thread_instance = MagicMock()
            mock_thread_cls.return_value = mock_thread_instance

            nb._webconsole_start_notebook(port=8000, grpc_port=50051)

        mock_thread_cls.assert_called()
        mock_thread_instance.start.assert_called()
        mock_atexit.assert_called()


# ---------------------------------------------------------------------------
# _webconsole_shutdown_notebook
# ---------------------------------------------------------------------------


class TestWebconsoleShutdownNotebook:
    def setup_method(self):
        nb._notebook_env.cache_clear()
        main_mod._server = None
        main_mod._server_thread = None
        main_mod._grpc_thread = None
        main_mod._grpc_server = None

    def teardown_method(self):
        nb._notebook_env.cache_clear()

    def test_displays_warning_when_no_server(self):
        main_mod._server = None
        with patch(
            "retrain_pipelines.dag_engine.web_console.main_notebook.display"
        ) as mock_display:
            nb._webconsole_shutdown_notebook()
        displayed = " ".join(
            a.data if hasattr(a, "data") else str(a)
            for call in mock_display.call_args_list
            for a in call.args
        )
        assert "not running" in displayed.lower()

    def test_sets_force_and_should_exit_on_server(self):
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv

        t = MagicMock()
        t.is_alive.return_value = False
        main_mod._server_thread = t
        main_mod._grpc_thread = t
        main_mod._grpc_server = None

        with patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"):
            nb._webconsole_shutdown_notebook()

        assert srv.should_exit is True
        assert srv.force_exit is True

    def test_stops_grpc_server(self):
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv

        grpc_srv = MagicMock()
        main_mod._grpc_server = grpc_srv

        t = MagicMock()
        t.is_alive.return_value = False
        main_mod._server_thread = t
        main_mod._grpc_thread = t

        with patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"):
            nb._webconsole_shutdown_notebook()

        grpc_srv.stop.assert_called_once()

    def test_calls_ngrok_stop_on_colab(self):
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv
        main_mod._grpc_server = None

        t = MagicMock()
        t.is_alive.return_value = False
        main_mod._server_thread = t
        main_mod._grpc_thread = t

        with (
            patch.object(nb, "_notebook_env", return_value="colab"),
            patch.object(nb, "_ngrok_stop") as mock_stop,
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
        ):
            nb._webconsole_shutdown_notebook()

        mock_stop.assert_called_once()

    def test_displays_server_down_confirmation(self):
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv
        main_mod._grpc_server = None

        t = MagicMock()
        t.is_alive.return_value = False
        main_mod._server_thread = t
        main_mod._grpc_thread = t

        with patch(
            "retrain_pipelines.dag_engine.web_console.main_notebook.display"
        ) as mock_display:
            nb._webconsole_shutdown_notebook()

        displayed = " ".join(
            a.data if hasattr(a, "data") else str(a)
            for call in mock_display.call_args_list
            for a in call.args
        )
        assert "down" in displayed.lower()

    def test_ctypes_kill_when_server_thread_stays_alive(self):
        """Server thread remains alive after join => ctypes async exc is raised."""
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv
        main_mod._grpc_server = None

        alive_thread = MagicMock()
        alive_thread.is_alive.return_value = True
        alive_thread.ident = 12345
        main_mod._server_thread = alive_thread

        dead_grpc = MagicMock()
        dead_grpc.is_alive.return_value = False
        main_mod._grpc_thread = dead_grpc

        mock_pythonapi = MagicMock()
        mock_pythonapi.PyThreadState_SetAsyncExc.return_value = 1

        with (
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
            patch("ctypes.pythonapi", mock_pythonapi),
        ):
            nb._webconsole_shutdown_notebook()

        mock_pythonapi.PyThreadState_SetAsyncExc.assert_called()

    def test_ctypes_kill_when_grpc_thread_stays_alive(self):
        """gRPC thread remains alive after its join => ctypes async exc is raised."""
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv
        main_mod._grpc_server = None

        dead_server_thread = MagicMock()
        dead_server_thread.is_alive.return_value = False
        main_mod._server_thread = dead_server_thread

        # grpc thread: first is_alive() True (enters block), second is_alive() True
        # (enters nested ctypes block), join() no-ops
        alive_grpc = MagicMock()
        alive_grpc.is_alive.return_value = True
        alive_grpc.ident = 99999
        main_mod._grpc_thread = alive_grpc

        mock_pythonapi = MagicMock()
        mock_pythonapi.PyThreadState_SetAsyncExc.return_value = 1

        with (
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
            patch("ctypes.pythonapi", mock_pythonapi),
        ):
            nb._webconsole_shutdown_notebook()

        mock_pythonapi.PyThreadState_SetAsyncExc.assert_called()

    def test_calls_ngrok_stop_on_kaggle(self):
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv
        main_mod._grpc_server = None

        t = MagicMock()
        t.is_alive.return_value = False
        main_mod._server_thread = t
        main_mod._grpc_thread = t

        with (
            patch.object(nb, "_notebook_env", return_value="kaggle"),
            patch.object(nb, "_ngrok_stop") as mock_stop,
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
        ):
            nb._webconsole_shutdown_notebook()

        mock_stop.assert_called_once()

    def test_deactivates_logger_controller_on_shutdown(self):
        srv = MagicMock(spec=["should_exit", "force_exit"])
        srv.should_exit = False
        srv.force_exit = False
        main_mod._server = srv
        main_mod._grpc_server = None

        t = MagicMock()
        t.is_alive.return_value = False
        main_mod._server_thread = t
        main_mod._grpc_thread = t

        with (
            patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
            patch.object(main_mod._logger_controller, "deactivate") as mock_deact,
        ):
            nb._webconsole_shutdown_notebook()

        mock_deact.assert_called()


# ---------------------------------------------------------------------------
# Inner helpers defined inside _webconsole_start_notebook (live_log_path path)
#
# Strategy: run _webconsole_start_notebook with RP_WEB_SERVER_LOGS set, capture
# the live objects the module installs (sys.stdout replacement, the live handler
# added to the root logger, and the real closures passed to atexit.register and
# threading.Thread), then invoke those real objects to hit the source lines.
# ---------------------------------------------------------------------------


def _run_start_with_live_log():
    """Run _webconsole_start_notebook with live_log_path active.

    Stubs TimedRotatingFileHandler.__init__ (no real file) and
    TimedRotatingFileHandler.emit at the *base-class* level only, so the
    module's _FlushingRotatingFileHandler subclass emit body still executes.
    Captures and returns:
      - stdout  : the _ServerThreadStream that replaced sys.stdout
      - stderr  : the _ServerThreadStream that replaced sys.stderr
      - atexit_fn   : the callable passed to atexit.register
      - cleanup_fn  : the target callable passed to threading.Thread
    """
    import logging.handlers as _lh

    captured = {}
    real_stdout = sys.stdout
    real_stderr = sys.stderr

    def _fake_handler_init(self, *args, **kwargs):
        self.stream = MagicMock()
        self.level = logging.DEBUG
        self.filters = []
        self.lock = threading.RLock()
        self.formatter = None
        self._name = None
        self.name = None

    def _capture_atexit(fn):
        captured["atexit_fn"] = fn

    def _capture_thread(**kwargs):
        captured["cleanup_fn"] = kwargs.get("target")
        m = MagicMock()
        m.start = MagicMock()
        return m

    with (
        patch.object(nb, "_wait_for_server", return_value=True),
        patch.object(nb, "_notebook_env", return_value="local"),
        patch.dict("os.environ", {"RP_WEB_SERVER_LOGS": "/tmp/rp_cov_test"}),
        patch("retrain_pipelines.dag_engine.web_console.main._webconsole_start"),
        patch("time.sleep"),
        patch("retrain_pipelines.dag_engine.web_console.main_notebook.display"),
        patch("retrain_pipelines.dag_engine.web_console.main_notebook.clear_output"),
        patch("os.makedirs"),
        patch.object(_lh.TimedRotatingFileHandler, "__init__", _fake_handler_init),
        patch.object(_lh.TimedRotatingFileHandler, "emit", lambda self, r: None),
        patch("atexit.register", side_effect=_capture_atexit),
        patch("threading.Thread", side_effect=_capture_thread),
    ):
        nb._webconsole_start_notebook(port=8000, grpc_port=50051)
        captured["stdout"] = sys.stdout
        captured["stderr"] = sys.stderr

    sys.stdout = real_stdout
    sys.stderr = real_stderr
    return captured


class TestInnerHelpersLiveLogPath:
    """Cover closures/inner classes only reachable when live_log_path is set."""

    def setup_method(self):
        nb._notebook_env.cache_clear()

    def teardown_method(self):
        nb._notebook_env.cache_clear()
        import io

        if not isinstance(sys.stdout, io.TextIOWrapper) and hasattr(
            sys.stdout, "_original"
        ):
            sys.stdout = sys.stdout._original
        if not isinstance(sys.stderr, io.TextIOWrapper) and hasattr(
            sys.stderr, "_original"
        ):
            sys.stderr = sys.stderr._original

    def _live_handler(self):
        """Return the _FlushingRotatingFileHandler the module added to root logger."""
        import logging.handlers as _lh

        for h in logging.getLogger().handlers:
            if isinstance(h, _lh.TimedRotatingFileHandler):
                return h
        return None

    # -- _plain / _PlainFormatter.format --

    def test_plain_formatter_format_strips_all_markup_variants(self):
        """_PlainFormatter.format() strips Rich markup, │=>|, ─=>-, ANSI=>'' via the real formatter."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            _run_start_with_live_log()
            h = self._live_handler()
            assert h is not None and h.formatter is not None
            record = logging.LogRecord(
                name="t",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="[bold]hi[/] \u2502 sep \u2500 \x1b[0m end",
                args=(),
                exc_info=None,
            )
            result = h.formatter.format(record)
            assert "\u2502" not in result and "|" in result
            assert "\u2500" not in result
            assert "[bold]" not in result
            assert "\x1b" not in result
        finally:
            root.handlers = handlers_before

    # -- _FlushingRotatingFileHandler.emit --

    def test_flushing_handler_emit_calls_super_then_flush(self):
        """_FlushingRotatingFileHandler.emit() on the installed handler hits the module's emit body."""
        import logging.handlers as _lh

        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            _run_start_with_live_log()
            h = self._live_handler()
            assert h is not None
            flushed = []
            record = logging.LogRecord(
                name="t",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="x",
                args=(),
                exc_info=None,
            )
            with (
                patch.object(
                    _lh.TimedRotatingFileHandler, "emit", lambda self, r: None
                ),
                patch.object(h, "flush", side_effect=lambda: flushed.append(True)),
            ):
                h.emit(record)
            assert flushed
        finally:
            root.handlers = handlers_before

    # -- _ServerThreadStream.write --

    def test_server_thread_stream_write_server_branch_happy_path(self):
        """write() in server-thread context writes to the handler stream."""
        captured = _run_start_with_live_log()
        stream = captured["stdout"]
        assert hasattr(stream, "_original")
        stream._original = MagicMock()
        with patch.object(main_mod, "_server_thread", threading.current_thread()):
            stream.write("server-data")
        stream._original.write.assert_not_called()

    def test_server_thread_stream_write_non_server_branch(self):
        """write() outside server-thread context delegates to _original."""
        captured = _run_start_with_live_log()
        stream = captured["stdout"]
        mock_original = MagicMock()
        stream._original = mock_original
        with patch.object(main_mod, "_server_thread", None):
            stream.write("cell-data")
        mock_original.write.assert_called_once_with("cell-data")

    def test_server_thread_stream_write_server_branch_value_error_suppressed(self):
        """write() server branch: ValueError on handler stream => silently passes."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            captured = _run_start_with_live_log()
            stream = captured["stdout"]
            stream._original = MagicMock()
            h = self._live_handler()
            assert h is not None
            h.stream.write.side_effect = ValueError("closed")
            with patch.object(main_mod, "_server_thread", threading.current_thread()):
                stream.write("x")  # must not raise
        finally:
            root.handlers = handlers_before

    def test_server_thread_stream_write_server_branch_attribute_error_suppressed(self):
        """write() server branch: AttributeError on handler stream => silently passes."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            captured = _run_start_with_live_log()
            stream = captured["stdout"]
            stream._original = MagicMock()
            h = self._live_handler()
            assert h is not None
            h.stream.write.side_effect = AttributeError("gone")
            with patch.object(main_mod, "_server_thread", threading.current_thread()):
                stream.write("x")  # must not raise
        finally:
            root.handlers = handlers_before

    # -- _ServerThreadStream.flush --

    def test_server_thread_stream_flush_calls_original_flush(self):
        """flush() always calls _original.flush()."""
        captured = _run_start_with_live_log()
        stream = captured["stdout"]
        mock_original = MagicMock()
        stream._original = mock_original
        stream.flush()
        mock_original.flush.assert_called()

    def test_server_thread_stream_flush_value_error_suppressed(self):
        """flush(): ValueError on handler stream.flush() => silently passes."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            captured = _run_start_with_live_log()
            stream = captured["stdout"]
            stream._original = MagicMock()
            h = self._live_handler()
            assert h is not None
            h.stream.flush.side_effect = ValueError("closed")
            stream.flush()  # must not raise
        finally:
            root.handlers = handlers_before

    def test_server_thread_stream_flush_attribute_error_suppressed(self):
        """flush(): AttributeError on handler stream.flush() => silently passes."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            captured = _run_start_with_live_log()
            stream = captured["stdout"]
            stream._original = MagicMock()
            h = self._live_handler()
            assert h is not None
            h.stream.flush.side_effect = AttributeError("gone")
            stream.flush()  # must not raise
        finally:
            root.handlers = handlers_before

    # -- _ServerThreadStream.__getattr__ --

    def test_server_thread_stream_getattr_proxies_to_original(self):
        """__getattr__ proxies unknown attributes to _original."""
        captured = _run_start_with_live_log()
        stream = captured["stdout"]
        mock_original = MagicMock()
        mock_original.encoding = "utf-8"
        stream._original = mock_original
        assert stream.encoding == "utf-8"

    # -- _write_termination_line (real closure via atexit capture) --

    def test_write_termination_line_value_error_suppressed(self):
        """_write_termination_line() (real closure): ValueError on closed stream => silently passes."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            captured = _run_start_with_live_log()
            h = self._live_handler()
            assert h is not None
            h.stream.write.side_effect = ValueError("closed")
            captured["atexit_fn"]()  # must not raise
        finally:
            root.handlers = handlers_before

    def test_write_termination_line_attribute_error_suppressed(self):
        """_write_termination_line() (real closure): AttributeError on closed stream => silently passes."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        try:
            captured = _run_start_with_live_log()
            h = self._live_handler()
            assert h is not None
            h.stream.write.side_effect = AttributeError("gone")
            captured["atexit_fn"]()  # must not raise
        finally:
            root.handlers = handlers_before

    # -- _cleanup_on_shutdown (real closure via threading.Thread target capture) --

    def test_cleanup_on_shutdown_joins_server_and_grpc_threads(self):
        """_cleanup_on_shutdown() (real closure): joins _server_thread and _grpc_thread."""
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        real_stdout = sys.stdout
        real_stderr = sys.stderr
        try:
            captured = _run_start_with_live_log()
            mock_server_t = MagicMock()
            mock_grpc_t = MagicMock()
            with (
                patch.object(main_mod, "_server_thread", mock_server_t),
                patch.object(main_mod, "_grpc_thread", mock_grpc_t),
            ):
                captured["cleanup_fn"]()
            mock_server_t.join.assert_called_once()
            mock_grpc_t.join.assert_called_once()
        finally:
            root.handlers = handlers_before
            sys.stdout = real_stdout
            sys.stderr = real_stderr
