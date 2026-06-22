"""
Unit tests for retrain_pipelines/dag_engine/web_console/main_cli_utility.py.

Only OS-level primitives that must not execute for real are mocked:
os.fork, os.kill, os.ttyname, socket.connect_ex, and file I/O where the test
itself controls the pid file via tmp_path.
"""

import os
import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

import retrain_pipelines.dag_engine.web_console.main_cli_utility as cli


@pytest.fixture(autouse=False)
def patch_stdio_fileno():
    """Patch sys.std{out,err,in}.fileno() to return stable fd integers.

    pytest redirects stdin to DontReadFromInput whose fileno() raises
    UnsupportedOperation. The source calls all three fileno()s to build
    the fd list for os.ttyname; patch them to return the conventional
    values (1, 2, 0) so the source can proceed normally.
    """
    with (
        patch.object(sys.stdout, "fileno", return_value=1),
        patch.object(sys.stderr, "fileno", return_value=2),
        patch.object(sys.stdin, "fileno", return_value=0),
    ):
        yield


# ---------------------------------------------------------------------------
# _pid_file
# ---------------------------------------------------------------------------


class TestPidFile:
    def test_uses_tty_from_stdout(self, patch_stdio_fileno):
        with patch("os.ttyname", return_value="/dev/pts/3"):
            result = cli._pid_file()
        assert result == "/tmp/webconsole_dev_pts_3.pid"

    def test_falls_back_to_stderr_when_stdout_raises(self, patch_stdio_fileno):
        call_count = [0]

        def _ttyname(fd):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError
            return "/dev/pts/5"

        with patch("os.ttyname", side_effect=_ttyname):
            result = cli._pid_file()
        assert result == "/tmp/webconsole_dev_pts_5.pid"

    def test_falls_back_to_ppid_when_all_fds_raise(self, patch_stdio_fileno):
        with (
            patch("os.ttyname", side_effect=OSError),
            patch("os.getppid", return_value=42),
        ):
            result = cli._pid_file()
        assert result == "/tmp/webconsole_42.pid"

    def test_slashes_replaced_with_underscores(self, patch_stdio_fileno):
        with patch("os.ttyname", return_value="/dev/pts/99"):
            result = cli._pid_file()
        assert "/" not in result[len("/tmp/") :]


# ---------------------------------------------------------------------------
# _daemonize
# ---------------------------------------------------------------------------


class TestDaemonize:
    def test_parent_branch_calls_sys_exit_0(self):
        """fork() > 0 → parent exits immediately."""
        with (
            patch("os.fork", return_value=99),
            patch("sys.exit", side_effect=SystemExit(0)) as mock_exit,
        ):
            with pytest.raises(SystemExit):
                cli._daemonize(port=8000)
        mock_exit.assert_called_once_with(0)

    def test_child_writes_pid_and_port_to_pid_file(self, tmp_path, patch_stdio_fileno):
        pid_path = str(tmp_path / "webconsole.pid")

        with (
            patch("os.fork", return_value=0),
            patch("threading.Thread") as mock_thread,
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.getppid", return_value=999),
            patch("os.kill"),
            patch("os.dup2"),
            patch("os.devnull", "/dev/null"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.webconsole_start",
                side_effect=SystemExit,
            ),
        ):
            with pytest.raises(SystemExit):
                cli._daemonize(port=8000)

        content = open(pid_path).read().strip()
        pid, port = content.split(":")
        assert int(pid) == os.getpid()
        assert port == "8000"
        mock_thread.assert_called_once()

    def test_child_redirects_all_stdio_to_devnull(self, tmp_path, patch_stdio_fileno):
        pid_path = str(tmp_path / "webconsole.pid")
        dup2_calls = []

        with (
            patch("os.fork", return_value=0),
            patch("threading.Thread"),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.getppid", return_value=999),
            patch("os.kill"),
            patch("os.dup2", side_effect=lambda a, b: dup2_calls.append(b)),
            patch("os.devnull", "/dev/null"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.webconsole_start",
                side_effect=SystemExit,
            ),
        ):
            with pytest.raises(SystemExit):
                cli._daemonize(port=8000)

        # stdin (0), stdout (1), stderr (2) must all be redirected
        for fd in (0, 1, 2):
            assert fd in dup2_calls

    def test_child_calls_shutdown_on_keyboard_interrupt(
        self, tmp_path, patch_stdio_fileno
    ):
        """KeyboardInterrupt after webconsole_start triggers shutdown."""
        pid_path = str(tmp_path / "webconsole.pid")
        sleep_calls = [0]

        def _sleep(n):
            sleep_calls[0] += 1
            if n == 1:
                raise KeyboardInterrupt

        with (
            patch("os.fork", return_value=0),
            patch("threading.Thread"),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.getppid", return_value=999),
            patch("os.dup2"),
            patch("os.devnull", "/dev/null"),
            patch("time.sleep", side_effect=_sleep),
            patch("retrain_pipelines.dag_engine.web_console.main.webconsole_start"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.webconsole_shutdown"
            ) as mock_shutdown,
        ):
            cli._daemonize(port=8000)

        mock_shutdown.assert_called_once()

    def test_child_calls_shutdown_on_system_exit(self, tmp_path, patch_stdio_fileno):
        """SystemExit after webconsole_start triggers shutdown."""
        pid_path = str(tmp_path / "webconsole.pid")

        def _sleep(n):
            if n == 1:
                raise SystemExit

        with (
            patch("os.fork", return_value=0),
            patch("threading.Thread"),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.getppid", return_value=999),
            patch("os.dup2"),
            patch("os.devnull", "/dev/null"),
            patch("time.sleep", side_effect=_sleep),
            patch("retrain_pipelines.dag_engine.web_console.main.webconsole_start"),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.webconsole_shutdown"
            ) as mock_shutdown,
        ):
            cli._daemonize(port=8000)

        mock_shutdown.assert_called_once()

    def test_watch_parent_calls_shutdown_and_exit_when_shell_dies(
        self, tmp_path, patch_stdio_fileno
    ):
        """_watch_parent triggers shutdown when shell_pid vanishes."""
        pid_path = str(tmp_path / "webconsole.pid")
        captured_thread_target = []

        def _capture_thread(**kwargs):
            captured_thread_target.append(kwargs.get("target"))
            m = MagicMock()
            m.start = MagicMock()
            return m

        with (
            patch("os.fork", return_value=0),
            patch("threading.Thread", side_effect=_capture_thread),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.getppid", return_value=999),
            patch("os.dup2"),
            patch("os.devnull", "/dev/null"),
            patch("time.sleep", side_effect=SystemExit),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.webconsole_start",
                side_effect=SystemExit,
            ),
        ):
            with pytest.raises(SystemExit):
                cli._daemonize(port=8000)

        assert captured_thread_target, "Thread was not created"
        watch_fn = captured_thread_target[0]

        # Run _watch_parent directly; sleep(2) returns, kill raises → shutdown + _exit.
        # os._exit must itself raise to break the while True loop (a no-op mock loops
        # forever since the real os._exit would have terminated the process).
        with (
            patch("time.sleep"),
            patch("os.kill", side_effect=ProcessLookupError),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.webconsole_shutdown"
            ) as mock_shutdown2,
            patch("os._exit", side_effect=SystemExit) as mock_exit2,
        ):
            with pytest.raises(SystemExit):
                watch_fn()

        mock_shutdown2.assert_called_once()
        mock_exit2.assert_called_once_with(0)


# ---------------------------------------------------------------------------
# webconsole_start_cli
# ---------------------------------------------------------------------------


class TestWebconsoleStartCli:
    def _sock_stub(self, port_in_use=False):
        mock_sock = MagicMock()
        mock_sock.__enter__ = lambda s: s
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_sock.connect_ex.return_value = 0 if port_in_use else 1
        return mock_sock

    def _base_env(self):
        return {"RP_WEB_SERVER_PORT": "8000", "RP_GRPC_SERVER_PORT": "50051"}

    def test_daemonizes_by_default(self, tmp_path):
        pid_path = str(tmp_path / "webconsole.pid")
        with (
            patch("sys.argv", ["webconsole_start"]),
            patch.dict("os.environ", self._base_env()),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("socket.socket", return_value=self._sock_stub()),
            patch.object(cli, "_daemonize") as mock_daemon,
        ):
            cli.webconsole_start_cli()
        mock_daemon.assert_called_once_with(8000)

    def test_foreground_skips_daemonize_and_calls_start(self, tmp_path):
        pid_path = str(tmp_path / "webconsole.pid")
        with (
            patch("sys.argv", ["webconsole_start", "--foreground"]),
            patch.dict("os.environ", self._base_env()),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("socket.socket", return_value=self._sock_stub()),
            patch(
                "retrain_pipelines.dag_engine.web_console.main.webconsole_start"
            ) as mock_start,
            patch("time.sleep", side_effect=KeyboardInterrupt),
            patch("retrain_pipelines.dag_engine.web_console.main.webconsole_shutdown"),
        ):
            cli.webconsole_start_cli()
        mock_start.assert_called_once()

    def test_refuses_duplicate_instance_same_terminal(self, tmp_path, capsys):
        pid_path = str(tmp_path / "webconsole.pid")
        with open(pid_path, "w") as f:
            f.write(f"{os.getpid()}:8000")  # current process => kill(pid, 0) succeeds

        with (
            patch("sys.argv", ["webconsole_start"]),
            patch.dict("os.environ", self._base_env()),
            patch.object(cli, "_pid_file", return_value=pid_path),
        ):
            cli.webconsole_start_cli()

        assert "already" in capsys.readouterr().out.lower()

    def test_removes_stale_pid_file_and_proceeds(self, tmp_path):
        pid_path = str(tmp_path / "webconsole.pid")
        with open(pid_path, "w") as f:
            f.write("99999:8000")  # dead process

        with (
            patch("sys.argv", ["webconsole_start"]),
            patch.dict("os.environ", self._base_env()),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("socket.socket", return_value=self._sock_stub()),
            patch.object(cli, "_daemonize"),
        ):
            cli.webconsole_start_cli()

        assert not os.path.exists(pid_path)

    def test_refuses_when_first_port_in_use(self, tmp_path, capsys):
        """port check fires on the first port in ports_list."""
        pid_path = str(tmp_path / "webconsole.pid")

        # connect_ex returns 0 (in-use) on the first call (web port), 1 on grpc
        mock_sock = MagicMock()
        mock_sock.__enter__ = lambda s: s
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_sock.connect_ex.side_effect = [0, 1]

        with (
            patch("sys.argv", ["webconsole_start"]),
            patch.dict("os.environ", self._base_env()),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("socket.socket", return_value=mock_sock),
        ):
            cli.webconsole_start_cli()

        assert "not available" in capsys.readouterr().out.lower()

    def test_refuses_when_second_port_in_use(self, tmp_path, capsys):
        """port check fires on the second port in ports_list."""
        pid_path = str(tmp_path / "webconsole.pid")

        # connect_ex returns 1 (free) for web port, 0 (in-use) for grpc port
        mock_sock = MagicMock()
        mock_sock.__enter__ = lambda s: s
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_sock.connect_ex.side_effect = [1, 0]

        with (
            patch("sys.argv", ["webconsole_start"]),
            patch.dict("os.environ", self._base_env()),
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("socket.socket", return_value=mock_sock),
        ):
            cli.webconsole_start_cli()

        assert "not available" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# webconsole_shutdown_cli
# ---------------------------------------------------------------------------


class TestWebconsoleShutdownCli:
    def test_warns_when_no_pid_file(self, capsys):
        with patch.object(cli, "_pid_file", return_value="/tmp/no_such_file.pid"):
            cli.webconsole_shutdown_cli()
        assert "no webconsole" in capsys.readouterr().out.lower()

    def test_sends_sigterm_to_correct_pid(self, tmp_path):
        pid_path = str(tmp_path / "webconsole.pid")
        with open(pid_path, "w") as f:
            f.write("1234:8000")

        with (
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.kill") as mock_kill,
        ):
            cli.webconsole_shutdown_cli()

        mock_kill.assert_called_once_with(1234, signal.SIGTERM)

    def test_removes_pid_file_after_sigterm(self, tmp_path):
        pid_path = str(tmp_path / "webconsole.pid")
        with open(pid_path, "w") as f:
            f.write("1234:8000")

        with patch.object(cli, "_pid_file", return_value=pid_path), patch("os.kill"):
            cli.webconsole_shutdown_cli()

        assert not os.path.exists(pid_path)

    def test_warns_when_process_not_found(self, tmp_path, capsys):
        pid_path = str(tmp_path / "webconsole.pid")
        with open(pid_path, "w") as f:
            f.write("99999:8000")

        with (
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            cli.webconsole_shutdown_cli()

        assert "no process" in capsys.readouterr().out.lower()

    def test_removes_pid_file_even_when_process_not_found(self, tmp_path):
        pid_path = str(tmp_path / "webconsole.pid")
        with open(pid_path, "w") as f:
            f.write("99999:8000")

        with (
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            cli.webconsole_shutdown_cli()

        assert not os.path.exists(pid_path)

    def test_tolerates_pid_file_already_gone_at_unlink(self, tmp_path, capsys):
        """FileNotFoundError during os.unlink is swallowed silently."""
        pid_path = str(tmp_path / "webconsole.pid")
        with open(pid_path, "w") as f:
            f.write("1234:8000")

        with (
            patch.object(cli, "_pid_file", return_value=pid_path),
            patch("os.kill"),
            patch("os.unlink", side_effect=FileNotFoundError),
        ):
            # Must not raise
            cli.webconsole_shutdown_cli()
