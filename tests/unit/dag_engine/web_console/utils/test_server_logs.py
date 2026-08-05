# test_server_logs_1.py
from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import time
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import boto3
import pytest
from botocore.exceptions import ClientError

from retrain_pipelines.dag_engine.web_console.utils import server_logs
from retrain_pipelines.utils import wsl_utils


class TestGetLogConfig:
    """Tests for the get_log_config function (local and S3 branches)."""

    def test_get_log_config(self, monkeypatch):
        """Ensure log configuration dictionary is correctly structured."""
        monkeypatch.setenv("RP_WEB_SERVER_LOGS", "/tmp/logs")

        cfg = server_logs.get_log_config()

        assert cfg["version"] == 1
        assert cfg["disable_existing_loggers"] is False

        assert cfg["handlers"]["file_default"]["filename"] == "/tmp/logs/server.log"
        assert cfg["handlers"]["file_access"]["filename"] == "/tmp/logs/access.log"

        assert "uvicorn" in cfg["loggers"]
        assert "uvicorn.error" in cfg["loggers"]
        assert "uvicorn.access" in cfg["loggers"]

    def test_get_log_config_s3(self, bucket_name):
        """
        When log_root is an S3 URI, the configuration uses S3RotatingLogHandler.
        """
        s3_uri = f"s3://{bucket_name}/logs/"
        with patch.object(
            server_logs.Config, "get_web_server_logs_root", return_value=s3_uri
        ):
            cfg = server_logs.get_log_config()
            assert (
                cfg["handlers"]["file_default"]["class"]
                == "retrain_pipelines.dag_engine.web_console.utils.server_logs.S3RotatingLogHandler"
            )
            assert cfg["handlers"]["file_default"]["bucket"] == bucket_name
            assert cfg["handlers"]["file_default"]["key"] == "logs/server.log"
            assert (
                cfg["handlers"]["file_access"]["class"]
                == "retrain_pipelines.dag_engine.web_console.utils.server_logs.S3RotatingLogHandler"
            )


class TestAccessLogEntry:
    """Tests for the AccessLogEntry model (extraction, parsing, rendering)."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("127.0.0.1:1234", "127.0.0.1"),
            ("127.0.0.1", "127.0.0.1"),
            ("2001:db8::1:9999", "2001:db8::1"),
            ("::1:9999", "::1"),
        ],
    )
    def test_extract_ip(self, value, expected):
        assert server_logs.AccessLogEntry.extract_ip(value) == expected

    def test_extract_ip_bracketed_ipv6(self):
        # IMPORTANT: matches the implementation's double-escaped regex semantics
        value = r"\[::1\]:12345"
        result = server_logs.AccessLogEntry.extract_ip(value)

        assert result == r"\[::1\]"

    def test_extract_ip_bracketed_ipv6_match(self):
        """Cover the bracketed-IPv6 branch where the regex matches.

        The implementation's regex uses double-escaped backslashes, so the
        matching input must contain literal backslash sequences that align
        with the pattern's escaping.
        """
        value = r"[X]\\]"
        result = server_logs.AccessLogEntry.extract_ip(value)
        assert result == "X]"

    def test_access_log_entry_to_json(self):
        """Verify JSON serialization of AccessLogEntry."""
        entry = server_logs.AccessLogEntry(
            raw_str="raw",
            timestamp=datetime(2025, 1, 1, 10, 0, 0),
            level="INFO",
            client_addr="127.0.0.1",
            method="GET",
            message="/health",
            status_code=200,
        )

        payload = entry.to_json()

        assert '"level": "INFO"' in payload
        assert '"client_addr": "127.0.0.1"' in payload
        assert '"status_code": 200' in payload
        assert '"message": "/health"' in payload

    @pytest.mark.parametrize(
        ("status_code", "method", "expected_title"),
        [
            (200, "GET", "200 - OK"),
            (404, "POST", "404 - Not Found"),
            (500, "DELETE", "500 - Internal Server Error"),
        ],
    )
    def test_to_fasthtml_div_http_titles(self, status_code, method, expected_title):
        """Ensure correct HTTP status titles are generated."""
        entry = server_logs.AccessLogEntry(
            raw_str="raw",
            timestamp=datetime.now(),
            level="INFO",
            client_addr="127.0.0.1",
            method=method,
            message="/path",
            status_code=status_code,
        )

        div = entry.to_fasthtml_div()

        assert getattr(div, "title") == expected_title

    @pytest.mark.parametrize(
        ("method", "expected_title"),
        [
            ("ws", "WebSocket"),
            ("sse", "Server-Side Event"),
        ],
    )
    def test_to_fasthtml_div_special_titles(self, method, expected_title):
        """Ensure correct titles for special connection types."""
        entry = server_logs.AccessLogEntry(
            raw_str="raw",
            timestamp=datetime.now(),
            level="INFO",
            client_addr="127.0.0.1",
            method=method,
            message="/path",
            status_code=200,
        )

        div = entry.to_fasthtml_div()

        assert getattr(div, "title") == expected_title

    def test_to_fasthtml_div_unknown_method(self):
        """Ensure fallback icon and title for unknown HTTP methods."""
        entry = server_logs.AccessLogEntry(
            raw_str="raw",
            timestamp=datetime.now(),
            level="INFO",
            client_addr="127.0.0.1",
            method="CUSTOM",
            message="/path",
            status_code=200,
        )

        div = entry.to_fasthtml_div()

        assert div is not None

    def test_to_fasthtml_div_3xx_status(self):
        """Ensure 3xx status uses the default dark colour."""
        entry = server_logs.AccessLogEntry(
            raw_str="raw",
            timestamp=datetime.now(),
            level="INFO",
            client_addr="127.0.0.1",
            method="GET",
            message="/redirect",
            status_code=304,
        )

        div = entry.to_fasthtml_div()
        assert getattr(div, "title") == "304 - Not Modified"
        style = div.style
        assert "51, 51, 51" in style or "51,51,51" in style

    def test_from_access_log_success(self):
        """Parse a valid access log string successfully."""
        line = "2025-07-06 15:41:53 [INFO] 127.0.0.1:36042 - 'POST /ui/update HTTP/1.1' 200"

        entry = server_logs.AccessLogEntry.from_access_log(line)

        assert entry.level == "INFO"
        assert entry.method == "POST"
        assert entry.status_code == 200
        assert entry.message == "/ui/update"

    def test_from_access_log_invalid(self):
        """Raise ValueError for malformed log strings."""
        with pytest.raises(ValueError):
            server_logs.AccessLogEntry.from_access_log("bad-log-line")


class TestReverseLines:
    """Tests for the reverse_lines generator (local file reads)."""

    def test_reverse_lines_empty_file(self, tmp_path):
        """Handle empty files gracefully."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        with open(f, "rb") as fp:
            assert list(server_logs.reverse_lines(fp)) == []

    def test_reverse_lines_ends_with_newline(self, tmp_path):
        """Correctly reverse lines when file ends with a newline."""
        f = tmp_path / "nl.txt"
        f.write_bytes(b"a\nb\n")
        with open(f, "rb") as fp:
            assert list(server_logs.reverse_lines(fp)) == [b"b", b"a"]

    def test_reverse_lines_no_newline(self, tmp_path):
        """Correctly reverse lines when file does not end with a newline."""
        f = tmp_path / "nonl.txt"
        f.write_bytes(b"a\nb")
        with open(f, "rb") as fp:
            assert list(server_logs.reverse_lines(fp)) == [b"b", b"a"]

    def test_reverse_lines_multi_block(self, tmp_path):
        """Handle files larger than the block size correctly."""
        f = tmp_path / "multi.txt"
        content = b"x" * 10000 + b"\nline1\nline2\n"
        f.write_bytes(content)
        with open(f, "rb") as fp:
            res = list(server_logs.reverse_lines(fp, blocksize=4096))
            assert res[0] == b"line2"
            assert res[1] == b"line1"


class TestReadLastNLines:
    """Tests for the _read_last_n_lines function (Linux and WSL paths)."""

    def test_read_last_n_lines_without_filter(self, tmp_path):
        """Read last N lines without any regex filtering."""
        f = tmp_path / "log.txt"
        f.write_text("a\nb\nc\nd\n")

        result = server_logs._read_last_n_lines(str(f), 2, None)

        assert result == [b"c", b"d"]

    def test_read_last_n_lines_with_filter(self, tmp_path):
        """Read last N lines matching a regex filter."""
        f = tmp_path / "log.txt"
        f.write_text("INFO one\nDEBUG two\nINFO three\nINFO four\n")
        result = server_logs._read_last_n_lines(str(f), 2, r"^INFO")
        assert result == [b"INFO three", b"INFO four"]

    def test_read_last_n_lines_close_error_ignored(self, monkeypatch, tmp_path):
        """Ensure os.close errors are caught and ignored."""
        f = tmp_path / "log.txt"
        f.write_text("a\n")

        original_close = server_logs.os.close

        def bad_close(fd):
            raise OSError("boom")

        monkeypatch.setattr(server_logs.os, "close", bad_close)

        try:
            result = server_logs._read_last_n_lines(str(f), 1, None)
            assert result == [b"a"]
        finally:
            monkeypatch.setattr(server_logs.os, "close", original_close)

    def test_read_last_n_lines_proc_version_open_error(self, monkeypatch, tmp_path):
        """Fallback to standard read when /proc/version is inaccessible.

        is_wsl() lives in wsl_utils and uses its own `os` reference, so we must
        patch wsl_utils.os.path.exists (not server_logs.os.path.exists).  We also
        short-circuit /etc/os-release, the fallback probe inside is_wsl().
        """
        f = tmp_path / "log.txt"
        f.write_text("hello\n")

        original_exists = wsl_utils.os.path.exists

        def patched_exists(path):
            if path in ("/proc/version", "/etc/os-release"):
                return False
            return original_exists(path)

        monkeypatch.setattr(wsl_utils.os.path, "exists", patched_exists)

        result = server_logs._read_last_n_lines(str(f), 1, None)

        assert result == [b"hello"]

    def test_read_last_n_lines_multi_block_loop_exhausted(self, tmp_path):
        """Handle multi-block reads when loop exhausts file."""
        f = tmp_path / "big.log"
        padding = "x" * 4000
        content = f"{padding}\nlineA\nlineB\n"
        f.write_text(content)

        result = server_logs._read_last_n_lines(str(f), 10, None)

        decoded = [r.decode() for r in result]
        assert "lineA" in decoded
        assert "lineB" in decoded

    def test_read_last_n_lines_multi_block_with_filter_loop_exhausted(self, tmp_path):
        """Handle multi-block reads with filter when loop exhausts file."""
        f = tmp_path / "big_filtered.log"
        padding = "y" * 4000
        content = f"{padding}\nINFO alpha\nDEBUG beta\n"
        f.write_text(content)

        result = server_logs._read_last_n_lines(str(f), 5, r"^INFO")

        assert result == [b"INFO alpha"]

    def test_read_last_n_lines_wsl_no_filter(self, monkeypatch):
        """WSL branch: read last N lines without filter."""
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        run_result = SimpleNamespace(returncode=0, stdout="a\nb\nc\n")
        fake_subprocess = SimpleNamespace(run=lambda *args, **kwargs: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 2, None)
        assert result == [b"b", b"c"]

    def test_read_last_n_lines_wsl_filtered(self, monkeypatch):
        """WSL branch: read last N lines with filter."""
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        run_result = SimpleNamespace(returncode=0, stdout="DEBUG\nINFO one\nINFO two\n")
        fake_subprocess = SimpleNamespace(run=lambda *args, **kwargs: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 2, r"^INFO")
        assert result == [b"INFO one", b"INFO two"]

    def test_read_last_n_lines_wsl_nonzero_returncode(self, monkeypatch, tmp_path):
        """WSL branch: fallback to os.open when subprocess fails."""
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        run_result = SimpleNamespace(returncode=1, stdout="")
        fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        real_file = tmp_path / "fallback.log"
        real_file.write_text("line1\n")

        real_os_open = server_logs.os.open

        def fake_os_open(path, flags):
            if path.startswith("/mnt/"):
                return real_os_open(str(real_file), flags)
            return real_os_open(path, flags)

        monkeypatch.setattr(server_logs.os, "open", fake_os_open)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 5, None)
        assert b"line1" in result

    def test_read_last_n_lines_wsl_filtered_exhausted(self, monkeypatch):
        """WSL branch: return available matches when filter exhausts file."""
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        run_result = SimpleNamespace(returncode=0, stdout="INFO only\n")
        fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 3, r"^INFO")
        assert result == [b"INFO only"]

    def test_read_last_n_lines_wsl_filtered_enough_lines_not_enough_matches(
        self, monkeypatch
    ):
        """WSL branch: return available matches when not enough lines match filter."""
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        stdout = "DEBUG a\nDEBUG b\nDEBUG c\nDEBUG d\nINFO one\n"
        run_result = SimpleNamespace(returncode=0, stdout=stdout)
        fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 3, r"^INFO")
        assert len(result) == 1
        assert result == [b"INFO one"]

    def test_read_last_n_lines_wsl_no_filter_exhausted(self, monkeypatch):
        """WSL branch: return available lines when file is shorter than N."""
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        run_result = SimpleNamespace(returncode=0, stdout="a\nb\n")
        fake_subprocess = SimpleNamespace(run=lambda *args, **kwargs: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 5, None)
        assert result == [b"a", b"b"]

    def test_read_last_n_lines_wsl_returns_early_when_enough_matches(self, monkeypatch):
        """WSL branch: return immediately when enough filtered lines are collected."""
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        stdout = "INFO one\nINFO two\nDEBUG three\n"
        run_result = SimpleNamespace(returncode=0, stdout=stdout)
        fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 2, r"^INFO")
        assert result == [b"INFO one", b"INFO two"]

    def test_read_last_n_lines_wsl_true_but_not_mount(self, monkeypatch, tmp_path):
        """
        Cover the branch where is_wsl() returns True but is_wsl_mount_path()
        returns False for the directory, so the function falls back to the
        os.open path instead of using subprocess.
        """
        f = tmp_path / "local.log"
        f.write_text("line1\nline2\nline3\n")

        monkeypatch.setattr(wsl_utils, "is_wsl", lambda: True)
        monkeypatch.setattr(wsl_utils, "is_wsl_mount_path", lambda p: False)

        result = server_logs._read_last_n_lines(str(f), 2, None)
        assert result == [b"line2", b"line3"]

    def test_read_last_n_lines_wsl_subprocess_stdout_empty(self, monkeypatch):
        """
        Cover the WSL branch when subprocess returns stdout that is empty or only
        whitespace, causing str_lines to be empty, and the function falls through
        to the end and returns an empty list (or whatever matches).
        """
        proc_file = mock_open(read_data="Linux version microsoft")
        monkeypatch.setattr("builtins.open", proc_file)

        run_result = SimpleNamespace(returncode=0, stdout="   \n\n")
        fake_subprocess = SimpleNamespace(run=lambda *a, **kw: run_result)
        monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

        result = server_logs._read_last_n_lines("/mnt/c/test.log", 2, None)
        assert result == []


class TestS3RotatingLogHandler:
    """Tests for the S3RotatingLogHandler logging handler."""

    def test_s3_rotating_handler_emit_flush(self, bucket_name):
        """Verify that emit() flushes after flush_every records and rotates on date change."""
        s3 = boto3.client("s3")
        bucket = bucket_name
        key = f"logs/server_{datetime.now().timestamp()}.log"
        handler = server_logs.S3RotatingLogHandler(
            bucket, key, backup_count=2, flush_every=2
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        handler.emit(record)
        assert handler._pending == 1

        handler.emit(record)
        assert handler._pending == 0

        resp = s3.get_object(Bucket=bucket, Key=key)
        assert resp["Body"].read().decode("utf-8") == "test message\ntest message\n"

        handler._current_date = date(2025, 1, 1)
        with patch.object(handler, "_rotate") as mock_rotate:
            handler.emit(record)
            mock_rotate.assert_called_once_with(date(2025, 1, 1))

    def test_s3_rotating_handler_flush_locked_no_existing(self, bucket_name):
        """_flush_locked creates a new object when the key does not exist."""
        s3 = boto3.client("s3")
        bucket = bucket_name
        key = f"logs/server_{datetime.now().timestamp()}.log"
        handler = server_logs.S3RotatingLogHandler(bucket, key)
        handler._buffer = io.StringIO("new content")
        handler._pending = 1

        handler._flush_locked()

        resp = s3.get_object(Bucket=bucket, Key=key)
        assert resp["Body"].read().decode("utf-8") == "new content"

    def test_s3_rotating_handler_flush_locked_with_existing(self, bucket_name):
        """_flush_locked appends to an existing object."""
        s3 = boto3.client("s3")
        bucket = bucket_name
        key = f"logs/server_{datetime.now().timestamp()}.log"
        s3.put_object(Bucket=bucket, Key=key, Body=b"existing\n")

        handler = server_logs.S3RotatingLogHandler(bucket, key)
        handler._buffer = io.StringIO("new\n")
        handler._pending = 1

        handler._flush_locked()

        resp = s3.get_object(Bucket=bucket, Key=key)
        assert resp["Body"].read().decode("utf-8") == "existing\nnew\n"

    def test_s3_rotating_handler_flush_locked_put_failure(self, bucket_name):
        """When put_object fails, _flush_locked logs a warning."""
        bucket = bucket_name
        key = f"logs/server_{datetime.now().timestamp()}.log"
        handler = server_logs.S3RotatingLogHandler(bucket, key)
        handler._buffer = io.StringIO("data")
        handler._pending = 1

        with patch.object(
            handler._s3, "put_object", side_effect=Exception("Put failed")
        ):
            with patch.object(server_logs.logger, "warning") as mock_warning:
                handler._flush_locked()
                mock_warning.assert_called_once()
                assert "S3 log flush failed" in mock_warning.call_args[0][0]

    def test_s3_rotating_handler_rotate(self, bucket_name):
        """_rotate copies the object to a dated key and removes the original."""
        s3 = boto3.client("s3")
        bucket = bucket_name
        key = f"logs/server_{datetime.now().timestamp()}.log"
        s3.put_object(Bucket=bucket, Key=key, Body=b"content")

        handler = server_logs.S3RotatingLogHandler(bucket, key, backup_count=1)
        rotated_date = date(2025, 1, 1)

        handler._rotate(rotated_date)

        dated_key = f"{key}.{rotated_date.isoformat()}"
        resp = s3.get_object(Bucket=bucket, Key=dated_key)
        assert resp["Body"].read().decode("utf-8") == "content"

        with pytest.raises(ClientError) as excinfo:
            s3.head_object(Bucket=bucket, Key=key)
        assert excinfo.value.response["Error"]["Code"] in ("NoSuchKey", "404")

    def test_s3_rotating_handler_prune_old_rotations(self, bucket_name):
        """Only the most recent backup_count rotated objects are kept."""
        s3 = boto3.client("s3")
        bucket = bucket_name
        key = f"logs/server_{datetime.now().timestamp()}.log"
        handler = server_logs.S3RotatingLogHandler(bucket, key, backup_count=2)

        for i in range(5):
            dated_key = f"{key}.2025-01-{i + 1:02d}"
            s3.put_object(Bucket=bucket, Key=dated_key, Body=b"old")

        handler._prune_old_rotations()

        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{key}.")
        keys = [obj["Key"] for obj in resp.get("Contents", [])]
        assert len(keys) == 2
        assert sorted(keys) == [f"{key}.2025-01-04", f"{key}.2025-01-05"]

    def test_s3_rotating_handler_prune_no_backup(self, bucket_name):
        """When backup_count is 0, no rotated objects are deleted."""
        s3 = boto3.client("s3")
        bucket = bucket_name
        key = f"logs/server_{datetime.now().timestamp()}.log"
        handler = server_logs.S3RotatingLogHandler(bucket, key, backup_count=0)

        for i in range(3):
            dated_key = f"{key}.2025-01-{i + 1:02d}"
            s3.put_object(Bucket=bucket, Key=dated_key, Body=b"old")

        handler._prune_old_rotations()

        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{key}.")
        assert len(resp.get("Contents", [])) == 3

    def test_s3_rotating_handler_flush_and_close(self):
        """flush() and close() delegate to _flush_locked."""
        handler = server_logs.S3RotatingLogHandler("bucket", "key")
        with patch.object(handler, "_flush_locked") as mock_flush:
            handler.flush()
            mock_flush.assert_called_once()
        with patch.object(handler, "flush") as mock_flush:
            handler.close()
            mock_flush.assert_called_once()

    def test_s3_rotating_handler_flush_locked_empty_buffer(self):
        """
        Cover the early return in _flush_locked when the buffer is empty.
        This ensures no S3 call is made and no warning is logged.
        """
        handler = server_logs.S3RotatingLogHandler("bucket", "key")
        handler._buffer = io.StringIO("")
        handler._pending = 0

        with patch.object(handler._s3, "put_object") as mock_put:
            with patch.object(server_logs.logger, "warning") as mock_warning:
                handler._flush_locked()
                mock_put.assert_not_called()
                mock_warning.assert_not_called()

    def test_s3_rotating_handler_flush_locked_get_object_raises_non_404(self):
        """
        Cover the branch where get_object raises a ClientError with a code other
        than NoSuchKey or 404, causing the exception to be re-raised and then
        caught by the outer handler, which logs a warning.
        """
        handler = server_logs.S3RotatingLogHandler("bucket", "key")
        handler._buffer = io.StringIO("data")
        handler._pending = 1

        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}}, "GetObject"
        )
        handler._s3 = mock_s3

        with patch.object(server_logs.logger, "warning") as mock_warning:
            handler._flush_locked()
            mock_warning.assert_called_once()
            assert handler._buffer.getvalue() == "data"
            assert handler._pending == 1

    def test_s3_rotating_handler_rotate_copy_fails(self):
        """
        Cover the branch in _rotate where copy_object raises a ClientError,
        resulting in a warning log.
        """
        handler = server_logs.S3RotatingLogHandler("bucket", "key", backup_count=1)
        handler._s3 = MagicMock()
        handler._s3.copy_object.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "Boom"}}, "CopyObject"
        )

        with patch.object(server_logs.logger, "warning") as mock_warning:
            handler._rotate(date(2025, 1, 1))
            mock_warning.assert_called_once()
            assert "S3 log rotation failed" in mock_warning.call_args[0][0]

    def test_s3_rotating_handler_prune_list_fails(self):
        """
        Cover the branch in _prune_old_rotations where list_objects_v2 raises
        a ClientError, causing a warning to be logged.
        """
        handler = server_logs.S3RotatingLogHandler("bucket", "key", backup_count=2)
        handler._s3 = MagicMock()
        handler._s3.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "Boom"}}, "ListObjectsV2"
        )

        with patch.object(server_logs.logger, "warning") as mock_warning:
            handler._prune_old_rotations()
            mock_warning.assert_called_once()
            assert "S3 log pruning failed" in mock_warning.call_args[0][0]

    def test_s3_rotating_handler_emit_format_fails(self):
        """
        Cover the except Exception block in emit when formatting the record
        raises an exception (e.g., due to a malformed record).
        """
        handler = server_logs.S3RotatingLogHandler("bucket", "key")
        handler.format = MagicMock(side_effect=Exception("format error"))

        with patch.object(handler, "handleError") as mock_handle:
            record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
            handler.emit(record)
            mock_handle.assert_called_once_with(record)


class TestFetchLastNLinesLocal:
    """Tests for the _fetch_last_n_lines_local function."""

    def test_fetch_last_n_lines_local(self, tmp_path):
        """
        Collect last N lines from rotated local log files.

        Files are processed in reverse modification time order.
        """
        base = "access.log"
        for i in range(3):
            f = tmp_path / (f"{base}.{i}" if i > 0 else base)
            lines = [f"line from file {i} - {j}\n" for j in range(1, 4)]
            f.write_bytes("".join(lines).encode())
            os.utime(f, (i, i))

        result = server_logs._fetch_last_n_lines_local(str(tmp_path), base, 5, None)
        decoded = [r.decode("utf-8").strip() for r in result]

        assert decoded == [
            "line from file 1 - 2",
            "line from file 1 - 3",
            "line from file 2 - 1",
            "line from file 2 - 2",
            "line from file 2 - 3",
        ]

    def test_fetch_last_n_lines_local_no_files(self, tmp_path):
        """
        Cover the case where no log files match the pattern; the function should
        return an empty list.
        """
        result = server_logs._fetch_last_n_lines_local(
            str(tmp_path), "nonexistent.log", 5, None
        )
        assert result == []


class TestFetchLastNLinesS3:
    """Tests for the _fetch_last_n_lines_s3 function."""

    def test_fetch_last_n_lines_s3_no_filter(self, bucket_name):
        """
        Fetch the last N lines from S3 objects without regex filtering.

        Objects are processed in LastModified descending order.
        """
        s3 = boto3.client("s3")
        bucket = bucket_name
        base = f"access_{datetime.now().timestamp()}.log"

        for i in range(3):
            key = f"logs/{base}.{i}" if i > 0 else f"logs/{base}"
            content = (
                "\n".join([f"line from obj {i} - {j}" for j in range(1, 4)]) + "\n"
            )
            s3.put_object(Bucket=bucket, Key=key, Body=content.encode())
            time.sleep(0.001)  # ensure distinct LastModified

        result = server_logs._fetch_last_n_lines_s3(
            f"s3://{bucket}/logs/", base, 5, None
        )
        decoded = [r.decode("utf-8").strip() for r in result]
        assert len(decoded) == 5
        assert all(line.startswith("line from obj") for line in decoded)

    def test_fetch_last_n_lines_s3_with_filter(self, bucket_name):
        """
        Fetch the last N lines from S3 objects with regex filtering.
        """
        s3 = boto3.client("s3")
        bucket = bucket_name
        base = f"access_{datetime.now().timestamp()}.log"
        key = f"logs/{base}"
        content = "INFO\nDEBUG\nINFO\nWARN\nINFO\n"
        s3.put_object(Bucket=bucket, Key=key, Body=content.encode())

        result = server_logs._fetch_last_n_lines_s3(
            f"s3://{bucket}/logs/", base, 2, "INFO"
        )
        decoded = [r.decode("utf-8").strip() for r in result]
        assert decoded == ["INFO", "INFO"]

    def test_fetch_last_n_lines_s3_no_objects(self, bucket_name):
        """
        Cover the branch where list_objects_v2 returns no Contents.
        """
        s3_uri = f"s3://{bucket_name}/logs/"

        unique_base = f"nonexistent_{datetime.now().timestamp()}.log"
        result = server_logs._fetch_last_n_lines_s3(s3_uri, unique_base, 5, None)
        assert result == []

    def test_fetch_last_n_lines_s3_list_fails(self, bucket_name):
        """
        Cover the branch where list_objects_v2 raises a ClientError,
        causing the function to log a warning and return an empty list.
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "InternalError", "Message": "Boom"}},
            "ListObjectsV2",
        )

        with patch.object(server_logs.boto3, "client", return_value=mock_s3):
            with patch.object(server_logs.logger, "warning") as mock_warning:
                result = server_logs._fetch_last_n_lines_s3(
                    f"s3://{bucket_name}/logs/", "access.log", 5, None
                )

        assert result == []
        mock_warning.assert_called_once()
        assert "S3 log listing failed" in mock_warning.call_args[0][0]

    def test_fetch_last_n_lines_s3_get_object_fails(self, bucket_name):
        """
        Cover the branch where get_object raises a ClientError for one
        object, causing the function to skip it and continue with the next.
        """
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "logs/access.log.1",
                    "LastModified": datetime(2025, 1, 2),
                },
                {
                    "Key": "logs/access.log",
                    "LastModified": datetime(2025, 1, 1),
                },
            ]
        }

        good_body = MagicMock()
        good_body.read.return_value = b"line one\nline two\n"

        def get_object_side_effect(**kwargs):
            if kwargs.get("Key") == "logs/access.log":
                raise ClientError(
                    {"Error": {"Code": "InternalError", "Message": "Boom"}},
                    "GetObject",
                )
            return {"Body": good_body}

        mock_s3.get_object.side_effect = get_object_side_effect

        with patch.object(server_logs.boto3, "client", return_value=mock_s3):
            with patch.object(server_logs.logger, "warning") as mock_warning:
                result = server_logs._fetch_last_n_lines_s3(
                    f"s3://{bucket_name}/logs/", "access.log", 5, None
                )

        decoded = [r.decode("utf-8").strip() for r in result]
        assert "line one" in decoded
        assert "line two" in decoded
        mock_warning.assert_called_once()
        assert "S3 log read failed" in mock_warning.call_args[0][0]


class TestReadLastAccessLogs:
    """Tests for the read_last_access_logs function (Local and S3 branches)."""

    def test_read_last_access_logs(self, monkeypatch):
        """Read and parse last access logs successfully."""
        monkeypatch.setattr(server_logs.glob, "glob", lambda *_: ["new.log", "old.log"])
        monkeypatch.setattr(
            server_logs.os.path, "getmtime", lambda p: 2 if p == "new.log" else 1
        )

        monkeypatch.setattr(
            server_logs,
            "_read_last_n_lines",
            lambda *args, **kwargs: [
                b"2025-07-06 15:41:53 [INFO] 127.0.0.1:36042 - 'GET /health HTTP/1.1' 200"
            ],
        )
        monkeypatch.setattr(server_logs, "strip_ansi_escape_codes", lambda s: s)

        result = server_logs.read_last_access_logs("/tmp", "access.log", 1, None)

        assert len(result) == 1

    def test_read_last_access_logs_skips_bad_lines(self, monkeypatch):
        """Skip malformed log lines during parsing."""
        monkeypatch.setattr(server_logs.glob, "glob", lambda *_: ["access.log"])
        monkeypatch.setattr(server_logs.os.path, "getmtime", lambda *_: 1)
        monkeypatch.setattr(
            server_logs, "_read_last_n_lines", lambda *args, **kwargs: [b"bad"]
        )
        monkeypatch.setattr(server_logs, "strip_ansi_escape_codes", lambda s: s)

        result = server_logs.read_last_access_logs("/tmp", "access.log", 10, None)
        assert result == []

    def test_read_last_access_logs_skips_empty_lines(self, monkeypatch):
        """Skip empty lines during parsing."""
        monkeypatch.setattr(server_logs.glob, "glob", lambda *_: ["access.log"])
        monkeypatch.setattr(server_logs.os.path, "getmtime", lambda *_: 1)
        monkeypatch.setattr(
            server_logs, "_read_last_n_lines", lambda *args, **kwargs: [b""]
        )
        monkeypatch.setattr(server_logs, "strip_ansi_escape_codes", lambda s: s)

        result = server_logs.read_last_access_logs("/tmp", "access.log", 10, None)
        assert result == []

    def test_read_last_access_logs_empty(self, monkeypatch):
        """Return empty list when no lines are read."""
        monkeypatch.setattr(server_logs.glob, "glob", lambda *_: ["access.log"])
        monkeypatch.setattr(server_logs.os.path, "getmtime", lambda *_: 1)
        monkeypatch.setattr(
            server_logs, "_read_last_n_lines", lambda *args, **kwargs: []
        )
        monkeypatch.setattr(server_logs, "strip_ansi_escape_codes", lambda s: s)

        result = server_logs.read_last_access_logs("/tmp", "access.log", 10, None)
        assert result == []

    def test_read_last_access_logs_s3(self, bucket_name):
        """
        Integration test for read_last_access_logs reading from S3.
        """
        lines = [
            "2025-07-06 15:41:53 [INFO] 127.0.0.1:36042 - 'POST /ui/update HTTP/1.1' 200 OK",
            "2025-07-06 15:41:54 [INFO] ::1:54321 - 'GET /api/data HTTP/1.1' 304 Not Modified",
        ]
        s3 = boto3.client("s3")
        bucket = bucket_name
        base = f"access_{datetime.now().timestamp()}.log"
        key = f"logs/{base}"
        content = "\n".join(lines) + "\n"
        s3.put_object(Bucket=bucket, Key=key, Body=content.encode())

        result = server_logs.read_last_access_logs(
            f"s3://{bucket}/logs/", base, 2, None
        )
        assert len(result) == 2
        for res in result:
            assert res.startswith("<div")
            assert "log-entry" in res


class TestWebSocketLogHandler:
    """Tests for the WebSocketLogHandler class."""

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_clients(self, monkeypatch):
        """Remove dead WebSocket clients during broadcast."""
        handler = server_logs.WebSocketLogHandler("/ws")

        good_ws = AsyncMock()
        good_ws.client = ("127.0.0.1", 1234)

        bad_ws = AsyncMock()
        bad_ws.client = ("127.0.0.1", 5678)
        bad_ws.send_text.side_effect = RuntimeError()

        handler.clients = {good_ws, bad_ws}

        entry = MagicMock()
        entry.message = "/other"

        div = MagicMock()
        monkeypatch.setattr(entry, "to_fasthtml_div", lambda: div)

        logger = MagicMock()
        monkeypatch.setattr(server_logs.logging, "getLogger", lambda *_: logger)

        await handler.broadcast(entry)

        assert bad_ws not in handler.clients
        logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_no_recursive_logging(self, monkeypatch):
        """Prevent recursive logging for the websocket's own route."""
        handler = server_logs.WebSocketLogHandler("/same")

        ws = AsyncMock()
        ws.client = ("127.0.0.1", 9999)

        handler.clients = {ws}

        entry = MagicMock()
        entry.message = "/same"
        monkeypatch.setattr(entry, "to_fasthtml_div", lambda: MagicMock())

        logger = MagicMock()
        monkeypatch.setattr(server_logs.logging, "getLogger", lambda *_: logger)

        await handler.broadcast(entry)

        logger.info.assert_not_called()

    def test_register_unregister(self):
        """Register and unregister WebSocket clients."""
        handler = server_logs.WebSocketLogHandler("/ws")

        ws = object()

        handler.register(ws)
        assert ws in handler.clients

        handler.unregister(ws)
        assert ws not in handler.clients

    @pytest.mark.asyncio
    async def test_broadcast_to_others(self, monkeypatch):
        """Broadcast message to all clients except the excluded one."""
        handler = server_logs.WebSocketLogHandler("/ws")
        handler._handlers_added = True

        excluded = AsyncMock()
        alive = AsyncMock()

        dead = AsyncMock()
        dead.send_text.side_effect = RuntimeError()

        handler.clients = {excluded, alive, dead}

        await handler.broadcast_to_others("msg", excluded)

        alive.send_text.assert_awaited_once_with("msg")
        assert dead not in handler.clients

    @pytest.mark.asyncio
    async def test_broadcast_to_others_resets_flag(self):
        """Reset _handlers_added flag when only excluded client remains."""
        handler = server_logs.WebSocketLogHandler("/ws")
        handler._handlers_added = True

        ws = AsyncMock()

        handler.clients = {ws}

        await handler.broadcast_to_others("msg", ws)

        assert handler._handlers_added is False

    def test_emit_non_running_loop(self, monkeypatch):
        """Handle emit when event loop is not running."""
        handler = server_logs.WebSocketLogHandler("/ws")
        loop = MagicMock()
        loop.is_running.return_value = False
        monkeypatch.setattr(server_logs.asyncio, "get_event_loop", lambda: loop)
        monkeypatch.setattr(handler, "broadcast", AsyncMock())

        run_mock = MagicMock()

        def _run_coro(coro):
            temp_loop = asyncio.new_event_loop()
            try:
                temp_loop.run_until_complete(coro)
            finally:
                temp_loop.close()

        run_mock.side_effect = _run_coro
        monkeypatch.setattr(server_logs.asyncio, "run", run_mock)
        monkeypatch.setattr(handler, "format", lambda r: "entry")

        handler.emit(object())

        run_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_running_loop_with_real_task(self, monkeypatch):
        """Handle emit when event loop is running."""
        handler = server_logs.WebSocketLogHandler("/ws")

        loop = MagicMock()
        loop.is_running.return_value = True
        loop.create_task.side_effect = lambda coro: asyncio.create_task(coro)

        broadcast_calls = []

        async def broadcast_mock(record):
            broadcast_calls.append(record)

        monkeypatch.setattr(server_logs.asyncio, "get_event_loop", lambda: loop)
        monkeypatch.setattr(handler, "broadcast", broadcast_mock)
        monkeypatch.setattr(handler, "format", lambda r: "entry")

        handler.emit(object())

        await asyncio.sleep(0)

        loop.create_task.assert_called_once()
        assert len(broadcast_calls) == 1
        assert broadcast_calls[0] == "entry"


class TestGetLogWebsocketEndpoint:
    """Tests for the get_log_websocket_endpoint factory function."""

    @pytest.mark.asyncio
    async def test_websocket_endpoint_first_connection_and_disconnect(
        self, monkeypatch
    ):
        """Handle first WebSocket connection and subsequent disconnect."""
        endpoint = server_logs.get_log_websocket_endpoint("logs")
        ws = AsyncMock()
        ws.client = ("127.0.0.1", 9000)
        ws.receive_text.side_effect = RuntimeError("client gone")
        monkeypatch.setattr(
            server_logs.WebSocketLogHandler, "broadcast_to_others", AsyncMock()
        )

        await endpoint(ws)

        ws.accept.assert_awaited_once()
        ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_handlers_already_added(self, monkeypatch):
        """Handle subsequent WebSocket connections without re-adding handlers."""
        endpoint = server_logs.get_log_websocket_endpoint("logs2")

        bcast = AsyncMock()
        monkeypatch.setattr(
            server_logs.WebSocketLogHandler, "broadcast_to_others", bcast
        )

        ws1 = AsyncMock()
        ws1.client = ("10.0.0.1", 9001)
        ws1.receive_text.side_effect = RuntimeError("ws1 gone")
        await endpoint(ws1)

        ws2 = AsyncMock()
        ws2.client = ("10.0.0.2", 9002)
        ws2.receive_text.side_effect = RuntimeError("ws2 gone")
        await endpoint(ws2)

        ws2.accept.assert_awaited_once()
        ws2.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_close_raises(self, monkeypatch):
        """Handle exceptions during WebSocket close gracefully."""
        endpoint = server_logs.get_log_websocket_endpoint("logs3")

        ws = AsyncMock()
        ws.client = ("192.168.1.1", 9003)
        ws.receive_text.side_effect = RuntimeError("disconnect")
        ws.close.side_effect = RuntimeError("close failed")

        monkeypatch.setattr(
            server_logs.WebSocketLogHandler, "broadcast_to_others", AsyncMock()
        )

        await endpoint(ws)  # must not raise

        ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_route_normalisation(self):
        """Normalize WebSocket route paths correctly."""
        endpoint = server_logs.get_log_websocket_endpoint("//ws/logs")
        assert asyncio.iscoroutinefunction(endpoint)

    @pytest.mark.asyncio
    async def test_access_log_entry_formatter_format_method(self, monkeypatch):
        """Verify AccessLogEntryFormatter correctly formats log records."""
        captured_handlers: list = []

        real_add_handler = logging.Logger.addHandler

        def capturing_add_handler(self, hdlr):
            captured_handlers.append(hdlr)
            real_add_handler(self, hdlr)

        monkeypatch.setattr(logging.Logger, "addHandler", capturing_add_handler)

        endpoint = server_logs.get_log_websocket_endpoint("capture_route")

        ws = AsyncMock()
        ws.client = ("127.0.0.1", 1111)
        ws.receive_text.side_effect = RuntimeError("bye")

        monkeypatch.setattr(
            server_logs.WebSocketLogHandler, "broadcast_to_others", AsyncMock()
        )

        await endpoint(ws)

        assert captured_handlers, "handler was not added to uvicorn.access"

        handler = captured_handlers[0]
        formatter = handler.formatter
        assert formatter is not None

        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='%s - "%s %s %s" %d',
            args=("127.0.0.1:9999", "GET", "/health", "HTTP/1.1", 200),
            exc_info=None,
        )

        log_entry = formatter.format(record)

        assert isinstance(log_entry, server_logs.AccessLogEntry)
        assert log_entry.method == "GET"
        assert log_entry.message == "/health"
        assert log_entry.status_code == 200
        assert log_entry.client_addr == "127.0.0.1"

        logging.getLogger("uvicorn.access").removeHandler(handler)
