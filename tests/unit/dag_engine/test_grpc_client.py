"""
Unit tests for retrain_pipelines.dag_engine.grpc_client
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from retrain_pipelines.dag_engine.grpc_client import (
    GRPC_CHANNEL_OPTIONS,
    GrpcClient,
    GrpcHealthClient,
)

import grpc
from grpc_health.v1 import health_pb2


# --- patch roots --------------------------------------------------------------

_MOD = "retrain_pipelines.dag_engine.grpc_client"


@pytest.fixture(autouse=True)
def _reset_grpc_client():
    """Restore GrpcClient class-level singleton state around every test."""
    saved = {
        k: getattr(GrpcClient, k) for k in ("_instance", "_channel", "_stub", "_pid")
    }
    yield
    for k, v in saved.items():
        setattr(GrpcClient, k, v)


# ══════════════════════════════════════════════════════════════════════════════
#  GrpcHealthClient
# ══════════════════════════════════════════════════════════════════════════════


class TestGrpcHealthClient:
    def _make(self):
        channel = MagicMock()
        with patch(f"{_MOD}.health_pb2_grpc.HealthStub") as MockStub:
            client = GrpcHealthClient(channel)
            client._stub = MockStub.return_value
        return client

    def test_check_returns_true_when_serving(self):
        client = self._make()
        resp = MagicMock()
        resp.status = health_pb2.HealthCheckResponse.SERVING
        client._stub.Check.return_value = resp
        assert client.check() is True

    def test_check_returns_false_when_not_serving(self):
        client = self._make()
        resp = MagicMock()
        resp.status = health_pb2.HealthCheckResponse.NOT_SERVING
        client._stub.Check.return_value = resp
        assert client.check() is False

    def test_check_propagates_rpc_error(self):
        client = self._make()
        client._stub.Check.side_effect = grpc.RpcError("boom")
        with pytest.raises(grpc.RpcError):
            client.check()

    def test_check_passes_timeout(self):
        client = self._make()
        resp = MagicMock()
        resp.status = health_pb2.HealthCheckResponse.SERVING
        client._stub.Check.return_value = resp
        client.check(service="svc", timeout=5.0)
        _, kw = client._stub.Check.call_args
        assert kw["timeout"] == 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  GrpcClient – singleton
# ══════════════════════════════════════════════════════════════════════════════


class TestGrpcClientSingleton:
    def test_new_returns_same_instance(self):
        a = GrpcClient()
        b = GrpcClient()
        assert a is b


# ══════════════════════════════════════════════════════════════════════════════
#  GrpcClient.initiated / stub
# ══════════════════════════════════════════════════════════════════════════════


class TestGrpcClientState:
    def test_initiated_false_when_stub_none(self):
        GrpcClient._stub = None
        assert GrpcClient.initiated() is False

    def test_initiated_false_when_pid_mismatch(self):
        GrpcClient._stub = MagicMock()
        GrpcClient._pid = os.getpid() + 1
        assert GrpcClient.initiated() is False

    def test_initiated_true_when_stub_and_pid_match(self):
        GrpcClient._stub = MagicMock()
        GrpcClient._pid = os.getpid()
        assert GrpcClient.initiated() is True

    def test_stub_raises_when_not_initialized(self):
        GrpcClient._stub = None
        with pytest.raises(RuntimeError, match="not initialized"):
            GrpcClient.stub()

    def test_stub_returns_stub_when_initialized(self):
        stub = MagicMock()
        GrpcClient._stub = stub
        assert GrpcClient.stub() is stub


# ══════════════════════════════════════════════════════════════════════════════
#  GrpcClient.init
# ══════════════════════════════════════════════════════════════════════════════


class TestGrpcClientInit:
    """Each test starts with a clean (uninitialized) GrpcClient."""

    @pytest.fixture(autouse=True)
    def _clear(self):
        GrpcClient._stub = None
        GrpcClient._channel = None
        GrpcClient._pid = None

    def _make_healthy_stub(self):
        stub = MagicMock()
        resp = MagicMock()
        resp.status = health_pb2.HealthCheckResponse.SERVING
        stub.Check.return_value = resp
        return stub

    def _env(self, scheme="http"):
        return {
            "RP_WEB_SERVER_URL": f"{scheme}://myhost:8080",
            "RP_GRPC_SERVER_PORT": "50051",
        }

    def test_init_creates_insecure_channel_for_http(self):
        with (
            patch.dict(os.environ, self._env("http")),
            patch(f"{_MOD}.grpc.insecure_channel") as mock_ch,
            patch(
                f"{_MOD}.health_pb2_grpc.HealthStub",
                return_value=self._make_healthy_stub(),
            ),
            patch(f"{_MOD}.task_trace_pb2_grpc.TaskTraceServiceStub"),
        ):
            GrpcClient.init()
        mock_ch.assert_called_once_with("myhost:50051", options=GRPC_CHANNEL_OPTIONS)

    def test_init_creates_secure_channel_for_https(self):
        with (
            patch.dict(os.environ, self._env("https")),
            patch(f"{_MOD}.grpc.secure_channel") as mock_ch,
            patch(f"{_MOD}.grpc.ssl_channel_credentials") as mock_creds,
            patch(
                f"{_MOD}.health_pb2_grpc.HealthStub",
                return_value=self._make_healthy_stub(),
            ),
            patch(f"{_MOD}.task_trace_pb2_grpc.TaskTraceServiceStub"),
        ):
            GrpcClient.init()
        mock_ch.assert_called_once_with(
            "myhost:50051", mock_creds.return_value, options=GRPC_CHANNEL_OPTIONS
        )

    def test_init_noop_when_already_initialized_same_pid(self):
        GrpcClient._stub = MagicMock()
        GrpcClient._pid = os.getpid()
        with patch(f"{_MOD}.grpc.insecure_channel") as mock_ch:
            GrpcClient.init()
        mock_ch.assert_not_called()

    def test_init_reinitializes_on_pid_change(self):
        """When _pid mismatches the current pid, init() calls shutdown().

        shutdown() guards on initiated() which requires _pid == os.getpid(),
        so it is a no-op for a stale pid ; the old channel is not closed by
        shutdown().  init() then proceeds to open a new channel and update
        _stub and _pid to reflect the current process.
        """
        old_channel = MagicMock()
        fake_stub = MagicMock()
        GrpcClient._stub = MagicMock()
        GrpcClient._channel = old_channel
        GrpcClient._pid = os.getpid() + 99  # stale pid → initiated() is False

        with (
            patch.dict(os.environ, self._env()),
            patch(f"{_MOD}.grpc.insecure_channel"),
            patch(
                f"{_MOD}.health_pb2_grpc.HealthStub",
                return_value=self._make_healthy_stub(),
            ),
            patch(
                f"{_MOD}.task_trace_pb2_grpc.TaskTraceServiceStub",
                return_value=fake_stub,
            ),
        ):
            GrpcClient.init()

        # shutdown() is a no-op when initiated() is False (pid mismatch),
        # so the old channel is never explicitly closed here.
        old_channel.close.assert_not_called()
        # A new stub and the current pid must be recorded.
        assert GrpcClient._stub is fake_stub
        assert GrpcClient._pid == os.getpid()

    def test_init_aborts_when_health_check_not_serving(self):
        stub = MagicMock()
        resp = MagicMock()
        resp.status = health_pb2.HealthCheckResponse.NOT_SERVING
        stub.Check.return_value = resp

        with (
            patch.dict(os.environ, self._env()),
            patch(f"{_MOD}.grpc.insecure_channel"),
            patch(f"{_MOD}.health_pb2_grpc.HealthStub", return_value=stub),
        ):
            GrpcClient.init()

        assert GrpcClient._stub is None

    def test_init_aborts_on_rpc_error(self):
        stub = MagicMock()
        stub.Check.side_effect = grpc.RpcError("unreachable")

        with (
            patch.dict(os.environ, self._env()),
            patch(f"{_MOD}.grpc.insecure_channel"),
            patch(f"{_MOD}.health_pb2_grpc.HealthStub", return_value=stub),
        ):
            GrpcClient.init()

        assert GrpcClient._stub is None

    def test_init_sets_stub_and_pid_on_success(self):
        fake_stub = MagicMock()
        with (
            patch.dict(os.environ, self._env()),
            patch(f"{_MOD}.grpc.insecure_channel"),
            patch(
                f"{_MOD}.health_pb2_grpc.HealthStub",
                return_value=self._make_healthy_stub(),
            ),
            patch(
                f"{_MOD}.task_trace_pb2_grpc.TaskTraceServiceStub",
                return_value=fake_stub,
            ),
        ):
            GrpcClient.init()

        assert GrpcClient._stub is fake_stub
        assert GrpcClient._pid == os.getpid()


# ══════════════════════════════════════════════════════════════════════════════
#  GrpcClient.shutdown
# ══════════════════════════════════════════════════════════════════════════════


class TestGrpcClientShutdown:
    def test_shutdown_closes_channel_and_clears_state(self):
        ch = MagicMock()
        GrpcClient._channel = ch
        GrpcClient._stub = MagicMock()
        GrpcClient._pid = os.getpid()

        GrpcClient.shutdown()

        ch.close.assert_called_once()
        assert GrpcClient._channel is None
        assert GrpcClient._stub is None
        assert GrpcClient._pid is None

    def test_shutdown_noop_when_not_initiated(self):
        GrpcClient._stub = None
        GrpcClient._pid = None
        GrpcClient.shutdown()  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
#  GrpcClient fork hooks
# ══════════════════════════════════════════════════════════════════════════════


class TestGrpcClientForkHooks:
    def test_before_fork_closes_channel_and_resets(self):
        ch = MagicMock()
        GrpcClient._channel = ch
        GrpcClient._stub = MagicMock()
        GrpcClient._pid = 99

        GrpcClient._before_fork()

        ch.close.assert_called_once()
        assert GrpcClient._channel is None
        assert GrpcClient._stub is None
        assert GrpcClient._pid is None

    def test_before_fork_noop_when_channel_none(self):
        GrpcClient._channel = None
        GrpcClient._stub = MagicMock()
        GrpcClient._before_fork()  # must not raise

    def test_before_fork_swallows_close_exception(self):
        ch = MagicMock()
        ch.close.side_effect = RuntimeError("bad close")
        GrpcClient._channel = ch
        GrpcClient._before_fork()  # must not raise
        assert GrpcClient._channel is None

    def test_after_fork_in_child_always_resets(self):
        GrpcClient._channel = MagicMock()
        GrpcClient._stub = MagicMock()
        GrpcClient._pid = 42

        GrpcClient._after_fork_in_child()

        assert GrpcClient._channel is None
        assert GrpcClient._stub is None
        assert GrpcClient._pid is None
