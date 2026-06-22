"""
Unit tests for retrain_pipelines.dag_engine.web_console.grpc_server
"""

import queue
import threading
from unittest.mock import MagicMock, mock_open, patch

import pytest

from retrain_pipelines.dag_engine.web_console.grpc_server import (
    TaskTraceServicer,
    _get_client_address,
    _thread_loops,
    create_health_servicer,
    logging__task_trace_received,
    serve_grpc,
    serve_grpc_secure,
)

import grpc
from grpc_health.v1 import health_pb2


# --- patch roots --------------------------------------------------------------


_MOD = "retrain_pipelines.dag_engine.web_console.grpc_server"
_EVENTS = f"{_MOD}.execution_events"
_IN_NB = f"{_MOD}.in_notebook"


@pytest.fixture(autouse=True)
def _clean_thread_loops():
    """Isolate per-test thread-loop registry state."""
    _thread_loops.clear()
    yield
    _thread_loops.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  _get_client_address
# ══════════════════════════════════════════════════════════════════════════════


class TestGetClientAddress:
    def _ctx(self, peer_str):
        ctx = MagicMock()
        ctx.peer.return_value = peer_str
        return ctx

    def test_ipv4(self):
        ip, port = _get_client_address(self._ctx("ipv4:127.0.0.1:12345"))
        assert ip == "127.0.0.1" and port == 12345

    def test_ipv6(self):
        ip, port = _get_client_address(self._ctx("ipv6:[::1]:9999"))
        assert ip == "::1" and port == 9999

    def test_unix_socket_returns_peer_and_none(self):
        ip, port = _get_client_address(self._ctx("unix:/tmp/sock"))
        assert ip == "unix:/tmp/sock" and port is None

    def test_unknown_transport_returns_peer_and_none(self):
        ip, port = _get_client_address(self._ctx("pipe:something"))
        assert ip == "pipe:something" and port is None

    def test_port_is_int_for_ipv4(self):
        _, port = _get_client_address(self._ctx("ipv4:10.0.0.1:8080"))
        assert isinstance(port, int)

    def test_port_is_int_for_ipv6(self):
        _, port = _get_client_address(self._ctx("ipv6:[fe80::1]:443"))
        assert isinstance(port, int)


# ══════════════════════════════════════════════════════════════════════════════
#  logging__task_trace_received
# ══════════════════════════════════════════════════════════════════════════════


class TestLoggingTaskTraceReceived:
    def _ctx(self, peer="ipv4:127.0.0.1:5000"):
        ctx = MagicMock()
        ctx.peer.return_value = peer
        return ctx

    def test_non_notebook_logs_via_uvicorn_logger(self):
        with (
            patch(_IN_NB, return_value=False),
            patch(f"{_MOD}.uvicorn_logger") as mock_log,
        ):
            logging__task_trace_received(self._ctx(), trace_id=7)
        mock_log.info.assert_called_once()
        args = mock_log.info.call_args[0]
        assert "7" in str(args)

    def test_non_notebook_assigns_event_loop_to_thread(self):
        with patch(_IN_NB, return_value=False):
            logging__task_trace_received(self._ctx(), trace_id=1)
        tid = threading.get_ident()
        assert tid in _thread_loops

    def test_non_notebook_reuses_existing_loop_for_same_thread(self):
        with patch(_IN_NB, return_value=False):
            logging__task_trace_received(self._ctx(), trace_id=1)
            loop1 = _thread_loops[threading.get_ident()]
            logging__task_trace_received(self._ctx(), trace_id=2)
            loop2 = _thread_loops[threading.get_ident()]
        assert loop1 is loop2

    def test_notebook_creates_loop_and_logs(self):
        with (
            patch(_IN_NB, return_value=True),
            patch(f"{_MOD}.uvicorn_logger") as mock_log,
        ):
            logging__task_trace_received(self._ctx(), trace_id=3)
        mock_log.info.assert_called_once()
        assert threading.get_ident() in _thread_loops

    def test_thread_loop_created_in_background_thread(self):
        results = {}

        def _run():
            with patch(_IN_NB, return_value=False):
                logging__task_trace_received(self._ctx(), trace_id=99)
            results["tid"] = threading.get_ident()

        t = threading.Thread(target=_run)
        t.start()
        t.join()
        assert results["tid"] in _thread_loops


# ══════════════════════════════════════════════════════════════════════════════
#  create_health_servicer
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateHealthServicer:
    def test_returns_health_servicer_instance(self):
        from grpc_health.v1.health import HealthServicer

        hs = create_health_servicer()
        assert isinstance(hs, HealthServicer)

    def test_overall_server_health_is_serving(self):
        hs = create_health_servicer()
        resp = hs.Check(health_pb2.HealthCheckRequest(service=""), None)
        assert resp.status == health_pb2.HealthCheckResponse.SERVING

    def test_task_trace_service_health_is_serving(self):
        from retrain_pipelines.dag_engine.db.grpc import task_trace_pb2

        svc_name = task_trace_pb2.DESCRIPTOR.services_by_name[
            "TaskTraceService"
        ].full_name
        hs = create_health_servicer()
        resp = hs.Check(health_pb2.HealthCheckRequest(service=svc_name), None)
        assert resp.status == health_pb2.HealthCheckResponse.SERVING


# ══════════════════════════════════════════════════════════════════════════════
#  TaskTraceServicer.SendTrace
# ══════════════════════════════════════════════════════════════════════════════


class TestTaskTraceServicerSendTrace:
    def _payload(self, trace_id=1, task_id=10, content="line", is_err=False):
        payload = MagicMock()
        payload.id = trace_id
        payload.task_id = task_id
        payload.microsec = 0
        payload.microsec_idx = 1
        payload.content = content
        payload.is_err = is_err
        payload.timestamp.ToDatetime.return_value = MagicMock(
            timestamp=MagicMock(return_value=1_000_000.0)
        )
        return payload

    def _ctx(self, peer="ipv4:127.0.0.1:1234"):
        ctx = MagicMock()
        ctx.peer.return_value = peer
        return ctx

    def _servicer(self):
        return TaskTraceServicer()

    def test_no_subscribers_returns_consumed_ack(self):
        ev = MagicMock()
        ev.task_trace_subscribers = []
        with patch(_EVENTS, ev), patch(_IN_NB, return_value=False):
            ack = self._servicer().SendTrace(self._payload(), self._ctx())
        assert ack.success is True
        assert "consumed" in ack.message.lower()

    def test_with_subscribers_puts_dict_on_queue(self):
        q = queue.Queue()
        ev = MagicMock()
        ev.task_trace_subscribers = [(q, None)]
        with patch(_EVENTS, ev), patch(_IN_NB, return_value=False):
            ack = self._servicer().SendTrace(self._payload(trace_id=5), self._ctx())
        assert ack.success is True
        item = q.get_nowait()
        assert item["id"] == 5

    def test_with_subscribers_dispatches_to_all_queues(self):
        q1, q2 = queue.Queue(), queue.Queue()
        ev = MagicMock()
        ev.task_trace_subscribers = [(q1, None), (q2, None)]
        with patch(_EVENTS, ev), patch(_IN_NB, return_value=False):
            self._servicer().SendTrace(self._payload(), self._ctx())
        assert not q1.empty()
        assert not q2.empty()

    def test_dispatched_dict_fields_complete(self):
        q = queue.Queue()
        ev = MagicMock()
        ev.task_trace_subscribers = [(q, None)]
        with patch(_EVENTS, ev), patch(_IN_NB, return_value=False):
            self._servicer().SendTrace(
                self._payload(trace_id=9, task_id=3, content="hello", is_err=True),
                self._ctx(),
            )
        d = q.get_nowait()
        for key in (
            "id",
            "task_id",
            "timestamp",
            "microsec",
            "microsec_idx",
            "content",
            "is_err",
        ):
            assert key in d
        assert d["content"] == "hello"
        assert d["is_err"] is True

    def test_queue_put_exception_returns_failure_ack(self):
        q = MagicMock()
        q.put_nowait.side_effect = RuntimeError("full")
        ev = MagicMock()
        ev.task_trace_subscribers = [(q, None)]
        ctx = self._ctx()
        with patch(_EVENTS, ev), patch(_IN_NB, return_value=False):
            ack = self._servicer().SendTrace(self._payload(), ctx)
        assert ack.success is False
        ctx.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)

    def test_error_ack_message_contains_exception_text(self):
        q = MagicMock()
        q.put_nowait.side_effect = RuntimeError("specific error msg")
        ev = MagicMock()
        ev.task_trace_subscribers = [(q, None)]
        with patch(_EVENTS, ev), patch(_IN_NB, return_value=False):
            ack = self._servicer().SendTrace(self._payload(), self._ctx())
        assert "specific error msg" in ack.message


# ══════════════════════════════════════════════════════════════════════════════
#  serve_grpc
# ══════════════════════════════════════════════════════════════════════════════


class TestServeGrpc:
    def test_returns_server(self):
        mock_server = MagicMock()
        with patch(f"{_MOD}.grpc.server", return_value=mock_server):
            result = serve_grpc(grpc_port=50051)
        assert result is mock_server

    def test_server_started(self):
        mock_server = MagicMock()
        with patch(f"{_MOD}.grpc.server", return_value=mock_server):
            serve_grpc(grpc_port=50051)
        mock_server.start.assert_called_once()

    def test_insecure_port_bound(self):
        mock_server = MagicMock()
        with patch(f"{_MOD}.grpc.server", return_value=mock_server):
            serve_grpc(grpc_port=12345)
        mock_server.add_insecure_port.assert_called_once_with("[::]:12345")

    def test_thread_pool_uses_max_workers(self):
        mock_server = MagicMock()
        with (
            patch(f"{_MOD}.grpc.server", return_value=mock_server),
            patch(f"{_MOD}.futures.ThreadPoolExecutor") as mock_tpe,
        ):
            serve_grpc(grpc_port=50051, max_workers=7)
        mock_tpe.assert_called_once_with(max_workers=7)

    def test_health_and_trace_servicers_added(self):
        mock_server = MagicMock()
        with (
            patch(f"{_MOD}.grpc.server", return_value=mock_server),
            patch(
                f"{_MOD}.health_pb2_grpc.add_HealthServicer_to_server"
            ) as mock_health,
            patch(
                f"{_MOD}.task_trace_pb2_grpc.add_TaskTraceServiceServicer_to_server"
            ) as mock_trace,
        ):
            serve_grpc(grpc_port=50051)
        mock_health.assert_called_once()
        mock_trace.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
#  serve_grpc_secure
# ══════════════════════════════════════════════════════════════════════════════


class TestServeGrpcSecure:
    def test_returns_server(self):
        mock_server = MagicMock()
        with (
            patch(f"{_MOD}.grpc.server", return_value=mock_server),
            patch(f"{_MOD}.grpc.ssl_server_credentials"),
            patch("builtins.open", mock_open(read_data=b"data")),
        ):
            result = serve_grpc_secure(grpc_port=50443)
        assert result is mock_server

    def test_server_started(self):
        mock_server = MagicMock()
        with (
            patch(f"{_MOD}.grpc.server", return_value=mock_server),
            patch(f"{_MOD}.grpc.ssl_server_credentials"),
            patch("builtins.open", mock_open(read_data=b"data")),
        ):
            serve_grpc_secure(grpc_port=50443)
        mock_server.start.assert_called_once()

    def test_secure_port_bound(self):
        mock_server = MagicMock()
        with (
            patch(f"{_MOD}.grpc.server", return_value=mock_server),
            patch(f"{_MOD}.grpc.ssl_server_credentials"),
            patch("builtins.open", mock_open(read_data=b"data")),
        ):
            serve_grpc_secure(grpc_port=9999)
        mock_server.add_secure_port.assert_called_once()
        args = mock_server.add_secure_port.call_args[0]
        assert "[::]:9999" in args[0]

    def test_ssl_credentials_built_from_key_and_cert(self):
        mock_server = MagicMock()
        key_data = b"KEY"
        cert_data = b"CERT"

        def _open(path, *a, **kw):
            if "key" in path:
                return mock_open(read_data=key_data)()
            return mock_open(read_data=cert_data)()

        with (
            patch(f"{_MOD}.grpc.server", return_value=mock_server),
            patch(f"{_MOD}.grpc.ssl_server_credentials") as mock_creds,
            patch("builtins.open", side_effect=_open),
        ):
            serve_grpc_secure(
                50443, private_key_path="server.key", certificate_path="server.crt"
            )
        mock_creds.assert_called_once_with(((key_data, cert_data),))

    def test_health_and_trace_servicers_added(self):
        mock_server = MagicMock()
        with (
            patch(f"{_MOD}.grpc.server", return_value=mock_server),
            patch(f"{_MOD}.grpc.ssl_server_credentials"),
            patch("builtins.open", mock_open(read_data=b"x")),
            patch(
                f"{_MOD}.health_pb2_grpc.add_HealthServicer_to_server"
            ) as mock_health,
            patch(
                f"{_MOD}.task_trace_pb2_grpc.add_TaskTraceServiceServicer_to_server"
            ) as mock_trace,
        ):
            serve_grpc_secure(grpc_port=50443)
        mock_health.assert_called_once()
        mock_trace.assert_called_once()
