
import os
import grpc
import logging
import time

from urllib.parse import urlparse

from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc


from .db.grpc import task_trace_pb2, \
    task_trace_pb2_grpc


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GRPC_CHANNEL_OPTIONS = [
    ('grpc.keepalive_time_ms', 60_000),          # ping every 60s
    ('grpc.keepalive_timeout_ms', 10_000),       # 10s timeout
    ('grpc.http2.max_pings_without_data', 0),    # allow aggressive pings if needed
    ('grpc.enable_retries', 1),                  # enable automatic retries (bool)
    ('grpc.initial_reconnect_backoff_ms', 100),  # Fail fast when server is down
    ('grpc.min_reconnect_backoff_ms', 100),      # Keep retry attempts fast
    ('grpc.max_reconnect_backoff_ms', 500),      # Prevent hanging on reconnect attempts
]


class GrpcHealthClient:
    def __init__(self, channel: grpc.Channel):
        self._stub = health_pb2_grpc.HealthStub(channel)

    def check(
        self,
        service: str = "",
        timeout: float = 2.0
    ) -> bool:
        """
        Returns True if service is SERVING.
        Raises grpc.RpcError on transport failure.
        """
        response = self._stub.Check(
            health_pb2.HealthCheckRequest(service=service),
            timeout=timeout,
        )
        return response.status == \
            health_pb2.HealthCheckResponse.SERVING


class GrpcClient:
    _instance = None
    _channel = None
    _stub = None
    _pid = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def init(cls):
        """Initialize gRPC channel and stub.

        Singleton."""

        current_pid = os.getpid()
        logger.debug(f"current_pid : {current_pid}")

        if cls._stub is not None:
            if cls._pid == current_pid:
                # Already initialized in this process
                return
            else:
                # Initialized in parent process
                # clean channel and stub
                cls.shutdown()

        server_url = os.environ['RP_WEB_SERVER_URL']
        parsed = urlparse(server_url)
        host = parsed.hostname or "localhost"
        is_secure = parsed.scheme == 'https'
        grpc_address = f'{host}:{os.environ["RP_GRPC_SERVER_PORT"]}'

        if is_secure:
            credentials = grpc.ssl_channel_credentials()
            cls._channel = grpc.secure_channel(
                grpc_address, credentials, options=GRPC_CHANNEL_OPTIONS)
        else:
            cls._channel = grpc.insecure_channel(
                grpc_address, options=GRPC_CHANNEL_OPTIONS)

        ###############################
        # Check gRPC server readiness #
        ###############################
        try:
            health_stub = health_pb2_grpc.HealthStub(cls._channel)
            logger.debug(f"current_pid : {current_pid} - health_stub : {health_stub}")
            response = health_stub.Check(
                health_pb2.HealthCheckRequest(
                    service=task_trace_pb2.DESCRIPTOR.services_by_name[
                        'TaskTraceService'
                    ].full_name
                ),
                timeout=0.5,
            )
            logger.debug(f"current_pid : {current_pid} - response : {response}")
            if response.status != health_pb2.HealthCheckResponse.SERVING:
                cls._channel.close()
                cls._channel = None
                return 
        except grpc.RpcError as e:
            logger.debug(f"current_pid : {current_pid} - health check failed: {e}")
            cls._channel.close()
            cls._channel = None
            return
        ###############################

        cls._stub = task_trace_pb2_grpc.TaskTraceServiceStub(cls._channel)
        cls._pid = current_pid

    @classmethod
    def stub(cls):
        """Get stub. Raises error if not initialized."""
        if cls._stub is None:
            raise RuntimeError(
                "GrpcClient not initialized. Call GrpcClient.init() first."
            )
        return cls._stub

    @classmethod
    def initiated(cls) -> bool:
        """Check if gRPC client is ready."""
        return cls._stub is not None and cls._pid == os.getpid()

    @classmethod
    def shutdown(cls):
        """Clean shutdown of gRPC channel."""
        if cls.initiated():
            cls._channel.close()
            cls._channel = None
            cls._stub = None
            cls._pid = None

    @classmethod
    def _before_fork(cls):
        """Close gRPC channel before forking.

        gRPC's C core is not fork-safe: background threads spawned by
        grpc.insecure_channel() / grpc.secure_channel() hold C-level
        mutexes that are inherited locked by the child, causing a
        deterministic deadlock when the child tries to init gRPC.
        Closing the channel before the fork lets the C core quiesce.
        """
        if cls._channel is not None:
            try:
                cls._channel.close()
            except Exception:
                pass
            cls._channel = None
            cls._stub = None
            cls._pid = None

    @classmethod
    def _after_fork_in_child(cls):
        """Reset gRPC state in child process after fork.

        Child must start with a clean slate regardless of parent state.
        """
        cls._channel = None
        cls._stub = None
        cls._pid = None


os.register_at_fork(
    before=GrpcClient._before_fork,
    after_in_child=GrpcClient._after_fork_in_child,
)

