
"""gRPC server for task traces.

Receives task traces from DAG engine
and dispatches to SSE subscribers.
"""

import grpc
import logging
import asyncio
import threading

from datetime import datetime
from concurrent import futures
from urllib.parse import unquote
from google.protobuf.timestamp_pb2 import Timestamp

from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1.health import HealthServicer


from .utils import ClientInfo
from ..db.grpc import task_trace_pb2
from ..db.grpc import task_trace_pb2_grpc

from ...utils import in_notebook

from .utils.execution import events as execution_events


logger = logging.getLogger(__name__)
uvicorn_logger = logging.getLogger("uvicorn.access")

_thread_loops = {}
_thread_loops_lock = threading.Lock()


def _get_client_address(grpc_context):
    peer = unquote(grpc_context.peer())

    if peer.startswith("ipv4:"):
        _, addr = peer.split("ipv4:", 1)
        ip, port = addr.rsplit(":", 1)

    elif peer.startswith("ipv6:"):
        _, addr = peer.split("ipv6:", 1)
        ip, port = addr.rsplit("]:", 1)
        ip = ip.lstrip("[")

    else:
        # unix socket or unknown transport
        return peer, None

    return ip, int(port)

def logging__task_trace_received(
    grpc_context: "_Context",
    trace_id: int
):
    """Add an entry to WebConsole server access log

    Params:
        - grpc_context (grpc._server._Context):
            gRPC context
        - trace_id (int):
    """
    grpc_client_host, grpc_client_port = \
        _get_client_address(grpc_context)

    client_info = ClientInfo(
        ip=grpc_client_host,
        port=grpc_client_port,
        url=""
    )

    # for logging, assign an event loop to current thread
    # In a Notebook, nest_asyncio patches get_running_loop()
    # so it never raises.
    # The except below never fires on bare gRPC threads.
    # Skip straight to the thread-local loop assignment.
    #
    # set_event_loop() is intentionally called on every invocation,
    # not only on first creation: gRPC's ThreadPoolExecutor
    # recycles OS thread IDs, so a new thread can hit
    # an existing _thread_loops entry whose loop was
    # bound to a now-dead thread, leaving the current thread
    # with no loop set.
    if in_notebook():
        thread_id = threading.get_ident()
        with _thread_loops_lock:
            if thread_id not in _thread_loops:
                _thread_loops[thread_id] = asyncio.new_event_loop()
            asyncio.set_event_loop(_thread_loops[thread_id])
            loop = _thread_loops[thread_id]
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            thread_id = threading.get_ident()
            with _thread_loops_lock:
                if thread_id not in _thread_loops:
                    _thread_loops[thread_id] = asyncio.new_event_loop()
                asyncio.set_event_loop(_thread_loops[thread_id])
                loop = _thread_loops[thread_id]
    
    uvicorn_logger.info(
        '%s - "%s %s %s" %d',
        f"{client_info['ip']}:{client_info['port']}",
        "grpc",
        f"{client_info['url']} taskTrace({trace_id})",
        "2.0", 200
    )


def create_health_servicer() -> HealthServicer:
    health_servicer = HealthServicer()

    # Overall server health
    health_servicer.set(
        "",
        health_pb2.HealthCheckResponse.SERVING
    )

    # TaskTrace service health
    health_servicer.set(
        task_trace_pb2.DESCRIPTOR.services_by_name[
            'TaskTraceService'
        ].full_name,
        health_pb2.HealthCheckResponse.SERVING
    )

    return health_servicer


class TaskTraceServicer(task_trace_pb2_grpc.TaskTraceServiceServicer):
    """Handles incoming task trace events from DAG engine.
    
    Forwards incoming gRPC payload to SSE subscribers."""

    def SendTrace(
        self,
        payload: task_trace_pb2.TaskTrace,
        grpc_context: "_Context"
    ):
        """Receive and process a single task trace.

        Params:
            - payload (task_trace_pb2.TaskTrace):
                TaskTrace protobuf message
            - grpc_context (grpc._server._Context):
                gRPC context

        Results:
             - (TraceAck):
                Acknowledgment with processing status
        """
        logging__task_trace_received(grpc_context, trace_id=payload.id)

        if len(execution_events.task_trace_subscribers) > 0:
            # Dispatch to SSE subscribers, if any
            try:
                # Convert protobuf timestamp to epoch milliseconds
                trace_timestamp = int(
                    payload.timestamp.ToDatetime().timestamp() * 1_000)

                # Build trace dict
                trace_dict = {
                    "id": payload.id,
                    "task_id": payload.task_id,
                    "timestamp": trace_timestamp,
                    "microsec": payload.microsec,
                    "microsec_idx": payload.microsec_idx,
                    "content": payload.content,
                    "is_err": payload.is_err
                }

                for q, _ in execution_events.task_trace_subscribers:
                    q.put_nowait(trace_dict)

                return task_trace_pb2.TraceAck(
                    success=True,
                    message="Trace dispatched"
                )

            except Exception as ex:
                logger.error(f"Error processing trace: {ex}",
                             exc_info=True)
                grpc_context.set_code(grpc.StatusCode.INTERNAL)
                grpc_context.set_details(str(ex))
                return task_trace_pb2.TraceAck(
                    success=False,
                    message=f"Error: {ex}"
                )
        else:
            return task_trace_pb2.TraceAck(
                success=True,
                message="Trace consumed"
            )


def serve_grpc(
    grpc_port: int,
    max_workers:int = 10
) -> grpc.Server:
    """Start the gRPC server.

    Params (int):
        grpc_port:
            Port to listen on
        max_workers (int):
            Thread pool size
        
    Results:
         - (grpc.Server):
            server instance
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    #################
    # Add servicers #
    #################

    # health check
    health_servicer = create_health_servicer()
    health_pb2_grpc.add_HealthServicer_to_server(
        health_servicer,
        server
    )

    # taskTrace
    task_trace_pb2_grpc.add_TaskTraceServiceServicer_to_server(
        TaskTraceServicer(), 
        server
    )
    #################

    # # Add reflection for grpcui/grpcurl
    # from grpc_reflection.v1alpha import reflection
    # SERVICE_NAMES = (
        # task_trace_pb2.DESCRIPTOR.services_by_name['TaskTraceService'].full_name,
        # reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Listen
    server.add_insecure_port(f'[::]:{grpc_port}')
    server.start()

    logger.info(f"gRPC TaskTraceService started on port {grpc_port}")
    return server


def serve_grpc_secure(
    grpc_port, 
    private_key_path='server.key', 
    certificate_path='server.crt',
    max_workers=10
):
    """Start the gRPC server with TLS.

    Params:
        port: Port to listen on
        private_key_path: Path to private key file
        certificate_path: Path to certificate file
        max_workers: Thread pool size

    Results:
        grpc.Server instance
    """
    server = grpc.server(futures.ThreadPoolExecutor(
                max_workers=max_workers))

    #################
    # Add servicers #
    #################

    # health check
    health_servicer = create_health_servicer()
    health_pb2_grpc.add_HealthServicer_to_server(
        health_servicer,
        server
    )

    # taskTrace
    task_trace_pb2_grpc.add_TaskTraceServiceServicer_to_server(
        TaskTraceServicer(), 
        server
    )
    #################

    # # Add reflection for grpcui/grpcurl
    # from grpc_reflection.v1alpha import reflection
    # SERVICE_NAMES = (
        # task_trace_pb2.DESCRIPTOR.services_by_name['TaskTraceService'].full_name,
        # reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Load credentials
    with open(private_key_path, 'rb') as f:
        private_key = f.read()
    with open(certificate_path, 'rb') as f:
        certificate_chain = f.read()

    server_credentials = grpc.ssl_server_credentials(
        ((private_key, certificate_chain),)
    )

    # Listen with TLS
    server.add_secure_port(f'[::]:{grpc_port}', server_credentials)
    server.start()

    logger.info(f"gRPC TaskTraceService started (TLS) on port {grpc_port}")
    return server

