
if updating "task_trace.proto", execute below commands on terminal :

    cd ... /pkg_src/retrain_pipelines/dag_engine/db/grpc

    source an appropriate venv or `pip install grpcio==1.76.0 grpcio-tools==1.64.1 protobuf==5.29.5`

    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. task_trace.proto

in the generated "task_trace_pb2_grpc.py" file, 
replace :
    import task_trace_pb2 as task__trace__pb2
with:
    from . import task_trace_pb2 as task__trace__pb2
