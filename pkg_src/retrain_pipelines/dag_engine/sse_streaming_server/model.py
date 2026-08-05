from pydantic import BaseModel


class TraceData(BaseModel):
    """Serialisable representation of a single task-trace event."""

    id: int
    task_id: int
    timestamp: int  # epoch milliseconds
    microsec: int
    microsec_idx: int
    content: str
    is_err: bool
