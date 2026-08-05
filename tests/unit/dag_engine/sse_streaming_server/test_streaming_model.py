"""
Unit tests for retrain_pipelines.dag_engine.sse_streaming_server.model.
"""

import pytest
from pydantic import ValidationError

from retrain_pipelines.dag_engine.sse_streaming_server.model import TraceData


class TestTraceData:
    def test_valid_instantiation(self):
        """TraceData accepts valid types and assigns them correctly."""
        data = TraceData(
            id=1,
            task_id=100,
            timestamp=1717238400000,
            microsec=123,
            microsec_idx=1,
            content="A trace message",
            is_err=False,
        )
        assert data.id == 1
        assert data.task_id == 100
        assert data.timestamp == 1717238400000
        assert data.microsec == 123
        assert data.microsec_idx == 1
        assert data.content == "A trace message"
        assert data.is_err is False

    def test_serialization(self):
        """TraceData can be dumped to dict and JSON."""
        data = TraceData(
            id=2,
            task_id=200,
            timestamp=1717238400001,
            microsec=456,
            microsec_idx=2,
            content="Another message",
            is_err=True,
        )
        dumped = data.model_dump()
        assert dumped == {
            "id": 2,
            "task_id": 200,
            "timestamp": 1717238400001,
            "microsec": 456,
            "microsec_idx": 2,
            "content": "Another message",
            "is_err": True,
        }

        json_dumped = data.model_dump_json()
        assert '"content":"Another message"' in json_dumped
        assert '"is_err":true' in json_dumped

    def test_invalid_type_raises_error(self):
        """Passing an invalid type for a field raises ValidationError."""
        with pytest.raises(ValidationError):
            TraceData(
                id="not-an-int",  # Invalid type
                task_id=1,
                timestamp=0,
                microsec=0,
                microsec_idx=0,
                content="x",
                is_err=False,
            )

    def test_missing_field_raises_error(self):
        """Omitting a required field raises ValidationError."""
        with pytest.raises(ValidationError):
            TraceData(
                id=1,
                # task_id missing
                timestamp=0,
                microsec=0,
                microsec_idx=0,
                content="x",
                is_err=False,
            )
