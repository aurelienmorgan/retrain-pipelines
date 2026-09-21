"""
Unit tests for retrain_pipelines.dag_engine.stores.payloads_store.
"""

import math
import os
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel as PydanticBaseModel

from retrain_pipelines.dag_engine.stores.commons import (
    load_from_disk,
    metadata_root,
)
from retrain_pipelines.dag_engine.stores.payloads_store import (
    serialize_task_payload,
)
from retrain_pipelines.utils.s3_utils import is_s3_path


class TestSerializeTaskPayload:
    """Tests for serialize_task_payload serialization logic."""

    def test_none_value(self, assets_cache):
        """Verify None values are stored as inline null without disk writes."""
        exec_id, task_id = 1, 1
        result = serialize_task_payload(exec_id, task_id, None)
        assert result == {
            "task_id": task_id,
            "sha": None,
            "disk_ref": None,
            "inline_val": None,
        }

    @pytest.mark.parametrize(
        "value, expected_inline",
        [
            (42, 42),
            (3.14, 3.14),
            (True, True),
            ("string", "string"),
            (
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                "2024-01-01T00:00:00+00:00",
            ),
        ],
    )
    def test_basic_json_serializable_value(self, assets_cache, value, expected_inline):
        """Natively JSON-serializable values should be returned as inline."""
        exec_id, task_id = 2, 2
        result = serialize_task_payload(exec_id, task_id, value)
        assert result["task_id"] == task_id
        assert result["sha"] is None
        assert result["disk_ref"] is None
        assert result["inline_val"] == expected_inline

    def test_dict_serializable(self, assets_cache):
        exec_id, task_id = 3, 3
        value = {"key": "val", "nested": {"num": 1}}
        result = serialize_task_payload(exec_id, task_id, value)
        assert result["sha"] is None
        assert result["disk_ref"] is None
        assert result["inline_val"] == value

    def test_tuple_serialized_to_list(self, assets_cache):
        exec_id, task_id = 4, 4
        value = (1, 2, 3)
        result = serialize_task_payload(exec_id, task_id, value)
        assert result["sha"] is None
        assert result["disk_ref"] is None
        assert result["inline_val"] == [1, 2, 3]

    def test_set_serialized_to_list(self, assets_cache):
        exec_id, task_id = 5, 5
        value = {1, 2, 3}
        result = serialize_task_payload(exec_id, task_id, value)
        assert result["sha"] is None
        assert result["disk_ref"] is None
        assert set(result["inline_val"]) == value
        assert isinstance(result["inline_val"], list)

    def test_pydantic_model_serialized_to_dict(self, assets_cache):
        class MyModel(PydanticBaseModel):
            x: int = 1
            y: str = "hello"

        exec_id, task_id = 6, 6
        model_instance = MyModel(x=10, y="world")
        result = serialize_task_payload(exec_id, task_id, model_instance)
        assert result["sha"] is None
        assert result["disk_ref"] is None
        assert result["inline_val"] == {"x": 10, "y": "world"}

    @pytest.mark.parametrize(
        "value",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_non_finite_float_pickled_to_disk(self, assets_cache, value):
        """Verify non-finite floats bypass JSON and trigger disk serialization."""
        exec_id, task_id = 7, 7
        result = serialize_task_payload(exec_id, task_id, value)

        assert result["task_id"] == task_id
        assert isinstance(result["sha"], str)
        assert len(result["sha"]) == 64
        assert result["inline_val"] is None

        expected_rel_path = (
            f"{exec_id}/{task_id}/payload.pkl"
            if is_s3_path(metadata_root())
            else os.path.join(str(exec_id), str(task_id), "payload.pkl")
        )
        assert result["disk_ref"] == expected_rel_path

        loaded = load_from_disk(metadata_root(), result["disk_ref"])
        if math.isnan(value):
            assert math.isnan(loaded)
        else:
            assert loaded == value

    def test_custom_object_pickled_to_disk(self, assets_cache):
        """Verify non-serializable objects are cloudpickled to disk across both backends."""

        class Custom:
            def __init__(self, data):
                self.data = data

        exec_id, task_id = 8, 8
        value = Custom("unserializable")
        result = serialize_task_payload(exec_id, task_id, value)

        assert result["task_id"] == task_id
        assert isinstance(result["sha"], str)
        assert len(result["sha"]) == 64
        assert result["inline_val"] is None

        expected_rel_path = (
            f"{exec_id}/{task_id}/payload.pkl"
            if is_s3_path(metadata_root())
            else os.path.join(str(exec_id), str(task_id), "payload.pkl")
        )
        assert result["disk_ref"] == expected_rel_path

        loaded = load_from_disk(metadata_root(), result["disk_ref"])
        assert isinstance(loaded, Custom)
        assert loaded.data == "unserializable"
