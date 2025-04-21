
from pydantic import BaseModel
from typing import List, Optional


class Request(BaseModel):
    adapter_name: Optional[str] = None
    queries_list: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "adapter_name": "func_caller_lora",
                "queries_list": [
                    "Hello there, how's it hanging?",
                    "Is 49 a perfect square?"
                ]
            }
        }


class QueryOutput(BaseModel):
    query: str
    input_tokens_count: int
    completion: str
    new_tokens_count: int


class Response(BaseModel):
    output: List[QueryOutput]

