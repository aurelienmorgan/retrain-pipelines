
from pydantic import BaseModel
from typing import List, Dict


class Request(BaseModel):
    adapter_name: str
    queries_batch: List[str]


class QueryOutput(BaseModel):
    query: str
    input_tokens_count: int
    completion: str
    new_tokens_count: int


class Response(BaseModel):
    output: List[QueryOutput]

