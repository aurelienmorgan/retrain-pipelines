
import os
import ast
import time

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import get_model_status
from peft.utils import ModulesToSaveWrapper

from pydantic import BaseModel
from typing import List, Dict

import litserve as ls

from litserve_serverconfig import Config


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


class UnslothLitAPI(ls.LitAPI):
    """
    Multi-LoRa (batch) inference server.
    Takes a yaml config file.
    
    Usage:
        ```sh
        curl -X 'POST' \
          'http://localhost:8765/predict' \
          -H 'accept: application/x-www-form-urlencoded' \
          -d 'adapter_name=func_caller' \
          -d 'queries=["Hello.", "Is 49 a perfect square?"]'
        ```
    """

    def setup(self, device):
        start_time = time.time()
        print("Loading weights. May take a small while.")

        # load specific version of base-model
        model, self.tokenizer = \
            FastLanguageModel.from_pretrained(
                model_name=(
                    Config.BASE_MODEL_PATH or
                    Config.BASE_MODEL_REPO_ID),
                revision=(
                    Config.BASE_MODEL_REVISION
                    if Config.BASE_MODEL_PATH is None
                    else None),
                max_seq_length=Config.MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=False,
                token = os.getenv("HF_TOKEN", None)
            )
        self.model = FastLanguageModel.for_inference(model)

        self.eos_token_id = self.tokenizer.all_special_ids[
            self.tokenizer.all_special_tokens.index(
                self.tokenizer.eos_token)]

        self.adapter_tokenizers = {}
        for adapter_name, adapter in Config.adapters.items():
            # load specific version of each adapter
            # and associated tokenizer
            adapter_repo_id = (
                path if (path:=adapter.get("path")) is not None
                else adapter["repo_id"])
            adapter_revision = (
                None if (path:=adapter.get("path")) is not None
                else adapter.get("revision"))
            self.model.load_adapter(
                peft_model_id=adapter_repo_id,
                revision=adapter_revision,
                adapter_name=adapter_name,
                #offload_folder=,
                token=os.getenv("HF_TOKEN", None)
            )
            self.adapter_tokenizers[adapter_name] = \
                AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=adapter_repo_id,
                    revision=adapter_revision,
                    token=os.getenv("HF_TOKEN", None)
                )

        print(f"Load time : {time.time()-start_time:.2f} seconds")

        print("---")
        for adapter_name, config in self.model.peft_config.items():
            print(f"Adapter : {adapter_name}")
            print(f"Type : {config.peft_type}")
            print(f"Task type : {config.task_type}")
            print(f"modules_to_save : {config.modules_to_save}")
            print(f"target_modules : {config.target_modules}")
            print("---")
        print(f"base_model : {self.model.base_model.name_or_path}")
        print("---")


    def decode_request(self, request) -> Request:
        adapter_name = request.get("adapter_name") or ""
        try:
            queries = ast.literal_eval(request["queries"])
        except Exception as e:
            return {"error": str(e)}, 500

        request_obj = Request(
            adapter_name=adapter_name, queries_batch=queries)
        print(f"request_obj : {request_obj}")

        return request_obj


    def predict(self, request: Request) -> Response:

        if (
            request.adapter_name in get_model_status(
                self.model).available_adapters
        ):
            if (
                set([request.adapter_name]) !=
                set(self.model.active_adapters())
            ):
                self.model.set_adapter(adapter_name=request.adapter_name)
            self.model.enable_adapters()
            #################
            # BUG FIX BEGIN #
            #################
            # comparative to old peft tuner (fixed) bug from
            # https://github.com/huggingface/peft/issues/493
            # disable/re-enable adapter re-enables LoRa layers
            # BUT fails to also re-enable "modules_to_save"
            for module in self.model.modules():
                if isinstance(module, ModulesToSaveWrapper):
                    module.enable_adapters(enabled=True)
            #################
            #  BUG FIX END  #
            #################
            print(f"active_adapters : {self.model.active_adapters()}")
            tokenizer = self.adapter_tokenizers[request.adapter_name]
            # print(f"chat_template : {tokenizer.chat_template}")
        else:
            self.model.disable_adapters()
            print("active_adapters : None")
            tokenizer = self.tokenizer

        formatted_inputs = [(tokenizer.chat_template or "{}").format(query, "")
                            for query in request.queries_batch]

        tokenized_inputs = tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to("cuda")
        input_tokens_count_list = [
            tokens.ne(tokenizer.pad_token_id).sum().item()
            for tokens in tokenized_inputs["input_ids"]]

        outputs = self.model.generate(
            input_ids=tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            max_new_tokens=Config.MAX_NEW_TOKENS,
            use_cache=True
        )

        decoded_outputs = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        
        new_tokens_count_list = [
            ((output_tokens != tokenizer.pad_token_id) &
             (output_tokens != self.eos_token_id)).sum().item() + 1 \
            - input_tokens_count
            for input_tokens_count, output_tokens
            in zip(input_tokens_count_list, outputs)
        ]
        print(f"Max new tokens : {max(new_tokens_count_list)}")

        batch_results = [
            QueryOutput(
                query=query,
                input_tokens_count=input_tokens_count,
                completion=output[len(formatted_input):].strip(),
                new_tokens_count=new_tokens_count
            )
            for query, formatted_input, output,
                new_tokens_count, input_tokens_count
            in zip(request.queries_batch, formatted_inputs,
                   decoded_outputs, new_tokens_count_list,
                   input_tokens_count_list)
        ]

        return Response(output=batch_results)


    def encode_response(self, output: Response) -> List[QueryOutput]:
        return output.output


if __name__ == "__main__":
    api = UnslothLitAPI()
    server = ls.LitServer(api, accelerator="cuda",
                          healthcheck_path="/status")
    server.run(port=Config.PORT)

