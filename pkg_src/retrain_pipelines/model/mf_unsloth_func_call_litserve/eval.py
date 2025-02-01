
import re
import csv
import json

import polars as pl
from tqdm.auto import tqdm
from collections import Counter

import torch

import datasets
import transformers


def infer_validation(
    tokenizer: transformers.PreTrainedTokenizer |
               transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    validation_data: datasets.Dataset,
    prompt_template: str,
    batch_size: int = 32,
    queries_attr_name: str = "query",
    answers_attr_name: str = "answers",
    max_new_tokens: int = 400,
    device: str = "cuda"
) -> list:
    """
    Generates inference on the validation dataset.
    Also provides input (incl. prompt_template)
    and new tokens count.

    Params:
        - tokenizer (transformers.PreTrainedTokenizer |
                     transformers.PreTrainedTokenizerFast):
        - model (transformers.PreTrainedModel);
        - validation_data (datasets.Dataset);
        - prompt_template (str):
        - batch_size (int):
        - queries_attr_name (str):
        - answers_attr_name (int):
        - max_new_tokens (int):
            maximum number of tokens the model
            can generate, excluding the input prompt.
            Note that larger values consume more
            computational resources
            (memory, processing time).
        - device (str):
            e.g. "cuda"

    Results:
        - list(dict):
            query,
            input_tokens_count
                Note : accounts for length
                of prompt_template.
            answer:
                Truth label (list of golden tool-calls).
                Passed-through from input validation data.
            completion:
                Inferred list of tool-calls
            new_tokens_count
    """

    torch.cuda.empty_cache()
    gc.collect()

    eos_token_id = tokenizer.all_special_ids[
        tokenizer.all_special_tokens.index(tokenizer.eos_token)]

    max_new_tokens_count = 0
    results = []

    for i in tqdm(range(0, len(validation_data), batch_size)):
        batch = validation_data[i:i + batch_size]
        queries = batch[queries_attr_name]
        formatted_inputs = [
            prompt.format(query, "") for query in queries]
        answers = batch[answers_attr_name]

        inputs = tokenizer(
            formatted_inputs, padding=True,
            truncation=True, return_tensors="pt"
        ).to(device)

        input_tokens_count_list = [
            tokens.ne(tokenizer.pad_token_id).sum().item()
            for tokens in inputs["input_ids"]]

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        decoded_outputs = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        new_tokens_count_list = [
            ((output_tokens != tokenizer.pad_token_id) &
             (output_tokens != eos_token_id)).sum().item() + 1 \
            - input_tokens_count
            for input_tokens_count, output_tokens
            in zip(input_tokens_count_list, outputs)
        ]
        max_new_tokens_count = \
            max(max_new_tokens_count, max(new_tokens_count_list))

        batch_results = [
            {
                "query": query,
                "input_tokens_count": input_tokens_count,
                "answer": answer,
                "completion": output[len(formatted_input):].strip(),
                "new_tokens_count": new_tokens_count
            }
            for query, answer, formatted_input, output,
            new_tokens_count, input_tokens_count
            in zip(queries, answers, formatted_inputs,
                   decoded_outputs, new_tokens_count_list,
                   input_tokens_count_list)
        ]
        results.extend(batch_results)

    print(f"observed max_new_tokens_count : {max_new_tokens_count}\n")

    return results


def _calculate_metrics(
    predicted: str,
    correct,
    is_format_fault_tolerant: bool = False
):
    """

    Params:
        predicted (str) :
            The predicted tool-calls list
            inferred string.
        correct (str) :
            The ground-truth tool-calls list
            string.
        is_format_fault_tolerant (bool):
            Whether or not a failure to abide
            to valid json formatting shall
            be considered a valid "no-tool-call"
            inference.
            Seems fair to assume so since,
            in an operationnal setting, such
            a model response could legitimately
            translate in such a behavior.
            Defaults to False.

    Results:
        (dict) :
            "ground_truth_tool_calls" (int),
            "predicted_tool_calls" (int),
            "precision" (float),
            "recall" (float),
            "f1": (float),
            "jaccard" (float)
    """

    try:
        predicted_tool_calls_list = json.loads(predicted)
    except Exception as ex:
        predicted_tool_calls_list = None

    try:
        true_tool_calls_list = json.loads(correct)
    except Exception as ex:
        print(ex)
        raise ex

    if predicted_tool_calls_list is None:
        if not is_format_fault_tolerant:
            return {
                "ground_truth_tool_calls":
                    len(true_tool_calls_list),
                "predicted_tool_calls": 0,
                "precision": .0,
                "recall": .0,
                "f1": .0,
                "jaccard": .0
            }
        else:
            predicted_tool_calls_list = []

    if (
        len(predicted_tool_calls_list) == 0 and
        len(true_tool_calls_list) == 0
    ):
            return {
                "ground_truth_tool_calls": 0,
                "predicted_tool_calls": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "jaccard": 1.0
            }

    def _make_hashable(obj) -> tuple:
        if isinstance(obj, dict):
            return tuple((k, _make_hashable(v))
                         for k, v in obj.items())
        elif isinstance(obj, list):
            return tuple(_make_hashable(i)
                         for i in obj)
        else:
            return obj

    hashable_predicted = map(
        _make_hashable, predicted_tool_calls_list)

    predicted_counter = Counter(hashable_predicted)
    hashable_correct = map(
        _make_hashable, true_tool_calls_list)
    correct_counter = Counter(hashable_correct)

    true_positives = sum(
        (predicted_counter & correct_counter).values())
    false_positives = sum(
        predicted_counter.values()) - true_positives
    false_negatives = sum(
        correct_counter.values()) - true_positives

    precision = true_positives/(true_positives+false_positives) \
                if (true_positives+false_positives)>0 else 0
    recall = true_positives/(true_positives+false_negatives) \
             if (true_positives+false_negatives)>0 else 0
    f1 = 2 * (precision*recall)/(precision+recall) \
         if (precision+recall) > 0 else 0
    jaccard = true_positives/ \
              (true_positives+false_positives+false_negatives) \
              if (true_positives+false_positives+false_negatives)>0 \
              else 0

    return {
        "ground_truth_tool_calls": len(true_tool_calls_list),
        "predicted_tool_calls": len(predicted_tool_calls_list),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard
    }


def _compute_metrics_per_row(
    row: dict,
    is_format_fault_tolerant: bool = False
) -> dict:
    """
    Computes counts and performance metrics
    of a given evaluation record
    by a tool-calling model, provided
    prediction vs. ground-truth.

    Every tool-call is accounted for
    and target main metric is jaccard
    for proper intersection over union
    consideration.

    Example usage :
        ```python
        test_eval_df = pl.DataFrame({
            "completion": [
                '[{"tool": "A"}, {"tool": "B"}, {"tool": "A"}]',
                '[{"tool": "AA"}, {"tool": "BB"}, {"tool": "AA"}]'
            ],
            "answer": [
                '[{"tool": "A"}, {"tool": "C"}, {"tool": "A"}]',
                '[{"tool": "AA"}, {"tool": "CC"}, {"tool": "AA"}]'
            ],
        })
        display(test_eval_df)

        test_eval_df.lazy().with_columns(
            pl.struct(["completion", "answer"]
                     ).map_elements(_compute_metrics_per_row,
                        return_dtype=pl.Object).alias("metrics")
        ).collect()
        ```

    Params:
        row (dict):
            A Polars dataframe rowize element.
            Mandatory attribute :
                "completion" :
                    the predicted tool-calls list
                "answer" :
                    the ground-truth tool-calls list
        is_format_fault_tolerant (bool):
            Whether or not a failure to abide
            to valid json formatting shall
            be considered a valid "no-tool-call"
            inference.
            Seems fair to assume so since,
            in an operationnal setting, such
            a model response could legitimately
            translate in such a behavior.
            Defaults to False.

    Results:
        (dict):
            "ground_truth_tool_calls" (int),
            "predicted_tool_calls" (int),
            "precision" (float),
            "recall" (float),
            "f1": (float),
            "jaccard" (float)
    """

    predicted = row["completion"]
    correct = row["answer"]

    return _calculate_metrics(
        predicted, correct, is_format_fault_tolerant)


def compute_counts_n_metrics(
    eval_df: pl.LazyFrame,
    is_format_fault_tolerant: bool = False
) -> pl.LazyFrame:
    """
    Computes counts and performance metrics
    of an evaluation dataset by a tool-calling
    model, provided prediction vs. ground-truth.

    Every tool-call is accounted for
    and target main metric is jaccard
    for proper intersection over union
    consideration.

    Params:
        eval_df (pl.LazyFrame) :
            A Polars dataframe with mandatory
            columns :
                "completion" :
                    the predicted tool-calls list
                "answer" :
                    the ground-truth tool-calls list
        is_format_fault_tolerant (bool):
            Whether or not a failure to abide
            to valid json formatting shall
            be considered a valid "no-tool-call"
            inference.
            Seems fair to assume so since,
            in an operationnal setting, such
            a model response could legitimately
            translate in such a behavior.
            Defaults to False.

    Results:
        (pl.LazyFrame) :
            The input evaluation dataframe
            to which are appended the follwowing
            columns:
                "ground_truth_tool_calls" (int),
                "predicted_tool_calls" (int),
                "precision" (float),
                "recall" (float),
                "f1": (float),
                "jaccard" (float)
    """

    return eval_df.with_columns(
            pl.struct(["completion", "answer"]
                     ).map_elements(
                        lambda row: _compute_metrics_per_row(
                            row, is_format_fault_tolerant),
                        return_dtype=pl.Object
                    ).alias("metrics")
        ).with_columns(
           pl.col("metrics").map_elements(
                   lambda x: x["ground_truth_tool_calls"],
                   return_dtype=pl.Int8
               ).alias("ground_truth_tool_calls"),
           pl.col("metrics").map_elements(
                   lambda x: x["predicted_tool_calls"],
                   return_dtype=pl.Int8
               ).alias("predicted_tool_calls"),
           pl.col("metrics").map_elements(
                   lambda x: x["precision"],
                   return_dtype=pl.Float32
               ).alias("precision"),
           pl.col("metrics").map_elements(
                   lambda x: x["recall"],
                   return_dtype=pl.Float32
               ).alias("recall"),
           pl.col("metrics").map_elements(
                   lambda x: x["f1"],
                   return_dtype=pl.Float32
               ).alias("f1"),
           pl.col("metrics").map_elements(
                   lambda x: x["jaccard"],
                   return_dtype=pl.Float32
               ).alias("jaccard")
        ).select(pl.exclude("metrics"))

