
import re
import json

import polars as pl
from collections import Counter


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

