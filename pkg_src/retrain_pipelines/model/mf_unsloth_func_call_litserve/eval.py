
import gc
import re
import sys
import csv
import json

import numpy as np
import polars as pl
from tqdm import tqdm
from collections import Counter

import torch

import datasets
import transformers

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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

    for i in tqdm(range(0, len(validation_data), batch_size),
                  file=sys.stdout):
        print("", end="\n", file=sys.stdout, flush=True)
        # print(f{i} / {len(validation_data)/batch_size}",
              # end="\n", file=sys.stdout, flush=True)
        batch = validation_data[i:i + batch_size]
        queries = batch[queries_attr_name]
        formatted_inputs = [
            prompt_template.format(query, "") for query in queries]
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


def _plot_bars(
    ax: plt.Axes,
    grouped: pl.DataFrame,
    xlabel: str,
    subtitle: str,
    max_x: int,
    show_legend: bool = True
) -> None:
    """
    100% stacked bar plots of correct/incorrect
    tool-calls completions, per tool-calls count.

    Overlayed in grey (bottom third of subplot)
    is total records count for each category.

    Note: for x-shared consistency between suplots
    in calling context, we ensure that
    their are as many tool-call counbt groups
    as "max_x".

    Params:
        - ax (plt.Axes)
        - grouped (pl.DataFrame)
        - xlabel (str)
        - subtitle (str)
        - max_x (int)
        - show_legend (bool)
    """

    # fill in with zeros to ensure shared x-axis
    # shows xticklabels for up to max_x
    zeros = pl.DataFrame({
        xlabel: pl.Series(range(max_x + 1),
                          dtype=pl.Int8),
        "correct_count": pl.Series([0] * (max_x + 1),
                                   dtype=pl.UInt32),
        "total_count": pl.Series([0] * (max_x + 1),
                                 dtype=pl.UInt32),
        "incorrect_count": pl.Series([0] * (max_x + 1),
                                     dtype=pl.UInt32)
    }).filter(pl.col(xlabel).is_in(grouped[xlabel]).not_())
    grouped = pl.concat([grouped, zeros]).sort(xlabel)

    total_counts = grouped["total_count"]
    correct_percent = (
        grouped["correct_count"] / 
        np.where(total_counts == 0, 1, total_counts))
    incorrect_percent = (
        grouped["incorrect_count"] / 
        np.where(total_counts == 0, 1, total_counts))
    ax.bar(grouped[xlabel], correct_percent,
           color="lightblue", label="Correct")
    ax.bar(grouped[xlabel], incorrect_percent,
           color="salmon", bottom=correct_percent,
           label="Incorrect")
    ax.set_xlabel(xlabel.replace("_", " ").title())
    ax.set_ylabel("Completions")
    ax.set_title(f"({subtitle})", fontsize=10,
                 pad=5, loc='right')
    ax.set_yscale("linear")
    ax.set_xticks(range(max_x + 1))
    ax.set_xticklabels(range(max_x + 1), fontsize=8)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([f'{x*100:.0f}%'
                        for x in np.arange(0, 1.1, 0.2)],
                       fontsize=8)

    if show_legend:
        ax.legend(
            loc='upper left', bbox_to_anchor=(0, 0.92),
            frameon=True, handletextpad=0.5,
            borderpad=0.2, labelspacing=0.2, fontsize=8)

    for i, (correct, incorrect, total) \
    in enumerate(zip(correct_percent, incorrect_percent,
                     total_counts)):
        if total > 0:
            ax.text(i, (correct + incorrect) / 2,
                    f'{total:,.0f}', ha="center",
                    va="center", color="#808080",
                    fontsize=9, rotation=90)
            ax.text(i, correct + incorrect - 0.03,
                    f'{int(incorrect_percent[i]*100)}%',
                    ha="center", va="top", color="black",
                    fontsize=8)

    ax2 = ax.twinx()
    ax2.set_ylabel("Records Count (log scale)",
                   rotation=270, labelpad=15,
                   loc="center", color="#666666",
                   fontsize=8)
    ax2.set_yscale("log")
    max_y = max([x for x in total_counts if x > 0])
    log_max_y = np.log10(max_y)
    ax2.set_ylim(.9, 1_000_000_000_000)
    ax2.bar(grouped[xlabel], total_counts,
            color="#808080", alpha=0.3, width=0.2)
    yticks = [
        int(
            np.round(i*10**-(len(str(int(i)))-1))
            *10**(len(str(int(i)))-1)
        )
        for i in np.logspace(0, log_max_y, num=4)]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'{int(i):,}' 
                         for i in yticks],
                        color="#CCAD00", fontsize=8)
    ax2.tick_params(axis='y', colors="#737373")
    ax.margins(x=0)
    ax2.margins(x=0)


def plot_validation_completions(
    eval_metrics_df: pl.LazyFrame,
    engine: str = "gpu"
) -> Figure:
    """
    100% stacked bar plots of correct/incorrect
    tool-calls completions, per tool-calls count.

    Overlayed in grey (bottom third of each subplot)
    is total records count for each category.

    Params:
        - eval_metrics_df (pl.LazyFrame):
            The result of `compute_counts_n_metrics`.
            Mandatory attributes being :
                - jaccard (float)
                - ground_truth_tool_calls (int)
                - predicted_tool_calls (int)
        - engine (str):
            Polars' engine (cpu or gpu).

    Results:
        - (Figure)
    """

    grouped_gt = eval_metrics_df.with_columns(
        correct=(pl.col("jaccard") == 1).alias("correct")
    ).group_by("ground_truth_tool_calls").agg(
        pl.sum("correct").alias("correct_count"),
        pl.len().alias("total_count")
    ).with_columns(
        incorrect_count=(pl.col("total_count") -
                         pl.col("correct_count"))
    ).collect(engine=engine)

    grouped_pred = eval_metrics_df.with_columns(
        correct=(pl.col("jaccard") == 1).alias("correct")
    ).group_by("predicted_tool_calls").agg(
        pl.sum("correct").alias("correct_count"),
        pl.len().alias("total_count")
    ).with_columns(
        incorrect_count=(pl.col("total_count") -
                         pl.col("correct_count"))
    ).collect(engine=engine)

    max_x = max(max(grouped_gt["ground_truth_tool_calls"]),
                max(grouped_pred["predicted_tool_calls"]))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(6, 4), sharex=True,
        gridspec_kw={'hspace': 0.3})

    fig.suptitle(
        "Distribution of Completions on Validation Records",
        fontsize=12, fontweight="bold", y=0.97)
    _plot_bars(
        ax_top, grouped_gt, "ground_truth_tool_calls",
        "per Ground Truth count", max_x)
    _plot_bars(
        ax_bottom, grouped_pred, "predicted_tool_calls",
        "per Predicted count", max_x, show_legend=False)

    fig.tight_layout()
    plt.close(fig)

    return fig

