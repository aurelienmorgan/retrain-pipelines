
import sys
import json

import numpy as np

import polars as pl

from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


def _normalize_parameters(params: Dict) -> str:
    """
    Normalize parameters dictionary by sorting keys
    and their values.
    Returns a canonical string representation.
    """
    if not params:
        return "{}"

    sorted_params = dict(sorted(params.items()))

    return json.dumps(sorted_params, sort_keys=True)


def _normalize_tool(
    tool: Dict
) -> Dict:
    """
    Create a normalized version of a tool
    with consistent parameters ordering.
    """

    normalized = tool.copy()
    if "parameters" in normalized:
        if isinstance(normalized["parameters"], str):
            try:
                params = json.loads(normalized["parameters"])
            except json.JSONDecodeError:
                try:
                    params = literal_eval(normalized["parameters"])
                except (ValueError, SyntaxError) as ex:
                    print(ex, file=sys.stderr)
                    print(tool, file=sys.stderr)
                    params = {}
        else:
            params = normalized["parameters"]

        normalized["parameters"] = _normalize_parameters(params)
    return normalized


def _parse_tools(
    tools_list_str: str
) -> List[Dict]:
    """
    Parse the 'list of tools' string into a
    list of normalized tools.
    """
    try:
        tools_list = json.loads(tools_list_str)
        return [_normalize_tool(tool)
                for tool in tools_list]
    # except (ValueError, SyntaxError):
    except Exception as ex:
        print(tools_list_str, file=sys.stderr)
        print(ex, file=sys.stderr)
        return []


def _normalize_answer_tool(
    tool: Dict
) -> Dict:
    """
    Create a normalized version of a tool with
    consistent parameter ordering.
    """
    normalized = tool.copy()
    if "arguments" in normalized:
        # If parameters is already a list of string
        # (from previous processing), parse it first
        if (
            isinstance(normalized["arguments"], list)
            and all(isinstance(i, str)
            for i in normalized["arguments"])
        ):
            try:
                params = json.loads(normalized["arguments"])
            except json.JSONDecodeError:
                try:
                    params = literal_eval(normalized["arguments"])
                except (ValueError, SyntaxError) as ex:
                    print(ex, file=sys.stderr)
                    print(tool, file=sys.stderr)
                    params = []
        else:
            params = normalized["arguments"]

        normalized["arguments"] = sorted(params.keys())

    return normalized


def _parse_answer_tools(
    tools_list_str: str
) -> List[Dict]:
    """
    Parse the 'list of tools' string into a
    list of normalized tools.
    Mostly, we don't consider arguments values
    in tool calls, only their respective names.
    """
    try:
        tools_list = json.loads(tools_list_str)
        return [_normalize_answer_tool(tool)
                for tool in tools_list]
    except Exception as ex:
        print(tools_list_str, file=sys.stderr)
        print(ex, file=sys.stderr)
        return []
    return tools_list


def count_tool_occurrences(
    lazy_df: pl.lazyframe.frame.LazyFrame,
    column_name: str,
    struct_schema: pl.Struct
) -> pl.lazyframe.frame.LazyFrame:
    """
    Count occurrences of each unique tool in the DataFrame.

    Tools with same parameters in different orders
    are counted as the same tool.
    Returns a DataFrame with tool details and their count.

    Params:
        - lazy_df (pl.lazyframe.frame.LazyFrame):
        - column_name (str):
            the column to count tools in.
            Can for instance take "tools" or "answers".
        - struct_schema (pl.Struct):
            The schema tools are abiding to.
            Can be tool declaration
            (with description and params description)
            or tool calls (with argument values).

    Results:
        - (pl.lazyframe.frame.LazyFrame)
    """

    with_columns = []
    for field in struct_schema.fields:
        with_columns.append(
            pl.col("tool").struct.field(field.name) \
                .alias(f"tool_{field.name}"),
        )
    # last tool field ; either "parameters" or "arguments"
    param_arg = field.name

    if "parameters" == param_arg:
        tool_parsing_method = _parse_tools
    elif True:
        tool_parsing_method = _parse_answer_tools
    else:
        raise ValueError(
            f"Unexpected value for 'param_arg': {param_arg}")

    return (
        lazy_df
        .with_columns([
            pl.col(column_name)
            .map_elements(
                tool_parsing_method,
                return_dtype=pl.List(struct_schema))
            .alias("parsed_tools")
        ])
        .select(
            pl.col("parsed_tools").explode().alias("tool")
        )
        .filter(
            pl.col("tool").is_not_null()
        )
        .group_by("tool")
        .agg(
            pl.len().alias("occurrences")
        )
        .with_columns(with_columns)
        .drop("tool")
        .sort(["occurrences", "tool_name"],
              descending=[True, False])
    )


def plot_tools_occurences(
    tools_occurences_df: pl.dataframe.frame.DataFrame,
    head_tail_size: int = 200,
    title_prefix: str = ""
) -> Figure:
    """
    Plots distribution of tools within set of records.
    "tool 'A' is referenced 'x' times, etc".

    Params:
        - tools_occurences_df (pl.dataframe.frame.DataFrame):
            at least columns "tool_name", "occurrences"
            (the plotting will not use other columns,
             however DEVELOPPER is free to group rows
             by other keys (such as list of parameters, etc)
             at will)
        - head_tail_size (int):
            since population of tools often-times get very large,
            we only print most-frequently and least-frenquently
            referenced once (head and tail of the distributuon)
        - title_prefix (str):

    Results:
        - (Figure)
    """

    head = tools_occurences_df.head(head_tail_size)
    tail = tools_occurences_df.tail(head_tail_size)

    # Create x-coordinates with a gap
    # of 50 units between head and tail
    x_coords = np.arange(2*head_tail_size)
    x_coords[head_tail_size:] += 50

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x_coords[:head_tail_size],
                   head["occurrences"], color='skyblue',
                   label='most frequent')
    bars2 = ax.bar(x_coords[head_tail_size:],
                   tail["occurrences"], color='lightsalmon',
                   label='least frequent')

    # x-axis
    all_labels = pl.concat([head["tool_name"],
                            tail["tool_name"]])
    tick_positions = x_coords[::10]  # Every 10th position
    tick_labels = all_labels[::10]  # Every 10th label
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right',
                       fontsize=14)
    # Set limits to remove empty areas
    ax.set_xlim(x_coords[0] - 0.5, x_coords[-1] + 0.5)

    ax.set_yscale('log')
    ax.set_ylabel('Occurrences (log scale)', fontsize=14)
    def format_func(value, tick_number):
        return f'{value:,.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    ax.tick_params(axis='y', labelsize=14)

    ax.set_xlabel('Tool Name (every 10th)', fontsize=14)
    ax.set_title(
        title_prefix+
        f"Top {head_tail_size} and Bottom {head_tail_size} "+
        "Tool Occurrences",
        fontsize=18
    )

    # gap band
    ax.axvspan(head_tail_size, head_tail_size+50,
               facecolor='lightgray', alpha=0.5)
    gap_text = \
        f"{len(tools_occurences_df) - 2*head_tail_size:,.0f} " + \
        "entries not shown"
    ax.text(head_tail_size+25, 0.5, gap_text,
            ha='center', va='center', color='red',
            fontweight='normal', fontsize=14,
            rotation=90, transform=ax.get_xaxis_transform())

    ax.legend(fontsize=14)

    return fig


def polars_df_column_words_stats(
    polars_df: pl.dataframe.frame.DataFrame,
    column_name: str
) -> pl.dataframe.frame.DataFrame:
    """
    
    Params:
        - polars_df (pl.dataframe.frame.DataFrame
                     or pl.lazyframe.frame.LazyFrame)
        - column_name (str)
          name of the text column for which
          to count words

    Results:
        - (pl.dataframe.frame.DataFrame
           or pl.lazyframe.frame.LazyFrame)
          object of same type as input "polars_df"
          parameter, with columns :
            - "max",
            - "q1" (number of records
              in the first quantile of words count)
            - "q2" (number of records
              in the second quantile of words count)
            - "q3" (number of records
              in the third quantile of words count)
    """
    return polars_df.select(
        [
            pl.col(column_name)
            .str.extract_all(r"\w+")
            .map_elements(lambda arr: len(arr), return_dtype=pl.Int16)
            .max()
            .alias("max"),

            pl.col(column_name)
            .str.extract_all(r"\w+")
            .map_elements(lambda arr: len(arr), return_dtype=pl.Int16)
            .quantile(0.25)  # 1st quartile
            .cast(pl.Int16)
            .alias("q1"),

            pl.col(column_name)
            .str.extract_all(r"\w+")
            .map_elements(lambda arr: len(arr), return_dtype=pl.Int16)
            .quantile(0.5)  # 2nd quartile
            .cast(pl.Int16)
            .alias("q2"),

            pl.col(column_name)
            .str.extract_all(r"\w+")
            .map_elements(lambda arr: len(arr), return_dtype=pl.Int16)
            .quantile(0.75)  # 3rd quartile
            .cast(pl.Int16)
            .alias("q3")
        ]
    )


def plot_words_count(
    lazy_df: pl.lazyframe.frame.LazyFrame,
    column_name: str,
    engine: str = "cpu"
) -> Figure:
    """

    Params:
        - lazy_df (pl.lazyframe.frame.LazyFrame):
        - column_name (str)
          name of the text column for which
          to count words
        - engine (str):
            Polars' engine (cpu or gpu)

    Results:
        - (Figure)
    """

    word_count_df = lazy_df.select(
        pl.col(column_name)
        .str.extract_all(r"\w+")  # Extract all words
        .map_elements(lambda arr: len(arr), return_dtype=pl.Int16)
        .alias("word_count")
    ).collect(engine=engine)
    word_counts = word_count_df["word_count"].to_list()
    word_counts.sort(reverse=True)


    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [2, 1]})
    bin_ranges = [(i, i + 5) for i in range(0, max(word_counts), 5)]
    binned_counts = [0] * len(bin_ranges)
    for count in word_counts:
        for i, (lower, upper) in enumerate(bin_ranges):
            if lower < count <= upper:
                binned_counts[i] += 1
                break
    ax1.barh(range(len(binned_counts)), binned_counts,
             color="skyblue", edgecolor="black")
    ax1.set_yticks(range(len(binned_counts)))
    ax1.set_yticklabels([f"{lower+1}-{upper}"
                         for lower, upper in bin_ranges], ha='right')
    ax1.set_xlabel("Number of Records in Range (log scale)",
                   fontsize=12)
    ax1.set_ylabel("Words Count Range", fontsize=12)
    ax1.set_xscale('log')
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}"))
    ax1.grid(axis="x", linestyle="--", alpha=0.7)

    box = ax2.boxplot(word_counts, vert=True)
    for median in box['medians']:
        median_val = median.get_ydata()[0]
        ax2.text(median.get_xdata()[0], median_val,
                 f'{int(median_val)}', horizontalalignment='center',
                 verticalalignment='bottom')
    for cap in box['caps']:
        cap_val = cap.get_ydata()[0]
        ax2.text(cap.get_xdata()[0], cap_val, f'{int(cap_val)}',
                 horizontalalignment='center',
                 verticalalignment='bottom')
    q1 = box['whiskers'][0].get_ydata()[0]
    q3 = box['whiskers'][1].get_ydata()[0]
    ax2.text(box['whiskers'][0].get_xdata()[0], q1, f'{int(q1)}',
             horizontalalignment='right', verticalalignment='bottom')
    ax2.text(box['whiskers'][1].get_xdata()[0], q3, f'{int(q3)}',
             horizontalalignment='left', verticalalignment='top')

    ax2.set_ylabel('Word Count')
    ax2.set_xticks([])
    ax2.set_xlabel('')
    ax2.grid(True)

    fig.suptitle("Queries Words Count Distribution", fontsize=16)

    return fig
