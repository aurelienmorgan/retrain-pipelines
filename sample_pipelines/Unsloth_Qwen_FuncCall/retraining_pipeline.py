
from unsloth import FastLanguageModel, \
    is_bfloat16_supported, UnslothTrainer, \
    UnslothTrainingArguments

import torch

import os
import sys

import gc
import json
import time
import shutil
import logging
import traceback
import subprocess
import importlib.util
from enum import Enum
from io import StringIO
from textwrap import dedent
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import polars as pl
from polars.exceptions import ComputeError

import matplotlib
import matplotlib.pyplot as plt

from jinja2 import Environment, FileSystemLoader

from metaflow import FlowSpec, step, Parameter, JSONType, \
    IncludeFile, current, metaflow_config as mf_config, \
    resources, Flow, Task, card
from metaflow.current import Current
from metaflow.cards import Image, Table, Markdown, \
    Artifact, get_cards

from datasets import load_dataset, Dataset, DatasetDict
from datasets.config import HF_DATASETS_CACHE, HF_CACHE_HOME
from huggingface_hub import list_repo_commits
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from retrain_pipelines import __version__
from retrain_pipelines.dataset.hf_utils import get_lazy_df, \
    get_column_info, iterable_dataset_multi_buffer_sampler, \
    push_dataset_version_to_hub
from retrain_pipelines.dataset.tool_calls import \
    get_unique_tools, count_tool_occurrences, \
    plot_tools_occurences, column_words_stats, \
    plot_words_count
from retrain_pipelines.utils.hf_utils import \
    get_new_repo_minor_version, push_files_to_hub_repo_branch
from retrain_pipelines.utils import create_requirements

 
class LocalServeReadinessEnum(Enum):
    """
    tracking local-serve (infra-validation)
    status using a "3+"-states enum :
        - "-1" for "not applicable"
          (i.e. "model version not blessed"),
        - "0/1" bool for failure/success.
    """
    NOT_APPLICABLE = -1
    FAILURE = 0
    FAILURE_NO_DOCKER = 2
    SUCCESS = 1


class UnslothFuncCallFlow(FlowSpec):
    """
    Training pipeline
    """
    # @see https://github.com/unslothai/unsloth/wiki

    #--- flow parameters -------------------------------------------------------

    RETRAIN_PIPELINE_TYPE = "mf_unsloth_func_call_litserve"
    # in order to share the config across subprocesses
    os.environ["retrain_pipeline_type"] = RETRAIN_PIPELINE_TYPE

    hf_dataset = Parameter(
        "hf_dataset",
        help="dict with 'repo_id' and 'commit_hash' keys. " + \
             "if 'commit_hash is None, falls back to latest version " +\
             "of the dataset available in parquet format.\n" +
             "Note that there are 3 required 'attributes' of type " + \
             "str, list[str], list[str]",
        type=JSONType,
        default=dedent("""{
            "repo_id": "Salesforce/xlam-function-calling-60k",
            "config_name": "",
            "commit_hash": "",
            "attributes": {
                "query_attr": "query",
                "answers_attr": "answers",
                "tools_attr": "tools"
            }
        }""").replace("'", '"').strip('"')
    )

    augmentation_rate = Parameter(
        "augmentation_rate",
        type=float,
        default=.05,
        help="proportion of records to be augmented "+\
             "(x% of original dataset is created"+\
             " as additional augmented datapoints), i.e. "+\
             "truncated queries to serve as negative examples, "+\
             "meaning they trigger no tool call "+\
             "due to info incompleteness."
    )

    hf_enrich_dataset = Parameter(
        "hf_enrich_dataset",
        help="dict with 'repo_id', 'config_name' and 'commit_hash', "+\
             "query_attribute' and 'query_attribute_handler' keys. "+\
             "if 'commit_hash is None, falls back to latest version "+\
             "of the dataset available in parquet format."+\
             "'query_attribute' depicts the dataset attribute "+\
             "from which 'queries' are to be sampled."+\
             "'query_attribute_handler' serves for attributes "+\
             "that have complex structure, "+\
             "other than 'string' datatype.",
        type=JSONType,
        # @see https://huggingface.co/datasets/google-research-datasets/natural_questions
        default=dedent("""{
            "repo_id": "lighteval/natural_questions_clean",
            "config_name": "",
            "commit_hash": "",
            "query_attribute": "question",
            "query_attribute_handler": "lambda x: x"
        }""").replace("'", '"').strip('"')
    )

    enrichment_rate = Parameter(
        "enrichment_rate",
        type=float,
        default=.1,
        help="proportion of records "+\
             "to be added from the 'hf_enrich_dataset'"+\
             "(x% of original dataset is sampled and"+\
             " added as enriching datapoints), i.e. "+\
             "queries to serve as negative examples, "+\
             "due to their complete disconnexion "+\
             "to tool calling situations."
    )

    dataset_repo_id = Parameter(
        "dataset_repo_id",
        type=str,
        default="retrain-pipelines/func_calls",
        help="The 'repo_id' to be used " + \
             "for the Hugging Face dataset version push " + \
             "(will be created at runtime" + \
             " if doesn't already exist)."
    )

    hf_base_model = Parameter(
        "hf_base_model",
        help="dict with 'repo_id' and 'commit_hash' keys."+\
             "if 'commit_hash is None, falls back "+\
             "to latest available version of the model.",
        type=JSONType,
        default=dedent("""{
            "repo_id": "unsloth/Qwen2.5-1.5B",
            "commit_hash": ""
        }""").replace("'", '"').strip('"')
    )

    cpt_training_args = Parameter(
        "cpt_training_args",
        help="dict with `TrainingArguments` params "+\
             "for the CPT job.",
        type=JSONType,
        default=dedent("""{
            "warmup_ratio": 0.1,
            "num_train_epochs": 1
        }""").replace("'", '"').strip('"')
    )

    sft_training_args = Parameter(
        "sft_training_args",
        help="dict with `TrainingArguments` params "+\
             "for the SFT job.",
        type=JSONType,
        default=dedent("""{
            "warmup_ratio": 0.1,
            "num_train_epochs": 1
        }""").replace("'", '"').strip('"')
    )

    model_repo_id = Parameter(
        "model_repo_id",
        type=str,
        default="retrain-pipelines/function_caller",
        help="The 'repo_id' to be used " + \
             "for the Hugging Face model version push " + \
             "(will be created at runtime" + \
             " if doesn't already exist)."
    )

    default_pipeline_card_module_dir = \
        os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{RETRAIN_PIPELINE_TYPE}"
            ).origin)
    pipeline_card_artifacts_path = Parameter(
        "pipeline_card_artifacts_path",
        type=str,
        default=default_pipeline_card_module_dir,
        help="pipeline_card artifacts location "+\
             "(i.e. dir hosting your optional " + \
             " custom documentation files :" + \
             " 'pipeline_card.py' and/or 'template.html'"+\
             " and/or 'model_readme.py'"+\
             " and/or 'model_readme_template.md'," +\
             " and/or 'dataset_readme.py'"+\
             " and/or 'dataset_readme_template.md' file), " +\
             "if different from default."
    )
    @staticmethod
    def copy_default_dataset_readme_module(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir, "dataset_readme.py"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    UnslothFuncCallFlow.default_pipeline_card_module_dir,
                    "dataset_readme.py"
                )
            shutil.copy(filefullname, target_dir)
            print(filefullname)
    @staticmethod
    def copy_default_dataset_readme_template(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir,
                                        "dataset_readme_template.md"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    UnslothFuncCallFlow.default_pipeline_card_module_dir,
                    "dataset_readme_template.md")
            shutil.copy(filefullname, target_dir)
            print(filefullname)
    @staticmethod
    def copy_default_model_readme_module(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir, "model_readme.py"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    UnslothFuncCallFlow.default_pipeline_card_module_dir,
                    "model_readme.py"
                )
            shutil.copy(filefullname, target_dir)
            print(filefullname)
    @staticmethod
    def copy_default_model_readme_template(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir,
                                        "model_readme_template.md"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    UnslothFuncCallFlow.default_pipeline_card_module_dir,
                    "model_readme_template.md")
            shutil.copy(filefullname, target_dir)
            print(filefullname)
    @staticmethod
    def copy_default_pipeline_card_module(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir, "pipeline_card.py"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    UnslothFuncCallFlow.default_pipeline_card_module_dir,
                    "pipeline_card.py"
                )
            shutil.copy(filefullname, target_dir)
            print(filefullname)
    @staticmethod
    def copy_default_pipeline_card_html_template(
        target_dir: str,
        exists_ok: bool = False
    ) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if (
            not exists_ok and
            os.path.exists(os.path.join(target_dir, "template.html"))
        ):
            print("File already exists. Skipping copy.")
        else:
            filefullname = os.path.join(
                    UnslothFuncCallFlow.default_pipeline_card_module_dir,
                    "template.html")
            shutil.copy(filefullname, target_dir)
            print(filefullname)

    del RETRAIN_PIPELINE_TYPE

    #---------------------------------------------------------------------------

    @step
    def start(self):
        print(f"{current.flow_name} - {current.run_id}")

        # GPU availability
        print(torch.cuda.get_device_name(0))
        print(torch.__version__)
        self.engine = "gpu" if torch.cuda.is_available() else "cpu"

        # hf_dataset
        hf_dataset_dict = \
            get_lazy_df(
                repo_id=self.hf_dataset["repo_id"],
                commit_hash=self.hf_dataset["commit_hash"],
                files_filter=(
                    self.hf_dataset['config_name']+"/.*\\.parquet"
                    if (
                        self.hf_dataset["config_name"] and
                        "" < self.hf_dataset["config_name"]
                    ) else ".*\\.parquet"
                ),
                hf_token=os.getenv("HF_TOKEN", None)
            )
        try:
            print(hf_dataset_dict["repo_id"], ", ",
                  hf_dataset_dict["commit_hash"], "  -  ",
                  hf_dataset_dict["commit_datetime"], "\n",
                  hf_dataset_dict["lazy_df"].explain())
        except ComputeError as ex:
            if "HF_TOKEN" not in os.environ:
                print("Does the Hugging Face-hosted dataset " +
                      "require authentication ?",
                      file=sys.stderr, flush=True)
            raise ex
        self.hf_dataset_dict = hf_dataset_dict

        # hf_enrich_dataset
        print(self.hf_enrich_dataset)
        hf_enrich_dataset_dict = \
            get_lazy_df(
                repo_id=self.hf_enrich_dataset["repo_id"],
                commit_hash=self.hf_enrich_dataset["commit_hash"],
                files_filter=(
                    self.hf_enrich_dataset['config_name']+"/.*\\.parquet"
                    if (
                        self.hf_enrich_dataset["config_name"] and
                        "" < self.hf_enrich_dataset["config_name"]
                    ) else ".*\\.parquet"
                ),
                hf_token=os.getenv("HF_TOKEN", None)
            )
        print(' ; '.join(f"{k}: {hf_enrich_dataset_dict[k]}"
                                 for k in ['commit_hash',
                                           'commit_datetime']))
        self.hf_enrich_dataset_dict = hf_enrich_dataset_dict

        # hf_base_model
        hf_base_model_commits = list_repo_commits(
            repo_id=self.hf_base_model["repo_id"],
            revision=(
                None if (rev_commit_hash:=self.hf_base_model["commit_hash"]) == ""
                else rev_commit_hash
            ),
            repo_type="model",
            token=os.getenv("HF_TOKEN", None))
        self.hf_base_model_dict = {
            "repo_id": self.hf_base_model["repo_id"],
            "commit_hash": hf_base_model_commits[0].commit_id,
            "commit_datetime": \
                hf_base_model_commits[0].created_at
        }

        self.model_version_blessed = False
        self.current_blessed_run = None
        self.current_blessed_version_dict = None
        current.run.remove_tag("model_version_blessed")

        self.retrain_pipelines = f"retrain-pipelines {__version__}"
        self.retrain_pipeline_type = os.environ["retrain_pipeline_type"]

        self.serving_artifacts_local_folder = \
            os.path.realpath(os.path.join(
                os.path.dirname(__file__),
                '..', '..', 'serving_artifacts',
                os.path.sep.join(current.run.path_components)
        ))

        if not os.path.exists(self.serving_artifacts_local_folder):
            os.makedirs(self.serving_artifacts_local_folder)

        self.unsloth_dir = os.path.join(
            self.serving_artifacts_local_folder,
            "Unsloth"
        )
        print(f"unsloth_dir : {self.unsloth_dir}")
        self.cpt_model_dir = os.path.join(
                self.unsloth_dir, "cpt_model")
        self.sft_model_dir = os.path.join(
                self.unsloth_dir, "sft_model")

        self.next(self.eda)


    @step
    def eda(self):
        """
        exploratory data analysis.
        """

        ############################
        #    features and label    #
        #       basic counts       #
        ############################
        self.records_count = self.hf_dataset_dict["lazy_df"] \
            .select(pl.len()).collect(engine=self.engine).item()
        self.data_schema = get_column_info(
            self.hf_dataset_dict["lazy_df"], engine=self.engine)
        ############################

        ############################
        #          Answers         #
        #        tools count       #
        ############################
        struct_schema = pl.Struct([
            pl.Field("name",
                     pl.String
                    ),
            pl.Field("arguments",
                     pl.List(pl.String)  # we retrieve list of args names
                                         # (without assigned values)
                    )
        ])
        tool_answer_occurrences_df = \
            count_tool_occurrences(
                self.hf_dataset_dict["lazy_df"],
                self.hf_dataset["attributes"]["answers_attr"],
                struct_schema) \
            .collect(engine=self.engine)
        print(f"{tool_answer_occurrences_df['occurrences'].sum():,} " +
              f"query/tool-calls pairs")
        fig = plot_tools_occurences(tool_answer_occurrences_df,
                                    title_prefix="Dataset answers - ")
        self.answers_tools_count_fig = fig
        ############################

        ############################
        #           Query          #
        #        words count       #
        ############################
        queries_max_length = self.hf_dataset_dict["lazy_df"].select(
            pl.col(
                self.hf_dataset["attributes"]["query_attr"]
            ).str.len_chars().max().alias("max_query_length")
        ).collect(engine=self.engine)
        print(f"longuest query counts " +
              f"{queries_max_length['max_query_length'][0]:,} characters")

        # queries length quartiles
        self.query_words_stats = \
            column_words_stats(
                self.hf_dataset_dict["lazy_df"],
                self.hf_dataset["attributes"]["query_attr"]
            ).collect(engine=self.engine)
        print(self.query_words_stats.to_pandas().to_string(index=False))
        print("Two thirds of the records have a query with less than " +
              f"{self.query_words_stats['q3'][0]} words.")

        fig = plot_words_count(
                self.hf_dataset_dict["lazy_df"],
                column_name=self.hf_dataset["attributes"]["query_attr"],
                engine=self.engine)
        self.words_count_fig = fig
        ############################

        ############################
        #     hf_enrich_dataset    #
        #    Query words count     #
        ############################
        enrich_question_words_stats = \
            column_words_stats(
                self.hf_enrich_dataset_dict['lazy_df'],
                self.hf_enrich_dataset["query_attribute"],
                column_attr_handler=eval(
                    self.hf_enrich_dataset["query_attribute_handler"])
            ).collect(engine=self.engine)
        print(enrich_question_words_stats.to_pandas()
                .to_string(index=False))
        del enrich_question_words_stats
        ############################

        self.next(self.augment_data)


    @step
    def augment_data(self):
        """
        Add 'negative' examples, where
        queries do not trigger any tool call.
        To achieve that, we sample long user queries,
        truncate at half words count, and
        associate this to an empty list of tool-calls.
        """
        """
        We only consider :
          - records with longuest queries,
            i.e. queries in the last quartile
            of "queries with most word-counts"
            (this is to avoid that 'truncated' queries
             get really short)
          - records with answers consisting
            in a single tool-call
            (in order to minimize the risk
             that truncating actually gives
             a valid answer with
             one tool-call [or more])

        Note on flow 'augmentation_rate' :
            we add that many records (at most),
            as quartiles size permits.
        """

        print("Sampling within the population with more than " +
              str(self.query_words_stats['q3'][0]) +
              " words (longest queries quartile) =>")

        samples_count = \
            int(self.records_count * self.augmentation_rate)
        print(f"would represent {samples_count:,.0f} " +
              f"records to be sampled")

        eligible_records_df = \
            self.hf_dataset_dict["lazy_df"].filter(
                pl.col(
                    self.hf_dataset["attributes"]["query_attr"]
                )
                .str.extract_all(r"\w+")
                .map_elements(
                    lambda arr: len(arr),
                    return_dtype=pl.Int16)
                .gt(self.query_words_stats['q3'][0])
                & pl.col("answers")
                .map_elements(
                    lambda x: len(json.loads(x)) == 1
                              if isinstance(x, str)
                              else False,
                    return_dtype=pl.Boolean)  
            ) \
            .collect(engine=self.engine)
        eligible_records_count = \
            eligible_records_df.select(pl.len())["len"][0]
        print(f"eligible_records_count : " +
              f"{eligible_records_count:,.0f}")
        samples_count = min(samples_count, eligible_records_count)
        self.actual_augmentation_rate = \
            samples_count / self.records_count
        print("actual augmentation rate : " +
              f"{self.actual_augmentation_rate:.1%}")
        sampled_records_df = eligible_records_df.sample(
            n=samples_count
        )

        self.augmented_records_df = \
            sampled_records_df.with_columns(
                pl.col("query")
                .map_elements(
                    lambda query:
                        " ".join(
                            query.split()[
                                :len(query.split()) // 2]),
                    return_dtype=pl.Utf8)
                .alias("truncated_query")
            ).select([
                pl.col("truncated_query").alias("query"),
                pl.lit("[]").alias("answers")
            ])
        print(self.augmented_records_df.height,
              self.augmented_records_df.columns)

        self.next(self.enrich_data)


    @step
    def enrich_data(self):
        """
        Further enrich our dataset with 'negative' records from
        another dataset (can be general-purpose text dataset)
        as specified by the the flow 'hf_enrich_dataset' argument.
        """
        """
        Note : we here use the Hugging Face `datasets` library
        in 'streaming' mode for records sampling.
        """

        hf_enrich_ds = load_dataset(
            path=self.hf_enrich_dataset["repo_id"],
            name=self.hf_enrich_dataset["config_name"],
            revision=self.hf_enrich_dataset_dict["commit_hash"],
            streaming=True)
        print(hf_enrich_ds["train"])

        samples_count = \
            int(self.records_count * self.enrichment_rate)
        print(f"Samplig {samples_count:,.0f} records")

        query_attribute_handler = \
            eval(self.hf_enrich_dataset["query_attribute_handler"])
        samples_iterator = iterable_dataset_multi_buffer_sampler(
                hf_enrich_ds["train"],
                total_samples=samples_count,
                attributes_selector=\
                    (lambda x:query_attribute_handler(
                        x[self.hf_enrich_dataset["query_attribute"]])),
                buffer_size=3_000,
                num_passes=3,
                seed=None
            )
        # Capitalize and add end punctuation if missing
        start_time = time.time()
        print("Starting sample enriching records, " +
              "this may take some time if the source dataset " +
              "has a complex structure..")
        samples_list = [
            s.capitalize() + ("" if s[-1] in ".!?" else "?")
            for s in samples_iterator]
        elapsed_time = time.time() - start_time
        print(f".. sampling completed " +
              f"({int(elapsed_time // 3_600)}h:" +
               f"{int((elapsed_time % 3_600) // 60)}m:" +
               f"{int(elapsed_time % 60)}s).")
        enriched_records_df = pl.DataFrame(
                {"query": samples_list,
                 "answers": \
                     ["[]"] * \
                     len(samples_list)}
            )
        self.enriched_records_df = enriched_records_df

        self.next(self.dataset_to_hub)


    @step
    def dataset_to_hub(self):
        """
        Push to hub dataset version
        - continued pre-training dataset
        - training and validation splits of the
        augmented and enriched
        supervised finetuning dataset
        - readme with versioning info
        """

        #############################
        #  case of user-provided    #
        # documentation artifact(s) #
        #############################
        # note that user can provide either
        # 'pipeline_card.py' or 'template.html'
        # or 'dataset_readme.py'
        # or 'dataset_readme_template.md'
        # or 'model_readme.py'
        # or 'model_readme_template.md'
        # or any combination of those
        # when specifying custom
        # 'pipeline_card_artifacts_path'
        if (
            "dataset_readme_template.md" in
                os.listdir(self.pipeline_card_artifacts_path)
        ):
            template_dir = self.pipeline_card_artifacts_path
        else:
            template_dir = os.path.dirname(
                importlib.util.find_spec(
                    f"retrain_pipelines.pipeline_card."+
                    f"{os.getenv('retrain_pipeline_type')}"
                ).origin)
        print(f"template_dir : '{template_dir}'")
        #############################
        if "dataset_readme.py" in os.listdir(
                self.pipeline_card_artifacts_path):
            from retrain_pipelines.utils import \
                get_get_dataset_readme_content
            get_dataset_readme_content = \
                get_get_dataset_readme_content(
                    self.pipeline_card_artifacts_path)
        else:
            from retrain_pipelines.pipeline_card import \
                    get_dataset_readme_content
        #############################
    

        #############################
        #    augmented & enriched   #
        #     finetuning dataset    #
        #############################
        merged_df = pl.concat([
                # dataset
                self.hf_dataset_dict["lazy_df"].select([
                        self.hf_dataset["attributes"]["query_attr"],
                        self.hf_dataset["attributes"]["answers_attr"]
                    ]).collect(engine=self.engine),
                # truncated queries augmentation
                self.augmented_records_df,
                # enriching dataset
                self.enriched_records_df
            ]).sample(
                # shuffling
                fraction=1,
                shuffle=True,
                with_replacement=False
            )
        merged_df = merged_df.sample(fraction=1, shuffle=True)
        merged_df.rechunk()
        print(("merged_df", f"{merged_df.shape[0]:,.0F}",
              merged_df.columns))

        pandas_df = merged_df.to_pandas()
        train_size = int(0.8 * len(pandas_df))
        print(f"validation : {len(pandas_df) - train_size}")
        sft_dataset = DatasetDict({
            "train": Dataset.from_pandas(pandas_df[:train_size]),
            "validation": Dataset.from_pandas(pandas_df[train_size:])
        })
        #############################

        #############################
        #   continued pre-training  #
        #          dataset          #
        #############################
        struct_schema = pl.Struct([
            pl.Field("name", pl.String),
            pl.Field("description", pl.String),
            pl.Field(
                "parameters",
                pl.String  # Use String to allow 
                           # for varying structures
                           # (different tools indeed having
                           #  different sets of parameters
                           #  i.e. different parameters counts,
                           #  datatypes and names)
                           # so parsing must be tolerant.
            )
        ])
        unique_tools_df = get_unique_tools(
                self.hf_dataset_dict["lazy_df"],
                tools_attr_name=\
                    self.hf_dataset["attributes"]["tools_attr"],
                struct_schema=struct_schema
            ).collect(engine=self.engine)
        unique_tools_arrow_table = unique_tools_df.to_arrow()
        self.unique_tools_dataset = \
            Dataset(unique_tools_arrow_table)
        print(self.unique_tools_dataset)
        #############################

        #############################
        #        DatasetDict        #
        #    with multiple tables   #
        #############################
        dataset_dict = DatasetDict({
            "continued_pre_training": \
                self.unique_tools_dataset,
            "supervised_finetuning": sft_dataset
        })
        print(dataset_dict, flush=True)
        #############################

        #############################
        #       dataset README      #
        #       from template       #
        #############################
        commit_datetime = datetime.utcnow()
        new_dataset_version_label = get_new_repo_minor_version(
            repo_id=self.dataset_repo_id,
            repo_type="dataset",
            hf_token=os.getenv("HF_TOKEN", None))
        readme_content = get_dataset_readme_content(
            template_folder=template_dir,

            hf_dataset_dict=self.hf_dataset_dict,
            hf_enrich_dataset_dict=self.hf_enrich_dataset_dict,
            dataset_dict=dataset_dict,

            augmentation_rate=self.actual_augmentation_rate,
            enrichment_rate=self.enrichment_rate,

            version_label=new_dataset_version_label,
            commit_datetime=commit_datetime,

            mf_flow_name=current.flow_name,
            mf_run_id=current.run.id,
            engine=self.engine
        )
        #############################

        dataset_commit_hash = push_dataset_version_to_hub(
            repo_id=self.dataset_repo_id,
            version_label=new_dataset_version_label,
            timestamp_str=commit_datetime.strftime(
                "%Y-%m-%d %H:%M:%S UTC"),
            dataset_dict=dataset_dict,
            dataset_readme_content=readme_content,
            hf_token=os.getenv("HF_TOKEN", None)
        )
        if not dataset_commit_hash:
            raise Exception(
                "Failed to publish dataset version.")
        print(f"https://huggingface.co/datasets/{self.dataset_repo_id}" +
              f"/blob/{dataset_commit_hash}/README.md")
        self.dataset_commit_dict = {
            "repo_id": self.dataset_repo_id,
            "commit_hash": dataset_commit_hash,
            "version_label": new_dataset_version_label,
            "commit_datetime": commit_datetime,
        }

        self.next(self.continued_pre_training)


    @step
    def continued_pre_training(self):
        """
        Gives the base model some additional intrinsic knowkledge
        through continued pre-training.
        See unsloth.ai/blog/contpretraining
        """
        from retrain_pipelines.model.hf_utils import \
            plot_log_history

        #######################################
        # base-model and associated tokenizer #
        #      from Hub (or local cache)      #
        #######################################
        self.max_seq_length = 2048
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.hf_base_model_dict["repo_id"],
            revision=self.hf_base_model_dict["commit_hash"],
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            # case of a gated or private base-model
            token=os.getenv("HF_TOKEN", None)
        )
        #######################################

        #######################################
        #   dataset prompt_template mapping   #
        #######################################
        tools_dataset = DatasetDict(
            {"train": self.unique_tools_dataset})
        print(tools_dataset)
        tool_prompt_template = "tool: {}"
        def formatting_prompts_func(tools_batch):
            tools_batch = tools_batch["tool"]
            outputs = []
            for tool in tools_batch:
                # Must add EOS_TOKEN,
                # otherwise generation will go on forever!
                text = tool_prompt_template.format(tool) + \
                       tokenizer.eos_token
                outputs.append(text)
            return { "tools" : outputs, }
        cpt_dataset = tools_dataset["train"].map(
            formatting_prompts_func, batched=True,)
        #######################################

        #######################################
        #             PEFT adapter            #
        #      for continued pre-training     #
        #######################################
        model = FastLanguageModel.get_peft_model(
            model,
            r = 128, # any number >0 ; 8, 16, 32, 64, 128, 256
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",
                              # Add for continued pretraining
                              "embed_tokens", "lm_head",],
            lora_alpha = 32,
            lora_dropout = 0,    # Supports any, 0 is optimized
            bias = "none",       # Supports any, "none" is optimized
            # True or "unsloth" for very long context
            use_gradient_checkpointing = "unsloth",
            use_rslora = True,   # rank-stabilized LoRA
            loftq_config = None, # LoftQ
            #random_state = 3407,
        )
        #######################################

        #######################################
        #             cpt_trainer             #
        #######################################
        if (
            "records_cap" in self.cpt_training_args and
            self.cpt_training_args["records_cap"] is not None and
            isinstance(self.cpt_training_args["records_cap"], int)
        ):
            cpt_dataset = cpt_dataset.take(
                self.cpt_training_args["records_cap"])
            print(f"cpt_dataset : {cpt_dataset}")

        train_args = UnslothTrainingArguments(
            # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_strategy
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,

            **{k: v for k, v in self.cpt_training_args.items()
                    if k != "records_cap"},

            # 2 to 10x smaller learning rate
            # for the embedding matrices
            learning_rate=5e-5,
            embedding_learning_rate=1e-5,

            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            #seed=3407,

            output_dir=os.path.join(
                self.unsloth_dir, "outputs", "cpt"),
            save_total_limit = 2,

            report_to="tensorboard",
            logging_dir=os.path.join(
                self.sft_model_dir,
                "runs", "cpt")
        )

        self.cpt_traces_file_fullname = os.path.join(
            self.unsloth_dir, "cpt_trainer_traces.txt")
        print("Training started. " +
              f"Check {self.cpt_traces_file_fullname} for live traces.",
              flush=True)

        trainer = UnslothTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=cpt_dataset,
            dataset_text_field="tools",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            args=train_args,
        )
        #######################################

        #######################################
        #      Show current memory stats      #
        #######################################
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        _ = gc.collect()

        gpu_stats = torch.cuda.get_device_properties(0)
        self.start_gpu_memory = \
            round(torch.cuda.max_memory_reserved()
                  / 1024 / 1024 / 1024, 3)
        self.max_memory = \
            round(gpu_stats.total_memory
                  / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. " +
              f"Max memory = {self.max_memory} GB.")
        print(f"{self.start_gpu_memory} GB of memory reserved.")
        #######################################

        with open(self.cpt_traces_file_fullname, 'w') as f:
            with redirect_stdout(f):
                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
                trainer_stats = trainer.train()
        hf_logging.set_verbosity_info()
        hf_logging.enable_progress_bar()
        print(f"{trainer_stats.metrics['train_runtime']} " +
              f"seconds used for training " +
              f"({round(trainer_stats.metrics['train_runtime']/60, 2)}" +
              f" minutes).")

        self.cpt_log_history = trainer.state.log_history
        # print(self.cpt_log_history)
        self.cpt_log_history_fig = \
            plot_log_history(
                self.cpt_log_history,
                title="Continued pretraining loss"
            )

        model.save_pretrained_merged(
            save_directory=self.cpt_model_dir,
            tokenizer=tokenizer,
            save_method="lora"
        )
        print(f"cpt_model_dir : {self.cpt_model_dir}\n")

        self.next(self.supervised_finetuning)


    @step
    def supervised_finetuning(self):
        """
        Trains the model on tool-calling
        task specialization.
        """
        from retrain_pipelines.model.hf_utils import \
            plot_log_history

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        _ = gc.collect()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cpt_model_dir,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        # !!!! bug fix BEGIN !!!!
        # otherwise, 'embed_tokens' and 'lm_head'
        # trained during CPT are "ignored",
        # i.e. not saved after SFT
        # (note that, alternatively, we could also 
        #  do this fix after sft-training and
        #  just before saving ;
        #  which would be equivalent to
        #  freezing embeddings during finetuning
        #  for better pretrained knowledge retention)
        # @see https://www.reddit.com/r/unsloth/comments/1dtzcd6/fastlanguagemodelpatch_peft_model_changing/
        model.model.model.embed_tokens.modules_to_save.default.to(
            device="cuda:0",
            dtype=torch.float32,
            non_blocking=True)
        model.model.model.embed_tokens.modules_to_save.default \
            .requires_grad_(True)
        model.model.lm_head.modules_to_save.default.to(
            device="cuda:0",
            dtype=torch.float32,
            non_blocking=True)
        model.model.lm_head.modules_to_save.default \
            .requires_grad_(True)
        # !!!! bug fix END !!!!

        #######################################
        #   dataset prompt_template mapping   #
        #######################################
        # download from Hub (or get from local cache)
        queries_dataset = load_dataset(
            path=self.dataset_commit_dict["repo_id"],
            name="supervised_finetuning",
            revision=self.dataset_commit_dict["commit_hash"],
            token=os.getenv("HF_TOKEN", None))
        print(f"HF_DATASETS_CACHE : {HF_DATASETS_CACHE}") # HF_CACHE_HOME
        self.sft_prompt_template = dedent("""
        You specialize in generating tool calls. Given a query, your task is to return a list of tool calls based on your knowledge of known tools.

        Rules:
        1. You can only use tools you know. Do not create new tools under any circumstances.
        2. If a query does not match any known tool, return an empty list ([]).
        3. If information is missing to use a known tool, do not attempt to use it.
        4. Your response must always be a valid JSON array, and nothing else.

        Be precise and do not guess.

        # query:
            {}
        # response:
            {}
        """).strip()
        tokenizer.chat_template = self.sft_prompt_template

        EOS_TOKEN = tokenizer.eos_token
        def formatting_prompts_func(records):
            query = records["query"]
            tools  = records["answers"]
            outputs = []
            for query, tools in zip(query, tools):
                # Must add EOS_TOKEN,
                # otherwise your generation will go on forever
                text = self.sft_prompt_template.format(query, tools) \
                       + EOS_TOKEN
                outputs.append(text)
            return { "text" : outputs, }
        sft_train_dataset = queries_dataset["train"].map(
            formatting_prompts_func, batched=True)
        sft_valid_dataset = queries_dataset["validation"].map(
            formatting_prompts_func, batched=True,)
        #######################################

        #######################################
        #             PEFT adapter            #
        #      for supervised finetuning      #
        #######################################
        # for cases where CPT has been merged into overall model
        # otherwize, keep on training current LoRa adapter
        # model = FastLanguageModel.get_peft_model(
            # model,
            # r = 128, # any number >0 ; 8, 16, 32, 64, 128, 256
            # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              # "gate_proj", "up_proj", "down_proj"],
            # lora_alpha = 32,
            # lora_dropout = 0, # Supports any, but = 0 is optimized
            # bias = "none",    # Supports any, but = "none" is optimized
            # # True or "unsloth" for very long context
            # use_gradient_checkpointing = "unsloth",
            # random_state = 3407,
            # use_rslora = True,   # rank stabilized LoRA
            # loftq_config = None, # LoftQ
        # )
        #######################################

        #######################################
        #             sft_trainer             #
        #######################################
        split = sft_train_dataset.train_test_split(
            test_size=1000,
            #seed=42
        )
        train_dataset = split['train']
        eval_dataset = split['test']
        if (
            "records_cap" in self.sft_training_args and
            self.sft_training_args["records_cap"] is not None and
            isinstance(self.sft_training_args["records_cap"], int)
        ):
            train_dataset = train_dataset.take(
                self.sft_training_args["records_cap"])
            eval_dataset = eval_dataset.take(
                self.sft_training_args["records_cap"])
            print(f"train_dataset : {train_dataset}")
            print(f"eval_dataset :  {eval_dataset}")

        train_args = UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,

            **{k: v for k, v in self.sft_training_args.items()
                    if k != "records_cap"},

            per_device_eval_batch_size=2,
            eval_steps=200,
            eval_strategy="steps",
            do_eval=True,

            learning_rate=5e-5,
            # embedding_learning_rate=1e-5, # Optionally here

            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            optim="adamw_8bit",
            weight_decay=0.00,
            lr_scheduler_type="linear",
            #seed=3407,

            output_dir=os.path.join(
                self.unsloth_dir, "outputs", "sft"),
            save_total_limit=2,

            logging_steps=1,
            report_to="tensorboard",
            logging_dir=os.path.join(
                self.sft_model_dir,
                "runs", "sft")
        )

        self.sft_traces_file_fullname = os.path.join(
            self.unsloth_dir, "sft_trainer_traces.txt")
        print("Training started. " +
              f"Check {self.sft_traces_file_fullname} for live traces.",
              flush=True)

        trainer = UnslothTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            eval_dataset=eval_dataset,
            max_seq_length=self.max_seq_length,
            dataset_num_proc=8,
            args=train_args
        )
        trainer.can_return_loss = True
        #######################################

        #######################################
        #      Show current memory stats      #
        #######################################
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        _ = gc.collect()

        used_memory = \
            round(torch.cuda.max_memory_reserved()
                  /1024/1024/1024, 3)
        used_memory_for_lora = \
            round(used_memory-self.start_gpu_memory, 3)
        used_percentage = \
            round(used_memory/self.max_memory*100, 3)
        lora_percentage = \
            round(used_memory_for_lora/self.max_memory*100,
                  3)
        print(f"Peak reserved memory = " +
              f"{used_memory} GB.")
        print(f"Peak reserved memory for " +
              f"training = {used_memory_for_lora} " +
              f"GB.")
        print(f"Peak reserved memory % of " +
              f"max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training " +
              f"% of max memory = {lora_percentage} %.")
        #######################################

        with open(self.sft_traces_file_fullname, 'w') as f:
            with redirect_stdout(f):
                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
                trainer_stats = trainer.train()
        hf_logging.set_verbosity_info()
        hf_logging.enable_progress_bar()
        print(f"{trainer_stats.metrics['train_runtime']} " +
              f"seconds used for training " +
              f"({round(trainer_stats.metrics['train_runtime']/60, 2)}" +
              f" minutes).")

        self.sft_log_history = trainer.state.log_history
        self.sft_log_history_fig = \
            plot_log_history(
                self.sft_log_history,
                title="Supervised finetuning loss"
            )

        model.save_pretrained_merged(
            self.sft_model_dir, tokenizer,
            save_method = "lora"
        )
        print(f"sft_model_dir : {self.sft_model_dir}\n")

        self.next(self.evaluate_model)


    @step
    def evaluate_model(self):
        """
        Batch inference on the SFT validation dataset.
        """
        from retrain_pipelines.model import \
            infer_validation, compute_counts_n_metrics, \
            plot_validation_completions

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        _ = gc.collect()


        ######################################################
        #              loading trained adapter               #
        ######################################################
        # Unsloth [and hf transformers before it]            #
        # (if loading both model & tokenizer at once         #
        # same as we did in prior tasks, but now             #
        # with tokenizer.chat_template being set             #
        # in tokenizer.config) is forcing on us some kind of #
        # chat_template format hard-requirements.            #
        ######################################################
        # load base from cache
        # (with base tokenizer, which we ignore)
        model, _ = FastLanguageModel.from_pretrained(
            model_name=self.hf_base_model_dict["repo_id"],
            revision=self.hf_base_model_dict["commit_hash"],
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
            # case of a gated or private base-model
            token=os.getenv("HF_TOKEN", None)
        )
        model = FastLanguageModel.for_inference(model)
        # load our CPT+SFT trained & locally-saved adapter
        model.load_adapter(peft_model_id=self.sft_model_dir)
        # Separately load our (potentially trained &)
        # locally-saved adapter-tokenizer
        # (loading it below via HF and not Unsloth)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.sft_model_dir
        )
        ######################################################

        ######################################################
        #                 validation dataset                 #
        ######################################################
        # download from Hub (or get from local cache)
        queries_dataset = load_dataset(
            path=self.dataset_commit_dict["repo_id"],
            name="supervised_finetuning",
            revision=self.dataset_commit_dict["commit_hash"],
            token=os.getenv("HF_TOKEN", None))
        if (
            "records_cap" in self.sft_training_args and
            self.sft_training_args["records_cap"] is not None and
            isinstance(self.sft_training_args["records_cap"], int)
        ):
            validation_data = queries_dataset["validation"].take(
                self.sft_training_args["records_cap"])
        else:
            validation_data = queries_dataset["validation"]
        print(validation_data, flush=True)
        ######################################################

        self.max_new_tokens = 400
        start_time = time.time()
        validation_results = infer_validation(
            tokenizer=tokenizer,
            model=model,
            validation_data=validation_data,
            prompt_template=tokenizer.chat_template,
            batch_size=32, # 64,
            queries_attr_name=\
                self.hf_dataset["attributes"]["query_attr"],
            answers_attr_name=\
                self.hf_dataset["attributes"]["answers_attr"],
            max_new_tokens=self.max_new_tokens,
            device="cuda"
        )
        print("infer_validation -   Elapsed time: " +
              f"{(time.time() - start_time):.2f} seconds")
        self.validation_results = validation_results #  <= to artifacts store

        eval_df  = pl.LazyFrame(validation_results)

        records = eval_df.with_columns(
            (pl.col("answer") == pl.col("completion")) \
                .alias("is_ground_truth_identical")
        ).collect() #engine=self.engine)
        print("perfect characters-match accuracy : " +
              str(records['is_ground_truth_identical'].mean()))

        eval_metrics_df = compute_counts_n_metrics(
            eval_df, is_format_fault_tolerant=True)
        overall_metrics_df = eval_metrics_df.select([
                pl.col("precision").mean(),
                pl.col("recall").mean(), 
                pl.col("f1").mean(),
                pl.col("jaccard").mean()
            ]).collect() #engine=self.engine)
        self.perf_metrics = overall_metrics_df.row(0, named=True)
        print(self.perf_metrics)

        self.validation_completions_fig = \
            plot_validation_completions(
                eval_metrics_df, engine=self.engine)

        del model
        del tokenizer
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        _ = gc.collect()

        self.next(self.model_version_blessing)


    @step
    def model_version_blessing(self):
        """
        Comparing newly-retrained model version
        against best-performing predecessor.
        """
        """
        Note: for Hugging Face integrated pipelines,
        we compare against lastest commit of main branch
        of the model repository there.
        When it comes to local "mf_run_id" of the pipeline run
        having generated that best prior model version
        (retrieved from model card metadata from HF yaml section),
        we check against records of the herein ML-framework instance,
        as "prior best version" of the model here beign retrained
        may have been originated from another one
        than the one executing the current retraining
        (in which case, we simply don't includ a "local" hyperlink
        in the model version pipeline_cards that will be
        produced later in the herein pipeline run).
        """
        from retrain_pipelines.model.hf_utils import \
            current_blessed_model_version_dict

        main_perf_metric_name = "jaccard"

        current_blessed_version_dict = \
            current_blessed_model_version_dict(
                repo_id=self.model_repo_id,
                hf_token=os.getenv("HF_TOKEN", None)
            )
        print("current_blessed_version_dict : " +
              str(current_blessed_version_dict))

        if current_blessed_version_dict is None:
            print("case 'no prior blessed model version found"
                  " => blessing.'")
            self.model_version_blessed = True

        elif (
            main_perf_metric_name in
                current_blessed_version_dict["perf_metrics"]
        ):
            current_blessed_run_id = \
                current_blessed_version_dict["mf_run_id"]
            print(f"current_blessed_run_id : {current_blessed_run_id}")
            current_blessed_metric_value = \
                current_blessed_version_dict[
                    "perf_metrics"][main_perf_metric_name]

            self.model_version_blessed = (
                self.perf_metrics[main_perf_metric_name] >=
                current_blessed_metric_value
            )

            if not self.model_version_blessed:
                self.current_blessed_version_dict = \
                    current_blessed_version_dict
                for run in Flow(self.__class__.__name__):
                    if str(run.id) == current_blessed_run_id:
                        run_steps = iter(run.steps())
                        last_run_step = next(run_steps)
                        last_task = next(iter(last_run_step.tasks()))

                        # tasks are listed backwards, so last task is first item :
                        # Has the run seen task "pipeline_card" prior to last task
                        # (meaning, "pipeline_card" completed successfully and
                        #  "run" has generated a sutom pipeline-card artifact) ?
                        # If not, hyperlink generation will later fail.
                        run_has_custom_card_artifact = False
                        for step in run_steps:
                            if "pipeline_card" == step.id:
                                run_has_custom_card_artifact = True
                                break

                        if not run_has_custom_card_artifact:
                            print(
                                f"Run #{current_blessed_run_id} " +
                                "Doesn't seem to have successfully " +
                                "generated a pipeline-card artifact.",
                                file=sys.stderr, flush=True)
                            break
                        else:
                            # further filtering on successful runs that are
                            # retraining of a prior version of the same model
                            # (to minimize the risk that this was obtained
                            #  on another ML-framework instance)
                            if (
                                # last_task.successful and
                                # may have failed after the "pipeline_card" step
                                # and been resumed
                                hasattr(last_task.artifacts,
                                        'model_version_blessed') and
                                last_task.artifacts.model_version_blessed.data and
                                hasattr(last_task.artifacts,
                                        'model_repo_id') and
                                last_task.artifacts.model_repo_id.data == \
                                    self.model_repo_id
                            ):
                                self.current_blessed_run = run
                            break

                if not self.current_blessed_run:
                    print(
                        "Couldn't find blessed run " +
                        f"{current_blessed_run_id} !\n" +
                        "It seems that prior blessed run was " +
                        "executed on another ML framework instance.",
                        file=sys.stderr, flush=True)

            print("new : " +
                    str(self.perf_metrics[main_perf_metric_name]) +
                  " - previous best : " +
                    str(current_blessed_metric_value) +
                  " - model_version_blessing : " +
                    str(self.model_version_blessed))

        else:
            raise Exception(
                "Performance metric '" +
                main_perf_metric_name +
                "' can't be found in eval results " +
                "from blessed run " +
                str(current_blessed_version_dict[
                    "mf_run_id"]) + " !")

        # self.model_version_blessed = True ### DEBUG - DELETE ###

        self.next(self.model_to_hub)


    @step
    def model_to_hub(self):
        """
        Push to hub model version, including
        readme with versioning info.
        """

        #############################
        #  case of user-provided    #
        # documentation artifact(s) #
        #############################
        # note that user can provide either
        # 'pipeline_card.py' or 'template.html'
        # or 'dataset_readme.py'
        # or 'dataset_readme_template.md'
        # or 'model_readme.py'
        # or 'model_readme_template.md'
        # or any combination of those
        # when specifying custom
        # 'pipeline_card_artifacts_path'
        if (
            "model_readme_template.md" in
                os.listdir(self.pipeline_card_artifacts_path)
        ):
            template_dir = self.pipeline_card_artifacts_path
        else:
            template_dir = os.path.dirname(
                importlib.util.find_spec(
                    f"retrain_pipelines.pipeline_card."+
                    f"{os.getenv('retrain_pipeline_type')}"
                ).origin)
        print(f"template_dir : '{template_dir}'")
        #############################
        if "model_readme.py" in os.listdir(
                self.pipeline_card_artifacts_path):
            from retrain_pipelines.utils import \
                get_get_model_readme_content
            get_model_readme_content = \
                get_get_model_readme_content(
                    self.pipeline_card_artifacts_path)
        else:
            from retrain_pipelines.pipeline_card import \
                    get_model_readme_content
        #############################
        from retrain_pipelines.model.hf_utils import \
            push_model_version_to_hub

        #############################
        #        model README       #
        #       from template       #
        #############################
        commit_datetime = datetime.utcnow()
        new_model_version_label = get_new_repo_minor_version(
            repo_id=self.model_repo_id,
            repo_type="model",
            hf_token=os.getenv("HF_TOKEN", None))
        readme_content = get_model_readme_content(
            template_folder=template_dir,

            model_repo_id=self.model_repo_id,

            base_model_dict=self.hf_base_model_dict,
            training_dataset_dict=self.dataset_commit_dict,

            version_label=new_model_version_label,
            commit_datetime=commit_datetime,
            perf_metrics=self.perf_metrics,

            mf_flow_name=current.flow_name,
            mf_run_id=current.run.id
        )
        #############################

        print("Pushing model version to HF hub " +
              ("(blessed). " if self.model_version_blessed
               else "(not blessed). ") +
              "May take a while..",
              flush=True)
        model_commit_hash = push_model_version_to_hub(
            repo_id=self.model_repo_id,
            model_version_blessed=\
                self.model_version_blessed,
            version_label=new_model_version_label,
            timestamp_str=commit_datetime.strftime(
                "%Y-%m-%d %H:%M:%S UTC"),
            model_dir=self.sft_model_dir,
            model_readme_content=readme_content,
            hf_token=os.getenv("HF_TOKEN", None)
        )
        if not model_commit_hash:
            raise Exception(
                "Failed to publish model version.")
        print("Push of model version to HF hub completed.",
              flush=True)
        print(f"https://huggingface.co/{self.model_repo_id}" +
              f"/blob/{model_commit_hash}/README.md")

        self.model_commit_dict = {
            "repo_id": self.model_repo_id,
            "commit_hash": model_commit_hash,
            "version_label": new_model_version_label,
            "commit_datetime": commit_datetime,
        }

        self.next(self.infra_validator)


    @step
    def infra_validator(self):
        """
        If the trained model version is blessed,
        validate serving.
        """
        """
        Note that using isolated virtual env
        (using @conda task decorator)
        is advisable to not embark the whole
        pipeline dependencies into the local server.
        We don't for educational purpose,
        keep things "simple" to grasp
        as well as to avoid forcing conda
        (for instance miniconda) as
        a virtual environment management mean
        to the user.
        """
        """
        Note : We load base model from HF-cache
        (mounted as /huggingface_hub_cache
        docker volume) and adapter from local dir
        (mounted as /FuncCallAdater docker volume.
        """

        self.local_serve_is_ready = LocalServeReadinessEnum.NOT_APPLICABLE

        if self.model_version_blessed:
            from retrain_pipelines.utils.docker import \
                env_has_docker

            if env_has_docker():
                model_module_dir = \
                    os.path.dirname(
                        importlib.util.find_spec(
                            "retrain_pipelines.model." +
                            os.getenv('retrain_pipeline_type')
                        ).origin)

                # server & data-model & server-config modules artifacts
                files_to_copy = [
                    "litserve_server.py",
                    "litserve_datamodel.py",
                    "litserve_serverconfig.py",
                    ".dockerignore" # docker context loading
                                    # at image-build time,
                                    # exclude model weights
                ]
                for filename in files_to_copy:
                    shutil.copy(
                        os.path.join(model_module_dir, "litserve",
                                     filename),
                        os.path.join(self.serving_artifacts_local_folder,
                                     filename)
                    )

                # save dependencies as artifact
                create_requirements(self.serving_artifacts_local_folder,
                                    exclude=["cudf-polars-.*", "cuda-python",
                                             "nvidia-.*", "(py)?libcudf-.*",
                                             "nvtx", "rmm-.*", "litserve",
                                             ".*retrain-pipelines.*"]
                )

                # server config yaml
                env = Environment(loader=FileSystemLoader(
                    os.path.join(model_module_dir, "litserve")))
                template = env.get_template(
                    "litserve_serverconfig_template.yaml")
                server_config_data = {
                    "port": "8000",
                    "max_seq_length": self.max_seq_length,
                    "max_new_token": self.max_new_tokens,
                    "base_model": {
                        "repo_id": self.hf_base_model_dict["repo_id"],
                        "revision": self.hf_base_model_dict["commit_hash"]
                    },
                    "adapters": [
                        {
                            "name": "func_caller",
                            "path": "/FuncCallAdapter"
                        }
                    ]
                }
                server_config_yaml = template.render(server_config_data)
                print(server_config_yaml)
                with open(os.path.join(
                    self.serving_artifacts_local_folder,
                    "litserve_serverconfig.yaml"), 'w'
                ) as output_file:
                    output_file.write(server_config_yaml)

                # Dockerfile
                env = Environment(loader=FileSystemLoader(
                    os.path.join(model_module_dir)))
                template = env.get_template(
                    "Dockerfile.litserve_template")
                # Change CUDA version here from available list
                # @see https://hub.docker.com/r/nvidia/cuda/tags
                dockerfile_content = template.render(
                    {"cuda_version": "12.0.0"})
                with open(os.path.join(
                    self.serving_artifacts_local_folder,
                    "Dockerfile.litserve"), 'w'
                ) as output_file:
                    output_file.write(dockerfile_content)

                os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

                ############################################
                #   actually deploy the inference service  #
                ############################################
                start_time = time.time()
                from retrain_pipelines.utils.docker import \
                    build_and_run_docker, print_container_log_tail, \
                    cleanup_docker
                from retrain_pipelines.model.litserve import \
                    endpoint_started, endpoint_is_ready

                self.port = 8765
                HF_HUB_CACHE = os.path.realpath(os.path.expanduser(
                    os.getenv(
                        "HF_HUB_CACHE",
                        os.path.join(os.getenv("HF_HOME",
                                               "~/.cache/huggingface"),
                                     "hub")
                    )))
                print(f"HF_HUB_CACHE : {HF_HUB_CACHE}")
                image_name = container_name = "litserve-model"

                serving_container = build_and_run_docker(
                    image_name=image_name, image_tag="1.0",
                    build_path=self.serving_artifacts_local_folder,
                    dockerfile="Dockerfile.litserve",
                    ports_publish_dict={'8000/tcp': self.port},
                    env_vars_dict={
                        "HF_HUB_CACHE": "/huggingface_hub_cache",
                        "HF_TOKEN": os.getenv("HF_TOKEN")
                    },
                    volumes_dict={
                        self.sft_model_dir:
                            {"bind": "/FuncCallAdapter",
                             "mode": "ro"},
                        HF_HUB_CACHE:
                            {"bind": "/huggingface_hub_cache",
                             "mode": "ro"}
                    }
                )

                if not serving_container:
                    print("failed spinning the LitServe container",
                          file=sys.stderr)
                    self.local_serve_is_ready = \
                        LocalServeReadinessEnum.FAILURE
                    try:
                        cleanup_docker(
                            container_name=container_name,
                            image_name=f"{image_name}:1.0",
                            no_pruning=True # for intermediate layers recycling
                                            # (during later re-runs)
                                            # to avoid long rebuild time
                                            # of exactly the same.
                        )
                    except Exception as cleanup_ex:
                        # fail silently
                        pass
                else:
                    print("Awaiting endpoint launch..")
                    start_time = time.time()
                    if not endpoint_started(
                        container_name, port=self.port, timeout=10*60
                    ):
                        print(
                            f"The endpoint '{container_name}' " +
                            f"did not start.")
                        self.local_serve_is_ready = \
                            LocalServeReadinessEnum.FAILURE
                    # health check on the spun-up endpoint
                    elif endpoint_is_ready(port=self.port):
                        self.local_serve_is_ready = \
                            LocalServeReadinessEnum.SUCCESS
                elapsed_time = time.time() - start_time
                print("deploy_local -   Elapsed time: " +
                      f"{elapsed_time:.2f} seconds")
                ############################################
            else:
                # env doesn't have docker
                self.local_serve_is_ready = \
                    LocalServeReadinessEnum.FAILURE_NO_DOCKER

            if LocalServeReadinessEnum.SUCCESS == self.local_serve_is_ready:
                from retrain_pipelines.model.litserve.litserve_datamodel \
                    import Response

                import requests

                url = f"http://localhost:{self.port}/predict"
                headers = {"accept": "application/x-www-form-urlencoded"}

                try:
                    start_time = time.time()
                    data = {
                        "adapter_name": "func_caller",
                        "queries": '["Hello.", "Is 49 a perfect square?"]'
                    }
                    print(f"inference test - data: {data}")
                    response = requests.post(url, headers=headers, data=data)
                    parsed_response = Response(**{"output": response.json()})
                    elapsed_time = time.time() - start_time
                    print("parsed_response ('func_caller' adapter ON) :" +
                          str(parsed_response) +
                          f"\t-\tElapsed time: {elapsed_time:.2f} seconds")

                    start_time = time.time()
                    data = {
                        "queries": '["Hello.", "Is 49 a perfect square?"]'
                    }
                    print(f"inference test - data: {data}")
                    response = requests.post(url, headers=headers, data=data)
                    parsed_response = Response(**{"output": response.json()})
                    elapsed_time = time.time() - start_time
                    print(f"parsed_response (no adapter) : {parsed_response}" +
                          f"\t-\tElapsed time: {elapsed_time:.2f} seconds")

                except Exception as ex:
                    print(ex, file=sys.stderr)
                    traceback.print_tb(ex.__traceback__, file=sys.stderr)
                    self.local_serve_is_ready = \
                        LocalServeReadinessEnum.FAILURE
                    pass

            try:
                cleanup_docker(
                    container_name=container_name,
                    image_name=f"{image_name}:1.0",
                    no_pruning=True # for intermediate layers recycling
                                    # (during later re-runs)
                                    # to avoid long rebuild time
                                    # of exactly the same.
                )
            except Exception as cleanup_ex:
                # fail silently
                pass

        self.next(self.pipeline_card)


    @card(id='default')
    @card(type='html', id='custom')
    @step
    def pipeline_card(self):
        import re
        import datetime
        import importlib.metadata

        #############################
        #   case of user-provided   #
        # documentation artifact(s) #
        #############################
        # note that user can provide either
        # 'pipeline_card.py' or 'template.html'
        # or 'dataset_readme.py'
        # or 'dataset_readme_template.md'
        # or 'model_readme.py'
        # or 'model_readme_template.md'
        # or any combination of those
        # when specifying custom
        # 'pipeline_card_artifacts_path'
        if "template.html" in os.listdir(
                                self.pipeline_card_artifacts_path
        ):
            template_dir = self.pipeline_card_artifacts_path
        else:
            template_dir = os.path.dirname(
                importlib.util.find_spec(
                    f"retrain_pipelines.pipeline_card."+
                    f"{os.getenv('retrain_pipeline_type')}"
                ).origin)
        #############################
        if "pipeline_card.py" in os.listdir(
                                    self.pipeline_card_artifacts_path
        ):
            from retrain_pipelines.utils import get_get_html
            get_html = \
                get_get_html(self.pipeline_card_artifacts_path)
        else:
            from retrain_pipelines.pipeline_card import \
                    get_html
        from retrain_pipelines.pipeline_card.helpers import \
                mf_dag_svg
        #############################


        #############################
        ##      "default" card     ##
        #############################
        self.metadata = {
            "name": "TabNet Model",
            "version": "1.0",
            "retrain_pipelines": f"retrain-pipelines {__version__}",
            "retrain_pipeline_type": os.environ["retrain_pipeline_type"],
            "description": "A PyTorch TabNet model retrained",
            "authors": [current.username],
            "tags": ["classification", "tabnet"],
            "license": "MIT License",
            "data_augmentation": [
                {
                    "name": "Augmentation",
                    "description": "Truncating queries and " + \
                                   "associate those to " + \
                                   "no tool-call answers. " + \
                                   "Intent being to instruct on " + \
                                   "not hallucinating missing " + \
                                   "tool-calls parameters values."
                },
                {
                    "name": "Enrichment",
                    "description": "Addition of records " + \
                                   "from an external data-source. " + \
                                   "Here to instruct on no tool-call."
                }
            ],
            "references": [
                {
                    "title": "Base model",
                    "link": f"https://hf.co/{self.hf_base_model_dict['repo_id']}"
                },
                {
                    "title": "Function-calling dataset",
                    "link": f"https://hf.co/{self.hf_dataset_dict['repo_id']}"
                },
                {
                    "title": "Data-enrichment dataset",
                    "link": f"https://hf.co/{self.hf_enrich_dataset_dict['repo_id']}"
                },
                {
                    "title": "Unsloth",
                    "link": "https://unsloth.ai/blog/contpretraining"
                }
            ]
        }

        current.card['default'].append(Markdown(
            "model_version_blessed : **%s**" % str(self.model_version_blessed)))
        current.card['default'].append(Artifact(
            {"model_version_blessed": self.model_version_blessed}))

        current.card['default'].append(
            Image.from_matplotlib(self.sft_log_history_fig))
        current.card['default'].append(
            Image.from_matplotlib(self.validation_completions_fig))
        #############################

        #############################
        ##    html "custom" card   ##
        #############################
        dt = datetime.datetime.now(tz=datetime.timezone.utc)
        formatted_dt = dt.strftime("%A %b %d %Y %I:%M:%S %p %Z")
        task_obj_python_cmd = f"metaflow.Task(" + \
            f"\"{current.pathspec}\", " + \
            f"attempt={str(current.retry_count)})"
        params={
            'template_dir': template_dir,
            'title': f"{current.flow_name}",
            "subtitle": f"(flow run # {len(list(current.run.parent.runs()))}," + \
                        f" run_id: {str(current.run.id)}  -  {formatted_dt})",

            # blessed status / current_blessed version
            'model_version_blessed': self.model_version_blessed,
            'current_blessed_version_label': (
                self.current_blessed_version_dict["version_label"]
                if self.current_blessed_version_dict
                else None
            ),
            'current_blessed_commit_datetime': (
                self.current_blessed_version_dict["commit_datetime"]
                if self.current_blessed_version_dict
                else None
            ),
            'current_blessed_model_commit_hash': (
                self.current_blessed_version_dict["commit_hash"]
                if self.current_blessed_version_dict
                else None
            ),
            'current_blessed_run': self.current_blessed_run,

            'LocalServeReadinessEnum': LocalServeReadinessEnum,
            'local_serve_is_ready': self.local_serve_is_ready,
            # EDA
            'main_dataset_repo_id': self.hf_dataset['repo_id'],
            'main_dataset_commit_hash': self.hf_dataset_dict['commit_hash'],
            'main_dataset_commit_datetime': \
                self.hf_dataset_dict['commit_datetime'],

            'records_count': self.records_count,
            'data_schema': self.data_schema,
            'answers_tools_count_fig': self.answers_tools_count_fig,
            'words_count_fig': self.words_count_fig,

            # model training
            'dataset_repo_id': self.dataset_repo_id,
            'dataset_version_label': self.dataset_commit_dict["version_label"],
            'dataset_commit_datetime': self.dataset_commit_dict["commit_datetime"],
            'dataset_commit_hash': self.dataset_commit_dict["commit_hash"],
            'dataset_augmentation_rate': self.actual_augmentation_rate,
            'dataset_enrichment_rate': self.enrichment_rate,

            'model_repo_id': self.model_repo_id,
            'model_version_label': self.model_commit_dict["version_label"],
            'model_commit_datetime': self.model_commit_dict["commit_datetime"],
            'model_commit_hash': self.model_commit_dict["commit_hash"],

            'cpt_log_history_fig': self.cpt_log_history_fig,
            'sft_log_history_fig': self.sft_log_history_fig,

            'validation_completions_fig': self.validation_completions_fig,

            'pipeline_parameters_dict': {"cpt": self.cpt_training_args,
                                         "sft": self.sft_training_args},

            'metrics_dict': self.perf_metrics,

            'task_obj_python_cmd': task_obj_python_cmd,
            'dag_svg': mf_dag_svg(self)
        }
        self.html = get_html(params)
        #############################
        current
        #############################

        self.next(self.pipeline_to_hub)


    @step
    def pipeline_to_hub(self):
        """
        publish versioned source-code and pipeline-card
        for ths run on the Hugging Face Hub.
        """

        model_commit_datetime = \
            self.model_commit_dict["commit_datetime"]
        timestamp_str = \
            "{:%Y%m%d_%H%M%S}".format(model_commit_datetime) + \
            "{:03d}".format(model_commit_datetime.microsecond//1000) + \
            "_UTC"
        subfolder_name = \
            "v" + self.model_commit_dict["version_label"] + \
            "_" + timestamp_str
        commit_datetime = datetime.utcnow()

        ###############################
        #         source-code         #
        ###############################
        # We upload only herein file  #
        # plus user-provided versions #
        # of the customizable ones    #
        # (if any).                   #
        ###############################
        custom_source_files = [os.path.abspath(__file__)]
        if (
            self.pipeline_card_artifacts_path != \
            self.default_pipeline_card_module_dir
        ):
            candidate_source_files = [
                "pipeline_card.py",
                "template.html",
                "dataset_readme.py",
                "dataset_readme_template.md",
                "model_readme.py",
                "model_readme_template.md"
            ]
            for candidate_source_file in candidate_source_files:
                file_fullpath = os.path.join(
                    self.pipeline_card_artifacts_path,
                    candidate_source_file)
                if os.path.exists(file_fullpath):
                    custom_source_files.append(file_fullpath)

        source_code_commit_hash = \
            push_files_to_hub_repo_branch(
                repo_id=self.model_repo_id,
                branch_name="retrain-pipelines_source-code",
                file_fullnames=custom_source_files,
                include_requirements_txt=True,
                path_in_repo=subfolder_name,
                commit_message=\
                    "source-code for model version " + \
                    subfolder_name + \
                    f"- retrain-pipelines {__version__}",
                repo_type="model",
                hf_token=os.getenv("HF_TOKEN", None)
            )
        print(source_code_commit_hash)
        self.source_code_commit_dict = {
            "repo_id": self.model_repo_id,
            "branch_name": "retrain-pipelines_source-code",
            "commit_datetime": commit_datetime,
            "commit_hash": source_code_commit_hash
        }
        ###############################

        ###############################
        #        pipeline-card        #
        ###############################
        pipeline_card_fullname = None
        for run_step in current.run.steps():
            task = list(run_step.tasks())[0]
            task_name = task.path_components[2]
            if "pipeline_card" == task_name:
                pipeline_card = get_cards(
                    task, id='custom', type='html')[0]
                pipeline_card_fullname = os.path.realpath(
                    os.path.join(
                        task.metadata_dict.get("ds-root", None),
                        mf_config.CARD_SUFFIX, pipeline_card.path
                    ))
                print(pipeline_card_fullname)
                break
        pipeline_card_commit_hash = \
            push_files_to_hub_repo_branch(
                repo_id=self.model_repo_id,
                branch_name="retrain-pipelines_pipeline-card",
                file_fullnames=[pipeline_card_fullname],
                path_in_repo=subfolder_name,
                commit_message=\
                    "pipeline-card for model version " + \
                    subfolder_name + \
                    f"- retrain-pipelines {__version__}",
                repo_type="model",
                hf_token=os.getenv("HF_TOKEN", None)
            )
        print(pipeline_card_commit_hash)
        self.pipeline_card_commit_dict = {
            "repo_id": self.model_repo_id,
            "branch_name": "retrain-pipelines_pipeline-card",
            "commit_datetime": commit_datetime,
            "commit_hash": pipeline_card_commit_hash
        }
        ###############################

        self.next(self.deploy)


    @step
    def deploy(self):
        """
        placeholder for the serving SDK deploy call
        (on the target production platform).
        Include any artifact you want,
        consider including the portable pipelione-card
        itself !
        """

        if (
            self.model_version_blessed and
            (self.local_serve_is_ready == LocalServeReadinessEnum.SUCCESS)
        ):
            pass # your code here

        self.next(self.load_test)


    @step
    def load_test(self):
        """
        placeholder
        """

        if (
            self.model_version_blessed and
            (self.local_serve_is_ready == LocalServeReadinessEnum.SUCCESS)
        ):
            pass # your code here

        self.next(self.end)


    @step
    def end(self):
        pass


if __name__ == "__main__":
    UnslothFuncCallFlow()

