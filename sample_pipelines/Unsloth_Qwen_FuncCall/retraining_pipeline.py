
from unsloth import FastLanguageModel, \
    is_bfloat16_supported, UnslothTrainer, \
    UnslothTrainingArguments

import torch

import os
import gc
import re
import sys
import json
import time
import shutil
import logging
import builtins

import importlib.util
from enum import Enum
from textwrap import dedent
from datetime import datetime, \
    timezone

import polars as pl
from polars.exceptions import ComputeError

from jinja2 import Environment, FileSystemLoader

from huggingface_hub import list_repo_commits
from datasets import load_dataset, \
    Dataset, DatasetDict
from datasets.config import HF_DATASETS_CACHE, \
    HF_CACHE_HOME
from transformers import AutoTokenizer

from retrain_pipelines import __version__
from retrain_pipelines.dataset.hf_utils import \
    get_lazy_df, get_column_info, \
    iterable_dataset_multi_buffer_sampler, \
    push_dataset_version_to_hub
from retrain_pipelines.dataset.tool_calls import \
    count_tool_occurrences, plot_tools_occurences, \
    column_words_stats, plot_words_count, \
    get_unique_tools
from retrain_pipelines.utils.hf_utils import \
    get_repo_version, get_new_repo_minor_version, \
    push_files_to_hub_repo_branch

from retrain_pipelines.dag_engine.core import \
    TaskPayload, task, dag, DagParam, ctx, UiCss

from retrain_pipelines.dag_engine.rp_logging import \
    rp_redirect_stdout

from retrain_pipelines.dag_engine.sdk import \
    ExecutionsIterator

from retrain_pipelines.utils import create_requirements


#--- helpers ----------------------------------------------------------------------------


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

 
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


def clear_gc():
    """Convenience method to clear
    the content of the garbage collector.
    Forcing it to actually clear
    any cuda tensor it holds.
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except:
            pass
    gc.collect()


#--- retraining-pipeline elements -------------------------------------------------------


@task
def start() -> TaskPayload:
    logger.info(f"{ctx.pipeline_name} - {ctx.exec_id}")
    logging.getLogger("retrain_pipelines").setLevel(logging.INFO)

    # inputs validation
    repo_id_pattern = re.compile(
        r"""
        ^                           # start
        (?!.*\.\.)                  # no '..' anywhere
        (?!.*--)                    # no '--' anywhere
        (?:                         # legacy: single segment OR namespace/repo
            [A-Za-z0-9._-]+         # legacy: gpt2, bert-base-uncased, etc.
            |
            [A-Za-z0-9._-]+/[A-Za-z0-9._-]+   # namespace/repo_name
        )
        $                           # end
        """,
        re.VERBOSE
    )
    assert repo_id_pattern.match(ctx.hf_dataset["repo_id"]) is not None, \
           f"Invalid repo_id format: {ctx.hf_dataset['repo_id']!r}"
    ctx.augmentation_rate = float(ctx.augmentation_rate)
    assert repo_id_pattern.match(ctx.hf_enrich_dataset["repo_id"]) is not None, \
           f"Invalid repo_id format: {ctx.hf_enrich_dataset['repo_id']!r}"
    ctx.enrichment_rate = float(ctx.enrichment_rate)
    assert repo_id_pattern.match(ctx.dataset_repo_id) is not None, \
           f"Invalid repo_id format: {dataset_repo_id!r}"
    assert ctx.polars_engine in ["gpu", "cpu"]
    assert repo_id_pattern.match(ctx.hf_base_model["repo_id"]) is not None, \
           f"Invalid repo_id format: {ctx.hf_base_model['repo_id']!r}"
    assert repo_id_pattern.match(ctx.model_repo_id) is not None, \
           f"Invalid repo_id format: {model_repo_id!r}"

    # GPU availability
    logger.info(torch.cuda.get_device_name(0))
    logger.info(torch.__version__)
    ctx.engine = "cpu" if (
                    ctx.polars_engine == "gpu" and
                    not torch.cuda.is_available()
                 ) else ctx.polars_engine
    logger.debug(f"Polars engine : {ctx.engine}")

    # hf_dataset
    hf_dataset_dict = \
        get_lazy_df(
            repo_id=ctx.hf_dataset["repo_id"],
            commit_hash=ctx.hf_dataset["commit_hash"],
            config_name=(
                ctx.hf_dataset["config_name"] and
                "" < ctx.hf_dataset["config_name"]
            ),
            hf_token=os.getenv("HF_TOKEN", None)
        )
    try:
        logger.info(f"hf_dataset_dict lazy_df : {hf_dataset_dict['lazy_df']}")
        logger.info(
            f"{hf_dataset_dict['repo_id']}, " +
            f"{hf_dataset_dict['commit_hash']}  -  " +
            f"{hf_dataset_dict['commit_datetime']}\n" +
            hf_dataset_dict["lazy_df"].explain()
        )
    except ComputeError as ex:
        if "HF_TOKEN" not in os.environ:
            logger.info("Does the Hugging Face-hosted dataset " +
                  "require authentication ?",
                  file=sys.stderr, flush=True)
        raise ex
    hf_dataset_version = get_repo_version(
        repo_id=hf_dataset_dict["repo_id"],
        revision=hf_dataset_dict["commit_hash"],
        repo_type="dataset",
        hf_token=os.getenv("HF_TOKEN", None)
    )
    hf_dataset_dict["version_label"] = (
        f"{hf_dataset_version[0]}.{hf_dataset_version[1]}"
        if sum(hf_dataset_version) > 0
        else None
    )
    ctx.hf_dataset_dict = hf_dataset_dict

    # hf_enrich_dataset
    hf_enrich_dataset_dict = \
        get_lazy_df(
            repo_id=ctx.hf_enrich_dataset["repo_id"],
            commit_hash=ctx.hf_enrich_dataset["commit_hash"],
            config_name=(
                ctx.hf_enrich_dataset["config_name"] and
                "" < ctx.hf_enrich_dataset["config_name"]
            ),
            hf_token=os.getenv("HF_TOKEN", None)
        )
    hf_enrich_dataset_version = get_repo_version(
        repo_id=hf_enrich_dataset_dict["repo_id"],
        revision=hf_enrich_dataset_dict["commit_hash"],
        repo_type="dataset",
        hf_token=os.getenv("HF_TOKEN", None)
    )
    hf_enrich_dataset_dict["version_label"] = (
        f"{hf_enrich_dataset_version[0]}.{hf_enrich_dataset_version[1]}"
        if sum(hf_enrich_dataset_version) > 0
        else None
    )
    logger.info(' ; '.join(f"{k}: {hf_enrich_dataset_dict[k]}"
                                   for k in ['commit_hash',
                                             'commit_datetime']))
    ctx.hf_enrich_dataset_dict = hf_enrich_dataset_dict

    # hf_base_model
    hf_base_model_revision=(
        None if (rev_commit_hash:=ctx.hf_base_model["commit_hash"]) == ""
        else rev_commit_hash
    )
    hf_base_model_commit = list_repo_commits(
            repo_id=ctx.hf_base_model["repo_id"],
            revision=hf_base_model_revision,
            repo_type="model",
            token=os.getenv("HF_TOKEN", None)
        )[0]
    # version major+minor=0 for non retrain-pipelines models
    hf_base_model_version = get_repo_version(
        repo_id=ctx.hf_base_model["repo_id"],
        revision=hf_base_model_revision,
        repo_type="model",
        hf_token=os.getenv("HF_TOKEN", None)
    )
    ctx.hf_base_model_dict = {
        "repo_id": ctx.hf_base_model["repo_id"],
        "version_label": (
            f"{hf_base_model_version[0]}.{hf_base_model_version[1]}"
            if sum(hf_base_model_version) > 0
            else None
        ),
        "commit_hash": hf_base_model_commit.commit_id,
        "commit_datetime": \
            hf_base_model_commit.created_at
    }


    ctx.model_version_blessed = False
    ctx.current_blessed_exec = None
    ctx.current_blessed_version_dict = None

    ctx.retrain_pipelines = f"retrain-pipelines {__version__}"
    ctx.retrain_pipeline_type = os.environ["retrain_pipeline_type"]


    ctx.serving_artifacts_local_folder = os.path.realpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "serving_artifacts",
        ctx.pipeline_name, str(ctx.exec_id)
    ))

    if not os.path.exists(ctx.serving_artifacts_local_folder):
        os.makedirs(ctx.serving_artifacts_local_folder)


    ctx.unsloth_dir = os.path.join(
        ctx.serving_artifacts_local_folder,
        "Unsloth"
    )
    logger.debug(f"unsloth_dir : {ctx.unsloth_dir}")
    ctx.cpt_model_dir = os.path.join(ctx.unsloth_dir, "cpt_model")
    ctx.sft_model_dir = os.path.join(ctx.unsloth_dir, "sft_model")

    return None


@task
def eda(_) -> None:
    """
    exploratory data analysis.
    """

    ############################
    #    features and label    #
    #       basic counts       #
    ############################
    ctx.records_count = ctx.hf_dataset_dict["lazy_df"] \
        .select(pl.len()).collect(engine=ctx.engine).item()
    ctx.data_schema = get_column_info(
        ctx.hf_dataset_dict["lazy_df"], engine=ctx.engine)
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
            ctx.hf_dataset_dict["lazy_df"],
            ctx.hf_dataset["attributes"]["answers_attr"],
            struct_schema) \
        .collect(engine=ctx.engine)
    print(f"{tool_answer_occurrences_df['occurrences'].sum():,} " +
          f"query/tool-calls pairs")
    fig = plot_tools_occurences(tool_answer_occurrences_df,
                                title_prefix="Dataset answers - ")
    ctx.answers_tools_count_fig = fig
    ############################

    ############################
    #           Query          #
    #        words count       #
    ############################
    queries_max_length = ctx.hf_dataset_dict["lazy_df"].select(
        pl.col(
            ctx.hf_dataset["attributes"]["query_attr"]
        ).str.len_chars().max().alias("max_query_length")
    ).collect(engine=ctx.engine)
    print(f"longuest query counts " +
          f"{queries_max_length['max_query_length'][0]:,} characters")

    # queries length quartiles
    ctx.query_words_stats = \
        column_words_stats(
            ctx.hf_dataset_dict["lazy_df"],
            ctx.hf_dataset["attributes"]["query_attr"]
        ).collect(engine=ctx.engine)
    print(ctx.query_words_stats.to_pandas().to_string(index=False))
    print("Two thirds of the records have a query with less than " +
          f"{ctx.query_words_stats['q3'][0]} words.")

    fig = plot_words_count(
            ctx.hf_dataset_dict["lazy_df"],
            column_name=ctx.hf_dataset["attributes"]["query_attr"],
            engine=ctx.engine)
    ctx.words_count_fig = fig
    ############################

    ############################
    #     hf_enrich_dataset    #
    #    Query words count     #
    ############################
    enrich_question_words_stats = \
        column_words_stats(
            ctx.hf_enrich_dataset_dict['lazy_df'],
            ctx.hf_enrich_dataset["query_attribute"],
            column_attr_handler=eval(
                ctx.hf_enrich_dataset["query_attribute_handler"])
        ).collect(engine=ctx.engine)
    print(enrich_question_words_stats.to_pandas()
            .to_string(index=False))
    del enrich_question_words_stats
    ############################

    return None


@task
def augment_data(_) -> None:
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
          str(ctx.query_words_stats['q3'][0]) +
          " words (longest queries quartile) =>")

    samples_count = \
        int(ctx.records_count * ctx.augmentation_rate)
    print(f"{ctx.augmentation_rate:.1%} would represent " +
          f"{samples_count:,.0f} records to be sampled")

    eligible_records_df = \
        ctx.hf_dataset_dict["lazy_df"].filter(
            pl.col(
                ctx.hf_dataset["attributes"]["query_attr"]
            )
            .str.extract_all(r"\w+")
            .map_elements(
                lambda arr: len(arr),
                return_dtype=pl.Int16)
            .gt(ctx.query_words_stats['q3'][0])
            & pl.col("answers")
            .map_elements(
                lambda x: len(json.loads(x)) == 1
                          if isinstance(x, str)
                          else False,
                return_dtype=pl.Boolean)  
        ) \
        .collect(engine=ctx.engine)
    eligible_records_count = \
        eligible_records_df.select(pl.len())["len"][0]
    print(f"eligible_records_count : " +
          f"{eligible_records_count:,.0f}")
    samples_count = min(samples_count, eligible_records_count)
    ctx.actual_augmentation_rate = \
        samples_count / ctx.records_count
    print("actual augmentation rate : " +
          f"{ctx.actual_augmentation_rate:.1%}")
    sampled_records_df = eligible_records_df.sample(
        n=samples_count
    )

    ctx.augmented_records_df = \
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
    print(ctx.augmented_records_df.height,
          ctx.augmented_records_df.columns)

    return None


@task
def enrich_data(_) -> None:
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
        path=ctx.hf_enrich_dataset["repo_id"],
        name=ctx.hf_enrich_dataset["config_name"],
        revision=ctx.hf_enrich_dataset_dict["commit_hash"],
        streaming=True)
    print(hf_enrich_ds["train"])

    samples_count = \
        int(ctx.records_count * ctx.enrichment_rate)
    print(f"Samplig {samples_count:,.0f} records")

    query_attribute_handler = \
        eval(ctx.hf_enrich_dataset["query_attribute_handler"])
    samples_iterator = iterable_dataset_multi_buffer_sampler(
            hf_enrich_ds["train"],
            total_samples=samples_count,
            attributes_selector=\
                (lambda x:query_attribute_handler(
                    x[ctx.hf_enrich_dataset["query_attribute"]])),
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
    ctx.enriched_records_df = enriched_records_df

    return None


@task(ui_css=UiCss(background="#FF9900", color="#111827", border="#1F2937"))
def dataset_to_hub(_) -> None:
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
            os.listdir(ctx.pipeline_card_artifacts_path)
    ):
        template_dir = ctx.pipeline_card_artifacts_path
    else:
        template_dir = os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{os.getenv('retrain_pipeline_type')}"
            ).origin)
    print(f"template_dir : '{template_dir}'")
    #############################
    if "dataset_readme.py" in os.listdir(
            ctx.pipeline_card_artifacts_path):
        from retrain_pipelines.utils import \
            get_get_dataset_readme_content
        get_dataset_readme_content = \
            get_get_dataset_readme_content(
                ctx.pipeline_card_artifacts_path)
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
            ctx.hf_dataset_dict["lazy_df"].select([
                    ctx.hf_dataset["attributes"]["query_attr"],
                    ctx.hf_dataset["attributes"]["answers_attr"]
                ]).collect(engine=ctx.engine),
            # truncated queries augmentation
            ctx.augmented_records_df,
            # enriching dataset
            ctx.enriched_records_df
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
    del merged_df
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
            ctx.hf_dataset_dict["lazy_df"],
            tools_attr_name=\
                ctx.hf_dataset["attributes"]["tools_attr"],
            struct_schema=struct_schema
        ).collect(engine=ctx.engine)
    unique_tools_arrow_table = unique_tools_df.to_arrow()
    ctx.unique_tools_dataset = \
        Dataset(unique_tools_arrow_table)
    print(ctx.unique_tools_dataset)
    #############################

    #############################
    #        DatasetDict        #
    #    with multiple tables   #
    #############################
    dataset_dict = DatasetDict({
        "continued_pre_training": \
            ctx.unique_tools_dataset,
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
        repo_id=ctx.dataset_repo_id,
        repo_type="dataset",
        hf_token=os.getenv("HF_TOKEN", None))
    readme_content = get_dataset_readme_content(
        template_folder=template_dir,

        hf_dataset_dict=ctx.hf_dataset_dict,
        hf_enrich_dataset_dict=ctx.hf_enrich_dataset_dict,
        dataset_dict=dataset_dict,

        augmentation_rate=ctx.actual_augmentation_rate,
        enrichment_rate=ctx.enrichment_rate,

        version_label=new_dataset_version_label,
        commit_datetime=commit_datetime,

        pipeline_name=ctx.pipeline_name,
        exec_id=ctx.exec_id,
        engine=ctx.engine
    )
    #############################

    dataset_commit_hash = push_dataset_version_to_hub(
        repo_id=ctx.dataset_repo_id,
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
    print(f"https://huggingface.co/datasets/{ctx.dataset_repo_id}" +
          f"/blob/{dataset_commit_hash}/README.md")
    ctx.dataset_commit_dict = {
        "repo_id": ctx.dataset_repo_id,
        "commit_hash": dataset_commit_hash,
        "version_label": new_dataset_version_label,
        "commit_datetime": commit_datetime,
    }

    return None


@task
def continued_pre_training(_) -> None:
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
    ctx.max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ctx.hf_base_model_dict["repo_id"],
        revision=ctx.hf_base_model_dict["commit_hash"],
        max_seq_length=ctx.max_seq_length,
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
        {"train": ctx.unique_tools_dataset})
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

    import dis as _dis_module
    # silence dis.py bytecode spam (Unsloth monkey-patch side-effect)
    _dis_module.print = lambda *a, **kw: None
    cpt_dataset = tools_dataset["train"].map(
        formatting_prompts_func, batched=True,)
    del _dis_module.print
    del tools_dataset
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
        "records_cap" in ctx.cpt_training_args and
        ctx.cpt_training_args["records_cap"] is not None and
        isinstance(ctx.cpt_training_args["records_cap"], int)
    ):
        cpt_dataset = cpt_dataset.take(
            ctx.cpt_training_args["records_cap"])
        print(f"cpt_dataset : {cpt_dataset}")

    train_args = UnslothTrainingArguments(
        # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.save_strategy
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        **{k: v for k, v in ctx.cpt_training_args.items()
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
            ctx.unsloth_dir, "outputs", "cpt"),
        save_total_limit = 2,

        report_to="tensorboard",
        logging_dir=os.path.join(
            ctx.sft_model_dir,
            "runs", "cpt")
    )

    # silence dis.py bytecode spam (Unsloth monkey-patch side-effect)
    _dis_module.print = lambda *a, **kw: None
    trainer = UnslothTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=cpt_dataset,
        dataset_text_field="tools",
        max_seq_length=ctx.max_seq_length,
        dataset_num_proc=2,
        args=train_args,
    )
    del _dis_module.print
    #######################################

    #######################################
    #      Show current memory stats      #
    #######################################
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    _ = gc.collect()

    gpu_stats = torch.cuda.get_device_properties(0)
    ctx.start_gpu_memory = \
        round(torch.cuda.max_memory_reserved()
              / 1024 / 1024 / 1024, 3)
    ctx.max_memory = \
        round(gpu_stats.total_memory
              / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. " +
          f"Max memory = {ctx.max_memory} GB.")
    print(f"{ctx.start_gpu_memory} GB of memory reserved.")
    #######################################

    ctx.cpt_traces_file_fullname = os.path.join(
        ctx.unsloth_dir, "cpt_trainer_traces.txt")
    logger.info(
        "Training started. " +
        f"Check [underline]{ctx.cpt_traces_file_fullname}[/] for live traces " +
        "or go watch your [white bold]TensorBoard[/] charts live updates !"
    )
    with open(ctx.cpt_traces_file_fullname, 'w') as f:
        with rp_redirect_stdout(f):
            trainer_stats = trainer.train()
    print(f"{trainer_stats.metrics['train_runtime']} " +
          f"seconds used for CPT training " +
          f"({round(trainer_stats.metrics['train_runtime']/60, 2)}" +
          f" minutes).")

    ctx.cpt_log_history = trainer.state.log_history
    ctx.cpt_log_history_fig = \
        plot_log_history(
            ctx.cpt_log_history,
            title="Continued pretraining loss"
        )
    del trainer
    # logger.debug(f"Continued pretraining loss curve : {ctx.cpt_log_history}")

    model.save_pretrained_merged(
        save_directory=ctx.cpt_model_dir,
        tokenizer=tokenizer,
        save_method="lora"
    )
    print(f"cpt_model_dir : {ctx.cpt_model_dir}\n")

    # vRAM & RAM cleanup
    # (incl. force-delete all CUDA tensors in gc)
    del model  
    del tokenizer
    clear_gc()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"After cleanup: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    return None


@task
def supervised_finetuning(_) -> None:
    """
    Trains the model on tool-calling
    task specialization.
    """
    from retrain_pipelines.model.hf_utils import \
        plot_log_history

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ctx.cpt_model_dir,
        max_seq_length=ctx.max_seq_length,
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
        path=ctx.dataset_commit_dict["repo_id"],
        name="supervised_finetuning",
        revision=ctx.dataset_commit_dict["commit_hash"],
        token=os.getenv("HF_TOKEN", None))
    print(f"HF_DATASETS_CACHE : {HF_DATASETS_CACHE}") # HF_CACHE_HOME
    ctx.sft_prompt_template = dedent("""
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
    tokenizer.chat_template = ctx.sft_prompt_template

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(records):
        query = records["query"]
        tools  = records["answers"]
        outputs = []
        for query, tools in zip(query, tools):
            # Must add EOS_TOKEN,
            # otherwise your generation will go on forever
            text = ctx.sft_prompt_template.format(query, tools) \
                   + EOS_TOKEN
            outputs.append(text)
        return { "text" : outputs, }

    import dis as _dis_module
    # silence dis.py bytecode spam (Unsloth monkey-patch side-effect)
    _dis_module.print = lambda *a, **kw: None
    sft_dataset = queries_dataset["train"].map(
        formatting_prompts_func, batched=True)
    del _dis_module.print
    del queries_dataset
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
    split = sft_dataset.train_test_split(
        test_size=1000,
        #seed=42
    )
    train_dataset = split['train']
    eval_dataset = split['test']
    del sft_dataset
    if (
        "records_cap" in ctx.sft_training_args and
        ctx.sft_training_args["records_cap"] is not None and
        isinstance(ctx.sft_training_args["records_cap"], int)
    ):
        train_dataset = train_dataset.take(
            ctx.sft_training_args["records_cap"])
        eval_dataset = eval_dataset.take(
            ctx.sft_training_args["records_cap"])
        print(f"train_dataset : {train_dataset}")
        print(f"eval_dataset :  {eval_dataset}")

    train_args = UnslothTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        **{k: v for k, v in ctx.sft_training_args.items()
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
            ctx.unsloth_dir, "outputs", "sft"),
        save_total_limit=2,

        disable_tqdm=True,
        logging_steps=1,
        report_to="tensorboard",
        logging_dir=os.path.join(
            ctx.sft_model_dir,
            "runs", "sft")
    )

    # silence dis.py bytecode spam (Unsloth monkey-patch side-effect)
    _dis_module.print = lambda *a, **kw: None
    trainer = UnslothTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=eval_dataset,
        max_seq_length=ctx.max_seq_length,
        dataset_num_proc=8,
        args=train_args
    )
    del _dis_module.print
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
        round(used_memory-ctx.start_gpu_memory, 3)
    used_percentage = \
        round(used_memory/ctx.max_memory*100, 3)
    lora_percentage = \
        round(used_memory_for_lora/ctx.max_memory*100,
              3)
    print(f"Peak reserved memory = " +
          f"{used_memory} GB.")
    print(f"Peak reserved memory for " +
          f"training = {used_memory_for_lora} " +
          f"GB.")
    print(f"Peak reserved memory % of " +
          f"max memory = {used_percentage} %.")
    print(f"Peak reserved memory for SFT training " +
          f"% of max memory = {lora_percentage} %.")
    #######################################

    ctx.sft_traces_file_fullname = os.path.join(
        ctx.unsloth_dir, "sft_trainer_traces.txt")
    logger.info(
        "Training started. " +
        f"Check [underline]{ctx.sft_traces_file_fullname}[/] for live traces " +
        "or go watch your [white bold]TensorBoard[/] charts live updates !"
    )
    with open(ctx.sft_traces_file_fullname, 'w') as f:
        with rp_redirect_stdout(f):
            trainer_stats = trainer.train()
    print(f"{trainer_stats.metrics['train_runtime']} " +
          f"seconds used for training " +
          f"({round(trainer_stats.metrics['train_runtime']/60, 2)}" +
          f" minutes).")

    ctx.sft_log_history = trainer.state.log_history
    ctx.sft_log_history_fig = \
        plot_log_history(
            ctx.sft_log_history,
            title="Supervised finetuning loss"
        )
    del trainer

    model.save_pretrained_merged(
        ctx.sft_model_dir, tokenizer,
        save_method = "lora"
    )
    print(f"sft_model_dir : {ctx.sft_model_dir}\n")

    # vRAM & RAM cleanup
    # (incl. force-delete all CUDA tensors in gc)
    del model  
    del tokenizer
    clear_gc()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"After cleanup: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    return None


@task
def evaluate_model(_) -> None:
    """
    Batch inference on the SFT validation dataset.
    """
    from retrain_pipelines.model import \
        infer_validation, compute_counts_n_metrics, \
        plot_validation_completions

    ######################################################
    #                 validation dataset                 #
    ######################################################
    # download from Hub (or get from local cache)
    queries_dataset = load_dataset(
        path=ctx.dataset_commit_dict["repo_id"],
        name="supervised_finetuning",
        revision=ctx.dataset_commit_dict["commit_hash"],
        token=os.getenv("HF_TOKEN", None))
    if (
        "records_cap" in ctx.sft_training_args and
        ctx.sft_training_args["records_cap"] is not None and
        isinstance(ctx.sft_training_args["records_cap"], int)
    ):
        validation_data = queries_dataset["validation"].take(
            ctx.sft_training_args["records_cap"])
    else:
        validation_data = queries_dataset["validation"]
    del queries_dataset
    logger.debug(validation_data)
    ######################################################

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
        model_name=ctx.hf_base_model_dict["repo_id"],
        revision=ctx.hf_base_model_dict["commit_hash"],
        max_seq_length=ctx.max_seq_length,
        dtype=None,
        load_in_4bit=False,
        # case of a gated or private base-model
        token=os.getenv("HF_TOKEN", None)
    )
    model = FastLanguageModel.for_inference(model)
    # load our CPT+SFT trained & locally-saved adapter
    model.load_adapter(peft_model_id=ctx.sft_model_dir)
    # Separately load our (potentially trained &)
    # locally-saved adapter-tokenizer
    # (loading it below via HF and not Unsloth)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=ctx.sft_model_dir
    )
    ######################################################

    ctx.max_new_tokens = 400
    start_time = time.time()
    validation_results = infer_validation(
        tokenizer=tokenizer,
        model=model,
        validation_data=validation_data,
        prompt_template=tokenizer.chat_template,
        batch_size=32, # 64,
        queries_attr_name=\
            ctx.hf_dataset["attributes"]["query_attr"],
        answers_attr_name=\
            ctx.hf_dataset["attributes"]["answers_attr"],
        max_new_tokens=ctx.max_new_tokens,
        device="cuda"
    )
    print("infer_validation -   Elapsed time: " +
          f"{(time.time() - start_time):.2f} seconds")
    ctx.validation_results = validation_results #  <= to artifacts store

    eval_df  = pl.LazyFrame(validation_results)

    records = eval_df.with_columns(
        (pl.col("answer") == pl.col("completion")) \
            .alias("is_ground_truth_identical")
    ).collect() #engine=ctx.engine)
    print("perfect characters-match accuracy : " +
          str(records['is_ground_truth_identical'].mean()))

    eval_metrics_df = compute_counts_n_metrics(
        eval_df, is_format_fault_tolerant=True)
    del eval_df
    overall_metrics_df = eval_metrics_df.select([
            pl.col("precision").mean(),
            pl.col("recall").mean(), 
            pl.col("f1").mean(),
            pl.col("jaccard").mean()
        ]).collect() #engine=ctx.engine)
    ctx.perf_metrics = overall_metrics_df.row(0, named=True)
    del overall_metrics_df
    print(ctx.perf_metrics)

    ctx.validation_completions_fig = \
        plot_validation_completions(
            eval_metrics_df, engine=ctx.engine)

    # vRAM & RAM cleanup
    # (incl. force-delete all CUDA tensors in gc)
    del model  
    del tokenizer
    clear_gc()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"After cleanup: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    return None


@task
def model_version_blessing(_) -> None:
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
            repo_id=ctx.model_repo_id,
            hf_token=os.getenv("HF_TOKEN", None)
        )
    print("current_blessed_version_dict : " +
          str(current_blessed_version_dict))

    if current_blessed_version_dict is None:
        print("case 'no prior blessed model version found"
              " => blessing.'")
        ctx.model_version_blessed = True

    elif (
        main_perf_metric_name in
            current_blessed_version_dict["perf_metrics"]
    ):
        current_blessed_exec_id = \
            current_blessed_version_dict["exec_id"]
        print(f"current_blessed_exec_id : {current_blessed_exec_id}")
        current_blessed_metric_value = \
            current_blessed_version_dict[
                "perf_metrics"][main_perf_metric_name]

        ctx.model_version_blessed = (
            ctx.perf_metrics[main_perf_metric_name] >=
            current_blessed_metric_value
        )

        ctx.model_version_blessed = False ### DEBUG - DELETE ###

        if not ctx.model_version_blessed:
            ctx.current_blessed_version_dict = \
                current_blessed_version_dict
            # may have failed after the "pipeline_card" task,
            # so we do not filter on success
            for execution in ExecutionsIterator(
                exec_name=ctx.pipeline_name,
                page_size=10
            ):
                if str(execution.id) == current_blessed_exec_id:
                    # Has the execution seen task "pipeline_card" which
                    # completed successfully
                    # ("execution" has generated a custom pipeline-card artifact) ?
                    # If not, hyperlink generation will later fail.
                    execution_has_custom_card_artifact = (len([
                        t for t in execution.get_tasks_with_name(
                                        task_type_name="pipeline_card")
                        if t.end_timestamp and t.success
                    ]) == 1)
                    if not execution_has_custom_card_artifact:
                        logger.warning(
                            f"Execution #{current_blessed_exec_id} " +
                            "Doesn't seem to have successfully " +
                            "generated a pipeline-card artifact.")

                    else:
                        # further filtering on successful executions that are
                        # retraining of a prior version of the same model
                        # (to minimize the risk that this was obtained
                        #  on another DAG-engine instance)
                        if (
                            execution.get_attr("model_version_blessed") and
                            execution.get_attr("model_repo_id") or "" == \
                                ctx.model_repo_id
                        ):
                            ctx.current_blessed_exec = execution

                    break

            if not ctx.current_blessed_exec:
                logger.warning(
                    "Couldn't find blessed execution " +
                    f"{current_blessed_exec_id} !\n" +
                    "It seems that prior blessed execution was " +
                    "executed on another DAG-engine instance.")
            else:
                logger.debug(
                    f"ctx.current_blessed_exec : {ctx.current_blessed_exec}")

        print("new : " +
                str(ctx.perf_metrics[main_perf_metric_name]) +
              " - previous best : " +
                str(current_blessed_metric_value) +
              " - model_version_blessing : " +
                str(ctx.model_version_blessed))

    else:
        raise Exception(
            "Performance metric '" +
            main_perf_metric_name +
            "' can't be found in eval results " +
            "from blessed execution " +
            str(current_blessed_version_dict[
                "exec_id"]) + " !")

    return None


@task(ui_css=UiCss(background="#FF9900", color="#111827", border="#1F2937"))
def model_to_hub(_) -> None:
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
            os.listdir(ctx.pipeline_card_artifacts_path)
    ):
        template_dir = ctx.pipeline_card_artifacts_path
    else:
        template_dir = os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{os.getenv('retrain_pipeline_type')}"
            ).origin)
    print(f"template_dir : '{template_dir}'")
    #############################
    if "model_readme.py" in os.listdir(
            ctx.pipeline_card_artifacts_path):
        from retrain_pipelines.utils import \
            get_get_model_readme_content
        get_model_readme_content = \
            get_get_model_readme_content(
                ctx.pipeline_card_artifacts_path)
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
        repo_id=ctx.model_repo_id,
        repo_type="model",
        hf_token=os.getenv("HF_TOKEN", None))
    readme_content = get_model_readme_content(
        template_folder=template_dir,

        model_repo_id=ctx.model_repo_id,

        base_model_dict=ctx.hf_base_model_dict,
        training_dataset_dict=ctx.dataset_commit_dict,

        version_label=new_model_version_label,
        commit_datetime=commit_datetime,
        perf_metrics=ctx.perf_metrics,

        pipeline_name=ctx.pipeline_name,
        exec_id=ctx.exec_id
    )
    #############################

    print("Pushing model version to HF hub " +
          ("(blessed). " if ctx.model_version_blessed
           else "(not blessed). ") +
          "May take a while..",
          flush=True)
    model_commit_hash = push_model_version_to_hub(
        repo_id=ctx.model_repo_id,
        model_version_blessed=\
            ctx.model_version_blessed,
        version_label=new_model_version_label,
        timestamp_str=commit_datetime.strftime(
            "%Y-%m-%d %H:%M:%S UTC"),
        model_dir=ctx.sft_model_dir,
        model_readme_content=readme_content,
        hf_token=os.getenv("HF_TOKEN", None)
    )
    if not model_commit_hash:
        raise Exception(
            "Failed to publish model version.")
    print("Push of model version to HF hub completed.",
          flush=True)
    print(f"https://huggingface.co/{ctx.model_repo_id}" +
          f"/blob/{model_commit_hash}/README.md")

    ctx.model_commit_dict = {
        "repo_id": ctx.model_repo_id,
        "commit_hash": model_commit_hash,
        "version_label": new_model_version_label,
        "commit_datetime": commit_datetime,
    }

    return None


@task
def infra_validator(_) -> None:
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

    ctx.local_serve_is_ready = LocalServeReadinessEnum.NOT_APPLICABLE

    if ctx.model_version_blessed:
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
                    os.path.join(ctx.serving_artifacts_local_folder,
                                 filename)
                )

            # save dependencies as artifact
            create_requirements(ctx.serving_artifacts_local_folder,
                                exclude=["numpy",  # version conflict
                                                   # quick fix
                                         "cudf-polars-.*", "cuda-python",
                                         "nvidia-.*", "(py)?libcudf-.*",
                                         "nvtx", "rmm-.*", "litserve",
                                         "protobuf", "grpc.*",
                                         "tensorboard",
                                         ".*retrain-pipelines.*"]
            )

            # server config yaml
            env = Environment(loader=FileSystemLoader(
                os.path.join(model_module_dir, "litserve")))
            template = env.get_template(
                "litserve_serverconfig_template.yaml")
            server_config_data = {
                "port": "8000",
                "max_seq_length": ctx.max_seq_length,
                "max_new_token": ctx.max_new_tokens,
                "base_model": {
                    "repo_id": ctx.hf_base_model_dict["repo_id"],
                    "revision": ctx.hf_base_model_dict["commit_hash"]
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
                ctx.serving_artifacts_local_folder,
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
                ctx.serving_artifacts_local_folder,
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

            ctx.port = 8765
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
                build_path=ctx.serving_artifacts_local_folder,
                dockerfile="Dockerfile.litserve",
                ports_publish_dict={'8000/tcp': ctx.port},
                env_vars_dict={
                    "HF_HUB_CACHE": "/huggingface_hub_cache",
                    "HF_TOKEN": os.getenv("HF_TOKEN")
                },
                volumes_dict={
                    ctx.sft_model_dir:
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
                ctx.local_serve_is_ready = \
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
                    container_name, port=ctx.port, timeout=10*60
                ):
                    print(
                        f"The endpoint '{container_name}' " +
                        f"did not start.")
                    ctx.local_serve_is_ready = \
                        LocalServeReadinessEnum.FAILURE
                # health check on the spun-up endpoint
                elif endpoint_is_ready(port=ctx.port):
                    ctx.local_serve_is_ready = \
                        LocalServeReadinessEnum.SUCCESS
            elapsed_time = time.time() - start_time
            print("deploy_local -   Elapsed time: " +
                  f"{elapsed_time:.2f} seconds")
            ############################################
        else:
            # env doesn't have docker
            ctx.local_serve_is_ready = \
                LocalServeReadinessEnum.FAILURE_NO_DOCKER

        if LocalServeReadinessEnum.SUCCESS == ctx.local_serve_is_ready:
            from retrain_pipelines.model.litserve.litserve_datamodel \
                import Response

            import requests

            url = f"http://localhost:{ctx.port}/predict"
            headers = {"accept": "application/x-www-form-urlencoded"}

            try:
                start_time = time.time()
                data = {
                    "adapter_name": "func_caller",
                    "queries_list": '["Hello.", "Is 49 a perfect square?"]'
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
                    "queries_list": '["Hello.", "Is 49 a perfect square?"]'
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
                ctx.local_serve_is_ready = \
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
    else:
        logger.info("model-version not blessed - skipping")

    return None


@task
def pipeline_card(_, task_id: int) -> None:
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
                            ctx.pipeline_card_artifacts_path
    ):
        template_dir = ctx.pipeline_card_artifacts_path
    else:
        template_dir = os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{os.getenv('retrain_pipeline_type')}"
            ).origin)
    #############################
    if "pipeline_card.py" in os.listdir(
                                ctx.pipeline_card_artifacts_path
    ):
        from retrain_pipelines.utils import get_get_html
        get_html = \
            get_get_html(ctx.pipeline_card_artifacts_path)
    else:
        from retrain_pipelines.pipeline_card import \
                get_html
    from retrain_pipelines.dag_engine.renderer import dag_svg
    #############################

    #############################
    ##    html "custom" card   ##
    #############################
    dt = datetime.now(tz=timezone.utc)
    formatted_dt = dt.strftime("%A %b %d %Y %I:%M:%S %p %Z")
    task_obj_python_cmd = f"sdk.Task({task_id})"
    executions_count = ExecutionsIterator(
        exec_name=ctx.pipeline_name).length()

    params={
        'template_dir': template_dir,
        'title': ctx.pipeline_name,
        "subtitle": f"(Pipeline execution # {executions_count}," + \
                    f" exec_id: {str(ctx.exec_id)}  -  {formatted_dt})",

        # blessed status / current_blessed version
        'model_version_blessed': ctx.model_version_blessed,
        'current_blessed_version_label': (
            ctx.current_blessed_version_dict["version_label"]
            if ctx.current_blessed_version_dict
            else None
        ),
        'current_blessed_commit_datetime': (
            ctx.current_blessed_version_dict["commit_datetime"]
            if ctx.current_blessed_version_dict
            else None
        ),
        'current_blessed_model_commit_hash': (
            ctx.current_blessed_version_dict["commit_hash"]
            if ctx.current_blessed_version_dict
            else None
        ),
        'current_blessed_run': ctx.current_blessed_exec,

        'LocalServeReadinessEnum': LocalServeReadinessEnum,
        'local_serve_is_ready': ctx.local_serve_is_ready,
        # EDA
        'main_dataset_repo_id': ctx.hf_dataset['repo_id'],
        'main_dataset_commit_hash': ctx.hf_dataset_dict['commit_hash'],
        'main_dataset_commit_datetime': \
            ctx.hf_dataset_dict['commit_datetime'],

        'records_count': ctx.records_count,
        'data_schema': ctx.data_schema,
        'answers_tools_count_fig': ctx.answers_tools_count_fig,
        'words_count_fig': ctx.words_count_fig,

        # model training
        'dataset_repo_id': ctx.dataset_repo_id,
        'dataset_version_label': ctx.dataset_commit_dict["version_label"],
        'dataset_commit_datetime': ctx.dataset_commit_dict["commit_datetime"],
        'dataset_commit_hash': ctx.dataset_commit_dict["commit_hash"],
        'dataset_augmentation_rate': ctx.actual_augmentation_rate,
        'dataset_enrichment_rate': ctx.enrichment_rate,

        # trained model version
        'model_repo_id': ctx.model_repo_id,
        'model_version_label': ctx.model_commit_dict["version_label"],
        'model_commit_datetime': ctx.model_commit_dict["commit_datetime"],
        'model_commit_hash': ctx.model_commit_dict["commit_hash"],

        'cpt_log_history_fig': ctx.cpt_log_history_fig,
        'sft_log_history_fig': ctx.sft_log_history_fig,

        'validation_completions_fig': ctx.validation_completions_fig,

        'hf_base_model_dict': ctx.hf_base_model_dict,
        'pipeline_parameters_dict': {"cpt": ctx.cpt_training_args,
                                     "sft": ctx.sft_training_args},

        'metrics_dict': ctx.perf_metrics,

        'task_obj_python_cmd': task_obj_python_cmd,
        'dag_svg': dag_svg(execution_id=ctx.exec_id)
    }
    html = get_html(params)

    filename = os.path.join(
        os.environ["RP_ARTIFACTS_STORE"],
        ctx.pipeline_name, str(ctx.exec_id),
        "pipeline_card.html"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(html)
    logger.debug(
        "pipeline_card - " +
        f"[bold]pipeline_card_file_fullname : {filename}[/]")

    ctx.pipeline_card_file_fullname = filename
    #############################

    return None


@task(ui_css=UiCss(background="#FF9900", color="#111827", border="#1F2937"))
def pipeline_to_hub(_) -> None:
    """
    publish versioned source-code and pipeline-card
    for ths run on the Hugging Face Hub.
    """
    model_commit_datetime = \
        ctx.model_commit_dict["commit_datetime"]
    timestamp_str = \
        "{:%Y%m%d_%H%M%S}".format(model_commit_datetime) + \
        "{:03d}".format(model_commit_datetime.microsecond//1000) + \
        "_UTC"
    subfolder_name = \
        "v" + ctx.model_commit_dict["version_label"] + \
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
        ctx.pipeline_card_artifacts_path != \
        ctx.params_definitions["pipeline_card_artifacts_path"].default
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
                ctx.pipeline_card_artifacts_path,
                candidate_source_file)
            if os.path.exists(file_fullpath):
                custom_source_files.append(file_fullpath)

    source_code_commit_hash = \
        push_files_to_hub_repo_branch(
            repo_id=ctx.model_repo_id,
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
    ctx.source_code_commit_dict = {
        "repo_id": ctx.model_repo_id,
        "branch_name": "retrain-pipelines_source-code",
        "commit_datetime": commit_datetime,
        "commit_hash": source_code_commit_hash
    }
    ###############################

    ###############################
    #        pipeline-card        #
    ###############################
    pipeline_card_commit_hash = \
        push_files_to_hub_repo_branch(
            repo_id=ctx.model_repo_id,
            branch_name="retrain-pipelines_pipeline-card",
            file_fullnames=[ctx.pipeline_card_file_fullname],
            path_in_repo=subfolder_name,
            commit_message=\
                "pipeline-card for model version " + \
                subfolder_name + \
                f"- retrain-pipelines {__version__}",
            repo_type="model",
            hf_token=os.getenv("HF_TOKEN", None)
        )
    print(pipeline_card_commit_hash)
    ctx.pipeline_card_commit_dict = {
        "repo_id": ctx.model_repo_id,
        "branch_name": "retrain-pipelines_pipeline-card",
        "commit_datetime": commit_datetime,
        "commit_hash": pipeline_card_commit_hash
    }
    ###############################

    return None


@task
def deploy(_):
    """
    placeholder for the serving SDK "deploy" call
    (on the target production platform).
    consider including the portable pipelione-card
    itself to the inference service endpoint !
    """

    if ctx.model_version_blessed and (ctx.local_serve_is_ready == 1):
        pass # your code here

    return None


@task
def load_test(_):
    """
    placeholder
    """

    if ctx.model_version_blessed and (ctx.local_serve_is_ready == 1):
        pass # your code here

    return None


@task
def end(_):
    pass


#--- retraining-pipeline params & DAG ---------------------------------------------------


@dag(ui_css=UiCss(color="#FFDD00", background="#7AD4FF", border="#C28E00"))
def retrain_pipeline():
    """
    Retraining pipeline with SFT & CPT. Small LLM with pluggable adapter specialized in tool-calling from intrinsic knowledge bank of tools and not from extended context. Model-version blessing. Serving via a custom LitServe toy-server.
    """
    # @see https://github.com/unslothai/unsloth/wiki

    #--- flow parameters -------------------------------------------------------


    RETRAIN_PIPELINE_TYPE = "mf_unsloth_func_call_litserve"
    # best way to share the config across subprocesses
    os.environ["retrain_pipeline_type"] = RETRAIN_PIPELINE_TYPE

    hf_dataset = DagParam(
        description="dict with 'repo_id' and 'commit_hash' keys. " + \
                    "if 'commit_hash is None, falls back to latest version " +\
                    "of the dataset available in parquet format.\n" +
                    "Note that there are 3 required 'attributes' of type " + \
                    "str, list[str], list[str]",
        default={
            "repo_id": "Salesforce/xlam-function-calling-60k",
            "config_name": "",
            "commit_hash": "",
            "attributes": {
                "query_attr": "query",
                "answers_attr": "answers",
                "tools_attr": "tools"
            }
        }
    )

    augmentation_rate = DagParam(
        description="(float) proportion of records to be augmented " + \
                    "(x% of original dataset is created" + \
                    " as additional augmented datapoints), i.e. " + \
                    "truncated queries to serve as negative examples, " + \
                    "meaning they trigger no tool call " + \
                    "due to info incompleteness.",
        default=.05
    )

    hf_enrich_dataset = DagParam(
        description="dict with 'repo_id', 'config_name' and 'commit_hash', " + \
                    "query_attribute' and 'query_attribute_handler' keys. " + \
                    "if 'commit_hash is None, falls back to latest version " + \
                    "of the dataset available in parquet format." + \
                    "'query_attribute' depicts the dataset attribute " + \
                    "from which 'queries' are to be sampled." + \
                    "'query_attribute_handler' serves for attributes " + \
                    "that have complex structure, " + \
                    "other than 'string' datatype.",
        # @see https://huggingface.co/datasets/google-research-datasets/natural_questions
        default={
            "repo_id": "lighteval/natural_questions_clean",
            "config_name": "",
            "commit_hash": "",
            "query_attribute": "question",
            "query_attribute_handler": "lambda x: x"
        }
    )

    enrichment_rate = DagParam(
        description="(float) proportion of records " + \
                    "to be added from the 'hf_enrich_dataset'" + \
                    "(x% of original dataset is sampled and" + \
                    " added as enriching datapoints), i.e. " + \
                    "queries to serve as negative examples, " + \
                    "due to their complete disconnexion " + \
                    "to tool calling situations.",
        default=.1
    )

    dataset_repo_id = DagParam(
        description="(str) The 'repo_id' to be used " + \
                    "for the Hugging Face dataset version push " + \
                    "(will be created at runtime" + \
                    " if doesn't already exist).",
        default="retrain-pipelines/func_calls"
    )

    polars_engine = DagParam(
        description="The engine used by Polars for " + \
                    "dataset querying and processing " + \
                    "(either 'gpu' or 'cpu').",
        default="gpu"
    )

    hf_base_model = DagParam(
        description="(str) dict with 'repo_id' and 'commit_hash' keys." + \
                    "if 'commit_hash is None, falls back " + \
                    "to latest available version of the model.",
        default={
            "repo_id": "unsloth/Qwen2.5-1.5B",
            "commit_hash": ""
        }
    )

    cpt_training_args = DagParam(
        description="dict with `TrainingArguments` params " + \
                    "for the CPT job.",
        default={
            "warmup_ratio": 0.1,
            "num_train_epochs": 1
        }
    )

    sft_training_args = DagParam(
        description="dict with `TrainingArguments` params " + \
                    "for the SFT job.",
        default={
            "warmup_ratio": 0.1,
            "num_train_epochs": 1
        }
    )

    model_repo_id = DagParam(
        description="(str) The 'repo_id' to be used " + \
                    "for the Hugging Face model version push " + \
                    "(will be created at runtime" + \
                    " if doesn't already exist).",
        default="retrain-pipelines/function_caller"
    )

    default_pipeline_card_module_dir = \
        os.path.dirname(
            importlib.util.find_spec(
                f"retrain_pipelines.pipeline_card."+
                f"{RETRAIN_PIPELINE_TYPE}"
            ).origin)
    pipeline_card_artifacts_path = DagParam(
        description="pipeline_card artifacts location " + \
                    "(i.e. dir hosting your optional " + \
                    " custom documentation files :" + \
                    " 'pipeline_card.py' and/or 'template.html'" + \
                    " and/or 'model_readme.py'"+\
                    " and/or 'model_readme_template.md'," + \
                    " and/or 'dataset_readme.py'" + \
                    " and/or 'dataset_readme_template.md' file), " + \
                    "if different from default.",
        default=default_pipeline_card_module_dir
    )
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def copy_default_dataset_readme_module(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir, "dataset_readme.py"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # default_pipeline_card_module_dir,
                    # "dataset_readme.py"
                # )
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def copy_default_dataset_readme_template(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir,
                                        # "dataset_readme_template.md"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # default_pipeline_card_module_dir,
                    # "dataset_readme_template.md")
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def copy_default_model_readme_module(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir, "model_readme.py"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # default_pipeline_card_module_dir,
                    # "model_readme.py"
                # )
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def copy_default_model_readme_template(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir,
                                        # "model_readme_template.md"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # default_pipeline_card_module_dir,
                    # "model_readme_template.md")
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def copy_default_pipeline_card_module(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir, "pipeline_card.py"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # default_pipeline_card_module_dir,
                    # "pipeline_card.py"
                # )
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)
    # TODO  -  convert from class method to TBD
    # @staticmethod
    # def copy_default_pipeline_card_html_template(
        # target_dir: str,
        # exists_ok: bool = False
    # ) -> None:
        # os.makedirs(target_dir, exist_ok=True)
        # if (
            # not exists_ok and
            # os.path.exists(os.path.join(target_dir, "template.html"))
        # ):
            # print("File already exists. Skipping copy.")
        # else:
            # filefullname = os.path.join(
                    # default_pipeline_card_module_dir,
                    # "template.html")
            # shutil.copy(filefullname, target_dir)
            # print(filefullname)

    del RETRAIN_PIPELINE_TYPE

    #---------------------------------------------------------------------------

    return start >> eda \
            >> augment_data >> enrich_data >> dataset_to_hub \
            >> continued_pre_training >> supervised_finetuning \
            >> evaluate_model >> model_version_blessing \
            >> model_to_hub >> infra_validator >> pipeline_card \
            >> pipeline_to_hub >> deploy >> load_test >> end

