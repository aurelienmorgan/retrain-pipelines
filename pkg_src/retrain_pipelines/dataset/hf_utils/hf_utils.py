
import os
import re
import sys
import json
import yaml
import random

import pandas as pd
import polars as pl

from datetime import datetime

from typing import Optional, Callable, Iterator

from huggingface_hub import list_repo_refs, list_repo_commits, \
    list_repo_files, hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, \
    RevisionNotFoundError, EntryNotFoundError

from datasets import IterableDataset, DatasetDict


def _dataset_repo_branch_commits_files(
    repo_id: str,
    repo_branch: str
) -> dict:
    """
    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - repo_branch (str):
            Branch (of the repository  of interest)
            to be considered.

    Results:
        - (dict)
            'commit_hash', 'created_at',
            'title', 'files'
    """
    commits = list_repo_commits(repo_id, revision=repo_branch,
                                repo_type="dataset",
                                token=os.environ["HF_TOKEN"])
    commits_dict = {}
    for commit in commits:
        files = list_repo_files(
            repo_id, revision=commit.commit_id,
            repo_type="dataset",
            token=os.environ["HF_TOKEN"])

        commits_dict[commit.commit_id] = {
            "created_at": commit.created_at.strftime(
                "%Y-%m-%d %H:%M:%S UTC"),
            "title": commit.title,
            "files": files
        }

    return commits_dict


def get_dataset_branches_commits_files(
    repo_id: str
) -> dict:
    """
    Selection of metadata for (litterally)
    all files of all commits of a given
    HF dataset repo.

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.

    Results:
        - (dict)
            'branches'
                (
                    'branch_name', 'commits',
                    (
                        'commit_hash', 'created_at',
                        'title', 'files'
                    )
                )
    """

    refs = list_repo_refs(repo_id, repo_type="dataset",
                          token=os.environ["HF_TOKEN"])

    dataset_repo_branches = {
        "repo_standard_branches": {},
        "repo_convert_branches": {}
    }
    for repo_standard_branches in refs.branches:
        dataset_repo_branches[
            "repo_standard_branches"
        ][repo_standard_branches.name] = {
            "branch_name": repo_standard_branches.ref,
            "commits": _dataset_repo_branch_commits_files(
                repo_id, repo_standard_branches.ref)
        }
        
    for repo_convert_branch in refs.converts:
        dataset_repo_branches[
            "repo_convert_branches"
        ][repo_convert_branch.name] = {
            "branch_name": repo_convert_branch.ref,
            "commits": _dataset_repo_branch_commits_files(
                repo_id, repo_convert_branch.ref)
        }

    return dataset_repo_branches


def get_latest_commit(
    repo_id: str,
    files_filter: str = ".*"
) -> dict:
    """
    Get the dataset version info for the latest commit
    for a given HF Dataset repo.
    Focus put on files matching the given regex pattern
    with associated metadata.

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - files_filter (str):
            only consider commits with any file
            matching this regex pattern.

    Results:
        - (dict):
            'commit_hash', 'commit_date',
            'branch_name', 'files'
    """

    dataset_repo_branches = \
        get_dataset_branches_commits_files(repo_id)

    latest_matching_commit = None
    regex_pattern = re.compile(files_filter)

    for branch_type, branches in dataset_repo_branches.items():
        for branch_name, branch_data in branches.items():
            for \
                commit_hash, commit_data \
                in branch_data["commits"].items() \
            :
                matching_files = [
                    f for f in commit_data["files"]
                      if regex_pattern.search(f)
                ]
                if matching_files:
                    commit_date = commit_data["created_at"]
                    if (
                        not latest_matching_commit
                        or commit_date >
                            latest_matching_commit["commit_date"]
                    ):
                        latest_matching_commit = {
                            "commit_hash": commit_hash,
                            "commit_date": commit_date,
                            "branch_name": \
                                branch_data["branch_name"],
                            "files": matching_files,
                        }
    
    return latest_matching_commit


def get_commit(
    repo_id: str,
    commit_hash: str = None,
    files_filter: str = ".*"
) -> dict:
    """
    Get the dataset version info for a given "revision"
    (commit_hash) of a given HF dataset.
    Focus put on files matching the given regex pattern
    with associated metadata.

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - commit_hash (Optional, str):
            Particular "revision" of the dataset
            to scan.
        - files_filter (str):
            Only consider files matching this regex pattern.

    Results:
        - (dict):
            'commit_hash', 'commit_date',
            'branch_name', 'files'
    """

    regex_pattern = re.compile(files_filter)

    if not commit_hash:
        matching_commit = get_latest_commit(
            repo_id, files_filter
        )
        return matching_commit
    else:
        dataset_repo_branches = \
            get_dataset_branches_commits_files(repo_id)
        for \
            branch_type, branches \
            in dataset_repo_branches.items() \
        :
            for branch_name, branch_data in branches.items():
                for \
                    branch_commit_hash, branch_commit_data \
                    in branch_data["commits"].items() \
                :
                    if commit_hash == branch_commit_hash:
                        matching_files = [
                            f for f
                              in branch_commit_data["files"]
                              if regex_pattern.search(f)
                        ]
                        if len(matching_files) > 0:
                            matching_commit = {
                                "commit_hash": commit_hash,
                                "commit_date": \
                                    branch_commit_data["created_at"],
                                "branch_name": \
                                    branch_data["branch_name"],
                                "files": matching_files,
                            }
                            return matching_commit
                        else:
                            print(f"commit '{commit_hash}' " +
                                  f"hosts no files matching pattern " +
                                  f"'{files_filter}'",
                                  file=sys.stderr)
                            return None

    return None


def get_lazy_df(
    repo_id: str,
    commit_hash: str = None,
    files_filter: str = ".*\.parquet",
    hf_token:str = None,
) -> dict:
    """
    Polars lazy dataframe object
    for a given HuggingFace-hosted dataset.

    Note:
        We look for parquet files in a given "revision"
        (i.e. a given repo commit).
        It could mean using the @~parquet branch
        if the dataset is not natively in parquet format
        (but needs to be "public"
         for the auto-convert bot to kick in).
        @see https://huggingface.co/docs/dataset-viewer/en/parquet#conversion-to-parquet

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - commit_hash (Optional, str):
            particular "revision" of the dataset
            to scan.
            Note:
                commit_hash is a "by reference"
                parameter. If None, it is updated
                to convey the actual (latest) value
                info back to the caller
        - files_filter (str):
            Only consider files matching this regex pattern.
            This can serve if the dataset has many tables
            (possibly each with several splits),
            with different formats, for instance.
        - hf_token (Optional, str):
            Needed if the HF dataset is "gated"
            (requires to be granted access).
            @see https://huggingface.co/docs/hub/en/datasets-gated

    Results:
        - (dict):
            - commit_hash (str):
                gets handy when no input value
                is given as input.
            - commit_date (str):
                24hrs, UTC format.
            - lazydf (pl.lazyframe.frame.LazyFrame):
    """

    parquet_commit = get_commit(
        repo_id=repo_id,
        commit_hash=commit_hash,
        files_filter=files_filter
    )
    if not parquet_commit:
        print(f"commit '{commit_hash}' " +
              "either does not exist or " +
              "hosts no parquet file",
              file=sys.stderr)
        return None

    polars_hf_urls = [
        "hf://datasets/{}@{}/{}".format(
            repo_id,
            parquet_commit['commit_hash'],
            parquet_file
        )
        for parquet_file \
            in parquet_commit['files']
    ]

    lazy_df = pl.scan_parquet(
        polars_hf_urls,
        storage_options={"token": hf_token})

    return {
            "commit_hash": parquet_commit['commit_hash'], \
            "commit_utc_date_str": parquet_commit['commit_date'], \
            "lazy_df": lazy_df
        }


def get_column_info(
    lazy_df: pl.lazyframe.frame.LazyFrame,
    engine: str = "cpu"
) -> pd.DataFrame:
    """
    Basic types description.
    Aggregate min/max for numericals and
    words count for strings.

    Params:
        - lazy_df (pl.lazyframe.frame.LazyFrame):
            the dataset table to consider.
        - engine (str):
            Polars' engine (cpu or gpu)

    Results:
        - (pd.DataFrame):
    """

    schema = lazy_df.limit(1).collect(engine=engine)[0].schema

    def count_words(text):
        if text is None:
            return 0
        return len(re.findall(r"\w+", str(text)))

    result = lazy_df.select([
        pl.col(c).min().alias(f"{c}_min")
        if t in [pl.Int64, pl.Float64]
        else pl.lit(None).alias(f"{c}_min")
        for c, t in schema.items()
    ] + [
        pl.col(c).max().alias(f"{c}_max")
        if t in [pl.Int64, pl.Float64]
        else pl.lit(None).alias(f"{c}_max")
        for c, t in schema.items()
    ] + [
        pl.col(c).map_elements(lambda x: count_words(x),
                               return_dtype=pl.Int64
                              ).min().alias(f"{c}_min_words")
        if t == pl.Utf8
        else pl.lit(None).alias(f"{c}_min_words")
        for c, t in schema.items()
    ] + [
        pl.col(c).map_elements(lambda x: count_words(x),
                               return_dtype=pl.Int64
                              ).max().alias(f"{c}_max_words")
        if t == pl.Utf8
        else pl.lit(None).alias(f"{c}_max_words")
        for c, t in schema.items()
    ])
    df = result.collect(engine=engine)

    column_info = {}
    for col, dtype in schema.items():
        if dtype in [pl.Int64, pl.Float64]:
            column_info[col] = \
                f"{str(dtype)} - " + \
                f"[{df[0][f'{col}_min'][0]}-" + \
                f"{df[0][f'{col}_max'][0]}]"
        elif dtype == pl.Utf8:
            column_info[col] = \
                f"[{df[0][f'{col}_min_words'][0]}-" + \
                f"{df[0][f'{col}_max_words'][0]}] words"
        else:
            raise NotImplementedError(
                f"Data type '{dtype}' not implemented")

    return pd.DataFrame(column_info, index=["Range"])


def iterable_dataset_multi_buffer_sampler(
    dataset: IterableDataset,
    total_samples: int,
    attributes_selector: Optional[Callable]=None,
    buffer_size: int = 1000,
    num_passes: int = 3,
    seed: int = None
) -> Iterator:
    """
    Lazy random sampling with multiple buffer passes.

    Randomizes via reservoir sampling
    across iterative buffer windows.

    Ensures broader distribution than single-pass shuffling.

    Supports reproducible sampling with optional seed.

    Enables selective attribute extraction.

    Usage:
    ```python
    samples = list(iterable_dataset_multi_buffer_sampler(
        hf_streaming_dataset['train'],
        total_samples=1000,
        buffer_size=2000,
        num_passes=3,
        selector=lambda x: {
            # Select desired attributes
            'key1': x['key1'],
            'key2': x['key2']
        }
    ))
    ```

    Parameters:
        - dataset (IterableDataset):
            Input streaming dataset
        - total_samples (int):
            Number of samples to return
        - selector: Optional[Callable] = None
            Transform input item.
            Optional function transforming input item
            to subset of desired attributes.
            Extract specific attributes pre-yield
        - buffer_size (int):
            Size of each buffer.
            Larger buffer for better randomization.
        - num_passes (int):
            Number of shuffling passes.
            Multiple passes for better coverage.
        - seed (int):
            Random seed for reproducibility

    Results:
        - (Iterator)
    """
    if seed is not None:
        random.seed(seed)

    samples_per_pass = total_samples // num_passes
    remainder = total_samples % num_passes

    for pass_idx in range(num_passes):
        # Adjust samples for last pass to include remainder
        current_samples = (
            samples_per_pass + remainder
            if pass_idx == num_passes - 1
            else samples_per_pass
        )

        # Get samples with a fresh buffer
        buffer = []
        for item in dataset:
            item = (
                attributes_selector(item)
                if attributes_selector else item
            )
            if len(buffer) < buffer_size:
                buffer.append(item)
            else:
                # Randomly replace items in buffer
                idx = random.randint(0, buffer_size - 1)
                buffer[idx] = item

            # Yield an item when buffer is full
            if len(buffer) == buffer_size:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer[idx]
                current_samples -= 1

            if current_samples <= 0:
                break

        while current_samples > 0 and buffer:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer[idx]
            current_samples -= 1


def get_new_dataset_minor_version(
    repo_id: str,
    hf_token: str = None
) -> str:
    """
    `retrain-pipelines` datasets on the HuggingFace Hub
    have a "version" tag in their metadata
    (str formatted "major.minor").
    We here retrieve the latest value and
    return it minor-incremented.

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - hf_token (Optional, str):
            Needed if the HF dataset is "gated"
            (requires to be granted access).
            @see https://huggingface.co/docs/hub/en/datasets-gated

    Results:
        - (str):
            new version label
    """
    api = HfApi()

    dataset_info = None
    new_version = None
    try:
        dataset_info = api.dataset_info(
            repo_id=repo_id, revision=None,
            token=hf_token
        )
    except RepositoryNotFoundError as err:
        print(f"repo {repo_id} not found.\n" +
              "If you are trying to access a " +
              "private or gated repo, " +
              "make sure you are authenticated " +
              "and your credentials allow it.",
              file=sys.stderr)
        print(err, file=sys.stderr)

    if (
        dataset_info is not None and
        "version" in dataset_info.card_data
    ):
        last_version = \
            str(dataset_info.card_data.get("version", {}))
        new_version = \
            last_version.split('=')[-1].strip('"').rsplit('.', 1)[0] + \
            '.' + \
            str(int(last_version.rsplit('.', 1)[-1].strip('"')) + 1)
    else:
        new_version = "0.1"

    return new_version


def dataset_dict_to_config_str(
    dataset_dict: DatasetDict
) -> str:
    """
    Intended use: readme yaml header 'configs' tag.
    """

    result = "configs:\n"
    for config_name, dataset in dataset_dict.items():
        result += f"  - config_name: {config_name}\n"
        result += "    data_files:\n"
        if isinstance(dataset, dict):
            for split, _ in dataset.items():
                result += f"      - split: {split}\n"
                result += f"        path: {config_name}/{split}.parquet\n"
        else:
            result += "      - split: train\n"
            result += f"        path: {config_name}/data.parquet\n"
    return result


def get_arxiv_codes(
    repo_id: str,
    commit_hash: str = None
) -> list:
    """
    Retrieve all arXiv codes associated with
    the dataset on the HFHub.

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - commit_hash (Optional, str):
            Specific "revision" of the dataset to scan.

    Returns:
        - List[str]:
            List of arXiv codes.
    """

    api = HfApi()
    arxiv_codes = []
    try:
        dataset_info = api.dataset_info(
            repo_id=repo_id, revision=commit_hash
        )
        arxiv_tags = [tag for tag in dataset_info.tags
                      if tag.startswith('arxiv:')]
        for arxiv_tag in arxiv_tags:
            arxiv_code = arxiv_tag.split(':')[1]
            arxiv_codes.append(arxiv_code)
    except Exception as err:
        print(err, file=sys.stderr)
        return []

    return arxiv_codes



def get_license_label(
    repo_id: str,
    commit_hash: str = None
):
    """
    @see @see https://huggingface.co/docs/hub/repositories-licenses

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - commit_hash (Optional, str):
            Specific "revision" of the dataset to scan.

    Returns:
        - (str)
    """

    api = HfApi()
    try:
        dataset_info = api.dataset_info(
            repo_id=repo_id, revision=commit_hash
        )
        license_tag = next((tag
                          for tag in dataset_info.tags
                          if tag.startswith('license:')),
                         None)
        
        if license_tag:
            license_label = license_tag.split(':')[1]
            return license_label
        else:
            return None
    except Exception as err:
        print(err, file=sys.stderr)
        return None

