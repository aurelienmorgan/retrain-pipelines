
import os
import re
import json

import polars as pl

from huggingface_hub import list_repo_refs, list_repo_commits, \
    list_repo_files


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
) -> (str, str, pl.lazyframe.frame.LazyFrame):
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
        files_filter=".*\.parquet"
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

    return parquet_commit['commit_hash'], \
           parquet_commit['commit_date'], \
           lazy_df

