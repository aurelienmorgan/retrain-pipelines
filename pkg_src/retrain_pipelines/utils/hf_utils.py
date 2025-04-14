
import os
import re
import sys
import shutil
import tempfile
import traceback
import subprocess
from datetime import datetime
from requests.exceptions import ReadTimeout

from huggingface_hub import list_repo_refs, \
    list_repo_commits, list_repo_files, HfApi, \
    hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from retrain_pipelines.utils import \
    create_requirements


def _repo_branch_commits_files(
    repo_id: str,
    repo_type: str = "model",
    repo_branch: str = "main"
) -> dict:
    """
    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - repo_type (str):
            can be "model", "dataset", "space".
        - repo_branch (str):
            Branch (of the repository  of interest)
            to be considered.

    Results:
        - (dict)
            'commit_hash', 'created_at',
            'title', 'files'
    """
    commits = list_repo_commits(repo_id, revision=repo_branch,
                                repo_type=repo_type,
                                token=os.environ["HF_TOKEN"])
    commits_dict = {}
    for commit in commits:
        files = list_repo_files(
            repo_id, revision=commit.commit_id,
            repo_type=repo_type,
            token=os.environ["HF_TOKEN"])

        commits_dict[commit.commit_id] = {
            "created_at": commit.created_at,
            "title": commit.title,
            "files": files
        }

    return commits_dict


def get_repo_branches_commits_files(
    repo_id: str,
    repo_type: str = "model"
) -> dict:
    """
    Selection of metadata for (litterally)
    all files of all commits of a given
    HF repo.

    Params:
        - repo_id (str):
            Path to the HuggingFace dataset.
        - repo_type (str):
            can be "model", "dataset", "space".

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

    refs = list_repo_refs(repo_id, repo_type=repo_type,
                          token=os.getenv("HF_TOKEN", None))

    repo_branches = {
        "repo_standard_branches": {},
        "repo_convert_branches": {}
    }
    for repo_standard_branches in refs.branches:
        repo_branches[
            "repo_standard_branches"
        ][repo_standard_branches.name] = {
            "branch_name": repo_standard_branches.ref,
            "commits": _repo_branch_commits_files(
                repo_id, repo_type,
                repo_standard_branches.ref)
        }
        
    for repo_convert_branch in refs.converts:
        repo_branches[
            "repo_convert_branches"
        ][repo_convert_branch.name] = {
            "branch_name": repo_convert_branch.ref,
            "commits": _repo_branch_commits_files(
                repo_id, repo_type,
                repo_convert_branch.ref)
        }

    return repo_branches


def get_latest_README_commit(
    repo_id: str,
    target_commit_hash: str,
    repo_type: str = "model",
    verbose: bool = True
) -> (str, datetime):
    """
    Using a given commit as a starting point,
    look for the latest prior commit for which
    there was a README.md file.

    This is to address cases where
        'the commit corresponding to this commit_hash
         didn't include a README and
         many entries are missing from
         `HfApi().dataset_info`, `HfApi().model_info`,
         `HfApi().space_info`..'.
    for instance, typical of datasets 'auto-convert bot'
    (think duckdb or parquet, 
     @see https://huggingface.co/docs/dataset-viewer/en/parquet#conversion-to-parquet).

    Params:
        - repo_id (str):
            Path to the HuggingFace repository.
        - commit_hash (Optional, str):
            particular "revision" of the repository
            to scan.
        - repo_type (str):
            can be "model", "dataset", "space".
        - verbose (bool):
            whether or not to print commit
            hash and date (target vs latest README)

    Results:
        - (str, datetime):
            latest_README_commit_hash,
            latest_README_commit_date
    """
    hf_repo_branches_commits_files = \
        get_repo_branches_commits_files(
            repo_id=repo_id, repo_type=repo_type)

    target_date = None
    for repo, repo_data in hf_repo_branches_commits_files.items():
        for branch, branch_data in repo_data.items():
            for commit_hash, commit_data in branch_data['commits'].items():
                if commit_hash == target_commit_hash:
                    target_date = commit_data['created_at']
                    break
            if target_date:
                break
        if target_date:
            break
    if verbose:
        print("target commit : ".ljust(25), target_commit_hash, target_date)

    README_date = None
    README_commit_hash = None
    for repo, repo_data in hf_repo_branches_commits_files.items():
        for branch, branch_data in repo_data.items():
            for commit_hash, commit_data in branch_data['commits'].items():
                if 'README.md' in commit_data['files']:
                    commit_date = commit_data['created_at']
                    if commit_date <= target_date:
                        README_date = commit_data['created_at']
                        README_commit_hash = commit_hash
                        if verbose:
                            print("lastest README commit : ".ljust(25),
                                  README_commit_hash, README_date)
                        break
            else:
                continue
            break
        else:
            continue
        break

    return README_commit_hash, README_date


def get_arxiv_codes(
    repo_id: str,
    repo_type: str = "model",
    commit_hash: str = None
) -> list:
    """
    Retrieve all arXiv codes associated with
    a given repository on the HFHub.

    Note: The "info" api is quite unstable server-side.
          The response structure is evolving a lot.
          For the same library version, we get
          different responses from one request to the next.
          @see https://discuss.huggingface.co/t/133438
          (even upgrading lib didn't fix it,
           contraryli to what we first  observed
           and reported).
          We go "best-effort" mode on this :
            - look into repo_info.card_data
            - fallback to looking into repo_info.tags
            - fallback to looking into readme content

    Params:
        - repo_id (str):
            Path to the HuggingFace repository.
        - repo_type (str):
            can be "model", "dataset", "space".
        - commit_hash (Optional, str):
            Specific "revision" of
            the repository to query.

    Returns:
        - List[str]:
            List of arXiv codes.
    """

    api = HfApi()
    arxiv_codes = []
    api_info_method = \
        api.model_info if "model" == repo_type \
        else api.dataset_info if "dataset" == repo_type \
        else api.space_info # if "space" == repo_type

    try:
        repo_info = api_info_method(
            repo_id=repo_id, revision=commit_hash
        )
        arxiv_codes = repo_info.card_data["arxiv"]
    except (ReadTimeout, HfHubHTTPError) as err:
        stack_trace = \
            ''.join(traceback.format_exception(
                type(err), err, err.__traceback__))
        print(stack_trace, file=sys.stderr)
        return []
    except Exception as err:
        print(("get_arxiv_codes", err), file=sys.stderr)

    if 0 == len(arxiv_codes):
        try:
            arxiv_tags = [tag for tag in repo_info.tags
                          if tag.startswith('arxiv:')]
            for arxiv_tag in arxiv_tags:
                arxiv_code = arxiv_tag.split(':')[1]
                arxiv_codes.append(arxiv_code)
        except Exception as err:
            print(("get_arxiv_codes", err), file=sys.stderr)

    if 0 == len(arxiv_codes):
        file_path = hf_hub_download(
            repo_id=repo_id, revision=commit_hash,
            repo_type=repo_type, filename="README.md")
        # arXiv:2406.18518
        # https://huggingface.co/papers/2406.18518
        # https://arxiv.org/abs/2406.18518
        # https://arxiv.org/pdf/2406.18518
        with open(file_path, 'r') as file:
            content = file.read()
        arxiv_references = re.findall(
            r'arXiv:(\d{4}\.\d{5})|https://(?:(huggingface|hf)\.(co|com)/papers|arxiv\.org/(?:abs|pdf))/(\d{4}\.\d{5})',
            content
        )
        arxiv_codes = list(set(
            [ref for ref_pair in arxiv_references
             for ref in ref_pair if ref]))

    return arxiv_codes


def get_license_label(
    repo_id: str,
    repo_type: str = "model",
    commit_hash: str = None
):
    """
    @see @see https://huggingface.co/docs/hub/repositories-licenses

    Params:
        - repo_id (str):
            Path to the HuggingFace repository.
        - repo_type (str):
            can be "model", "dataset", "space".
        - commit_hash (Optional, str):
            Specific "revision" of
            the repository to query.

    Returns:
        - (str)
    """

    api = HfApi()
    api_info_method = \
        api.model_info if "model" == repo_type \
        else api.dataset_info if "dataset" == repo_type \
        else api.space_info # if "space" == repo_type

    try:
        repo_info = api_info_method(
            repo_id=repo_id, revision=commit_hash
        )
        return repo_info.card_data["license"]
    except (ReadTimeout, HfHubHTTPError) as err:
        stack_trace = \
            ''.join(traceback.format_exception(
                type(err), err, err.__traceback__))
        print(stack_trace, file=sys.stderr)
        return None
    except Exception as err:
        print(("get_license_label", err), file=sys.stderr)
        return None


def get_pretty_name(
    repo_id: str,
    repo_type: str = "model",
    commit_hash: str = None
) -> str:
    """
    Falls back to capitalized repo_id.

    Params:
        - repo_id (str):
            Path to the HuggingFace repository.
        - repo_type (str):
            can be "model", "dataset", "space".
        - commit_hash (Optional, str):
            Specific "revision" of
            the repository to query.

    Returns:
        - (str)
    """

    api = HfApi()
    pretty_name = None
    api_info_method = \
        api.model_info if "model" == repo_type \
        else api.dataset_info if "dataset" == repo_type \
        else api.space_info # if "space" == repo_type

    try:
        repo_info = api_info_method(
            repo_id=repo_id, revision=commit_hash
        )
        pretty_name = repo_info.card_data[
            "model_name" if "model" == repo_type
            else "pretty_name"]
    except (ReadTimeout, HfHubHTTPError) as err:
        stack_trace = \
            ''.join(traceback.format_exception(
                type(err), err, err.__traceback__))
        print(stack_trace, file=sys.stderr)
    except Exception as err:
        print(("get_pretty_name", err), file=sys.stderr)

    if not pretty_name:
        pretty_name = ' '.join(
            [word.capitalize()
             for word in re.sub(
                '[-_]', ' ',
                repo_id.split("/", 1)[-1]).split()
            ])

    return pretty_name


def get_repo_version(
    repo_id: str,
    revision: str = None,
    repo_type: str = "model",
    hf_token: str = os.getenv("HF_TOKEN", None)
) -> (int, int):
    """
    """

    if revision is None:
        return get_repo_latest_version(
            repo_id=repo_id, repo_type=repo_type, hf_token=hf_token
        )

    api = HfApi()
    api_info_method = \
        api.model_info if "model" == repo_type \
        else api.dataset_info if "dataset" == repo_type \
        else api.space_info # if "space" == repo_type

    repo_rev_info = api_info_method(
        repo_id=repo_id,
        revision=revision,
        token=hf_token
    )
    repo_rev_card_data = repo_rev_info.card_data

    if repo_rev_card_data and "version" in repo_rev_card_data:
        version_label = \
            repo_rev_card_data["version"]
        major, minor =  \
            map(int, version_label.split('.'))

        return major, minor

    return 0, 0


def get_repo_latest_version(
    repo_id: str,
    repo_type: str = "model",
    hf_token: str = os.getenv("HF_TOKEN", None)
) -> (int, int):
    """
    `retrain-pipelines` repositories
    on the Hugging Face Hub
    have a "version" tag in their metadata
    (str formatted "major.minor").
    We here retrieve the latest value and
    return it minor-incremented.

    Note : We look into last commit
           of all branches.

    Params:
        - repo_id (str):
            Path to the HuggingFace repository.
        - repo_type (str):
            can be "model", "dataset", "space".
        - hf_token (Optional, str):
            Needed if the HF repository is "gated"
            (requires to be granted access).
            @see https://huggingface.co/docs/hub/en/datasets-gated

    Results:
        - (int, int):
            version as (minor, major) pair
    """

    refs = None
    refs = list_repo_refs(
        repo_id=repo_id,
        repo_type=repo_type,
        token=hf_token
    )

    if refs:
        api = HfApi()
        api_info_method = \
            api.model_info if "model" == repo_type \
            else api.dataset_info if "dataset" == repo_type \
            else api.space_info # if "space" == repo_type

        latest_version_major = 0
        latest_version_minor = 0
        for branch in refs.branches:
            branch_repo_info = api_info_method(
                repo_id=repo_id,
                revision=branch.target_commit,
                token=hf_token
            )
            branch_card_data = branch_repo_info.card_data
            if branch_card_data and "version" in branch_card_data:
                branch_version_label = \
                    branch_card_data["version"]
                branch_major, branch_minor =  \
                    map(int, branch_version_label.split('.'))
                if branch_major > latest_version_major:
                    latest_version_major, latest_version_minor = \
                        branch_major, branch_minor
                elif branch_minor > latest_version_minor:
                    latest_version_major, latest_version_minor = \
                        branch_major, branch_minor
        #         print(branch_card_data["version"])

        return latest_version_major, latest_version_minor

    return 0, 0


def get_new_repo_minor_version(
    repo_id: str,
    repo_type: str = "model",
    hf_token: str = os.getenv("HF_TOKEN", None)
) -> str:
    """
    `retrain-pipelines` repositories
    on the Hugging Face Hub
    have a "version" tag in their metadata
    (str formatted "major.minor").
    We here retrieve the latest value and
    return it minor-incremented.

    Note : We look into last commit
           of all branches.

    Params:
        - repo_id (str):
            Path to the HuggingFace repository.
        - repo_type (str):
            can be "model", "dataset", "space".
        - hf_token (Optional, str):
            Needed if the HF repository is "gated"
            (requires to be granted access).
            @see https://huggingface.co/docs/hub/en/datasets-gated

    Results:
        - (str):
            new version label
    """

    latest_version_major, latest_version_minor = \
        get_repo_latest_version(repo_id, repo_type, hf_token)

    if latest_version_major + latest_version_minor > 0:
        new_version_label = \
            f"{latest_version_major}.{latest_version_minor+1}"
    else:
        new_version_label = "0.1"
    
    return new_version_label


def create_repo_if_not_exists(
    hf_api: HfApi,
    repo_id: str,
    repo_type: str = "model",
    hf_token: str = os.getenv("HF_TOKEN", None)
) -> bool:
    """
    Note : For repositories of type 'model',
    we create two branches in addition to default 'main' :
      - "retrain-pipelines/source_code" and
      - "retrain-pipelines/pipeline_card"
    Those serve for artifacts stores for model versions.

    Params:
        - hf_api (HfApi):
            An instanciated FH api object.
        - repo_id (str):
            Path to the HuggingFace repository.
        - repo_type (str):
            can be "model", "dataset", "space".
        - hf_token (Optional, str):
            "create on namespace" permission required.

    Results:
        - (bool):
            success/failure
    """

    try:
        hf_api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True,
            private=True,
            token=hf_token
        )
    except HfHubHTTPError as err:
        print(f"Failed to create {repo_type} `{repo_id.split('/')[1]}` "+
              f"under the `{repo_id.split('/')[0]}` namespace " +
              "on the HuggingFace Hub.\n" +
              "Does the HF_TOKEN you use have the permission " +
              "on that namespace ?",
              file=sys.stderr)
        print(''.join(traceback.format_exception(
                    type(err), err, err.__traceback__)))
        return False
    except Exception as err:
        print(''.join(traceback.format_exception(
                    type(err), err, err.__traceback__)))
        return False

    if "model" == repo_type:
        for branch_name in [
            "retrain-pipelines_not-blessed",
            "retrain-pipelines_source-code",
            "retrain-pipelines_pipeline-card"
        ]:
            try:
                hf_api.create_branch(
                    repo_id=repo_id,
                    branch=branch_name,
                    repo_type="model",
                    token=os.environ["HF_TOKEN"]
                )
            except HfHubHTTPError as err:
                if (
                    f"Reference already exists: refs/heads/{branch_name}"
                    != err.server_message.strip()
                ):
                    print(f"Failed to create branch {branch_name} for " +
                          f"{repo_type} `{repo_id.split('/')[1]}` "+
                          f"under the `{repo_id.split('/')[0]}` namespace " +
                          "on the HuggingFace Hub.",
                          file=sys.stderr)
                    print(''.join(traceback.format_exception(
                                type(err), err, err.__traceback__)))
                    return False
            except Exception as err:
                print(f"Failed to create branch {branch_name} for " +
                      f"{repo_type} `{repo_id.split('/')[1]}` "+
                      f"under the `{repo_id.split('/')[0]}` namespace " +
                      "on the HuggingFace Hub.",
                      file=sys.stderr)
                print(''.join(traceback.format_exception(
                            type(err), err, err.__traceback__)))
                return False

    return True


def local_repo_folder_to_hub(
    repo_id: str,
    local_folder: str,
    commit_message: str = "new commit",
    branch_name: str = "main",
    repo_type: str = "model",
    hf_token: str = os.getenv("HF_TOKEN", None),
) -> str:
    """
    Upload all files in a single commit.

    Note : We do not go the "run_as_future" way
           (for asynchronous upload) despite
           it being Advisable when publishing
           models to the HF hub (since usually
           models with many params are slow to upload).
           We do rely on commit_hash and need it
           to continue the documentation process
           for the `retrain-pipelines` run.

    Params:
        - repo_id (str):
            Path to the HuggingFace repository version
            (is created if needed and if authorized).
        - local_folder (str):
            path to the source folder to be pushed.
        - commit_message (str):
            the message associated to the 'push_to_hub'
            commit.
        - branch_name (str):
            The repo-branch on which to publish.
            Defaults to 'main'.
        - repo_type (str):
            can be "model", "dataset", "space".
        - hf_token (Optional, str):
            "create on namespace" permission required.
    """

    api = HfApi()
    repository_new_commit_hash = None

    if not create_repo_if_not_exists(
        hf_api=api,
        repo_id=repo_id,
        repo_type=repo_type,
        hf_token=hf_token
    ):
        return None

    try:
        repository_new_commit = api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=branch_name,
            path_in_repo="",
            folder_path=local_folder,
            delete_patterns=["**"],
            commit_message=commit_message,
            token=hf_token
        )
        repository_new_commit_hash = repository_new_commit.oid
    except HfHubHTTPError as err:
        print(f"Failed to upload {repo_type} to HuggingFace Hub.\n" +
              "Is 'write' permission associated to " +
              "the HF_TOKEN you use ?\n" +
              f"On the `{repo_id.split('/')[0]}` namespace ?",
              file=sys.stderr)
        print(''.join(traceback.format_exception(
                    type(err), err, err.__traceback__)))
        return None

    return repository_new_commit_hash


def push_files_to_hub_repo_branch(
    repo_id: str,
    branch_name: str,
    file_fullnames: list,
    include_requirements_txt: bool = False,

    path_in_repo: str = "",
    commit_message: str = "new commit",
    repo_type: str = "model",
    hf_token:str = os.getenv("HF_TOKEN", None)
) -> str:
    """
    pushes files to single parent folder
    on a HF Hub repo branch.

    Note :
        Fails silently, in that sense that
        if any of the listed file
        does not exist, it moves on.

    Params:
        - repo_id (str):
            Path to the target HuggingFace
            repository.
        - branch_name (str):
            Name of the target branch
            on the HuggingFace repository.
        - file_fullnames (List[str]):
            list of local fullpaths to
            files to be uploaded
        - include_requirements_txt (bool):
            whether to includ a snapshot of
            the env installed librairies
            in the folder upload.
        - path_in_repo (str):
            subdir of the targetted branch
            to upload files into.
            If it exists on the remote repo,
            its content is overriden
            (i.e. any file present there will be
             excluded from the snapshot
             after the herein commit).
        - commit_message (str):
            the message associated to the 'push_to_hub'
            commit.
        - repo_type (str):
            can be "model", "dataset", "space".
        - hf_token (Optional, str):
            "create on namespace" permission required.

    Results:
        - (str):
            commit_hash
    """

    tmp_src_dir = tempfile.mkdtemp()
    # print(tmp_src_dir)

    for file_fullname in file_fullnames:
        if (
            os.path.isabs(file_fullname) and
            os.path.exists(file_fullname)
        ):
            shutil.copy(
                file_fullname,
                os.path.join(
                    tmp_src_dir,
                    os.path.basename(file_fullname)
                )
            )

    if include_requirements_txt:
        create_requirements(tmp_src_dir)

    api = HfApi()
    folder_branch_commit = api.upload_folder(
        repo_id=repo_id,
        revision=branch_name,
        path_in_repo=path_in_repo,
        folder_path=tmp_src_dir,
        delete_patterns=["**"],
        commit_message=commit_message,
        repo_type=repo_type,
        token=hf_token
    )
    folder_branch_commit_hash = \
        folder_branch_commit.oid

    shutil.rmtree(tmp_src_dir, ignore_errors=True)

    return folder_branch_commit_hash


def get_commit_created_at(
    hf_api: HfApi,
    repo_id: str,
    revision: str,
    repo_type: str = "model",
    hf_token: str = os.getenv("HF_TOKEN", None)
):
    """

    Params:
        - hf_api (HfApi):
            An instanciated FH api object.
        - repo_id (str):
            Path to the HuggingFace repository.
        - revision (str):
            commit hash or branch name.
            If branch name, we consider
            the latest commit.
        - repo_type (str):
            can be "model", "dataset", "space".
        - hf_token (Optional, str):

    Results:
        - (bool):
            success/failure
    """
    commits = api.list_repo_commits(
        repo_id, repo_type="model", revision=revision, token=hf_token)
    if commits:
        # Commits are sorted by date, latest first
        return commits[0].created_at

    return None

