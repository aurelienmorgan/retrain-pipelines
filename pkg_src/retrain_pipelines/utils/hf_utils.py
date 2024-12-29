
import os
import sys

import re
import traceback

from requests.exceptions import ReadTimeout

from huggingface_hub.utils import \
    RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi


def get_arxiv_codes(
    repo_id: str,
    repo_type: str = "model",
    commit_hash: str = None
) -> list:
    """
    Retrieve all arXiv codes associated with
    a given repository on the HFHub.

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
        return []

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
        pretty_name = repo_info.card_data["pretty_name"]
    except (ReadTimeout, HfHubHTTPError) as err:
        stack_trace = \
            ''.join(traceback.format_exception(
                type(err), err, err.__traceback__))
        print(stack_trace, file=sys.stderr)
    except Exception as err:
        print(err, file=sys.stderr)

    if not pretty_name:
        pretty_name = ' '.join(
            [word.capitalize()
             for word in re.sub(
                '[-_]', ' ',
                repo_id.split("/", 1)[-1]).split()
            ])

    return pretty_name


def get_new_repo_minor_version(
    repo_id: str,
    repo_type: str = "model",
    hf_token: str = None
) -> str:
    """
    `retrain-pipelines` repositories
    on the HuggingFace Hub
    have a "version" tag in their metadata
    (str formatted "major.minor").
    We here retrieve the latest value and
    return it minor-incremented.

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
    api = HfApi()

    repo_info = None
    new_version = None
    api_info_method = \
        api.model_info if "model" == repo_type \
        else api.dataset_info if "dataset" == repo_type \
        else api.space_info # if "space" == repo_type

    try:
        repo_info = api_info_method(
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
        repo_info is not None and
        "version" in repo_info.card_data
    ):
        last_version = \
            str(repo_info.card_data.get("version", {}))
        new_version = \
            last_version.split('=')[-1].strip('"').rsplit('.', 1)[0] + \
            '.' + \
            str(int(last_version.rsplit('.', 1)[-1].strip('"')) + 1)
    else:
        new_version = "0.1"

    return new_version


def _create_repo_if_not_exists(
    hf_api: HfApi,
    repo_id: str,
    repo_type: str = "model",
    hf_token: str = None,
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
    repo_type: str = "model",
    hf_token: str = None,
) -> str:
    """
    Upload all files in a single commit.

    Params:
        - repo_id (str):
            Path to the HuggingFace repository version
            (is created if needed and if authorized).
        - local_folder (str):
            path to the source folder to be pushed.
        - commit_message (str):
            the message associated to the 'push_to_hub'
            commit.
        - repo_type (str):
            can be "model", "dataset", "space".
        - hf_token (Optional, str):
            "create on namespace" permission required.
    """

    api = HfApi()
    repository_new_commit_hash = None

    if not _create_repo_if_not_exists(
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

