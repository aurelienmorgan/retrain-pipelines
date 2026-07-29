import logging
import os
import re
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

from .s3_utils import is_s3_path, parse_s3_uri

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@lru_cache
def build_path(root: str, parts: tuple[str, ...]) -> str:
    """
    Build via a convenience method.

    Joins *root* and *parts* into a single path or S3 URI string.
    Uses '/' separator for S3 URIs and OS-specific separator for local paths.

    Note: ``parts`` must be a tuple (not a list) to be hashable
    for ``lru_cache``.
    """
    if is_s3_path(root):
        bucket, key_prefix = parse_s3_uri(root)
        key = key_prefix + "/".join(parts)
        return f"s3://{bucket}/{key}"
    else:
        return os.path.join(root, *parts)


def read_text_file(root: str, parts: list[str], encoding: str = "utf-8") -> str:
    """Read a text file from either a local path or an S3 URI.

    Constructs the full path by joining *root* with *parts*, dispatching
    to the appropriate backend based on whether *root* is an S3 URI.

    Parameters
    ----------
    root : str
        Local directory path or S3 URI (``s3://bucket/prefix/``).
    parts : list[str]
        Ordered path components appended to *root*
        (e.g. ``[pipeline_name, "42", "pipeline_card.html"]``).
    encoding : str
        Text encoding for decoding the file content. Defaults to ``"utf-8"``.

    Returns
    -------
    str
        Full text content of the file.

    Raises
    ------
    FileNotFoundError:
        If the file does not exist (locally or in S3).
    """
    if is_s3_path(root):
        bucket, key_prefix = parse_s3_uri(root)
        key = key_prefix + "/".join(parts)
        try:
            resp = boto3.client("s3").get_object(Bucket=bucket, Key=key)
            return resp["Body"].read().decode(encoding)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"s3://{bucket}/{key}") from e
            raise
    else:
        path = os.path.join(root, *parts)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, encoding=encoding) as f:
            return f.read()


def write_text_file(root: str, parts: list[str], content: str, encoding: str = "utf-8") -> str:
    """Write text content to a file at either a local path or an S3 URI.

    Constructs the full path by joining *root* with *parts*, dispatching
    to the appropriate backend based on whether *root* is an S3 URI.
    Parent directories are created automatically for local paths.

    Parameters
    ----------
    root : str
        Local directory path or S3 URI (``s3://bucket/prefix/``).
    parts : list[str]
        Ordered path components appended to *root*
        (e.g. ``[pipeline_name, "42", "pipeline_card.html"]``).
    content : str
        Text content to write.
    encoding : str
        Text encoding. Defaults to ``"utf-8"``.

    Returns
    -------
    str
        Full local path or S3 URI (``s3://bucket/key``) of the written file.

    Examples
    --------
    >>> path = write_text_file(
    ...     Config.get_artifacts_store_root(),
    ...     ["my_pipeline", "42", "pipeline_card.html"],
    ...     content="<html>...</html>",
    ... )
    >>> print(path)
    /home/user/.cache/retrain-pipelines/artifacts/my_pipeline/42/pipeline_card.html
    """
    if is_s3_path(root):
        bucket, key_prefix = parse_s3_uri(root)
        key = key_prefix + "/".join(parts)
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=content.encode(encoding),
        )
        return f"s3://{bucket}/{key}"
    else:
        path = os.path.join(root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return path


def read_binary_file(root: str, parts: list[str]) -> bytes:
    """Read binary content from a local path or S3 URI built from *root* and *parts*.

    Parameters
    ----------
    root : str
        Local directory path or S3 URI (``s3://bucket/prefix/``).
    parts : list[str]
        Ordered path components appended to *root*.

    Returns
    -------
    bytes

    Raises
    ------
    FileNotFoundError:
        If the file does not exist (locally or in S3).
    """
    logger.debug(f"{root} - {parts}")
    if is_s3_path(root):
        bucket, key_prefix = parse_s3_uri(root)
        key = key_prefix + "/".join(parts)
        try:
            resp = boto3.client("s3").get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"s3://{bucket}/{key}") from e
            raise
    else:
        path = os.path.join(root, *parts)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            return f.read()


def write_binary_file(root: str, parts: list[str], content: bytes) -> None:
    """Write binary content to a local path or S3 URI built from *root* and *parts*.

    Parent directories are created automatically for local paths.

    Parameters
    ----------
    root : str
        Local directory path or S3 URI (``s3://bucket/prefix/``).
    parts : list[str]
        Ordered path components appended to *root*.
    content : bytes
        Binary content to write.
    """
    if is_s3_path(root):
        bucket, key_prefix = parse_s3_uri(root)
        key = key_prefix + "/".join(parts)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=content)
    else:
        path = os.path.join(root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)


def validate_path(path: str) -> None:
    """Validate a user-provided local directory path or S3 prefix.

    Ensures the path is a non-empty string. For local paths, validates against
    invalid filesystem characters, rejects malformed paths, and creates the
    directory if it does not exist. For S3 URIs, validates the URI format and
    bucket naming rules, then checks that the bucket exists and is accessible.

    Parameters
    ----------
    path : str
        Local directory path or S3 URI (``s3://bucket/prefix/``).

    Raises
    ------
    TypeError:
        If ``path`` is not a string.
    ValueError:
        If the path is empty, malformed, violates naming constraints,
        contains invalid characters, cannot be created, or if the S3 bucket is
        inaccessible.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Path must be a non-empty string.")

    if is_s3_path(path):
        bucket, prefix = parse_s3_uri(path)
        if not bucket:
            raise ValueError(f"Invalid S3 URI, missing bucket name: {path}")

        if not re.match(r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$", bucket):
            raise ValueError(f"Invalid S3 bucket name '{bucket}' in URI: {path}")
        if ".." in bucket:
            raise ValueError(f"Invalid S3 bucket name '{bucket}' (consecutive dots) in URI: {path}")

        s3 = boto3.client("s3")
        try:
            s3.head_bucket(Bucket=bucket)
        except ClientError as e:
            raise ValueError(f"S3 bucket '{bucket}' is not accessible: {e}") from e
    else:
        if "\x00" in path:
            raise ValueError(f"Local path contains illegal null byte: {path}")
        if any(c in path for c in '<>:"|?*'):
            raise ValueError(f"Local path contains invalid characters: {path}")

        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Failed to create or access local directory '{path}': {e}") from e


def list_files(root: str, recursive: bool = True) -> list[str]:
    """List files and directories under a local path or S3 URI.

    Parameters
    ----------
    root : str
        Local directory path or S3 URI (``s3://bucket/prefix/``).
    recursive : bool
        If True (default), lists all files and directories recursively.
        If False, lists only files and directories immediately under the path.

    Returns
    -------
    list[str]
        Names of files and directories. When recursive, paths are relative
        to *root* using ``/`` as the separator.
    """
    if is_s3_path(root):
        bucket, prefix = parse_s3_uri(root)
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        files_set: set[str] = set()

        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if not recursive:
            kwargs["Delimiter"] = "/"

        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                # Exclude the prefix itself if it matches a key exactly
                # (S3 pseudo-directory marker)
                if key == prefix:
                    continue
                rel_key = key[len(prefix) :]
                if rel_key:
                    files_set.add(rel_key)
            if not recursive:
                for common_prefix in page.get("CommonPrefixes", []):
                    p: str = common_prefix["Prefix"]
                    rel_p = p[len(prefix) :]
                    if rel_p.endswith("/"):
                        rel_p = rel_p[:-1]
                    if rel_p:
                        files_set.add(rel_p)
        return list(files_set)
    else:
        if recursive:
            files_list: list[str] = []
            for dirpath, dirnames, filenames in os.walk(root):
                for name in dirnames + filenames:
                    full_path = os.path.join(dirpath, name)
                    rel_path = os.path.relpath(full_path, root)
                    # Use forward slash for consistency with S3 paths
                    files_list.append(rel_path.replace(os.sep, "/"))
            return files_list
        else:
            return os.listdir(root)
