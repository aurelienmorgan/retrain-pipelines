import io
import logging
import os
from functools import lru_cache
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

logging.getLogger("botocore").setLevel(logging.WARNING)


@lru_cache(maxsize=128)
def is_s3_path(path: str) -> bool:
    """Return True if *path* is an S3 URI (``s3://…``)."""
    return path.startswith("s3://")


@lru_cache(maxsize=128)
def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 URI into ``(bucket, key_prefix)``.

    If the last path component has a file extension, it is treated as a file
    and no trailing slash is appended. Otherwise, it is treated as a directory
    prefix and a trailing slash is ensured.
    """
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    if prefix:
        if not prefix.endswith("/") and not os.path.splitext(prefix)[1]:
            prefix += "/"

    return bucket, prefix


def validate_s3_env() -> None:
    """Raise ``EnvironmentError`` if no AWS credential configuration is detectable.

    Accepted configurations (in order of precedence, mirroring boto3):
    - ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY``
    - ``AWS_PROFILE``
    - ``AWS_ROLE_ARN`` (assumed via STS)
    - ``AWS_WEB_IDENTITY_TOKEN_FILE`` (IRSA / pod identity)
    - Instance metadata credentials (EC2 / ECS) ; not verifiable from env alone,
      so this check is necessarily best-effort for static credentials.

    Note
    ----
    ``AWS_ENDPOINT_URL`` is optional and enables S3-compatible stores (e.g. MinIO).
    """
    has_keys = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
    has_profile = bool(os.environ.get("AWS_PROFILE"))
    has_role = bool(os.environ.get("AWS_ROLE_ARN"))
    has_web_identity = bool(os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"))

    if not any([has_keys, has_profile, has_role, has_web_identity]):
        raise OSError(
            "No AWS credentials found. "
            "Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, AWS_PROFILE, "
            "AWS_ROLE_ARN, or configure an instance/pod identity role."
        )


def ensure_s3_bucket(uri: str) -> None:
    """Ensure the S3 bucket referenced by *uri* exists, creating it if necessary.

    Analogous to ``Config._ensure_dir`` for local paths.

    Parameters
    ----------
    uri : str
        An S3 URI of the form ``s3://<bucket>/…``.

    Raises
    ------
    EnvironmentError:
        If no AWS credential configuration is detectable.
    botocore.exceptions.ClientError:
        On unexpected S3 errors (access denied, service error, etc.).
    """
    validate_s3_env()
    bucket, _ = parse_s3_uri(uri)
    s3 = boto3.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            try:
                s3.create_bucket(Bucket=bucket)
            except ClientError as ce:
                print(f"Error creating S3 bucket {bucket}: {ce}")
                raise ce
        else:
            print(f"Error accessing S3 bucket {bucket}: {e}")
            raise


def create_s3_prefix_symlink(bucket: str, src_prefix: str, dst_prefix: str) -> None:
    """Create a zero-byte S3 object at *dst_prefix* whose key name is *src_prefix*.

    S3 equivalent of a directory symlink / junction: browsing to *dst_prefix*
    reveals a single entry whose name is the full *src_prefix* path, making
    the location of the original objects immediately navigable.
    No objects are copied.

    Note
    ----
    The standard slashes in ``src_prefix`` are replaced with a full-width
    solidus (``"／"`` (U+FF0F)). This preserves human readability while forcing S3
    to store it as a single literal object name rather than interpreting
    the slashes as a raw navigable folder structure.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    src_prefix : str
        Source key prefix (e.g. ``".cache/metadata/<temp_id>/params/defaults/"``).
        Used as the marker object's name within *dst_prefix*.
    dst_prefix : str
        Destination key prefix (e.g. ``".cache/metadata/<exec_id>/params/defaults/"``).
        Where the symlink marker is placed.

    Examples
    --------
    >>> bucket = "my-bucket"
    >>> dst_prefix = "destination/path/"
    >>> src_prefix = "my/source/prefix/"
    >>> create_s3_prefix_symlink(bucket, src_prefix, dst_prefix)
    """
    readable_src_prefix = src_prefix.replace("/", "／")

    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=f"{dst_prefix}{readable_src_prefix}",
        Body=b"",
    )


def s3_prefix_has_objects(bucket: str, prefix: str) -> bool:
    """Return True if at least one object exists under the given S3 prefix.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        Key prefix to check (should end with ``"/"``).
    """
    resp = boto3.client("s3").list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(resp.get("Contents"))


def copy_local_dir_to_s3(local_dir: str, s3_uri: str) -> None:
    """Recursively copy the contents of a local directory to an S3 prefix.

    Traverses *local_dir* and uploads all files to *s3_uri*, preserving the
    relative file structure. Uses ``/`` as the separator for S3 keys.

    Parameters
    ----------
    local_dir : str
        The local filesystem directory to copy from.
    s3_uri : str
        The target S3 URI (e.g. ``s3://bucket/prefix/``). The contents of
        *local_dir* will be placed under this prefix.

    Raises
    ------
    ValueError
        If *local_dir* does not exist or is not a directory.
    """
    if not os.path.isdir(local_dir):
        raise ValueError(f"Local directory does not exist: {local_dir}")

    bucket, key_prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")

    for dirpath, _, filenames in os.walk(local_dir):
        for filename in filenames:
            local_file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(local_file_path, local_dir)
            # S3 keys use '/' as the separator, regardless of OS
            s3_key = key_prefix + rel_path.replace(os.sep, "/")
            s3.upload_file(local_file_path, bucket, s3_key)


def copy_s3_prefix_to_local(s3_uri: str, local_dir: str, recursive: bool = True) -> None:
    """Copy the contents of an S3 prefix to a local directory.

    Traverses *s3_uri* and downloads files to *local_dir*, preserving
    the relative file structure.

    Parameters
    ----------
    s3_uri : str
        The source S3 URI (e.g. ``s3://bucket/prefix/``).
    local_dir : str
        The target local filesystem directory. Will be created if it does
        not exist.
    recursive : bool
        If True (default), recursively copies all files under the prefix.
        If False, only copies files immediately under the prefix.

    Raises
    ------
    ValueError
        If *s3_uri* is not a valid S3 URI.
    """
    if not is_s3_path(s3_uri):
        raise ValueError(f"Source path must be an S3 URI: {s3_uri}")

    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    os.makedirs(local_dir, exist_ok=True)

    pagination_kwargs = {"Bucket": bucket, "Prefix": prefix}
    if not recursive:
        pagination_kwargs["Delimiter"] = "/"

    for page in paginator.paginate(**pagination_kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip directory placeholders
            if key.endswith("/"):
                continue

            rel_path = os.path.relpath(key, prefix)
            local_file_path = os.path.join(local_dir, rel_path)

            # Ensure any nested local directories exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(bucket, key, local_file_path)


class S3TextStream(io.TextIOBase):
    """File-like object that appends text to an S3 object in real-time.

    Maintains a local in-memory copy, then pushes the fully
    consolidated object to S3 on each write.
    """

    def __init__(self, uri: str):
        self.uri = uri
        self.bucket, self.key = parse_s3_uri(uri)
        self.s3 = boto3.client("s3")
        self._closed = False
        self._local_copy = ""

    def write(self, text: str) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed stream.")

        self._local_copy += text

        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.key,
            Body=self._local_copy.encode("utf-8"),
        )
        return len(text)

    def flush(self) -> None:
        # No network buffer to flush, writes are immediate overwrites
        pass

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "S3TextStream":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
