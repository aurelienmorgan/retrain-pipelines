import os

from ..utils.s3_utils import ensure_s3_bucket, is_s3_path


class NotSupportedError(Exception):
    """Raised when an unsupported database scheme is detected."""


class Config:
    """Application configuration."""

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """Create the directory if it does not exist."""
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError as e:
            print(f"Permission denied on {path}: {e}")
            raise e
        except Exception as e:
            print(f"Error creating directory {path}: {e}")
            raise e

    ################################################################
    #                        RP_ASSETS_CACHE                       #
    ################################################################
    @staticmethod
    def get_assets_cache_root() -> str:
        """
        Return the assets cache directory path.

        Which is home to inner-workings serialization stores
        (DAG.params, execution context, tasks payloads, etc.)

        The value is taken from the environment variable RP_ASSETS_CACHE if set,
        otherwise a default under ``~/.cache/retrain-pipelines/`` is used.
        The path is expanded and normalized to end with a path separator.
        S3 URIs (``s3://…``) are accepted; the bucket is created if it does not exist.
        """
        default = "~/.cache/retrain-pipelines/"
        path = os.environ.get("RP_ASSETS_CACHE", default)

        if is_s3_path(path):
            ensure_s3_bucket(path)
            return path.rstrip("/") + "/"

        path = os.path.expanduser(path)
        Config._ensure_dir(path)
        return path.rstrip(os.sep) + os.sep

    ################################################################
    #                      RP_ARTIFACTS_STORE                      #
    ################################################################
    @staticmethod
    def get_artifacts_store_root() -> str:
        """
        Return the artifacts store directory path.

        Which is home to non-model, non-dataset, and non-metadata artifacts
        (Training code, scripts, and configuration files,
         Data validation reports, charts, and documentation,
         Feature engineering outputs, etc.).

        The value is taken from the environment variable RP_ARTIFACTS_STORE if set,
        otherwise a default under ``<assets_cache>/artifacts/`` is used.
        The path is expanded and normalized to end with a path separator.
        S3 URIs (``s3://…``) are accepted; the bucket is created if it does not exist.
        """
        default = os.path.join(Config.get_assets_cache_root(), "artifacts")
        path = os.environ.get("RP_ARTIFACTS_STORE", default)

        if is_s3_path(path):
            ensure_s3_bucket(path)
            return path.rstrip("/") + "/"

        path = os.path.expanduser(path)
        Config._ensure_dir(path)
        return path.rstrip(os.sep) + os.sep

    ################################################################
    #                      RP_WEB_SERVER_LOGS                      #
    ################################################################
    @staticmethod
    def get_web_server_logs_root() -> str:
        """
        Return the web server logs directory path.

        Which is used for "local" installs of the web UI.

        The value is taken from the environment variable RP_WEB_SERVER_LOGS if set,
        otherwise a default under ``<assets_cache>/logs/web_server/`` is used.
        The path is expanded and normalized to end with a path separator.
        S3 URIs (``s3://…``) are accepted; the bucket is created if it does not exist.
        """
        default = os.path.join(Config.get_assets_cache_root(), "logs", "web_server")
        path = os.environ.get("RP_WEB_SERVER_LOGS", default)

        if is_s3_path(path):
            ensure_s3_bucket(path)
            return path.rstrip("/") + "/"

        path = os.path.expanduser(path)
        Config._ensure_dir(path)
        return path.rstrip(os.sep) + os.sep

    ##########################################
    # RP_WEB_SERVER_URL / RP_WEB_SERVER_PORT #
    ##########################################
    @staticmethod
    def get_web_server_port() -> int:
        """
        Return the web server port.

        The value is taken from the environment variable RP_WEB_SERVER_PORT if set,
        otherwise defaults to "5001".
        """
        return int(os.environ.get("RP_WEB_SERVER_PORT", "5001"))

    @staticmethod
    def get_web_server_url() -> str:
        """
        Return the web server URL ``<scheme>://<hotname>:<port>``.

        Which serves for the internal dag-engine.

        The value is taken from the environment variable RP_WEB_SERVER_URL if set
        (in which case RP_WEB_SERVER_PORT is ignored),
        otherwise constructed as ``http://localhost:{RP_WEB_SERVER_PORT}/`` where
        RP_WEB_SERVER_PORT defaults to "5001".

        Note:
        -----
        The URL is normalized to not end with a trailing slash.
        """
        port = Config.get_web_server_port()
        default = f"http://localhost:{port}/"
        url = os.environ.get("RP_WEB_SERVER_URL", default)
        return url.rstrip("/")

    ################################################################
    #                   RP_METADATASTORE_URL                   #
    ################################################################
    @staticmethod
    def get_metadatastore_url() -> str:
        """
        Return the metadata store URL.

        The value is taken from the environment variable RP_METADATASTORE_URL if set
        (beware that we perform no connextion-string validation against it),
        otherwise constructed as ``sqlite:///<assets_cache_root>local_metadatastore.db?timeout=10.0``
        where assets_cache_root is provided by get_assets_cache_root().

        We instruct SQLite to wait for a lock to be released before raising a "db locked" error
        on concurrency issue. Setting a timeout in the URL.

        Raises
        ------
            NotSupportedError: If RP_METADATASTORE_URL is unset and the default assets cache
                is an S3 path, since SQLite cannot operate over S3.
            FileNotFoundError: If the resolved URL is a ``sqlite:///`` path whose parent
                directory does not exist.

        Note:
        -----
        Typical PostgreSql connection-string reads :
        ```
        >>> os.environ["RP_METADATASTORE_URL"] = os.environ.get(
        ...    "RP_METADATASTORE_URL",
        ...    "postgresql://postgres:mypassword@host:port/postgres",
        ...)
        ```
        Avoid Pgpooler "transaction" or "statement" modes.
        They do not support prepared statements properly.
        """
        assets_cache_root = Config.get_assets_cache_root()

        if is_s3_path(assets_cache_root) and "RP_METADATASTORE_URL" not in os.environ:
            raise NotSupportedError(
                "RP_ASSETS_CACHE is an S3 URI ; SQLite cannot operate over S3. "
                "Set RP_METADATASTORE_URL to a local path (``sqlite:///….db?timeout=10.0``) "
                "or a network database (e.g. ``postgresql://…``)."
            )

        default = f"sqlite:///{assets_cache_root}local_metadatastore.db?timeout=10.0"
        url = os.environ.get("RP_METADATASTORE_URL", default)

        if url.startswith("sqlite:///"):
            # The .db file need not exist yet ; validate only that the parent directory does.
            # strip scheme and query string:
            # ``sqlite:///path/to/file.db?timeout=10.0`` => ``path/to/file.db``
            parent = os.path.dirname(url[len("sqlite:///") :].split("?")[0]) or "."
            if not os.path.isdir(parent):
                raise FileNotFoundError(
                    f"SQLite parent directory does not exist: {parent!r} "
                    f"(resolved from RP_METADATASTORE_URL={url!r})."
                )

        return url

    ################################################################
    #                RP_METADATASTORE_ASYNC_URL                #
    ################################################################
    # For the WebConsole connection pool
    @staticmethod
    def get_metadatastore_async_url() -> str:
        """
        Return the async metadata store URL for the WebConsole connection pool.

        The value is taken from the environment variable RP_METADATASTORE_ASYNC_URL if set
        (highly discouraged since sync and async shall go in pair. Also beware that
        we perform no connextion-string validation against it), otherwise it is derived from
        `` get_metadatastore_url`` by converting the database scheme to its async counterpart:
          - sqlite:// -> sqlite+aiosqlite:// (query parameters like ?timeout=... are stripped)
          - postgresql:// -> postgresql+asyncpg://
          - postgres:// -> postgresql+asyncpg:// (normalization)

        Raises
        ------
            NotSupportedError: If the scheme in RP_METADATASTORE_URL is not recognised,
                or if RP_METADATASTORE_URL is unset and the default assets cache is an S3 path.

        Note:
        -----
        Typical PostgreSql connection-string reads :
        ```
        >>> os.environ["RP_METADATASTORE_ASYNC_URL"] = os.environ.get(
        ...    "RP_METADATASTORE_ASYNC_URL",
        ...    "postgresql+asyncpg://postgres:mypassword@host:port/postgres",
        ...)
        ```
        """
        # First check if environment variable is set
        env_url = os.environ.get("RP_METADATASTORE_ASYNC_URL")
        if env_url is not None:
            return env_url

        # Otherwise derive from sync URL
        # (get_metadatastore_url raises NotSupportedError
        #  if assets cache is S3 with no explicit URL)
        sync_url = Config.get_metadatastore_url()

        if sync_url.startswith("sqlite://"):
            # Strip any query parameters (e.g., ?timeout=10.0)
            base = sync_url.split("?")[0]
            async_url = base.replace("sqlite://", "sqlite+aiosqlite://", 1)
            return async_url

        if sync_url.startswith("postgresql://"):
            async_url = sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return async_url

        if sync_url.startswith("postgres://"):
            async_url = sync_url.replace("postgres://", "postgresql+asyncpg://", 1)
            return async_url

        # Unsupported scheme – raise NotSupportedError
        raise NotSupportedError(f"Unsupported database scheme in RP_METADATASTORE_URL: {sync_url}")
