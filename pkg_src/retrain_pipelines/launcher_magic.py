
"""
Importable Jupyter line magic: %retrain_pipelines

Usage in a notebook:
    %load_ext retrain_magic

    %retrain_pipelines filefullname start --exec-params params
    %retrain_pipelines /path/to/file start --exec-params params
    %retrain_pipelines filefullname restart checkpoint_name --exec-params params
    %retrain_pipelines /path/to/file restart checkpoint_name --exec-params params
"""

import os
import sys
import argparse

from IPython import get_ipython
from IPython.core.magic import register_line_magic

from .launcher import launch, dag_help, dag_render


def _tokenize(command: str) -> list:
    """
    Split a magic command line into tokens, keeping
    brace-delimited dict literals intact as single tokens.
    """
    tokens = []
    i = 0
    n = len(command)
    while i < n:
        if command[i].isspace():
            i += 1
        elif command[i] == '{':
            depth, j = 0, i
            while j < n:
                if   command[j] == '{': depth += 1
                elif command[j] == '}': depth -= 1
                j += 1
                if depth == 0:
                    break
            tokens.append(command[i:j])
            i = j
        else:
            j = i
            while j < n and not command[j].isspace():
                j += 1
            tokens.append(command[i:j])
            i = j
    return tokens


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrain_pipelines",
        description=(
            "Launch a retrain-pipelines DAG execution in a clean subprocess.\n"
            "Returns the pipeline payload so the result is assignable\n"
            "directly in the notebook cell:\n"
            "  result = %retrain_pipelines pipeline.py start --exec-params params"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "filefullname",
        help="variable name holding the path " +
              "or path string (absolute or relative) " +
              "to the python module holding the DAG declaration.")
    parser.add_argument(
        "action",
        choices=["start", "restart", "help", "render"],
        help="'start' to execute the pipeline, " +
             "'restart <from>' to re-execute from a task or task-group, " +
             "'help' to return the pipeline's DAG help, "
             "'render' to display the DAG graph inline.")
    parser.add_argument(
        "restart_from",
        nargs="?",
        default=None,
        help="required when action is 'restart': " +
             "plain identifier string. " +
             "task name (or task-group name) from which " +
             "last execution shall be re-initiated.")
    parser.add_argument("--exec-params",  dest="exec_params",
                        required=False)
    parser.add_argument("--exec-workers", dest="exec_workers",
                        required=False)
    return parser


def _resolve_path_str(raw: str, ns: dict) -> str:
    """turns "Variable name" (if exists) into its string value,
    otherwise treat as raw path string (quoted or unquoted)."""
    if raw in ns:
        value = ns[raw]
        if not isinstance(value, str):
            raise TypeError(
                f"%retrain_pipelines: {raw!r} must resolve to a str, "
                f"got {type(value).__name__}")
        return value
    if len(raw) >= 2 and (
        (raw[0] == '"'  and raw[-1] == '"') or
        (raw[0] == "'"  and raw[-1] == "'")
    ):
        return raw[1:-1]
    return raw


def _validate_path(path: str) -> str | None:
    """Resolve to an absolute real path
    and verify the file exists.
    Returns the resolved path on success,
    prints to stderr and returns
    None on failure."""
    path = os.path.realpath(path)
    if not os.path.exists(path):
        print(
            f"Pipeline file not found: {path}",
            file=sys.stderr, flush=True)
        return None
    return path


def _resolve_dict(raw: str, ns: dict) -> dict:
    """Variable name or inline dict literal → dict."""
    try:
        value = eval(raw, ns)  # noqa: S307
    except Exception as e:
        raise ValueError(
                f"%retrain_pipelines: could not evaluate {raw!r}"
            ) from e
    if not isinstance(value, dict):
        raise TypeError(
            f"%retrain_pipelines: expected a dict, "
            f"got {type(value).__name__}")
    return value


@register_line_magic
def retrain_pipelines(command: str) -> None:
    """Line magic implementation."""
    ## ###################################################
    # retrieve parameters values from notebook namespace #
    ################################################### ##
    ip = get_ipython()
    if ip is None:
        raise RuntimeError(
            "%retrain_pipelines must be run " +
            "inside a Jupyter environment.")

    parser = _make_parser()
    try:
        args = parser.parse_args(_tokenize(command))
    except SystemExit:
        # error message already written to stderr at this point
        return

    if args.action == "help":
        filefullname = _validate_path(_resolve_path_str(
                            args.filefullname, ip.user_ns))
        if filefullname is None:
            return
        result = dag_help(filefullname)
        if result:
            print(result)
        return

    if args.action == "render":
        filefullname = _validate_path(_resolve_path_str(
                            args.filefullname, ip.user_ns))
        if filefullname is None:
            return
        dag_render(filefullname)
        return

    if args.action == "restart" and args.restart_from is None:
        print(
            "%retrain_pipelines: 'restart' must be followed" +
            " by a task name (or task-group name) argument.",
            file=sys.stderr, flush=True
        )
        return

    ns = ip.user_ns

    filefullname  = _validate_path(_resolve_path_str(
                        args.filefullname, ns))
    if filefullname is None:
        return
    restart_from  = args.restart_from
    exec_params   = _resolve_dict(args.exec_params,  ns) \
                    if args.exec_params  is not None else None
    exec_workers  = _resolve_dict(args.exec_workers, ns) \
                    if args.exec_workers is not None else None
    ######################################################

    return launch(filefullname, params=exec_params)


def load_ipython_extension(ipython):
    """
    Any module file that define a function
    named `load_ipython_extension`
    can be loaded via `%load_ext module.path`
    or be configured to be autoloaded
    by IPython at startup time.
    """
    ipython.register_magic_function(
        retrain_pipelines, magic_kind="line")

