
"""
execute() runs in a dedicated subprocess rather than in-process.

Some pipeline dependencies initialize C-level runtime state (GPU contexts,
distributed cluster sockets, native thread pools, …) that cannot be safely
inherited across fork boundaries or cleanly re-initialized within the same
process lifetime.  Running execute() in a subprocess guarantees a clean
slate for every pipeline run, regardless of what the calling process has
already imported or initialized.

params and result are exchanged through temp files (cloudpickle) because
that avoids any shared-memory or socket-based IPC that the C runtimes
above may interfere with.  retrain_pipeline_type is passed back separately
so it is available in os.environ before the result file is deserialized
(retrain-pipelines uses it to route module imports at load time).
"""
import os
import sys
import shutil
import logging
import tempfile
import threading
import subprocess
import cloudpickle

_dag_render_cell_hash = None


def _run_helper(pipeline_fullpath: str, helper_body_tpl) -> object:
    """
    Spawn a subprocess from helper_body_tpl(result_file), a callable
    that receives the result_file path and returns the helper script body.
    The helper script is placed under a subdir named after the pipeline's
    parent directory so that core.DAG.init derives the correct
    pipeline_name from __file__.
    Streams stdout/stderr, unpickles and returns the result payload.
    """
    from retrain_pipelines.utils import in_notebook

    result_file = tempfile.mktemp(suffix='.pkl')
    _tmp_dir    = tempfile.mkdtemp()
    # on Kaggle, mkdtemp() and the working dir both live under /tmp;
    # if mkdtemp() returns dirname(launch_dir), helper_dir == launch_dir,
    # causing a circular import; the nested mkdtemp works around that
    helper_dir  = os.path.join(
        tempfile.mkdtemp(dir=_tmp_dir),
        os.path.basename(os.path.dirname(pipeline_fullpath))
    )
    os.makedirs(helper_dir)
    helper_file = os.path.join(
        helper_dir, os.path.basename(pipeline_fullpath)
    )

    with open(helper_file, 'w') as f:
        f.write(helper_body_tpl(result_file))

    env  = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    proc = subprocess.Popen(
        [sys.executable, helper_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )

    if in_notebook():
        from rich.text import Text
        from rich.console import Console
        _nb_console = Console(
            file=sys.stdout,
            soft_wrap=False,
            width=1_000_000,
            force_terminal=True,
            legacy_windows=False,
        )
        def _relay(src, dst):
            for line in src:
                _nb_console.print(
                    Text.from_ansi(line.decode('utf-8',
                                   errors='replace')),
                    sep="", end=""
                )
    else:
        def _relay(src, dst):
            for line in src:
                dst.write(line.decode('utf-8', errors='replace'))
                dst.flush()

    t_out = threading.Thread(target=_relay,
                             args=(proc.stdout, sys.stdout))
    t_err = threading.Thread(target=_relay,
                             args=(proc.stderr, sys.stderr))
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    os.unlink(helper_file)
    shutil.rmtree(helper_dir, ignore_errors=True)
    shutil.rmtree(_tmp_dir, ignore_errors=True)

    if not os.path.exists(result_file):
        raise RuntimeError(
            "Helper subprocess failed to produce a result.")

    with open(result_file, 'rb') as f:
        status, payload = cloudpickle.load(f)
    os.unlink(result_file)

    if status == 'err':
        raise payload

    return payload


def launch(pipeline_fullpath: str,
           params: dict = None):
    """Launch a pipeline execution.

    Params:
        - pipeline_fullpath (str):
            absolute path to the module
            containing the "retrain_pipeline" DAG
            object declaration.
            NOTE: file existence is to be ensured
                  by the caller.
        - params (dict):
            the DAG parameter values to be applied.
    """
    params = params or {}

    from retrain_pipelines.utils import animate_wave
    from retrain_pipelines.__version__ import __version__
    from retrain_pipelines.dag_engine.rp_logging import \
        RichLoggingController

    animate_wave(f"retrain-pipelines {__version__}",
                 wave_length=6, delay=0.01, loops=2)

    dag_module_name   = os.path.splitext(os.path.basename(
                            pipeline_fullpath))[0]
    launch_dir        = os.path.dirname(pipeline_fullpath)

    params_file = tempfile.mktemp(suffix='.pkl')
    env_file    = tempfile.mktemp(suffix='.txt')

    with open(params_file, 'wb') as f:
        cloudpickle.dump(params, f)

    def _helper_body(result_file):
        return f"""
import os, sys
os.environ['RP_LAUNCHER_SUBPROCESS'] = '1'
sys.path.insert(0, {repr(launch_dir)})
import cloudpickle
with open({repr(params_file)}, 'rb') as f:
    params = cloudpickle.load(f)
from retrain_pipelines.dag_engine.runtime import execute
from {dag_module_name} import retrain_pipeline
try:
    result = ('ok', execute(retrain_pipeline, params=params))
except Exception as e:
    result = ('err', e)
with open({repr(env_file)}, 'w') as f:
    f.write(os.environ.get('retrain_pipeline_type', ''))
with open({repr(result_file)}, 'wb') as f:
    cloudpickle.dump(result, f)
os._exit(0)
"""

    payload = _run_helper(pipeline_fullpath, _helper_body)

    retrain_pipeline_type = ''
    if os.path.exists(env_file):
        with open(env_file) as f:
            retrain_pipeline_type = f.read().strip()
        os.unlink(env_file)

    if retrain_pipeline_type:
        # ensure retrain-pipelines python modules
        # load properly for "context_dump"
        # and "final_result" (below) to unpickle sucessfully
        # if holding any reference to custom classes
        os.environ['retrain_pipeline_type'] = retrain_pipeline_type

    final_result, context_dump = payload

    logging_controller = RichLoggingController()
    logging_controller.activate()
    logging.getLogger().info(
        f"{context_dump['username']} - execution {context_dump['exec_id']} - " +
        f"{context_dump['pipeline_name']} - final result : {final_result}\n" +
        (
            f"model version blessed : {context_dump['model_version_blessed']}" \
            if "model_version_blessed" in context_dump else ""
        )
    )
    logging_controller.deactivate()

    return payload


def dag_help(pipeline_fullpath: str):
    """
    Invoke retrain_pipeline.help()
    in a dedicated subprocess.

    logging.Logger.handle is patched
    before the pipeline module is imported
    to suppress retrain_pipelines log records
    emitted at DAG instantiation time,
    so only the help string itself is
    returned to the caller.

    Params:
        - pipeline_fullpath (str):
            absolute path to the module
            containing the "retrain_pipeline" DAG
            object declaration.
            NOTE: file existence is to be ensured
                  by the caller.
    """
    dag_module_name   = os.path.splitext(os.path.basename(
                            pipeline_fullpath))[0]
    launch_dir        = os.path.dirname(pipeline_fullpath)

    def _helper_body(result_file):
        return f"""
import os, sys, logging
os.environ['RP_LAUNCHER_SUBPROCESS'] = '1'
sys.path.insert(0, {repr(launch_dir)})
import cloudpickle
_orig_handle = logging.Logger.handle
logging.Logger.handle = lambda self, record: \
    None if self.name.startswith('retrain_pipelines') \
    else _orig_handle(self, record)
from {dag_module_name} import retrain_pipeline
logging.Logger.handle = _orig_handle
try:
    result = ('ok', retrain_pipeline.help())
except Exception as e:
    result = ('err', e)
with open({repr(result_file)}, 'wb') as f:
    cloudpickle.dump(result, f)
os._exit(0)
"""
    return _run_helper(pipeline_fullpath, _helper_body)


def dag_render(pipeline_fullpath: str) -> None:
    """
    Render the pipeline's DAG as an interactive SVG in the notebook.

    Generates the HTML rendering in a dedicated subprocess
      - (see dag_help for the Logger.handle patch rationale).

    Params:
        - pipeline_fullpath (str):
            absolute path to the module
            containing the "retrain_pipeline" DAG
            object declaration.
            NOTE: file existence is to be ensured
                  by the caller.
    """
    dag_module_name   = os.path.splitext(os.path.basename(
                            pipeline_fullpath))[0]
    launch_dir        = os.path.dirname(pipeline_fullpath)

    def _helper_body(result_file):
        return f"""
import os, sys, logging
os.environ['RP_LAUNCHER_SUBPROCESS'] = '1'
sys.path.insert(0, {repr(launch_dir)})
import cloudpickle
_orig_handle = logging.Logger.handle
logging.Logger.handle = lambda self, record: \
    None if self.name.startswith('retrain_pipelines') \
    else _orig_handle(self, record)
from {dag_module_name} import retrain_pipeline
logging.Logger.handle = _orig_handle
try:
    from retrain_pipelines.dag_engine.renderer import dag_svg
    html_body_str = dag_svg(retrain_pipeline)
    result = ('ok', html_body_str)
except Exception as e:
    result = ('err', e)
with open({repr(result_file)}, 'wb') as f:
    cloudpickle.dump(result, f)
os._exit(0)
"""

    html_body_str = _run_helper(pipeline_fullpath, _helper_body)

    from IPython.display import display, HTML

    display(HTML("""
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lobster&amp;display=swap">
        <div style="
            background: rgba(128, 0, 128, 0.8);
            border: 4px solid #800080;
            border-radius: 8px;
            position: relative;
        ">
            """ + html_body_str + """
            <div style="
                position: absolute;
                bottom: 10px;
                right: 10px;
                color: #800080;
                font-family: 'Lobster';
                font-weight: bold;
                font-size: 18px;
                text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
            ">
                retrain-pipelines
            </div>
        </div>
    """))


if __name__ == "__main__":
    launch(os.path.realpath(__file__))

