
import os
import sys

from .utils import animate_wave
from .__version__ import __version__
from .legacy_launcher import retrain_pipelines_legacy \
                                as _retrain_pipelines_legacy

from IPython.core.magic import register_line_magic


@register_line_magic
def retrain_pipelines_legacy(
    command
):
    animate_wave(f"retrain-pipelines {__version__}",
                 wave_length=6, delay=0.01, loops=2)

    env = os.environ.copy()
    ############################################
    #    replace default python bin in PATH    #
    ############################################
    # drop existing python bin directory from PATH
    path_dirs = env['PATH'].split(os.pathsep)
    path_dirs = [d for d in path_dirs
                 if not os.path.exists(os.path.join(d, 'python'))]
    # prepend current environment python bin directory to PATH
    python_path = sys.executable
    new_path = os.pathsep.join([os.path.dirname(python_path)] +
                               path_dirs)
    env['PATH'] = new_path
    ############################################
    env['launched_from_magic'] = 'True'

    _ = _retrain_pipelines_legacy(command, env)


# register magic with running IPython.
def load_ipython_extension(ipython):
    """
    Any module file that define a function
    named `load_ipython_extension`
    can be loaded via `%load_ext module.path`
    or be configured to be autoloaded
    by IPython at startup time.
    """
    ipython.register_magic_function(
                retrain_pipelines_legacy, 'line')

