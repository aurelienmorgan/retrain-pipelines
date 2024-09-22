
import os
import sys

from .local_launcher import retrain_pipelines_local \
                                as _retrain_pipelines_local

from IPython.core.magic import register_line_magic


@register_line_magic
def retrain_pipelines_local(
    command
):

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

    _ = _retrain_pipelines_local(command, env)


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
                retrain_pipelines_local, 'line')

