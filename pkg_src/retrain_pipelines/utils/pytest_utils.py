
import os

import platform
import subprocess


abort_cmd = (
    '("Command+." to abort)' if platform.system() == "Darwin"
    else '("Ctrl+C" to abort)'
)


def _find_env_python(
    env_name: str
) -> str:
    """
    Returns the fullpath to the python binary
    (might be a symlink) for a given
    virtual environment.

    Params:
        - env_name (str):
            the name of the virtual envionment
            to consider.
            Works for conda or venv
            kind of virtual env.
    Results:
        - (str)
    """

    def _find_venv_python(venv_name):
        cmd = f'find / -path "*/{venv_name}/bin/python"'+\
               ' -print -quit 2>/dev/null'
        try:
            output = subprocess.check_output(
                cmd, shell=True, universal_newlines=True)
            return output.strip()
        except subprocess.CalledProcessError as e:
            if 1 == e.returncode:
                # not actual error(s) but warning(s)
                # probably non-sudo permission
                # on some directories: not an actual issue
                # in the herein context
                return e.output.strip()
            else:
                logger.info(f"Error finding directories: {e}")
                logger.info(f"Command output: {e.output}")
                logger.info(f"Command error output: {e.stderr}")

        return ""

    try:
        result = subprocess.run(
            ['conda', 'run', '-n', env_name, 'which', 'python'],
            capture_output=True, text=True)
        python_path = result.stdout.strip()
    finally:
        if not python_path:
            python_path = _find_venv_python(env_name)

    return python_path


def get_venv(
    virtual_env_name: str
) -> os._Environ:
    """
    Returns an instance of a virtual environment
    (with proper Python version and
     installed packages and all).

    Params:
        - virtual_env_name (str):

    Results:
        - (os._Environ)
    """

    print(f"Looking for virtual environment '{virtual_env_name}' " +
          f"{abort_cmd}.. ", end="")
    python_path = _find_env_python(virtual_env_name)
    assert python_path, \
           "Virtual environment for this test is missing."
    print("Found.")

    env = os.environ.copy()
    ############################################
    #    replace default python bin in PATH    #
    ############################################
    # drop existing python bin directory from PATH
    path_dirs = env['PATH'].split(os.pathsep)
    path_dirs = [d for d in path_dirs
                 if not os.path.exists(os.path.join(d, 'python'))]
    # prepend current environment python bin directory to PATH
    new_path = os.pathsep.join([os.path.dirname(python_path)] +
                               path_dirs)
    env['PATH'] = new_path
    ############################################

    return env

