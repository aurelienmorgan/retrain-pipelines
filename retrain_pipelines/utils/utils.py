
import os
import sys

import re
import stat
import shutil
import logging
import tempfile
import itertools
import subprocess
import importlib.util
from textwrap import dedent

# conditional import of the "torch" package
# required for some callables in hp_dict
# (but not necessarily installed
#  on virtual env that do not require it)
if importlib.util.find_spec("torch") is not None:
    import torch

from contextlib import contextmanager

retrain_pipeline_type = os.getenv("retrain_pipeline_type")

logger = logging.getLogger(__name__)


def _load_and_get_function(
    module_path:str,
    qualified_module_name:str,
    function_name: str
) -> callable:
    """
    Loads a python module,
    which can be user-provided
    (path given through flow
     "pipeline_card_module_dir" parameter)
    and returns its "get_html" function.

    Params:
        - module_path (str):
            the full path to the source
            python module (.py) file
        - qualified_module_name (str):
            the module full qualified name
            to be set.
        - function_name (str):
            the (non-qualified/short)*
            function name.

    Results:
        - (callable)
    """

    # Load the module from the file path
    spec = importlib.util.spec_from_file_location(
                qualified_module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the function
    if hasattr(module, function_name):
        function = getattr(module, function_name)
    else:
        raise AttributeError(
                f"Function 'get_html' not found" +
                f"in module '{qualified_module_name}'")

    # Create a proxy module
    proxy_module = type(sys)(qualified_module_name)
    sys.modules[qualified_module_name] = proxy_module

    # Assign the function to the proxy module
    setattr(proxy_module, function_name, function)

    # function is fully qualified
    # import inspect
    # logger.debug(inspect.getmodule(function))

    return function


def get_get_html(
    pipeline_card_module_dir: str
):
    """
    Loads the "pipeline_card" module,
    which can be user-provided
    (path given through flow
     "pipeline_card_module_dir" parameter)
    and returns its "get_html" function.
    """

    pipeline_card_module_path = \
        os.path.realpath(os.path.join(pipeline_card_module_dir,
                                      "pipeline_card.py"))

    get_html = \
        _load_and_get_function(
            pipeline_card_module_path,
            f"retrain_pipelines.pipeline_card."+
                f"{retrain_pipeline_type}.pipeline_card",
            "get_html"
        )

    return get_html


def get_preprocess_data_fct(
    preprocess_module_dir: str
):
    """
    Loads the "preprocessing" module,
    which can be user-provided
    (path given through flow
     "preprocess_module_dir" parameter).
    and returns its "preprocess_data_fct"
    function.
    """

    preprocessing_module_path = \
        os.path.realpath(os.path.join(preprocess_module_dir,
                                      "preprocessing.py"))

    preprocess_data_fct = \
        _load_and_get_function(
            preprocessing_module_path,
            f"retrain_pipelines.model."+
                f"{retrain_pipeline_type}.preprocessing",
            "preprocess_data_fct"
        )

    return preprocess_data_fct


def _create_requirements_from_conda(
    target_dir: str,
    exclude: list = []
):
    """
    Params:
        - target_dir (str):
            Where the "requirements.txt" file
            shall be placed.
        - exclude (list[str]):
            Names of packages to be excluded
            from the resulting file
            if present in the active environement.
    """
    """
    Tried other (more straightforward) ways to do it,
    none didn't mess with version numbers
    so "pip install -n requirments.txt"
    succeeds in satisfying them all.
    Recall how pip and conda don't co-exist nicely..
    """

    ######################
    #  conda list freeze #
    ######################
    # retrieve exhaustive list of installed packages
    # on the active conda virtual environment
    # the ones installed via "conda"
    # and the ones installed via "pip"
    result = subprocess.run(
        ['conda', 'list', '-n', os.path.basename(sys.prefix), '--export'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        logger.debug(f"Error running command: {result.stderr}")
        return

    # store entries as {package: version} key/value paris
    lines = result.stdout.splitlines()
    conda_env_packages = {}

    for line in lines:
        if '=' in line:
            line_splits = line.split('=')
            package_name = line_splits[0]
            if "pytorch" == package_name:
                package_name = "torch"
            elif "pytorch-tabnet" == package_name:
                package_name = "pytorch_tabnet"
            conda_env_packages[package_name] = line_splits[1]
    ######################

    ######################
    #   pip list freeze  #
    ######################
    # pip freeze on the active virtual environment
    # we need to truncate long version-numbering
    # such as "1.13.1.post200" or "0.11.1b20240718"
    pip_freeze = subprocess.run(["pip", "list", "--format", "freeze"],
                                capture_output=True, text=True)
    pip_freeze_output_lines = pip_freeze.stdout.splitlines()

    # fix the package version-numbering issue
    updated_packages = []
    for line in pip_freeze_output_lines:
        if '==' in line:
            pkg_name, pip_version = line.split('==')
            if pkg_name in exclude:
                logger.info(f"excluding package '{pkg_name}' "+
                            f"from generated requirements.txt")
                continue
            if pkg_name in conda_env_packages:
                conda_version = conda_env_packages[pkg_name]
                
                # Extract major and minor versions for comparison
                pip_major_minor = '.'.join(pip_version.split('.')[:2])
                conda_major_minor = '.'.join(conda_version.split('.')[:2])

                # Swap versions if both major and minor versions match
                if pip_major_minor == conda_major_minor:
                    updated_packages.append(f"{pkg_name}=={conda_version}")
                else:
                    updated_packages.append(line)
            else:
                updated_packages.append(line)
    ######################

    ######################
    #  requirements.txt  #
    ######################
    with open(os.path.join(target_dir, 'requirements.txt'),
              'w') as f:
        f.write("\n".join(updated_packages))
    ######################

def _create_requirements_from_pip(
    target_dir: str,
    exclude: list = []
):
    """
    Params:
        - target_dir (str):
            Where the "requirements.txt" file
            shall be placed.
        - exclude (list[str]):
            Names of packages to be excluded
            from the resulting file
            if present in the active environement.
    """
    python_executable = sys.executable
    command = [python_executable, "-m", "pip", "freeze"]
    result = subprocess.run(
        command, capture_output=True, text=True)
    entries = result.stdout.splitlines()
    filtered_entries = [entry for entry in entries
                        if re.split(r'(==| @ )', entry)[0] not in exclude]
    with open(os.path.join(target_dir, 'requirements.txt')
              , 'w') as f:
        f.write('\n'.join(filtered_entries) + '\n')

def create_requirements(
    target_dir: str,
    exclude: list = []
):
    """
    Params:
        - target_dir (str):
            Where the "requirements.txt" file
            shall be placed.
        - exclude (list[str]):
            Names of packages to be excluded
            from the resulting file
            if present in the active environement.
    """

    # Check for Conda environment
    if is_conda_env():
        _create_requirements_from_conda(
            target_dir, exclude)
    else:
        _create_requirements_from_pip(
            target_dir, exclude)


def _dict_list_get_combinations(
    param_dict: dict
) -> list:
    """
    Returns list of dicts, where
    each dict represents one combination
    of sets of key/value.

    Params:
        - param_dict (dict):
            dictionnary whose values
            all are lists (of various length)

    Results:
        -list[dict]:
            all possible combinations of
            dicts with key/value possible
            (where all keys are preserved).

    Example usage:
        {"a": [1, 2], "b": [3]}
        return the folowwing :
        [
            {"a": 1, "b": 3},
            {"a": 2, "b": 3}
        ]
    """

    # Recursively generate combinations
    keys, values = zip(*param_dict.items())
    sub_combinations = []
    for combination in itertools.product(*values):
        combo_dict = dict(zip(keys, combination))
        # Evaluate 'optimizer_fn' and/or 'scheduler_fn'
        # if it exists in the current dictionary
        if "optimizer_fn" in combo_dict:
            combo_dict["optimizer_fn"] = eval(combo_dict["optimizer_fn"])
        if "scheduler_fn" in combo_dict:
            combo_dict["scheduler_fn"] = eval(combo_dict["scheduler_fn"])
        sub_combinations.append(combo_dict)
    return sub_combinations

def dict_dict_list_get_all_combinations(
    nested_dict: dict
) -> list:
    """
    Returns list of dict (of nested dict), where
    each dict represents one combination
    of sets of key/key/value.

    Params:
        -nested_dict (dict):
            1-level nested dictionary
            where nested dictionaries
            are of the form "dict of list".

    Results:
        - list[dict[dict]]:
            all possible combinations of
            dicts with key/key/value possible
            (where all keys are preserved).

    Example usage:
        {"AA": {"a": [1, 2], "b": [3]}, "BB" {"d": [4]}
        return the folowwing :
        [
            {"AA": {"a": 1, "b": 3}, "BB" {"d": 4},
            {"AA": {"a": 2, "b": 3}, "BB" {"d": 4}
        ]
    """

    # Generate combinations for each sub-dictionary
    sub_combinations = {key: _dict_list_get_combinations(value)
                        for key, value in nested_dict.items()}
    # Cartesian product of the combinations of the sub-dictionaries
    all_combinations = [
        {key: combo
         for key, combo in zip(sub_combinations.keys(), value)}
        for value in itertools.product(*sub_combinations.values())
    ]

    return all_combinations


def flatten_dict(
    dict_: dict,
    callable_to_name: bool = False,
    parent_key: str = ''
):
    """
    Recursive flatten of nested dictionnaries.
    Joins keys with "_" separator.

    If callable_to_name is True,
    callables are replaced with their names.

    Example usage:
        - {"AA": {"a0": 1, "a1": 2},
           "BB": {"b0": 0, "b1": 4, "b2": 10.0}}
          gives
          {"AA_a0": 1, "AA_a1": 2,
           "BB_b0": 0, "BB_b1": 4, "BB_b2": 10.0}
    """

    items = {}
    for k, v in dict_.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(
                flatten_dict(v, callable_to_name, new_key))
        else:
            if callable_to_name:
                if callable(v):
                    v = v.__name__
                elif isinstance(v, list):
                    v = [eval(elmt).__name__
                         if (
                            isinstance(elmt, str) and
                            callable(eval(elmt))
                         ) else elmt
                         for elmt in v]
            items[new_key] = v

    return items


def system_has_conda() -> bool:
    try:
        # Try to run 'conda --version'
        subprocess.run(
            ['conda', '--version'], check=True,
            stderr=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True
        )
        return True  # 'conda' is present
    except:
        # FileNotFoundError or subprocess.CalledProcessError
        # or whatever if the above fails (for whatever reason)
        return False  # 'conda' is absent


def is_conda_env() -> bool:
    """
    Whether or not the current python environment
    is a conda virtual one.
    """
    conda_prefix_set = 'CONDA_PREFIX' in os.environ

    conda_env = os.environ.get('CONDA_PREFIX', '')
    sys_prefix_is_conda = conda_env and sys.prefix.startswith(conda_env)
    
    return conda_prefix_set and sys_prefix_is_conda


def venv_as_conda(venv_path, conda_env_name):
    """
    Duplicate current venv as a conda env
    (typical use case is MLserver conda.pack
     on the current (non-conda) virtual environment).
    """
    """
    Note: requires a conda install (e.g. miniforge).
    """

    # retrieve python version
    python_executable = os.path.join(venv_path, 'bin', 'python')
    python_version = subprocess.check_output(
        [python_executable, '--version']).decode().strip().split()[1]
    # create new conda env
    subprocess.run(['conda', 'create', '-n', conda_env_name,
                    f'python={python_version}', '--yes'], check=True)
    # list dependencies
    requirements_path = os.path.join(tempfile.gettempdir(), 'requirements.txt')
    # ~/venv_root/metaflow_venv/bin/pip freeze
    # /home/organization/miniforge3/envs/_metaflow_venv/bin/pip freeze
    subprocess.run([os.path.join(venv_path, 'bin', 'pip'), 'freeze'],
                   stdout=open(requirements_path, 'w'), check=True)
    # install into the new conda env
    install_cmd = dedent(f"""
    bash -c "
    source $(conda info --base)/etc/profile.d/conda.sh && \
    conda activate {conda_env_name} && \
    $(conda info --base)/envs/{conda_env_name}/bin/python -m \
        pip install -r {requirements_path}
    "
    """)
    logger.info(install_cmd)
    subprocess.run(install_cmd, shell=True, check=True)


def find_env_python(
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


def grant_read_access(file_path: str):
    """
    set read access to all users.

    Params:
        - file_path(str)
    """

    # Get the current file permissions
    current_permissions = stat.S_IMODE(os.lstat(file_path).st_mode)
    # Add read permission for all users
    new_permissions = \
        current_permissions | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    # Set the new permissions on the file
    os.chmod(file_path, new_permissions)


@contextmanager
def tmp_os_environ(env_updates: dict):
    """
    context to execute code snipet
    within an OS environement
    with specific set of variables.

    Params:
        - env_updates (dict):
            env_var/env_var_value pairs

    Example:
    ```
    with tmp_os_environ({'TITI': 'toto'}):
        print(os.environ['TITI'])
    print(os.environ['TITI'])
    ```
    """

    old_environ = dict(os.environ)
    os.environ.update(env_updates)
    yield
    os.environ.clear()
    os.environ.update(old_environ)

