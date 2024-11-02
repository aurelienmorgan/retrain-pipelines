
"""
Wrap the Metaflow package.

Set environement variables for local Metaflow service
prior to importing the package.

Also, wraps methods to ensure ml-pipelines specifics
are dealt with.
"""

import os
import sys
import urllib

import logging
logger = logging.getLogger()

from functools import wraps

os.environ['METAFLOW_SERVICE_URL'] = \
    os.environ.get('METAFLOW_SERVICE_URL', 'http://localhost:8080/')
os.environ['METAFLOW_DEFAULT_METADATA'] = 'service'

# in case user is behind a proxy =>
os.environ['NO_PROXY'] = os.environ['no_proxy'] = \
    'localhost,0.0.0.0,' + (urllib.request.getproxies()["no"] \
                            if "no" in urllib.request.getproxies() else '')

import metaflow

# Create aliases for all metaflow modules
package_name = __name__
for name in dir(metaflow):
    if not name.startswith('_'):  # Skip privates
        module = getattr(metaflow, name)
        if isinstance(module, type(metaflow)):
            # if a module
            sys.modules[f'{package_name}.{name}'] = module

from metaflow import *
# Expose all attributes
__all__ = [name for name in dir(metaflow) if not name.startswith('_')]
# Add all to the local namespace
for name in __all__:
    if name not in globals():
        globals()[name] = getattr(metaflow, name)

##############################################################################

"""
When browsing artifacts, Metaflow requires to
access a whole lot of modules for unpickle.
Nebulous but, factual.
ml-pipelines must thus be mapped according to
the ml-pipeline-type that applies.
Setting it below (if not otherwise done already)
and doing it via methods wrapping
(maintain docstrings and all).
"""

def _set_retrain_pipeline_type_env(
    flow_start_task: metaflow.Task
):
    """
    Allow support (with warning) for non-"retrain-pipelines" flows.
    """

    if "retrain_pipeline_type" in flow_start_task:
        os.environ["retrain_pipeline_type"] = \
            flow_start_task["retrain_pipeline_type"].data
    else:
        logger.warn("not recognized as a retrain-pipelines flow.")


@wraps(metaflow.Flow)
def Flow(*args, **kwargs):
    flow = metaflow.Flow(*args, **kwargs)
    if not os.getenv("retrain_pipeline_type", None):
        _set_retrain_pipeline_type_env(
            list(flow.latest_run.steps())[-1].task)
    return flow

@wraps(metaflow.Run)
def Run(*args, **kwargs):
    run = metaflow.Run(*args, **kwargs)
    if not os.getenv("retrain_pipeline_type", None):
        _set_retrain_pipeline_type_env(list(run.steps())[-1].task)
    return run

@wraps(metaflow.Task)
def Task(*args, **kwargs):
    task = metaflow.Task(*args, **kwargs)
    if not os.getenv("retrain_pipeline_type", None):
        run = task.parent.parent
        _set_retrain_pipeline_type_env(list(run.steps())[-1].task)
    return task

