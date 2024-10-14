
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

@wraps(metaflow.Flow)
def Flow(*args, **kwargs):
    flow = metaflow.Flow(*args, **kwargs)
    if not os.getenv("retrain_pipeline_type", None):
        os.environ["retrain_pipeline_type"] = \
            list(flow.latest_run.steps())[-1] \
                        .task["retrain_pipeline_type"].data
    return flow

@wraps(metaflow.Run)
def Run(*args, **kwargs):
    run = metaflow.Run(*args, **kwargs)
    if not os.getenv("retrain_pipeline_type", None):
        print(list(run.steps())[-1].task)
        os.environ["retrain_pipeline_type"] = \
            list(run.steps())[-1].task["retrain_pipeline_type"].data
    return run

@wraps(metaflow.Task)
def Task(*args, **kwargs):
    task = metaflow.Task(*args, **kwargs)
    if not os.getenv("retrain_pipeline_type", None):
        os.environ["retrain_pipeline_type"] = \
            task["retrain_pipeline_type"].data
    return task

