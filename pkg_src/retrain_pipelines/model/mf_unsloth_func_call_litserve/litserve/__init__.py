from importlib.util import find_spec

__all__ = []

# conditional import of the "docker" package
# required for some infra_validation tasks
# (but not necessarily installed
#  on virtual env that do not require it)
if find_spec("docker") is not None:
    from .model_info import (
        endpoint_is_ready as endpoint_is_ready,
    )
    from .model_info import (
        endpoint_started as endpoint_started,
    )

    __all__ += [
        "endpoint_is_ready",
        "endpoint_started",
    ]
