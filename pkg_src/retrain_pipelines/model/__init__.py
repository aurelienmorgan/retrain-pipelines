import os
import sys
from typing import Any, Protocol


class PreprocessFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


retrain_pipeline_type = os.getenv("retrain_pipeline_type")


preprocess_data_fct: PreprocessFn


if "mf_unsloth_func_call_litserve" == retrain_pipeline_type:
    from .mf_unsloth_func_call_litserve import litserve
    from .mf_unsloth_func_call_litserve.eval import (
        compute_counts_n_metrics,
        infer_validation,
        plot_validation_completions,
    )

    sys.modules["retrain_pipelines.model.litserve"] = litserve

    __all__ = [
        "litserve",
        "compute_counts_n_metrics",
        "infer_validation",
        "plot_validation_completions",
    ]

    __path__.append(os.path.join(__path__[0], "mf_unsloth_func_call_litserve", "litserve"))


elif "mf_tabnet_classif_torchserve" == retrain_pipeline_type:
    from .mf_tabnet_classif_torchserve import torchserve
    from .mf_tabnet_classif_torchserve.preprocessing import preprocess_data_fct
    from .mf_tabnet_classif_torchserve.trainer_utils import (
        PrintLR,
        WandbCallback,
    )

    sys.modules["retrain_pipelines.model.torchserve"] = torchserve

    __all__ = [
        "torchserve",
        "preprocess_data_fct",
        "PrintLR",
        "WandbCallback",
    ]

    __path__.append(os.path.join(__path__[0], "mf_tabnet_classif_torchserve", "torchserve"))


elif "mf_lightgbm_regress_mlserver" == retrain_pipeline_type:
    from .mf_lightgbm_regress_mlserver import mlserver
    from .mf_lightgbm_regress_mlserver.dask_cleanup import nuke_dask_cpp
    from .mf_lightgbm_regress_mlserver.dask_trainer import dask_regressor_fit
    from .mf_lightgbm_regress_mlserver.preprocessing import preprocess_data_fct

    sys.modules["retrain_pipelines.model.mlserver"] = mlserver

    __all__ = [
        "mlserver",
        "nuke_dask_cpp",
        "dask_regressor_fit",
        "preprocess_data_fct",
    ]

    __path__.append(os.path.join(__path__[0], "mf_lightgbm_regress_mlserver", "mlserver"))

else:
    raise ValueError(f"retrain_pipeline_type {retrain_pipeline_type} not recognized.")
