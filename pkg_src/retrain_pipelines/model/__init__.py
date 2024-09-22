
import os
import sys

retrain_pipeline_type = os.getenv("retrain_pipeline_type")


if "mf_tabnet_classif_torchserve" == retrain_pipeline_type:
    from .mf_tabnet_classif_torchserve.preprocessing import \
                preprocess_data_fct
    from .mf_tabnet_classif_torchserve.trainer_utils import \
                PrintLR, WandbCallback

    from .mf_tabnet_classif_torchserve import \
                torchserve, endpoint_test

    sys.modules["retrain_pipelines.model.torchserve"] = torchserve
    sys.modules["retrain_pipelines.model.endpoint_test"] = endpoint_test

    __all__ = ["torchserve", "endpoint_test"]
    __path__.append(os.path.join(
        __path__[0], "mf_tabnet_classif_torchserve", "torchserve"))
    __path__.append(os.path.join(
        __path__[0], "mf_tabnet_classif_torchserve", "endpoint_test"))


elif "mf_lightgbm_regress_tempo" == retrain_pipeline_type:
    from .mf_lightgbm_regress_tempo.preprocessing import \
                preprocess_data_fct
    from .mf_lightgbm_regress_tempo.dask_trainer import \
                dask_regressor_fit
    from .mf_lightgbm_regress_tempo.tempo_model import \
                get_tempo_artifact
    from .mf_lightgbm_regress_tempo.tempo_helpers import \
                tempo_wait_ready, tempo_predict


else:
    raise ValueError(f"retrain_pipeline_type {retrain_pipeline_type}" +
                      "not recognized.")

