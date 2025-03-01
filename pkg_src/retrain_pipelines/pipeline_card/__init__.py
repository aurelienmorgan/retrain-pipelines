
import os
import sys

retrain_pipeline_type = os.getenv("retrain_pipeline_type")


if "mf_unsloth_func_call_litserve" == retrain_pipeline_type:
    from . import helpers
    sys.modules[
            "retrain_pipelines.pipeline_card.mf_unsloth_func_call_litserve.helpers"
        ] = helpers
    __all__ = ["helpers"]

    from .mf_unsloth_func_call_litserve.dataset_readme import \
                get_dataset_readme_content
    from .mf_unsloth_func_call_litserve.model_readme import \
                get_model_readme_content
    from .mf_unsloth_func_call_litserve.pipeline_card import \
                get_html


elif "mf_tabnet_classif_torchserve" == retrain_pipeline_type:
    from . import helpers
    sys.modules[
            "retrain_pipelines.pipeline_card.mf_tabnet_classif_torchserve.helpers"
        ] = helpers
    __all__ = ["helpers"]

    from .mf_tabnet_classif_torchserve.pipeline_card import \
                get_html
    from .mf_tabnet_classif_torchserve.masks_plotting import \
                plot_masks_to_dict


elif "mf_lightgbm_regress_tempo" == retrain_pipeline_type:
    from . import helpers
    sys.modules[
            "retrain_pipelines.pipeline_card.mf_lightgbm_regress_tempo.helpers"
        ] = helpers
    __all__ = ["helpers"]

    from .mf_lightgbm_regress_tempo.pipeline_card import \
                get_html

else:
    raise ValueError(f"retrain_pipeline_type {retrain_pipeline_type} " +
                     " not recognized.")
