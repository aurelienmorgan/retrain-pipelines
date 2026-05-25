import os
import sys

retrain_pipeline_type = os.getenv("retrain_pipeline_type")


if "mf_unsloth_func_call_litserve" == retrain_pipeline_type:
    from . import helpers as helpers

    sys.modules["retrain_pipelines.pipeline_card.mf_unsloth_func_call_litserve.helpers"] = helpers

    from .mf_unsloth_func_call_litserve.dataset_readme import (
        get_dataset_readme_content as get_dataset_readme_content,
    )
    from .mf_unsloth_func_call_litserve.model_readme import (
        get_model_readme_content as get_model_readme_content,
    )
    from .mf_unsloth_func_call_litserve.pipeline_card import (
        get_html as get_html,
    )

    __all__ = [
        "helpers",
        "get_dataset_readme_content",
        "get_model_readme_content",
        "get_html",
    ]


elif "mf_tabnet_classif_torchserve" == retrain_pipeline_type:
    from . import helpers as helpers

    sys.modules["retrain_pipelines.pipeline_card.mf_tabnet_classif_torchserve.helpers"] = helpers

    from .mf_tabnet_classif_torchserve.masks_plotting import (
        plot_masks_to_dict as plot_masks_to_dict,
    )
    from .mf_tabnet_classif_torchserve.pipeline_card import (
        get_html as get_html,
    )

    __all__ = [
        "helpers",
        "plot_masks_to_dict",
        "get_html",
    ]


elif "mf_lightgbm_regress_mlserver" == retrain_pipeline_type:
    from . import helpers as helpers

    sys.modules["retrain_pipelines.pipeline_card.mf_lightgbm_regress_mlserver.helpers"] = helpers

    from .mf_lightgbm_regress_mlserver.pipeline_card import (
        get_html as get_html,
    )

    __all__ = [
        "helpers",
        "get_html",
    ]


else:
    raise ValueError(f"retrain_pipeline_type {retrain_pipeline_type} not recognized.")
