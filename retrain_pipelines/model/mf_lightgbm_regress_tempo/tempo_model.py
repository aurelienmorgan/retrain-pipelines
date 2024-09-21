
import os
import sys
import logging

import json
import joblib

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

import tempo
from tempo.serve.metadata import ModelDataArgs


def get_tempo_artifact(
    conda_env: str,
    local_folder: str,
    description: str,
    inputs: ModelDataArgs,
    outputs: ModelDataArgs,
    verbosity: int = logging.INFO
):
    @tempo.model(
        name='lightgbm-model',
        platform=tempo.ModelFramework.Custom,
        conda_env=conda_env,
        local_folder=local_folder,
        # uri="s3://tbd",
        description=description,
        inputs=inputs,
        outputs=outputs
    )
    def lightgbm_model(
        **kwargs
    ) -> int:
        """
        'predict' method.

        Params:
        **kwargs : dict
            Arbitrary keyword arguments. Each key-value pair represents a feature,
            where the key is the feature name (column name in the resulting DataFrame)
            and the value is the feature data (which should be a list or array-like object).

        Returns:
        predicted class (int)
        """

        from preprocessing import preprocess_data_fct

        lightgbm_model.context.logger.debug(kwargs)

        X_raw = pd.DataFrame(kwargs)

        X_preprocessed = preprocess_data_fct(
                            X_raw,
                            lightgbm_model.encoder,
                            lightgbm_model.buckets
        )
        preds = lightgbm_model.model.predict(X_preprocessed)

        return preds


    @lightgbm_model.loadmethod
    def load_lightgbm_model():
        logging.basicConfig()
        lightgbm_model.context.logger = logging.getLogger(__name__)
        lightgbm_model.context.logger.setLevel(verbosity)

        lightgbm_model.context.logger.debug(lightgbm_model.details)

        local_folder = os.getenv('MLSERVER_MODEL_URI',
                                 lightgbm_model.details.local_folder)

        # allow the inference runtime
        # to find the "preprocessing" module
        sys.path.append(local_folder)
        lightgbm_model.context.logger.debug(sys.path)

        # load the preprocessing fitted buckets edges
        buckets_dict_path = \
            os.path.join(local_folder, 'buckets_params.json')
        if os.path.exists(buckets_dict_path):
            with open(buckets_dict_path, "r") as json_file:
                buckets_dict = json.load(json_file)
            lightgbm_model.buckets = buckets_dict
        else:
            lightgbm_model.buckets = {}

        # load the preprocessing fitted OrdinalEncoder
        encoder_file = os.path.join(local_folder, 'encoder_params.json')
        lightgbm_model.context.logger.debug(encoder_file)
        with open(encoder_file, "r") as json_file:
            encoder_dict = json.load(json_file)
        if encoder_dict:
            # case "raw features count at least one
            #       that is categorical"
            encoder = OrdinalEncoder(
                categories=[encoder_dict[feature]
                            for feature in encoder_dict.keys()],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            # still needs to be considered "fitted" at this stage =>
            # Manually set the fit flag and other necessary attributes
            encoder.fit(np.array([encoder_dict[feature][0]
                                  for feature in encoder_dict.keys()]
                                ).reshape(1, -1))
            # Set n_features_in_ & feature_names_in_ attributes
            encoder.n_features_in_ = len(encoder_dict)
            encoder.feature_names_in_ = np.array(list(encoder_dict.keys()))
        else:
            encoder = None
        lightgbm_model.encoder = encoder

        # load the model itself
        model_file = os.path.join(local_folder, 'model.joblib')
        lightgbm_model.context.logger.debug(model_file)
        lightgbm_model.model = joblib.load(model_file)

    return lightgbm_model