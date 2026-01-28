
import importlib
import os
import sys
import logging
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import OrdinalEncoder

from mlserver import MLModel
from mlserver.types import InferenceRequest, \
    InferenceResponse, ResponseOutput


class LightGBMModel(MLModel):

    async def load(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        local_folder = self.settings.parameters.uri

        # allow the inference runtime
        # to find the "preprocessing" module
        sys.path.append(local_folder)
        self.logger.debug(sys.path)

        from preprocessing import preprocess_data_fct
        self.preprocess_data_fct = preprocess_data_fct

        # load the preprocessing fitted buckets edges
        buckets_dict_path = \
            os.path.join(local_folder, "buckets_params.json")
        if os.path.exists(buckets_dict_path):
            with open(buckets_dict_path, "r") as json_file:
                buckets_dict = json.load(json_file)
            self.buckets = buckets_dict
        else:
            self.buckets = {}

        # load the preprocessing fitted OrdinalEncoder
        encoder_file = os.path.join(local_folder,
                                    "encoder_params.json")
        self.logger.debug(encoder_file)
        with open(encoder_file, "r") as json_file:
            encoder_dict = json.load(json_file)
        if encoder_dict:
            # case "raw features count at least one
            #       that is categorical"
            encoder = OrdinalEncoder(
                categories=[encoder_dict[feature]
                            for feature in encoder_dict.keys()],
                handle_unknown="use_encoded_value",
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
        self.encoder = encoder

        # load the model itself
        self.model = lgb.Booster(
            model_file=os.path.join(local_folder,
                                    "model.txt")
        )
        self.logger.info(
            f"LightGBM expected feature names: {self.model.feature_name()}")

    async def predict(
        self, request: InferenceRequest
    ) -> InferenceResponse:
        """
        'predict' method.

        Params:
            - **kwargs (dict):
                Arbitrary keyword arguments.
                Each key-value pair represents a feature,
                where the key is the feature name
                (column name in the resulting DataFrame)
                and the value is the feature data
                (which should be a list or array-like object).

        Results:
            - (int):
                predicted class
        """
        self.logger.debug(request)

        kwargs = {}
        for inp in request.inputs:
            if inp.datatype == "FP32" or inp.datatype == "FP64":
                kwargs[inp.name] = np.array(inp.data, dtype=float)
            elif inp.datatype.startswith("INT"):
                kwargs[inp.name] = np.array(inp.data, dtype=int)
            elif inp.datatype == "BYTES":
                kwargs[inp.name] = np.array(inp.data, dtype=str)
            else:
                kwargs[inp.name] = np.array(inp.data)

        X_raw = pd.DataFrame(kwargs)

        X_preprocessed = self.preprocess_data_fct(
            X_raw,
            self.encoder,
            self.buckets
        )

        preds = self.model.predict(X_preprocessed)

        return InferenceResponse(
            model_name=self.name,
            outputs=[
                ResponseOutput(
                    name="prediction",
                    datatype="FP32",
                    shape=[len(preds)],
                    data=preds.tolist(),
                )
            ],
        )

