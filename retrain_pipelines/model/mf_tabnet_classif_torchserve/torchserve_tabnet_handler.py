
import zipfile
import logging
import json
import os

import numpy as np
import pandas as pd

import torch
from ts.torch_handler.base_handler import BaseHandler

from pytorch_tabnet.tab_model import TabNetClassifier


from sklearn.preprocessing import StandardScaler, \
                                  OneHotEncoder

from preprocessing import preprocess_data_fct


class TabnetHandler(BaseHandler):
    def __init__(self):
        super(TabnetHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
                "cuda:" + str(properties.get("gpu_id"))
                if torch.cuda.is_available()
                else "cpu"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # load the fitted model
        with zipfile.ZipFile("model.zip", 'w') as zipf:
            zipf.write("network.pt", arcname='network.pt')
            zipf.write("model_params.json", arcname='model_params.json')

        model = TabNetClassifier()
        model.load_model(filepath="model.zip")
        self.model = model
        self.initialized = True
        self.logger.debug(model)

        # load the raw (inference) feature names
        if os.path.exists('feature_names.json'):
            with open('feature_names.json', "r") as json_file:
                raw_feature_names = json.load(json_file)
            self.raw_features = raw_feature_names
        else:
            self.buckets = {}

        # load the preprocessing fitted buckets edges
        if os.path.exists('buckets_params.json'):
            with open('buckets_params.json', "r") as json_file:
                buckets_dict = json.load(json_file)
            self.buckets = buckets_dict
        else:
            self.buckets = {}

        # load the preprocessing fitted StandardScaler
        with open('scaler_params.json', "r") as json_file:
            scaler_params = json.load(json_file)
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_params["mean"])
        scaler.scale_ = np.array(scaler_params["std_dev"])
        self.scaler = scaler

        # load the preprocessing fitted OneHotEncoder
        with open('encoder_params.json', "r") as json_file:
            encoder_dict = json.load(json_file)
        if encoder_dict:
            # case "raw features count at least one
            #       that is categorical"
            encoder = OneHotEncoder(
                sparse_output=False,
                categories=[encoder_dict[feature]
                            for feature in encoder_dict.keys()],
                handle_unknown='ignore',
                drop=None # drop='first' for linear models, @see the doc
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


    def preprocess(self, data):
        # Extract the input data from the JSON request
        input_data = data[0]["body"]
        if isinstance(input_data, str):
            input_data = input_data.decode('utf-8')

        self.logger.debug(f"preprocess : {input_data}")

        # Convert to list of lists
        if not isinstance(input_data, list):
            input_data = [input_data['data']]
        else:
            input_data = [request_input_item['data']
                          for request_input_item in input_data]

        input_data_df = pd.DataFrame(input_data,
                                     columns=self.raw_features)
        self.logger.debug(f"preprocess : {input_data_df}")

        # Apply the preprocessing function
        preprocessed_data = preprocess_data_fct(
            input_data_df,
            scaler=self.scaler,
            encoder=self.encoder,
            buckets=self.buckets
        )

        self.logger.debug(f"preprocessed_data : {preprocessed_data}")

        return preprocessed_data


    def inference(self, data):
        results = self.model.predict(data.values)
        results = results.tolist()
        # to handle batch inference for a single query =>
        results = [results]
        self.logger.debug(f"inference results : {results}")
        return results


    def postprocess(self, inference_output):
        return inference_output


























































