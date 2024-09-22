
import os
import json

import pandas as pd
import numpy as np

import inspect
import shutil

from sklearn.preprocessing import OrdinalEncoder


def preprocess_data_fct(
    X_raw: pd.DataFrame,
    encoder: OrdinalEncoder,
    buckets: dict = {},
    is_training: bool = False,
    local_path: str=""
) -> pd.DataFrame:
    """
    We apply feature engineering in
    bucketizing 'num_feature1'.

    In addition, since we here train
    a distance-based classifier,
    we normalize numerical features and
    one-hot encode categorical ones.

    Params:
        - X_raw (pd.DataFrame):
            raw input data.
            2D dataframe (1 row per record,
            n columns for n features).
        - encoder (OrdinalEncoder):
            ordinal encoder
        - buckets (dict):
            option to bucketize one or more features.
            Key-value pairs being featurename-bins.
            Bins can be either a number or a list of edges.
            See `pandas.cut`
        - is_training (bool):
            whether or not the call to the herein function
            is made from the model training loop
        - local_path (str): (Optional)
            path to be used for the serialization
            of the fitted artifacts.
            Ignored if 'is_training' is false.

    Results:
        - pd.DataFrame
            transformed dataset, to be fed
            to the ML-classifier
    """
    X_raw = X_raw.copy()

    # bucketize features
    buckets_dict = {}
    for bucketize_feature in buckets:
        bins = buckets[bucketize_feature]
        if isinstance(buckets[bucketize_feature], list):
            # case : bin edges provided =>
            # where it is up to the developper
            # to choose for "training",
            # in any event, "inference" falls here
            num_buckets = len(bins) - 1
            num_digits = len(str(num_buckets))

            labels = [f'bucket{str(i+1).zfill(num_digits)}'
                      for i in range(num_buckets)]
            # we make sure any "non-training" datapoint is handled
            # if outside the "training" range of observed values
            buckets[bucketize_feature][0] = min(
                buckets[bucketize_feature][0], min(X_raw[bucketize_feature]))
            buckets[bucketize_feature][-1] = max(
                buckets[bucketize_feature][-1], max(X_raw[bucketize_feature]))
        elif isinstance(buckets[bucketize_feature], (int)):
            # case : number of bins provided =>
            # Generate dynamic labels
            num_digits = len(str(bins))
            labels = [f'bucket{str(i+1).zfill(num_digits)}'
                      for i in range(bins)]
        # assign a bucket_label to each datapoints =>
        # (recall that "bins" below can take either
        #  a bins-count or bins-edges)
        X_raw[f"bucketized_{bucketize_feature}"], bin_bounds = pd.cut(
            X_raw[bucketize_feature], bins=bins, labels=labels, retbins=True)
        del X_raw[bucketize_feature]
        buckets_dict = {**buckets_dict,
                        **{bucketize_feature: bin_bounds.tolist()}}
    if is_training:
        # refresh the dict content
        # without changing its address in memory
        # to allow for upward artifact saving !
        buckets.clear()
        buckets.update(buckets_dict)
        #serialize bukets edges info
        buckets_dict_path = \
            os.path.join(local_path, 'buckets_params.json')
        with open(buckets_dict_path, "w") as json_file:
            json.dump(buckets_dict, json_file)
    print(f"buckets_dict : {buckets_dict}")

    # Separate numerical and categorical columns
    numerical_features = X_raw.select_dtypes(include=[np.number]).columns
    categorical_features = X_raw.select_dtypes(exclude=[np.number]).columns

    # Capture raw feature names for later use
    raw_features = X_raw.columns.values

    # Ordinal encode categorical features during training
    if not categorical_features.empty:
        if is_training:
            X_encoded = pd.DataFrame(
               encoder.fit_transform(X_raw[categorical_features]),
                columns=encoder.get_feature_names_out(categorical_features))
            # Serialize the fitted encoder categories
            encoder_dict = {feature: list(categories)
                            for feature, categories
                            in zip(categorical_features, encoder.categories_)}
            print(f"encoder_dict : {encoder_dict}")
            encoder_dict_path = \
                os.path.join(local_path, 'encoder_params.json')
            with open(encoder_dict_path, "w") as json_file:
                json.dump(encoder_dict, json_file)
        else:
            X_encoded = pd.DataFrame(
                encoder.transform(X_raw[categorical_features]),
                columns=\
                    encoder.get_feature_names_out(categorical_features)
            )
    else:
        if is_training:
            encoder_dict = {}
            encoder_dict_path = \
                os.path.join(local_path, 'encoder_params.json')
            with open(encoder_dict_path, "w") as json_file:
                json.dump(encoder_dict, json_file)
        X_encoded = pd.DataFrame()


    # Combine numerical and encoded categorical features
    X_preprocessed = pd.concat([X_encoded, X_raw[numerical_features]], axis=1)

    if is_training:
        # save preprocessing as artefact
        src_path = inspect.getfile(preprocess_data_fct)
        shutil.copy(src_path, os.path.join(
                                local_path,
                                os.path.basename(src_path)
        ))

    return X_preprocessed
