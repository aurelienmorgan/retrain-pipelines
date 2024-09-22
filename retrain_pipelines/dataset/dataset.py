
from enum import Enum, auto

import numpy as np
import pandas as pd

import random
from math import ceil, sqrt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class DatasetType(Enum):
    TABULAR_REGRESSION = auto()
    TABULAR_CLASSIFICATION = auto()

class InvalidDatasetTypeError(ValueError):
    def __init__(self, dataset_type):
        super().__init__(
            f"Invalid dataset type: {dataset_type}."+
            " Valid types are: "+
            ', '.join([e.name for e in DatasetType])
        )


def pseudo_random_generate(
    dataset_type: DatasetType,
    num_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generates a pseudo-randomized dataset.
    """

    if not isinstance(dataset_type, DatasetType):
        raise InvalidDatasetTypeError(dataset_type)

    if dataset_type == DatasetType.TABULAR_REGRESSION:
        return _tabular_regression_generate(
                    num_samples, seed)
    elif dataset_type == DatasetType.TABULAR_CLASSIFICATION:
        return _tabular_classification_generate(
                    num_samples, seed)


def _tabular_regression_generate(
    num_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generates a pseudo-randomized tabular dataset
    with continous target variable.

    Params:
        - num_samples (int)
            number of records to be generated
        -seed (int - optional)
    Results:
        - pdDataFrame
            categ_feature0:
                categorical, with categories 'value1..N'
                where N is random
            num_feature1
                numerical, sampled from a normal distribution
            num_feature2
                numerical, sampled from a logistic distribution
            num_feature3
                numerical, sampled from a uniform distribution
            num_feature4
                numerical, sampled from a lognormal distribution
            target:
                the dataset continuous target variable,
                a weighted/morphed combination of the input features
    """

    if seed is not None:
        # Set a random seed for reproducibility
        np.random.seed(seed)

    #########################
    # 1 categorical feature #
    #########################
    # Generate random categories count
    N = random.randint(4, 10)
    # Generate random categorical feature values
    num_digits = len(str(N))
    categories = [f'value{str(i+1).zfill(num_digits)}' for i in range(N)]
    categ_feature0 = np.random.choice(
            categories, size=num_samples,
            p=np.random.dirichlet(np.ones(N), size=1).flatten()
    )

    #########################
    #  4 numeical features  #
    #########################
    # Generate random features with different distributions
    feature1 = np.random.normal(loc=0, scale=4, size=num_samples)
    feature2 = np.random.logistic(loc=5, scale=1, size=num_samples)
    # induce some linearities between
    # the following 2 numerical features
    low3, high3 = -3, 3 
    mean4, sigma4 = 0, 1
    # correlation coefficient
    correlation = 0.2
    # Generate the uniform feature
    feature3 = np.random.uniform(
        low=low3, high=high3, size=num_samples)
    # Generate the lognormal feature with a
    # slight correlation to the above uniform feature
    feature4 = np.exp(mean4 +
                      sigma4 * np.random.normal(size=num_samples))
    feature4 = feature4 + \
               correlation * (feature3 - feature3.mean()) \
                             / feature3.std()
    # Normalize for desired mean and variance
    feature4 = (feature4 - feature4.mean()) \
                / feature4.std() * sigma4 + mean4

    # Generate random (some non-linear) transformations
    transformation_functions = [np.random.choice(
                                    [np.square, np.tanh,
                                     lambda x: np.exp(-x**2),
                                     lambda x: x])
                                for _ in range(4)]
    # Apply the non-linear transformation functions to the features
    transformed_num_features = \
        [np.round(func(feature), 1)
         for (func, feature)
         in zip(transformation_functions,
                [feature1, feature2, feature3, feature4])
        ]

    #########################
    #    target variable    #
    #########################
    # Create a dictionary to assign a
    # specific impact score to each category
    category_impact = {category: np.random.rand()
                       for category in categories}
    # print(category_impact)
    categorical_impact = np.array([category_impact[cat]
                                   for cat in categ_feature0])
    # Combine the categorical feature impact
    # with the combination of the transformed numerical features
    normalized_num_features = [(f - np.mean(f)) / np.std(f)
                               for f in transformed_num_features]
    combined_score = (
        25*categorical_impact +
        np.sum(normalized_num_features, axis=0)
    )
    # print(combined_score)
    # Create the target variable based on the transformed features
    # plus a tiny bit of noise
    # raw_target = np.tanh(combined_score + np.random.normal(0, 1, num_samples))
    raw_target = combined_score + np.random.normal(
                                    0, 1/2 * np.std(combined_score)*5,
                                    num_samples)
    # raw_target = np.tanh(raw_target / max(raw_target))
    scaled_target = (raw_target - np.min(raw_target)) / \
                    (np.max(raw_target) - np.min(raw_target))
    raw_target = np.tanh(scaled_target)

    # Create a DataFrame to store the data
    data = pd.DataFrame({
        'categ_feature0': categ_feature0,
        'num_feature1': transformed_num_features[0],
        'num_feature2': transformed_num_features[1],
        'num_feature3': transformed_num_features[2],
        'num_feature4': transformed_num_features[3],
        'target': np.round(scaled_target, 3)
    })

    return data


def _tabular_classification_generate(
    num_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generates a pseudo-randomized classification dataset
    (4 classes).

    Params:
        - num_samples (int)
            number of records to be generated
        -seed (int - optional)
    Results:
        - pdDataFrame
            categ_feature0:
                categorical, with categories 'value1..N'
                where N is random
            num_feature1
                numerical, sampled from a normal distribution
            num_feature2
                numerical, sampled from a logistic distribution
            num_feature3
                numerical, sampled from a uniform distribution
            num_feature4
                numerical, sampled from a lognormal distribution
            target:
                the dataset target variable,
                categorical, 4 classes,
                a weighted/morphed combination
                of the input features
    """

    if seed is not None:
        # Set a random seed for reproducibility
        np.random.seed(seed)

    #########################
    # 1 categorical feature #
    #########################
    # Generate random categories count
    N = random.randint(4, 10)
    # Generate random categorical feature values
    num_digits = len(str(N))
    categories = [f'value{str(i+1).zfill(num_digits)}' for i in range(N)]
    categ_feature0 = [random.choice(categories) for _ in range(num_samples)]

    #########################
    #  4 numeical features  #
    #########################
    # Generate random features with different distributions
    feature1 = np.random.normal(loc=0, scale=2, size=num_samples)
    feature2 = np.random.logistic(loc=5, scale=1, size=num_samples)
    feature3 = np.random.uniform(low=-3, high=3, size=num_samples)
    feature4 = np.random.lognormal(mean=0, sigma=1, size=num_samples)

    # Generate random coefficients for each feature
    coefficients = np.random.rand(4)
    # Generate random non-linear transformation functions for each feature
    transformation_functions = [np.random.choice(
                                    [np.sin, np.exp, np.square, np.tanh])
                                for _ in range(4)]
    # Apply the non-linear transformation functions to the features
    transformed_num_features = [np.round(func(feature), 1)
                                for (func, feature) in zip(
                                        transformation_functions,
                                        [feature1, feature2, feature3, feature4])
                               ]

    #########################
    #    target variable    #
    #########################
    # Create a dictionary to assign a specific impact score to each category
    category_impact = {category: np.random.rand() for category in categories}
    # print(category_impact)
    categorical_impact = np.array([category_impact[cat] for cat in categ_feature0])
    # Combine the categorical feature impact
    # with the combination of the transformed nulmerical features
    combined_score = categorical_impact + np.sum(transformed_num_features, axis=0)

    # Create the target variable based on the transformed features and coefficients
    raw_target = (combined_score + np.random.normal(0, 1, num_samples))
    raw_target = np.tanh(raw_target / max(raw_target))

    # Create 4 classes by splitting the distribution of `raw_target`
    # for a "randomly" balanced; use 3 random percentilesbetween 10% and 90%
    random_percentiles = sorted(np.random.uniform(10, 90, 3))
    thresholds = np.percentile(raw_target, random_percentiles)
    target = np.digitize(raw_target, bins=thresholds)

    # print classes prior probabilities
    classes_counts = np.bincount(target)
    classes_priors = [round(100 * count / num_samples, 1)
                      for count in classes_counts]
    print(f"Classes priors: {classes_priors}%")

    # Create a DataFrame to store the data
    data = pd.DataFrame({
        'categ_feature0': categ_feature0,
        'num_feature1': transformed_num_features[0],
        'num_feature2': transformed_num_features[1],
        'num_feature3': transformed_num_features[2],
        'num_feature4': transformed_num_features[3],
        'target': [f"class_{target_val}" for target_val in target]
    })

    return data


def features_desc(
    data: pd.DataFrame
) -> dict:
    """
    features and label basic counts.

    Params (pd.DataFrame):
        input recordset
    Results:
        dict
            features/description key/value pairs
    """

    column_types = {}
    for column in data.columns:
        if pd.api.types.is_integer_dtype(data[column]):
            col_min, col_max = data[column].min(), data[column].max()
            column_types[column] = f"numerical (int) [{col_min}, {col_max}]"
        elif pd.api.types.is_float_dtype(data[column]):
            col_min, col_max = data[column].min(), data[column].max()
            column_types[column] = f"numerical (float) [{col_min}, {col_max}]"
        else:
            unique_count = data[column].nunique()
            column_types[column] = f"categorical ({unique_count})"

    return column_types


def features_distri_plot(
    data: pd.DataFrame
) -> Figure:
    """
    plotting the respective distributions
    of the features and label (all provided columns)
    of the input dataset.
    Log-scale y-axis.

    Params:
        data (pd.DataFrame):
    Results:
        Figure
    """

    columns = data.columns
    num_columns = len(columns)

    sqrt_n = sqrt(num_columns)
    ncols = int(ceil(sqrt_n))
    nrows = int(num_columns / ncols)
    if nrows * ncols < num_columns:
        nrows += 1

    categorical_columns = data.select_dtypes(
        include=['object', 'category']).columns

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes.flatten()
    for i, col in enumerate(columns):
        if col in categorical_columns:
            # Sort categorical features
            sorted_values = data[col].value_counts().sort_index()
            axes[i].bar(sorted_values.index, sorted_values.values)
            axes[i].set_xticks(sorted_values.index)
            axes[i].set_xticklabels(sorted_values.index,
                                    rotation=45, ha='right')
        else:
            # Plot numerical features
            data[col].hist(bins=100, ax=axes[i])

        axes[i].set_yscale('log')
        axes[i].set_ylim(bottom=1e-1) # can't be zero but, close
        axes[i].set_title(col)

    # Hide unused subplots if any
    for j in range(num_columns, len(axes)):
        fig.delaxes(axes[j])

    fig.supylabel('datapoints count (log scale)', fontsize='x-large')
    plt.gcf().set_tight_layout(True)

    plt.close()

    return fig

