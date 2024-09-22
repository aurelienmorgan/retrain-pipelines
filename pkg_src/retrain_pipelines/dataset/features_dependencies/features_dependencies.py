
"""
Module to help plot inter-features and feature-against-target
dependencies in a consolidated (controversial?) uniform manner.
"""

import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

from contextlib import contextmanager
import logging
import warnings


@contextmanager
def temp_warn_level():
    # module logging level
    logger = logging.getLogger(__name__)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(
        level=min(original_level, logging.WARNING))

    # make sure it escalates
    original_root_level = logging.root.level
    logging.basicConfig(
        level=min(original_root_level,
                  logging.WARNING))

    try:
        # Redirect warnings to the logging system
        logging.captureWarnings(True)
        yield
    finally:
        logging.captureWarnings(False)
        # Restore the original logging level
        logging.getLogger().setLevel(level=original_root_level)
        logging.basicConfig(level=logging.WARNING)

class InterpretabilityWarning(Warning):
    pass
warnings.simplefilter("always", InterpretabilityWarning)

INTERPRETABILITY_DISCLAIMER = \
"""
We're here reconciling apples with oranges.
The intent is to visualize in the [0-1] range
measures of inter-dependencies between features
using a mix of three different approcahes :
Cramér's V, η², and absolute value of correlation.
Despite them all falling in the [0-1] range,
they measure different things, respectively:
strength of association between
pairs of categorical features,
proportion of variance in a numerical feature
explained by a categorical feature, and
(unsigned) linear relationship between
pairs of numerical features.
""".replace("\n", " ")


def get_full_correlation_matrix(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Standard correlation matrix, except that
    the outcome retains the structure (set of columns)
    of the input, whether they are numerical or not
    (filled with NaNs where appropriate).

    Params:
        - data (pd.DataFrame)
            input dataframe, can host
            any variation of categorical
            and numerical columns.

    Results:
        - pd.DataFrame:
    """

    numerical_columns = \
        data.select_dtypes(include=[np.number]).columns
    corr_matrix = data[numerical_columns].corr()
    full_corr_matrix = pd.DataFrame(
        np.nan, index=data.columns, columns=data.columns)
    full_corr_matrix.loc[numerical_columns,
                         numerical_columns] = corr_matrix

    return full_corr_matrix


def get_full_eta_squared_matrix(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Standard ETA-squarred matrix, except that
    the outcome retains the structure
    (set of columns) of the input,
    whether they are categorical or not
    (filled with NaNs where appropriate).
    Convinience method to compute ETA Squared
    where applicable for tabular datasets.
    η² (ETA-Squared) measures the
    proportion of the variance
    in a continuous variable explained
    by a categorical feature.
    In the [0-1] range

    Params:
        - data (pd.DataFrame)
            input dataframe, can host
            any variation of categorical
            and numerical columns.

    Results:
        - pd.DataFrame:
    """

    def _eta_squared_vector(categories, continuous):
        categories = pd.Categorical(categories).codes
        ss_total = np.sum((continuous - continuous.mean())**2)
        
        unique_categories, category_counts = \
            np.unique(categories, return_counts=True)
        category_means = \
            np.bincount(categories, weights=continuous) \
            / category_counts
        
        ss_between = np.sum(category_counts \
                     * (category_means - continuous.mean())**2)
        
        return ss_between / ss_total

    cat_columns = \
        data.select_dtypes(include=['object',
                                  'category']).columns
    num_columns = \
        data.select_dtypes(include=np.number).columns
    all_columns = \
        cat_columns.union(num_columns)

    n_features = len(all_columns)
    eta_squared_matrix = np.full((n_features, n_features),
                                 np.nan)
    
    for i, col1 in enumerate(all_columns):
        for j, col2 in enumerate(all_columns):
            if i < j:  # Compute only upper triangle
                if (
                    col1 in cat_columns and
                    col2 in num_columns
                ):
                    eta_sq = _eta_squared_vector(
                                data[col1].values,
                                data[col2].values)
                elif (
                    col2 in cat_columns and
                    col1 in num_columns
                ):
                    eta_sq = _eta_squared_vector(
                                data[col2].values,
                                data[col1].values)
                else:
                    # eta_sq = 0
                    eta_sq = np.nan
                eta_squared_matrix[i, j] = \
                    eta_squared_matrix[j, i] = eta_sq
    
    eta_squared_matrix = pd.DataFrame(
        eta_squared_matrix,
        index=all_columns, columns=all_columns)

    return eta_squared_matrix


def get_full_cramers_v_matrix(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Standard Cramer's V matrix, except that
    the outcome retains the structure
    (set of columns) of the input,
    whether they are categorical or not
    (filled with NaNs where appropriate).
    Convinience method to compute ETA Squared
    where applicable for tabular datasets.
    Cramer's V is a measure of association
    between two nominal variables,
    based on Pearson's chi-squared statistic,
    it takes a value between 0 and 1.

    Params:
        - data (pd.DataFrame)
            input dataframe, can host
            any variation of categorical
            and numerical columns.

    Results:
        - pd.DataFrame:
    """

    def _cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        min_dim = min(confusion_matrix.shape) - 1
        return np.sqrt(chi2 / (n * min_dim))
    
    # Identify categorical and numerical columns
    categorical_columns = data.select_dtypes(
                              include=['object', 'category']
                          ).columns
    numerical_columns = data.select_dtypes(
                              include=[np.number]
                          ).columns

    # Initialize with NaNs
    all_columns = data.columns
    cramer_matrix = pd.DataFrame(
        np.nan, index=all_columns, columns=all_columns)
    
    # Compute Cramer's V for categorical-categorical pairs
    n = len(categorical_columns)
    for i in range(n):
        for j in range(i, n):
            col1 = categorical_columns[i]
            col2 = categorical_columns[j]
            cramer_matrix.loc[col1, col2] = \
                _cramers_v(data[col1], data[col2])
            # matrix being symmetric =>
            cramer_matrix.loc[col2, col1] = \
                cramer_matrix.loc[col1, col2]

    return cramer_matrix


def get_reconciled_matrix_matrix(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    
    Params:
        - data (pd.DataFrame)
            input dataframe, can host
            any variation of categorical
            and numerical columns.

    Results:
        - pd.DataFrame:
    """

    with temp_warn_level():
        warnings.warn(INTERPRETABILITY_DISCLAIMER,
                      InterpretabilityWarning)

    full_corr_matrix = get_full_correlation_matrix(data)
    full_eta_squared_matrix = get_full_eta_squared_matrix(data)
    full_cramers_v_matrix = get_full_cramers_v_matrix(data)

    reconciled_matrix = (
        full_corr_matrix.fillna(0).abs() +
        full_eta_squared_matrix.fillna(0) +
        full_cramers_v_matrix.fillna(0)
    )

    return reconciled_matrix


def dataset_to_heatmap_fig(
    data: pd.DataFrame
) -> (Figure, Axes):
    """
    Create a heatmap figure from the given dataset.
    Visual representation of relationships
    among features and with the target variable.
    
    Params:
        - data (pd.DataFrame)
            input dataframe, can host
            any variation of categorical
            and numerical columns.

    Results:
        - Figure:
            The heatmap figure object.
        - Axes:
            The axes object of the heatmap.
    """

    reconciled_matrix = get_reconciled_matrix_matrix(data)

    # Generate mask for upper triangle
    mask = np.triu(np.ones_like(reconciled_matrix, dtype=bool))

    plt.ioff()

    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.matshow(np.where(mask, reconciled_matrix, np.nan),
                     cmap='cool')

    cbar = fig.colorbar(cax, fraction=0.0455, pad=0.04)

    # Set the ticks and labels correctly
    ax.set_xticks(np.arange(len(reconciled_matrix.columns)))
    ax.set_xticklabels(reconciled_matrix.columns, rotation=45,
                       ha='left')
    ax.set_yticks(np.arange(len(reconciled_matrix.columns)))
    ax.set_yticklabels(reconciled_matrix.columns)

    ax.set_title(
        "Relationships Among Features and With Target",
        pad=20)
    fig.text(
        0.55, 0.88, # 0.5, 1.01, # 
        "Combining Cramér's V, η² and (unsigned) Correlation",
        ha='center', va='center',
        fontsize=9, style='italic'
    )
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    #fig.subplots_adjust(top=0.8)

    plt.close(fig)

    return (fig, ax)

