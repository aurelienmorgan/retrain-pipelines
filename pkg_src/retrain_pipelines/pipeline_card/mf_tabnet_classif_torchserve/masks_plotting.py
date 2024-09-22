
import numpy as np
import pandas as pd

import warnings
from contextlib import contextmanager

from matplotlib import pyplot as plt

from pytorch_tabnet.tab_model import TabNetClassifier


@contextmanager
def _tight_layout_manager(fig, **kwargs):
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield fig.tight_layout(**kwargs)
        if len(w) > 0 and issubclass(w[-1].category,
                                     UserWarning):
            print("Warning: tight_layout " +
                  "may not be fully compatible " +
                  "with all Axes.")
    except Exception as e:
        print(f"Error in tight_layout: {e}")


def _masks_column_idx(
    grouped_features: list,
    features_count: int
) -> list:
    """
    Per the PyTorch TabNet documentation
    (https://pypi.org/project/pytorch-tabnet/),
    grouped features are assigned the same activation value so,
    we only keep one for each.

    Params:
        - grouped_features (list[list[int]])
            indices in the preprocessed input features
            dataframe (one list per grouped feature).
        - features_count (int):
            total number of features (after preprocessing).
    """

    categ_features_mask_column_idx = [
        inner_list[0]
        for inner_list in grouped_features if inner_list
    ]
    excluded_indices = {
        idx for sublist in grouped_features for idx in sublist
    }
    columns_to_keep = [
        idx for idx in range(features_count)
        if (
            idx in categ_features_mask_column_idx
            or idx not in excluded_indices
        )
    ]
    return columns_to_keep


def plot_masks_to_dict(
    model: TabNetClassifier,
    X_transformed: pd.DataFrame,
    grouped_features: list,
    raw_feature_names: str,
    y: np.ndarray
) -> dict:
    """

    Params:
        - model (TabNetClassifier):
        - X_transformed (pd.DataFrame):
            records used to generate predictions.
        - grouped_features (list[list[int]]):
        - raw_feature_names (str):
            names of the features (before preprocessing)
            used for x-axis labelling.
        - y (np.ndarray):
            label used to identify true positives.

    Results:
        - (dict[Figure]):
            key/value pairs
            of target_class/masks_plots.
    """

    max_displayed_masks_count = 4
    max_true_positives_display_count = 10

    columns_to_keep = _masks_column_idx(grouped_features,
                                        X_transformed.shape[1])

    target_class_plots = {}
    target_classes = np.unique(y).tolist()
    for target_class in target_classes:
        target_class_idx = np.where(y == target_class)[0]
        predictions = \
            model.predict(X_transformed.iloc[target_class_idx].values)
        target_class_true_positive_idx = \
            target_class_idx[predictions == target_class]
        true_positives_display_count = \
            min(max_true_positives_display_count,
                len(target_class_true_positive_idx))
        print(f"subset of true positives for '{target_class}' : " +
              f"{true_positives_display_count}")
        true_positives = X_transformed.values[
            target_class_true_positive_idx[:true_positives_display_count]]
        true_positives_count = len(true_positives)
        if true_positives_count > 0:
            explain_matrix, masks = model.explain(true_positives)

            # view first 5 masks at most
            # each layer of TabNet producing one,
            # it depends on the model how many masks are available.
            displayed_masks_count = min(model.n_steps,
                                        max_displayed_masks_count)
            max_label_length = max(len(label)
                                   for label in raw_feature_names)
            # Adjust the below 2 to control figure height
            x_label_height_factor = 0.15
            plot_area_height_factor = .3 * (2/displayed_masks_count)**3

            # Reserve space for title and subtitle
            title_space = .6 # inches
            fig_height = (
                plot_area_height_factor * true_positives_count +
                x_label_height_factor * max_label_length +
                title_space
            )
            fig, axs = plt.subplots(nrows=1, ncols=displayed_masks_count,
                                    figsize=(7, fig_height))

            if displayed_masks_count == 1:
                axs = [axs]

            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=0, vmax=1)

            for i, ax in enumerate(axs):
                im = ax.imshow(masks[i][:, columns_to_keep],
                               cmap=cmap, norm=norm)
                ax.set_title(f"Mask {i}", fontsize=9)
                ax.set_yticks(np.arange(len(true_positives)))
                ax.set_yticklabels([""]*len(true_positives))
                ax.set_xticks(np.arange(len(raw_feature_names)))
                ax.set_xticklabels(raw_feature_names, fontsize=9,
                                   rotation=45, ha='right')

            # main title
            first_masks_only = (
                "" if model.n_steps <= max_displayed_masks_count
                else f"(first {max_displayed_masks_count}) "
            )
            main_title = f"{first_masks_only}Masks " + \
                         f"for Class '{target_class}'"

            cbar_ax = fig.add_axes([0.85, 0.85, 0.1, 0.02])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Attention\nscore', fontsize=9,
                           labelpad=-40, x=.5, ha='center')
            cbar.ax.tick_params(labelsize=7)

            fig.suptitle(main_title, fontsize=11,
                         y= 1 - (0.1 * title_space / fig_height))
            # subtitle
            subtitle = "Number of true positives plotted: " + \
                       str(true_positives_display_count)
            fig.text(0.5, 1.03 - (0.8 * title_space / fig_height),
                     subtitle, ha='center', fontsize=8)
            
            with _tight_layout_manager(
                    fig, rect=[0, 0, 1, 1 - (title_space / fig_height)],
                    pad=0.01, h_pad=0.01, w_pad=0.01):
                pass  # tight_layout applied within context manager

            target_class_plots[target_class] = fig
            print(f"fig dims : {fig.get_size_inches()}")
            plt.close(fig)
        else:
            target_class_plots[target_class] = None

    return target_class_plots







































