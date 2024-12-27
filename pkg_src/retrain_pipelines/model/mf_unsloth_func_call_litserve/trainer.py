

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_training_loss(
    log_history: list
) -> Figure:
    """
    Params:
        - cpt_log_history (list):
            List[dict('loss', 'epoch', 'step')]

    Results:
        - (Figure)
    """

    training_data = \
        [d for d in log_history
         if all(k in d
                for k in ['loss', 'epoch', 'step'])]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([d['epoch'] for d in training_data],
            [d['loss'] for d in training_data],
            'c-o')

    ax.set_xticks(np.arange(
        0, max(d['epoch'] for d in training_data)+1))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(color='lightgrey', linestyle='-',
            linewidth=0.5)

    ax.set_title('Continued Pretraining')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.close(fig)

    return fig

