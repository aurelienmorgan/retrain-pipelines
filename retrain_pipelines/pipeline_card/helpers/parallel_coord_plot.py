
import pandas as pd

import plotly.graph_objects as go


def _dimensions_from_df(
    input_df: pd.DataFrame
) -> list:
    """
    Generate and return plotly-friendly dimensions
    from given input dataframe
    """

    dimensions = []
    for i, col in enumerate(input_df.columns):
        if (
            # For chart readability,
            # we keep columns with more than one distinct value
            input_df[col].nunique() > 1 or
            # indeed, we keep the perf metric in any event
            i == len(input_df.columns)-1
        ):
            if input_df[col].dtype == 'object':
                col_categ = input_df[col].astype('category').cat
                categ_dim = {
                    'range': [0, max(col_categ.codes.values)],
                    'tickvals': list(range(len(input_df))),
                    'ticktext': col_categ.categories.unique() \
                                    .values.tolist(),
                    'label': col,
                    'values': col_categ.codes.values
                }
                dimensions.append(categ_dim)
            else:
                # Numerical features
                if i != len(input_df.columns)-1:
                    # Extract unique values for tick marks
                    unique_vals = sorted(input_df[col].unique())
                    num_dim = {
                        'label': col,
                        'values': input_df[col],
                        'tickvals': unique_vals,
                        'ticktext': [str(val)
                                     for val in unique_vals]
                    }
                else:
                    # last column (performance metric)
                    num_dim = {
                        'label': col,
                        'values': input_df[col]
                    }

                dimensions.append(num_dim)

    return dimensions


def parallel_coord_plot(
    perf_df: pd.DataFrame
) -> str:
    """
    Plots the parallel coordinate plot for
    a given hyperparameters tuning settings.

    Param:
        - perf_df (pd.DataFrame): a dataframe with non-null columns.
          the n-1 first columns being hyperparameter columns
          the last column in the (numerical) performance column
          each row correspond to a training run.

    Result:
        - str: portable html string
    """

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = perf_df.iloc[:,-1:],
                        colorscale = 'Portland', # 'Tealgrn', #'Electric',
                        cmin = min(perf_df.iloc[:, -1]),
                        cmax = max(perf_df.iloc[:, -1]),
                        showscale = True,
                        colorbar=dict(len=1, ypad=0,
                                      xpad=0, x=1, # xanchor='right',
                                      ticktext=[],
                                      tickvals=[]
                                     )
                   ),
            dimensions = list(_dimensions_from_df(perf_df)),
            unselected = dict(line = dict(color = 'lightgrey',
                                          opacity = 0.4))
        )
    )

    fig.update_layout(
        plot_bgcolor = 'white',
        paper_bgcolor = 'white',
        font=dict(
            family="Roboto, sans-serif",
            size=14, # seems to only apply to the scale (gradient legend)
            color="RebeccaPurple"
        ),
        margin=go.layout.Margin(l=70, r=10, b=10, t=40, pad=4)
    )
    fig.update_traces(
        tickfont_size=14,
        labelfont_size=14
    )
    # DEBUG, when on a Jupyter notebook =>
    #display(HTML(fig.to_html(full_html=False)))

    return fig.to_html(full_html=False)
