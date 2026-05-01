
import os
import copy
import pytz

from textwrap import dedent, indent

import pandas as pd

import importlib                                                    ## LEGACY  -  DELETE ##
if importlib.util.find_spec("metaflow") is not None:                ## LEGACY  -  DELETE ##
    import warnings                                                 ## LEGACY  -  DELETE ##
    with warnings.catch_warnings():                                 ## LEGACY  -  DELETE ##
        warnings.filterwarnings(                                    ## LEGACY  -  DELETE ##
            "ignore", message="pkg_resources is deprecated")        ## LEGACY  -  DELETE ##
        from metaflow import cards                                  ## LEGACY  -  DELETE ##

from jinja2 import Environment, FileSystemLoader

from retrain_pipelines import __version__
from .helpers import apply_args_color_format, \
    highlight_min_max_cells, fig_to_base64, \
    parallel_coord_plot
from ...dag_engine.sdk.core import Execution


def get_html(
    params: list
) -> str:
    """
    Generates the html for the portable pipeline card.
    at runtime for the DAG-execuction being run.
    """

    NONE_HTML_STR = \
        "<center><font color=white><em>None</em></font></center>"
    DEFAULT_HTML_STR = \
        "<center><font color=white><em>Default</em></font></center>"

    ##########################
    # model version blessing #
    ##########################
    model_version_blessed = params['model_version_blessed']
    if not model_version_blessed:
        current_blessed_exec = params['current_blessed_run']        ## LEGACY  -  RENAME PARAM ##
        if not isinstance(current_blessed_exec, Execution):
            ## LEGACY  -  DELETE start ##
            previous_blessed_model_card_task = [
                step.task for step in current_blessed_exec.steps()
                if step.id == 'pipeline_card'][0]
            previous_blessed_custom_card = \
                cards.get_cards(previous_blessed_model_card_task,
                                id='custom', type='html')[0]

            current_blessed_run_finished = \
                current_blessed_exec \
                    .finished_at.astimezone(pytz.utc) \
                    .strftime('%A %b %d %Y %I:%M:%S %p %Z')
            previsous_blessed_card_href = '/flows/' + \
                previous_blessed_custom_card.path[
                    :previous_blessed_custom_card.path.rfind("/")+1] + \
                previous_blessed_custom_card.hash
            previsous_blessed_card_url = \
                previous_blessed_model_card_task.pathspec + \
                '?direction=asc&group=true&order=startTime&section=' + \
                previous_blessed_custom_card.hash
            ## LEGACY  -  DELETE end ##
        else:
            current_blessed_run_finished = (                        ## LEGACY  -  RENAME VARIABLE ##
                current_blessed_exec
                    .end_timestamp.astimezone(pytz.utc)
                    .strftime('%A %b %d %Y %I:%M:%S %p %Z')
            )
            previsous_blessed_card_href = \
                f"/execution?id={current_blessed_exec.id}"
            previsous_blessed_card_url = ""                         ## LEGACY  -  NOT USED ANYMORE ##
                                                                    ## REMOVE SECTION FROM HTML TEMPLATE ##

    ##########################

    local_serve_is_ready = params['local_serve_is_ready']

    ##########################
    #          EDA           #
    ##########################
    records_count = "{:,}".format(params['records_count'])
    features_desc_table = pd.DataFrame(
            [params['features_desc']]
        ).to_html(classes='wide', escape=False, index = False)

    data_distri_fig = copy.copy(params['data_distri_fig'])
    # Set the RGBA background white color with partial transparency
    data_distri_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    data_distri_curve = fig_to_base64(data_distri_fig)

    data_heatmap_fig = copy.copy(params['data_heatmap_fig'])
    # Set the RGBA background white color with partial transparency
    data_heatmap_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    data_heatmap_curve = fig_to_base64(data_heatmap_fig,
                                       extra_tight=True)
    ##########################

    ##########################
    #     model training     #
    ##########################
    if not params['buckets_dict']:
        buckets_table = NONE_HTML_STR
    else:
        buckets_table = pd.DataFrame(
            [params['buckets_dict']]).to_html(classes='wide',
                                              escape=False, index = False)
    if params['hyperparameters_dict']:
        hyperparameters_table = \
            pd.DataFrame([params['hyperparameters_dict']]
                        ).to_html(classes='wide', escape=False, index = False)
    else:
        hyperparameters_table = DEFAULT_HTML_STR

    training_plt_fig = copy.copy(params['training_plt_fig'])
    # Set the RGBA background white color with partial transparency
    training_plt_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    training_curve = fig_to_base64(training_plt_fig)

    features_plt_fig = copy.copy(params['features_plt_fig'])
    # Set the RGBA background white color with partial transparency
    features_plt_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    feature_imp_curve = fig_to_base64(features_plt_fig)

    perf_metrics_df = pd.DataFrame([params['perf_metrics_dict']])
    perf_metrics_df = perf_metrics_df.map(
        lambda x: '{:.3f}'.format(x).rstrip('0').rstrip('.'))
    perf_metrics_table = perf_metrics_df.to_html(
        classes='wide', escape=False, index = False)

    if params['sliced_perf_metrics_dict'] is None:
        slice_feature_name = None
        styled_sliced_perf_metrics_table = None
    else:
        slice_feature_name = params['slice_feature_name']
        sliced_perf_metrics_df = \
            pd.DataFrame.from_dict(
                params['sliced_perf_metrics_dict'],
                orient='index'
            ).sort_index(ascending=True)
        styled_sliced_perf_metrics_table = \
            highlight_min_max_cells(sliced_perf_metrics_df)
    ##########################

    ##########################
    # hyperparameter tunning #
    ##########################
    if params['pipeline_hp_grid_dict']:
        hp_grid_table = pd.DataFrame(
                [params['pipeline_hp_grid_dict']]
            ).to_html(classes='wide', escape=False, index = False)
        hp_perfs_df = pd.DataFrame(params['hp_perfs_list'])
        hp_perfs_curve = parallel_coord_plot(hp_perfs_df)
        hp_perfs_table = hp_perfs_df.sort_values(by='rmse', ascending=True) \
                                    .to_html(classes='wide', escape=False,
                                             index = False)
    else:
        hp_grid_table = None
        hp_perfs_curve = None
        hp_perfs_table = None
    ##########################

    env = Environment(loader=FileSystemLoader(params['template_dir']))
    template = env.get_template('template.html')

    return template.render(
        title=params['title'], subtitle=params['subtitle'],
        blessed_color="#008000" if model_version_blessed else "#811331",
        blessed_background="#7CFC00" if model_version_blessed else "#FF3131",
        model_version_not_blessed="" if model_version_blessed else "NOT ",

        # if model version not blessed => #
        current_blessed_run_id=(                                    ## LEGACY  -  RENAME VARIABLE ##
            None if model_version_blessed
            else current_blessed_exec.id),
        current_blessed_run_finished=(
            None if model_version_blessed
            else current_blessed_run_finished),
        previsous_blessed_card_href=(
            None if model_version_blessed
            else previsous_blessed_card_href),
        previsous_blessed_card_url =(
            None if model_version_blessed
            else previsous_blessed_card_url),
        ###################################

        # infra validation status =>      #
        local_serve_color=(
            None if (-1 == local_serve_is_ready)
            else "#008000" if (1 == local_serve_is_ready)
            else "#811331"),
        local_serve_background=(
            None if (-1 == local_serve_is_ready)
            else "#7CFC00" if (1 == local_serve_is_ready)
            else "#FF3131"),
        local_serve_status=(
            None if (-1 == local_serve_is_ready)
            else 'Passed' if (1 == local_serve_is_ready)
            else 'Failed'),
        ###################################

        records_count=records_count,
        features_desc_table=indent(features_desc_table, ' '*36),
        data_distri_curve=data_distri_curve,
        data_heatmap_curve=data_heatmap_curve,

        training_curve=training_curve,
        buckets_table=indent(buckets_table, ' '*52),
        hyperparameters_table=indent(hyperparameters_table, ' '*52),
        feature_imp_curve=feature_imp_curve,
        perf_metrics_table=indent(perf_metrics_table, ' '*52),
        slice_feature_name=slice_feature_name,
        styled_sliced_perf_metrics_table=(
            None if styled_sliced_perf_metrics_table is None else
            indent(styled_sliced_perf_metrics_table, ' '*52)
        ),

        hp_grid_table=indent(hp_grid_table, ' '*52) \
                      if hp_grid_table is not None else None,
        cv_folds=str(params['cv_folds']),
        hp_perfs_curve=indent(hp_perfs_curve, ' '*60) \
                       if hp_perfs_curve is not None else None,
        hp_perfs_table=indent(hp_perfs_table, ' '*52) \
                       if hp_perfs_table is not None else None,

        wandb_project_ui_url=params['wandb_project_ui_url'],
        wandb_filter_run_id=params['wandb_filter_run_id'],
        wandb_need_sync_dir=params['wandb_need_sync_dir'],

        task_obj_python_cmd=apply_args_color_format(params['task_obj_python_cmd']),
        dag_svg=indent(params['dag_svg'], ' '*40),
        __version__=__version__
    )
