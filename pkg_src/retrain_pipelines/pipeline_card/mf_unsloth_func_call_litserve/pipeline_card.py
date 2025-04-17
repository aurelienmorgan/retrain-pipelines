
import os
import copy
import pytz
import importlib.util
from textwrap import dedent, indent

import pandas as pd

from metaflow import cards

from jinja2 import Environment, BaseLoader, TemplateNotFound

from retrain_pipelines import __version__
from .helpers import apply_args_color_format, highlight_min_max_cells, \
                     fig_to_base64, parallel_coord_plot

def get_html(
    params: list
) -> str:
    """
    Generates the html for the portable pipeline card.
    at runtime for the flow run being executed.
    """

    ##########################
    # model version blessing #
    ##########################
    model_version_blessed = params['model_version_blessed']
    previous_blessed_run = params['current_blessed_run']
    if previous_blessed_run is not None:
        previous_blessed_model_card_task = [
            step.task for step in previous_blessed_run.steps()
            if step.id == 'pipeline_card'][0]
        previous_blessed_custom_card = \
            cards.get_cards(previous_blessed_model_card_task,
                            id='custom', type='html')[0]

        previsous_blessed_card_href = '/flows/' + \
            previous_blessed_custom_card.path[
                :previous_blessed_custom_card.path.rfind("/")+1] + \
            previous_blessed_custom_card.hash
        previsous_blessed_card_url = \
            previous_blessed_model_card_task.pathspec + \
            '?direction=asc&group=true&order=startTime&section=' + \
            previous_blessed_custom_card.hash
    ##########################

    LocalServeReadinessEnum = params['LocalServeReadinessEnum']
    local_serve_is_ready = params['local_serve_is_ready']

    ##########################
    #          EDA           #
    ##########################
    data_schema_table = params['data_schema'].to_html(escape=False,
                                                      index = False)

    answers_tools_count_fig = copy.copy(params['answers_tools_count_fig'])
    # Set the RGBA background white color with partial transparency
    answers_tools_count_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    answers_tools_count_curve = fig_to_base64(answers_tools_count_fig,
                                              extra_tight=True)

    words_count_fig = copy.copy(params['words_count_fig'])
    # Set the RGBA background white color with partial transparency
    words_count_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    words_count_curve = fig_to_base64(words_count_fig,
                                      extra_tight=True)
    ##########################

    ##########################
    #     model training     #
    ##########################
    base_model_dict = params['hf_base_model_dict']
    base_model_repo_id = base_model_dict['repo_id']
    base_model_version_label = base_model_dict['version_label']
    base_model_commit_hash = base_model_dict['commit_hash']
    base_model_commit_datetime = base_model_dict['commit_datetime']

    pipeline_parameters_table = \
        pd.DataFrame([params['pipeline_parameters_dict']]
                    ).to_html(classes='wide', escape=False,
                              index = False)

    cpt_log_history_fig = copy.copy(params['cpt_log_history_fig'])
    # Set the RGBA background white color with partial transparency
    cpt_log_history_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    cpt_log_history_curve = fig_to_base64(cpt_log_history_fig)

    sft_log_history_fig = copy.copy(params['sft_log_history_fig'])
    # Set the RGBA background white color with partial transparency
    sft_log_history_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    sft_log_history_curve = fig_to_base64(sft_log_history_fig)
    ##########################

    ##########################
    #     sft performance    #
    ##########################
    validation_completions_fig = copy.copy(params['validation_completions_fig'])
    # Set the RGBA background white color with partial transparency
    validation_completions_fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.6))
    validation_completions_curve = fig_to_base64(validation_completions_fig)

    metrics_table = pd.DataFrame(
            [params['metrics_dict']]).to_html(classes='wide',
                                              escape=False, index = False)
    ##########################

    bootstrap_js_dependencies = [
        # 'https://code.jquery.com/jquery-3.2.1.slim.min.js',
        'jquery-3.2.1.slim.min.js',
        # 'https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js',
        'popper-1.12.9.min.js',
        # 'https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js',
        'bootstrap-4.0.0.min.js'
    ]

    ##########################
    #     Jinja template     #
    # rendering starts here  #
    ##########################
    class MultiFileSystemLoader(BaseLoader):
        """
        allow for more than one dir
        to serve as possible origins
        for Jinja input source files.
        """

        def __init__(self, searchpath):
            self.searchpath = searchpath

        def get_source(self, environment, template):
            for path in self.searchpath:
                template_path = os.path.join(path, template)
                if os.path.exists(template_path):
                    with open(template_path, 'r', encoding='utf-8') as file:
                        source = file.read()
                    return source, template_path, lambda: True

            raise TemplateNotFound(template)

    file_loader = MultiFileSystemLoader([
        params['template_dir'],
        os.path.realpath(os.path.join(
            os.path.dirname(importlib.util.find_spec(f"retrain_pipelines").origin),
            "pipeline_card", "static"))
    ])
    env = Environment(loader=file_loader)
    template = env.get_template('template.html')

    return template.render(
        bootstrap_js_dependencies=bootstrap_js_dependencies,
        title=params['title'], subtitle=params['subtitle'],
        blessed_color="#008000" if model_version_blessed \
                      else "#811331",
        blessed_background="#7CFC00" if model_version_blessed \
                           else "#FF3131",
        model_version_not_blessed="" if model_version_blessed \
                                  else "NOT ",
        ###################################

        # if model version not blessed => #
        previous_blessed_version_label=(
            None if model_version_blessed
            else params['current_blessed_version_label']),
        previous_blessed_commit_datetime=(
            None if model_version_blessed
            else params['current_blessed_commit_datetime']),
        previous_blessed_model_commit_hash=(
            None if model_version_blessed
            else params['current_blessed_model_commit_hash']),
        ## if ML framework instance is the same
        ## between non-blessed and previous-blessed
        previous_blessed_run_id=(
            None if previous_blessed_run is None
            else previous_blessed_run.id),
        previsous_blessed_card_href=(
            None if previous_blessed_run is None
            else previsous_blessed_card_href),
        previsous_blessed_card_url=(
            None if previous_blessed_run is None
            else previsous_blessed_card_url),
        ###################################

        # infra validation status =>      #
        local_serve_color=(
            None if (LocalServeReadinessEnum.NOT_APPLICABLE
                     == local_serve_is_ready)
            else "#008000" if (LocalServeReadinessEnum.SUCCESS
                               == local_serve_is_ready)
            else "#A34700" if (LocalServeReadinessEnum.FAILURE_NO_DOCKER
                               == local_serve_is_ready)
            else "#811331"),
        local_serve_background=(
            None if (LocalServeReadinessEnum.NOT_APPLICABLE
                     == local_serve_is_ready)
            else "#7CFC00" if (LocalServeReadinessEnum.SUCCESS
                               == local_serve_is_ready)
            else "#FFA500" if (LocalServeReadinessEnum.FAILURE_NO_DOCKER
                               == local_serve_is_ready)
            else "#FF3131"),
        local_serve_status=(
            None if (LocalServeReadinessEnum.NOT_APPLICABLE
                     == local_serve_is_ready)
            else "Passed" if (LocalServeReadinessEnum.SUCCESS
                               == local_serve_is_ready)
            else "Skipped" if (LocalServeReadinessEnum.FAILURE_NO_DOCKER
                               == local_serve_is_ready)
            else "Failed"),
        local_serve_reason=(
            "(docker missing on host)" if (
                LocalServeReadinessEnum.FAILURE_NO_DOCKER
                == local_serve_is_ready
            )
            else None),
        ###################################

        # EDA =>                          #
        main_dataset_repo_id=params['main_dataset_repo_id'],
        main_dataset_commit_hash=\
            params['main_dataset_commit_hash'],
        main_dataset_commit_datetime=\
            params['main_dataset_commit_datetime'],
        records_count="{:,}".format(params['records_count']),
        data_schema_table=indent(data_schema_table, ' '*36),
        answers_tools_count_curve=answers_tools_count_curve,
        words_count_curve=words_count_curve,

        ###################################

        # model training =>               #
        dataset_repo_id=params['dataset_repo_id'],
        dataset_version_label=params['dataset_version_label'],
        dataset_commit_datetime=\
            params['dataset_commit_datetime'],
        dataset_commit_hash=params['dataset_commit_hash'],
        dataset_augmentation_rate=params['dataset_augmentation_rate'],
        dataset_enrichment_rate=params['dataset_enrichment_rate'],

        model_repo_id=params['model_repo_id'],
        model_commit_hash=params['model_commit_hash'],
        model_version_label=params['model_version_label'],
        model_commit_datetime=params['model_commit_datetime'],

        cpt_log_history_curve=cpt_log_history_curve,
        sft_log_history_curve=sft_log_history_curve,

        ###################################

        # pipeline execution params
        base_model_repo_id=base_model_repo_id,
        base_model_version_label=base_model_version_label,
        base_model_commit_hash=base_model_commit_hash,
        base_model_commit_datetime=base_model_commit_datetime,
        pipeline_parameters_table=indent(pipeline_parameters_table, ' '*28),

        # validation perf =>              #
        validation_completions_curve=validation_completions_curve,
        metrics_table=indent(metrics_table, ' '*28),

        ###################################

        task_obj_python_cmd= \
            apply_args_color_format(params['task_obj_python_cmd']),
        dag_svg=indent(params['dag_svg'], ' '*40),
        __version__=__version__
    )