
import os
import logging

from huggingface_hub import list_repo_files, list_repo_commits, \
    hf_hub_download

from flask import Flask, request, render_template_string, \
    render_template


BRANCH_NAME = "retrain-pipelines_pipeline-card"
HF_TOKEN = os.getenv("HF_TOKEN", None)
ERROR_PAGE = "index.html"

app = Flask(__name__, template_folder='/app')

@app.errorhandler(Exception)
def handle_exception(error):
    # status code, default to 500
    status_code = getattr(error, 'code', 500)
    error_msg = str(error) or 'An unknown error occurred.'
    app.logger.error(
        f"{request.remote_addr} - Exception occurred: {error_msg}",
        exc_info=error)

    return render_template(ERROR_PAGE, error_msg=error_msg), \
           status_code


@app.route('/', methods=['GET'])
def preview_html():
    model_repo_id = request.args.get('model_repo_id')
    if not model_repo_id:
        error_msg="Please provide a \"model_repo_id\" parameter"
        app.logger.error(f"{request.remote_addr} - {error_msg}",
                         exc_info=False)
        return render_template(
                    ERROR_PAGE, error_msg=error_msg), \
               400
    subfolder = request.args.get('subfolder')
    if not subfolder:
        error_msg="Please provide a \"subfolder\" parameter"
        app.logger.error(f"{request.remote_addr} - {error_msg}",
                         exc_info=False)
        return render_template(
                    ERROR_PAGE, error_msg=error_msg), \
               400

    try:
        pipeline_card_filename = get_pipeline_card_filename(
            model_repo_id=model_repo_id,
            banch_name=BRANCH_NAME,
            subfolder=subfolder,
            hf_token=HF_TOKEN
        )

        content = open(hf_hub_download(
                repo_id=model_repo_id,
                repo_type="model",
                revision=BRANCH_NAME,
                subfolder=subfolder,
                filename=pipeline_card_filename,
                token=HF_TOKEN,
                # cache_dir="/usr/local/.cache"
            )).read()
        return render_template_string(content)
    except Exception as e:
        error_msg = f"Error fetching pipeline-card: {str(e)}"
        app.logger.error(f"{request.remote_addr} - {error_msg}",
                         exc_info=False)
        return render_template(
                    ERROR_PAGE, error_msg=error_msg), \
               500


def get_pipeline_card_filename(
    model_repo_id: str,
    banch_name: str,
    subfolder: str,
    hf_token: str
) -> str:
    """
    There can be more than one pipeline-card
    portable html file per model version
    since run can be 'resumed', for instance
    at the 'pipeline_card' step itself.
    In such cases, we consider the latest.
    """
    
    files = list_repo_files(
        repo_id=model_repo_id,
        repo_type="model",
        revision=banch_name,
        token=hf_token
    )
    
    pipeline_card_files = [f for f in files if f.startswith(f"{subfolder}/")]
    if 0 == len(pipeline_card_files):
        raise Exception(f"No pipeline-card exists for \"subfolder\".")
    elif 1 < len(pipeline_card_files):
        commits = list_repo_commits(
            repo_id=model_repo_id,
            revision=banch_name,
            repo_type="model",
            token=hf_token
        )
        latest_commit = None
        latest_commit_time = None
        pipeline_card = None
        for commit in commits:
            files = list_repo_files(
                repo_id=model_repo_id,
                revision=commit.commit_id,
                repo_type="model",
                token=hf_token
            )
            for file in files:
                if file in pipeline_card_files:
                    commit_time = commit.created_at
                    if latest_commit_time is None or commit_time > latest_commit_time:
                        latest_commit_time = commit_time
                        html_pipeline_card_path = file
                        latest_commit = commit
    else:
        html_pipeline_card_path = pipeline_card_files[0]
    
    return os.path.basename(html_pipeline_card_path)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)


