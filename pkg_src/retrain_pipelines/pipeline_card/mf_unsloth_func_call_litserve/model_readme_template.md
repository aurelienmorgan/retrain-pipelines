---
# @see https://github.com/huggingface/hub-docs/blob/main/modelcard.md
# @see https://huggingface.co/docs/huggingface_hub/guides/model-cards#update-metadata

{% set timestamp_str = commit_datetime.strftime('%Y%m%d_%H%M%S') ~ '%03d'|format(commit_datetime.microsecond // 1000) ~ '_UTC' -%}

version: '{{ new_version_label }}'

timestamp: '{{ timestamp_str }}'

model_name: {{ pretty_name }}

base_model: {{ base_model_repo_id }}
library_name: peft
datasets:
- {{ dataset_repo_id }}

license: {{ base_model_license_label }}

language:
- en

task_categories:
- question-answering

tags:
- retrain-pipelines
- function-calling
- LLM Agent
- code
- Unsloth

thumbnail: https://cdn-avatars.huggingface.co/v1/production/uploads/651e93137b2a2e027f9e55df/96hzBved0YMjCq--s0kad.png


# @see https://huggingface.co/docs/hub/models-widgets#enabling-a-widget
widget:
- text: "Is this review positive or negative? Review: Best cast iron skillet you will ever buy."
  output:
      text: "Hello my name is Julien"


# @see https://huggingface.co/docs/huggingface_hub/guides/model-cards#include-evaluation-results
model-index:
- name: {{ pretty_name }}
  results:
  - task:
      type: question-answering
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
    - type: f1
      value: 0.65

---

# {{ pretty_name }}

`version {{ new_version_label }}`  -  `{{ commit_datetime.strftime("%Y-%m-%d %H:%M:%S UTC") }}`
(retraining
<a target="_blank"
   href="https://huggingface.co/retrain-pipelines/function_caller/tree/retrain-pipelines_source-code/v{{ new_version_label }}_{{ timestamp_str }}">source-code</a> |
<a target="_blank"
   href="https://huggingface.co/retrain-pipelines/function_caller/tree/retrain-pipelines_pipeline-card/v{{ new_version_label }}_{{ timestamp_str }}">pipeline-card</a>)

Training dataset&nbsp;:
- <code>{{ dataset_repo_id }} v{{ dataset_version_label }}</code>
(<a href="https://huggingface.co/datasets/{{ dataset_repo_id }}/blob/{{ dataset_commit_hash }}/README.md"
    target="_blank">{{ dataset_commit_hash[:7] }}</a> -
    {{ dataset_commit_datetime.strftime("%Y-%m-%d %H:%M:%S UTC") }})

Base model&nbsp;:
- <code>{{ base_model_repo_id }}</code>
(<a href="https://huggingface.co/{{ base_model_repo_id }}/blob/{{ base_model_commit_hash }}/README.md"
    target="_blank">{{ base_model_commit_hash[:7] }}</a> -
    {{ base_model_commit_datetime.strftime("%Y-%m-%d %H:%M:%S UTC") }})
{% if base_model_arxiv_codes -%}
arxiv&nbsp;:<br />
{%- for base_model_arxiv_code in base_model_arxiv_codes %}
  - <a href="https://huggingface.co/papers/{{ base_model_arxiv_code }}"
       target="_blank">https://huggingface.co/papers/{{ base_model_arxiv_code }}</a><br />
{% endfor -%}
{% endif -%}


<br />
<br />
<br />
<br />
<br />
<br />
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; !! Section on Eval goes here !!
<br />
<br />
<br />
<br />
<br />
<br />

<hr />
Powered by
<code><a target="_blank"
         href="https://pypi.org/project/retrain-pipelines/">retrain-pipelines
      {{ __version__ }}</a></code> - 
<code>Run by <a target="_blank" href="https://huggingface.co/{{ run_user }}">{{ run_user }}</a></code> -
<em><b>{{ mf_flow_name }}</b></em> - mf_run_id&nbsp;: <code>{{ mf_run_id }}</code>

