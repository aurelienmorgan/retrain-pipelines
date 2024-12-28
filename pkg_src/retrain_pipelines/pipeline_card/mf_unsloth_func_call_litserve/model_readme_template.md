---
# @see https://github.com/huggingface/hub-docs/blob/main/modelcard.md
# @see https://huggingface.co/docs/huggingface_hub/guides/model-cards#update-metadata

version: '{{ new_version_label }}'

timestamp: {{ utc_timestamp }}

model_name: {{ pretty_name }}

base_model: {{ base_model_repo_id }}
library_name: peft

license: {{license_label}}

language:
- en

task_categories:
- question-answering

tags:
- function-calling
- LLM Agent
- code
- Unsloth



thumbnail: https://cdn-avatars.huggingface.co/v1/production/uploads/651e93137b2a2e027f9e55df/96hzBved0YMjCq--s0kad.png

datasets:
- {{ dataset_repo_id }}

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

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; !!! TEMPLATE UNDER CONSTRUCTION !!!

# {{ pretty_name }}

`version {{ new_version_label }}`  -  `{{ utc_timestamp }}`

Training dataset :
&nbsp; &nbsp; {{ dataset_repo_id }}
<code>{{ dataset_version_label }}</code> - <code>{{ dataset_commit_commit_utc_date_str }}</code>
{{ dataset_commit_hash }}
(<a href="https://huggingface.co/datasets/{{ dataset_repo_id }}/blob/{{ dataset_commit_hash }}/README.md"
    target="_blank">{{ dataset_commit_hash[:7] }}</a>)


Source code&nbsp;:
https://huggingface.co/retrain-pipelines/function_caller/tree/retrain-pipelines_source-code/{{ new_version_label }}

Pipeline-card&nbsp;:
https://huggingface.co/retrain-pipelines/function_caller/tree/retrain-pipelines_pipeline-card/{{ new_version_label }}




<hr />
Powered by <code>retrain-pipelines v{{ __version__ }}</code> - 
<code>Run by <a target="_blank" href="https://huggingface.co/{{ run_user }}">{{ run_user }}</a></code> -
<em><b>{{ mf_flow_name }}</b></em> - mf_run_id&nbsp;: <code>{{ mf_run_id }}</code>

