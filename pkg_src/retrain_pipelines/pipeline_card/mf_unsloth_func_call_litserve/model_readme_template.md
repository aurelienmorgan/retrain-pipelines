---
# @see https://github.com/huggingface/hub-docs/blob/main/modelcard.md
# @see https://huggingface.co/docs/huggingface_hub/guides/model-cards#update-metadata
# @see https://huggingface.co/docs/hub/model-cards#model-card-metadata

{% set timestamp_str = commit_datetime.strftime('%Y%m%d_%H%M%S') ~ '%03d'|format(commit_datetime.microsecond // 1000) ~ '_UTC' -%}

version: '{{ new_version_label }}'

timestamp: '{{ timestamp_str }}'

model_name: {{ pretty_name }}

base_model: {{ base_model_repo_id }}
base_model_relation: adapter
library_name: transformers
datasets:
- {{ dataset_repo_id }}

license: {{ base_model_license_label }}

language:
- en

task_categories:
- text2text-generation

tags:
- retrain-pipelines
- function-calling
- LLM Agent
- code
- unsloth

thumbnail: https://cdn-avatars.huggingface.co/v1/production/uploads/651e93137b2a2e027f9e55df/96hzBved0YMjCq--s0kad.png


# @see https://huggingface.co/docs/hub/models-widgets#enabling-a-widget
# @see https://huggingface.co/docs/hub/models-widgets-examples
# @see https://huggingface.co/docs/hub/en/model-cards#specifying-a-task--pipelinetag-
pipeline_tag: text2text-generation
widget:
  - text: >-
      Hello
    example_title: No function call
    output:
      text: '[]'
  - text: >-
      Is 49 a perfect square?
    example_title: Perfect square
    output:
      text: '[{"name": "is_perfect_square", "arguments": {"num": 49}}]'

mf_run_id: '{{ mf_run_id }}'

# @see https://huggingface.co/docs/huggingface_hub/guides/model-cards#include-evaluation-results
# @see https://huggingface.co/docs/huggingface_hub/main/en/package_reference/cards#huggingface_hub.EvalResult
model-index:
- name: {{ pretty_name }}
  results:
  - task:
      type: text2text-generation
      name: Text2Text Generation
    dataset:
      name: {{ dataset_pretty_name }}
      type: {{ dataset_repo_id }}
      split: validation
      revision: {{ dataset_commit_hash }}
{{ perf_metrics }}

---

<div 
  class="
    p-6 mb-4 rounded-lg 
    pt-6 sm:pt-9
    bg-gradient-to-b
    from-purple-500 
    dark:from-purple-500/20
  "
>
  <div 
    class="
      pl-4 rounded-lg 
      border-2 border-gray-100 
      bg-gradient-to-b
      from-purple-500 
      dark:from-purple-500/20
    "
  >
    <b>{{ pretty_name }}</b>
</div>
  <code>version {{ new_version_label }}</code>  -  <code>{{ commit_datetime.strftime("%Y-%m-%d %H:%M:%S UTC") }}</code>
  (retraining
  <a target="_blank"
     href="https://huggingface.co/{{ model_repo_id }}/tree/retrain-pipelines_source-code/v{{ new_version_label }}_{{ timestamp_str }}">source-code</a> |
  <a target="_blank"
     href="https://huggingface.co/spaces/retrain-pipelines/online_pipeline_card_renderer/?model_repo_id={{ model_repo_id }}&version_id=v{{ new_version_label }}_{{ timestamp_str }}">pipeline-card</a>)
</div>

Training dataset&nbsp;:
- <code>{{ dataset_repo_id }} v{{ dataset_version_label }}</code>
(<a href="https://huggingface.co/datasets/{{ dataset_repo_id }}/blob/{{ dataset_commit_hash }}/README.md"
    target="_blank">{{ dataset_commit_hash[:7] }}</a> -
    {{ dataset_commit_datetime.strftime("%Y-%m-%d %H:%M:%S UTC") }})
    <br />
    <img alt="" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2F{{ dataset_repo_id | urlencode }}&amp;query=%24.downloads&amp;logo=huggingface&amp;label=downloads"  class="inline-block" />&nbsp;<img alt="" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2F{{ dataset_repo_id | urlencode }}&amp;query=%24.likes&amp;logo=huggingface&amp;label=likes"  class="inline-block" />

Base model&nbsp;:
- <code>{{ base_model_repo_id }}{% if base_model_version_label is not none %} v{{ base_model_version_label }}{% endif %}</code>
(<a href="https://huggingface.co/{{ base_model_repo_id }}/blob/{{ base_model_commit_hash }}/README.md"
    target="_blank">{{ base_model_commit_hash[:7] }}</a> -
    {{ base_model_commit_datetime.strftime("%Y-%m-%d %H:%M:%S UTC") }})
    <br />
    <img alt="" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2F{{ base_model_repo_id | urlencode }}&amp;query=%24.downloads&amp;logo=huggingface&amp;label=downloads"  class="inline-block" />&nbsp;<img alt="" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2F{{ base_model_repo_id | urlencode }}&amp;query=%24.likes&amp;logo=huggingface&amp;label=likes"  class="inline-block" /><br />
{% if base_model_arxiv_codes -%}
arxiv&nbsp;:<br />
{%- for base_model_arxiv_code in base_model_arxiv_codes %}
  - <code><a href="https://huggingface.co/papers/{{ base_model_arxiv_code }}"
             target="_blank">{{ base_model_arxiv_code }}</a></code><br />
{% endfor -%}
{% endif -%}

The herein LoRa adapter can for instance be used as follows&nbsp;:<br />
```python
{{ main_usage_snippet }}
```

<br />
<br />

<div 
  class="
    p-6 mb-4 rounded-lg 
    pt-6 sm:pt-9
    px-4
    pb-1 
    bg-gradient-to-t
    from-purple-500 
    dark:from-purple-500/20
  "
>
  <div 
    class="
      p-6 mb-4 rounded-lg 
      border-2 border-gray-100 
      pt-6 sm:pt-9
      bg-gradient-to-t
      from-purple-500 
      dark:from-purple-500/20
    "
  >
    Powered by
    <code><a target="_blank"
             href="https://github.com/aurelienmorgan/retrain-pipelines">retrain-pipelines
          {{ __version__ }}</a></code> - 
    <code>Run by <a target="_blank" href="https://huggingface.co/{{ run_user }}">{{ run_user }}</a></code> -
    <em><b>{{ mf_flow_name }}</b></em> - mf_run_id&nbsp;: <code>{{ mf_run_id }}</code>
  </div>
</div>


