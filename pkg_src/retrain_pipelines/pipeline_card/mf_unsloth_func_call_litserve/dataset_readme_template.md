---
# @see https://github.com/huggingface/hub-docs/blob/main/datasetcard.md

{% set license_label = main_license_label if main_license_label == enrich_license_label or not (main_license_label and enrich_license_label) else none -%}
{% if license_label is none -%}
  {% set license_label = "unknown" -%}
{% endif -%}

{{configs }}

version: '{{ new_version_label }}'

timestamp: {{ utc_timestamp }}

pretty_name: {{ pretty_name }}

source_datasets:
- {{ main_repo_id }}
- {{ enrich_repo_id }}

license: {{license_label}}

language:
- en

task_categories:
- question-answering
- text-generation
- reinforcement-learning

tags:
- function-calling
- LLM Agent
- code
- synthetic

size_categories:
- {{ size_category }}

---

# {{ pretty_name }}

`version {{ new_version_label }}`  -  `{{ utc_timestamp }}`

Source datasets :
  - main&nbsp;:
    - <b>{{ main_pretty_name }}</b><br />
    `{{ main_repo_id }}`
    (<a href="https://huggingface.co/datasets/{{ main_repo_id }}/blob/{{ main_commit_hash }}/README.md"
        target="_blank">{{ main_commit_hash[:7] }}</a> - {{ main_commit_utc_date_str }})
    <br />
    license&nbsp;:
    {% if main_license_label -%}
    <b><code>{{main_license_label}}</code></b><br />
    {% else -%}
    <b><code>unknown</code></b><br />
    {% endif -%}
    {% if main_arxiv_codes -%}
    arxiv&nbsp;:<br />
    {%- for main_arxiv_code in main_arxiv_codes %}
      - <a href="https://huggingface.co/papers/{{ main_arxiv_code }}"
           target="_blank">https://huggingface.co/papers/{{ main_arxiv_code }}</a><br />
    {% endfor -%}
    {% endif -%}
    <br />
  - data-enrichment&nbsp;:
    - <b>{{ enrich_pretty_name }}</b><br />
    `{{ enrich_repo_id }}`
    (<a href="https://huggingface.co/datasets/{{ enrich_repo_id }}/blob/{{ enrich_commit_hash }}/README.md"
        target="_blank">{{ enrich_commit_hash[:7] }}</a> - {{ enrich_commit_utc_date_str }})
    <br />
    license&nbsp;:
    {% if enrich_license_label -%}
    <b><code>{{enrich_license_label}}</code></b><br />
    {% else -%}
    <b><code>unknown</code></b><br />
    {% endif -%}
    {% if enrich_arxiv_codes -%}
    arxiv&nbsp;:<br />
    {%- for enrich_arxiv_code in enrich_arxiv_codes %}
      - <a href="https://huggingface.co/papers/{{ enrich_arxiv_code }}"
           target="_blank">https://huggingface.co/papers/{{ enrich_arxiv_code }}</a><br />
    {% endfor -%}
    {% endif -%}
    <br />

The herein dataset has 2 configs : `continued_pre_training` and `supervised_finetuning`.<br />
The former serves for added intrinsic knowledge. Typical entries look like&nbsp;:<br />
```python
{{ main_format_description }}
```
The latter is a classic question/answer text dataset. Only tool calls are in the answers. May be an empty list.<br />
Data-augmentation rate&nbsp;: +{{ (augmentation_rate * 100)|round(1) ~ '%' }}<br />
Data-enrichment rate&nbsp;: +{{ (enrichment_rate * 100)|round(1) ~ '%' }}<br />

<hr />
Powered by
<a target="_blank" href="https://pypi.org/project/retrain-pipelines/">retrain-pipelines {{ __version__ }}</a> - 
<code>Run by <a target="_blank" href="https://huggingface.co/{{ run_user }}">{{ run_user }}</a></code> -
<em><b>{{ mf_flow_name }}</b></em> - mf_run_id&nbsp;: <code>{{ mf_run_id }}</code>

