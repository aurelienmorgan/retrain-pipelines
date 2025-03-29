
![PyPI - Monthly Downloads](https://img.shields.io/pypi/dm/retrain-pipelines)
![PyPI - Total Downloads](https://static.pepy.tech/badge/retrain-pipelines)
![PyPI - Release version](https://img.shields.io/pypi/v/retrain-pipelines)
![GitHub - License](https://img.shields.io/github/license/aurelienmorgan/retrain-pipelines?logo=github&style=flat&color=purple)

![logo_large](https://github.com/user-attachments/assets/19725866-13f9-48c1-b958-35c2e014351a)

<b>retrain-pipelines</b> simplifies the creation and management of machine learning retraining pipelines. 
The package is designed to remove the complexity of building end-to-end ML retraining pipelines, allowing users to focus on their data and model-architecture. 
With pre-built, highly adaptable pipeline examples that work out of the box, users can easily integrate their own data and begin retraining models with minimal-to-no setup. 

<center><a href="https://hf.co/retrain-pipelines" target="_blank"><img src="https://github.com/user-attachments/assets/d34a9f58-309f-4c1f-9cd1-f576a56c3b24" width="40%"/><br />
https://hf.co/retrain-pipelines</a></center>

### Key features of retrain-pipelines include&nbsp;:
- **Model version blessing**: Automatically compare the performance of retrained models against previous best versions to ensure only superior models are deployed.
- **Infrastructure validation**: Each retraining pipeline includes inference pipeline packaging, local Docker container deployment, and request/response validation to ensure that models are production-ready.
- **Comprehensive documentation**: Every retraining pipeline is fully documented with sections covering Exploratory Data Analysis (EDA), hyperparameter tuning, retraining steps, model performance metrics, and key commands for retrieving training artifacts. 
  Additionally, DAG information for the retraining process is readily available for pipeline transparency and debugging.

In essence, <b>retrain-pipelines</b> offers a seamless solution: "Come with your data and it works" with the added benefit of flexibility for more advanced users to adjust and extend pipelines as needed.

### Customizability & Adaptability
<b>retrain-pipelines</b> offers a high degree of flexibility, allowing users to tailor the pre-shipped pipelines to their specific needs:
- **Custom Preprocessing Functions**&nbsp;: Users can provide their own Python functions for custom data preprocessing. For example, some built-in pipelines for tabular data allow optional bucketization of numerical features by name, but you can easily modify or extend these preprocessing steps to suit your dataset and feature requirements.
- **Custom Pipeline Card Generation**&nbsp;: You can specify custom Python functions to generate pipeline cards, such as including specific performance charts or metrics relevant to your use case.
- **Custom HTML Templates**&nbsp;: For further personalization, `retrain-pipelines` supports customizable HTML templates, enabling you to adjust formatting, insert specific charts, change page colors, or even add your company's logo to documentation pages. 

<b>retrain-pipelines</b> doesn't just streamline the retraining process, it empowers teams to innovate faster, iterate smarter, and deploy more robust models with confidence. Whether you're looking for an out-of-the-box solution or a highly customizable pipeline, <b>retrain-pipelines</b> is your ultimate companion for continuous model improvement.


## Getting Started

You can trigger a <b>retrain-pipelines</b> launch from many different places.

[local_launcher.webm](https://github.com/user-attachments/assets/4164abfd-4cd6-4e8a-a720-07267241b9f6)


## Sample pipelines

the <b>retrain-pipelines</b> package comes with off-the-shelf Machine Learning retraining pipelines. Find them at <code><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines" target="_blank">/sample_pipelines</a></code>. For instance&nbsp;:

| framework | modality | task | model lib | Serving |  |
|----------|----------|----------|----------|----------|--|
| <a href="https://metaflow.org/" target="_blank">Metaflow</a> <img src="https://github.com/user-attachments/assets/30f4f382-3032-4bf7-b697-f6dbcab35fd7" height=20px /> | text, NLP   | function&nbsp;calling | <a href="https://unsloth.ai/" target="_blank">Unsloth</a> <img src="https://github.com/user-attachments/assets/3bb9244b-8c89-41fa-8b38-c4862763eea1" width=20px /> / <a href="https://github.com/dreamquark-ai/tabnet/tree/develop" target="_blank">Qwen&nbsp;2.5</a>&nbsp;<img src="https://github.com/user-attachments/assets/3067f88e-3064-470f-9c8e-2d80c40b3d5c" width=20px /> | <a href="https://lightning.ai/docs/litserve/home/" target="_blank">LitServe</a> <img src="https://github.com/user-attachments/assets/b5abcd66-9cb4-420c-ad2c-29bafb0f3b62" width=20px /> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/Unsloth_Qwen_FuncCall" target="_blank">starter-kit</a></b> |
| <a href="https://metaflow.org/" target="_blank">Metaflow</a> <img src="https://github.com/user-attachments/assets/30f4f382-3032-4bf7-b697-f6dbcab35fd7" height=20px /> | Tabular   | regression   | <a href="https://www.dask.org/" target="_blank">Dask</a> <img src="https://github.com/user-attachments/assets/a94807e7-cc67-4415-9a9e-da1ed4755cb1" width=20px /> / <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank">LightGBM</a> <img src="https://github.com/user-attachments/assets/92ac0b53-17f8-470d-9c73-619657db42bd" width=20px />   | <a href="https://www.seldon.io/solutions/seldon-mlserver" target="_blank">ML Server</a> <img src="https://github.com/user-attachments/assets/69c57bce-cd38-4f8c-8730-e5171e842d13" width=20px /> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/LightGBM_hp_cv_WandB" target="_blank">starter-kit</a></b> |
| <a href="https://metaflow.org/" target="_blank">Metaflow</a> <img src="https://github.com/user-attachments/assets/30f4f382-3032-4bf7-b697-f6dbcab35fd7" height=20px /> | Tabular   | classification | <a href="https://pytorch.org/" target="_blank">Pytorch</a> <img src="https://github.com/user-attachments/assets/bfa9b38e-e9b3-41ff-8370-e64a0a0a4a93" width=20px /> / <a href="https://github.com/dreamquark-ai/tabnet/tree/develop" target="_blank">TabNet</a> | <a href="https://pytorch.org/serve/" target="_blank">TorchServe</a>&nbsp;<img src="https://github.com/user-attachments/assets/bfa9b38e-e9b3-41ff-8370-e64a0a0a4a93" width=20px /> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/TabNet_hp_cv_WandB" target="_blank">starter-kit</a></b> |


You can simply give one of those your data and it just runs. The only manual change you need to do is regarding the endpoint request &amp; serving signatures, since it is purposely hard-coded.<br />
<small>Indeed, the <code>infra_validator</code> step is here to ensure that <u>your inference pipeline</u> (the one you're working on building a continuous-retraining automation for) keeps adhering to the schema expected by consumers of the inference endpoint. So, if you break the format of the required input raw data, you need to create a somehow new retraining pipeline and assign it a new unique name. This is to ensure that any interface disruption between the inference endpoint and its consumer(s) is intentional.</small>

## some important markers

One of the things that make <b>retrain-pipelines</b> stand is its focus on strong MLOps fundamentals.

<details>
  <summary>model blessing&nbsp;🔽</summary>
<b>retrain-pipelines</b> cares for the newly-retrained model version to be evaluated against the previous model versions from that retraining pipeline. We indeed ensure that no lesser-performing model ever gets into production.<br />
Default sample pipelines each come with certain built-in evaluation criteria but, you can customize those per your own requirement. You can for instance choose to include evaluation of model performance on a particular sub-population, so as to serve as a gateway against potential incoming biases.
<hr width=60% />
</details>

<details>
  <summary>infrastructure validation&nbsp;🔽</summary>
<b>retrain-pipelines</b> cares for the inference endpoint to be tested prior to deployment. We pack the preprocessing engine together with the newly retrained (and blessed) model version with the ML-server of choice and deploy it locally. We then send an inference request to that temp endpoint and check for a <code>200 http-ok</code> response with a valid payload format.
<hr width=60% />
</details>

<details>
  <summary>pipeline cards&nbsp;🔽</summary>
<b>retrain-pipelines</b> is strongly opinionated around ease of quick-access to information ML-engineers care for when it comes to retraining and serving.<br />
That's why it offers a central place and minimal amounts of clicks to navigate efficiently.
<table width=100%>
  <tr width=100%>
    <td>
      <a href="https://github.com/user-attachments/assets/fc4b94a5-8178-49b0-822a-a8088dbf1b6d" target="_blank"><img src="https://github.com/user-attachments/assets/fc4b94a5-8178-49b0-822a-a8088dbf1b6d" width=100 height=80 /></a><br />
      overview
    </td>
    <td>
      <a href="https://github.com/user-attachments/assets/b5ce6b19-df87-4486-ac71-3cf06776452e" target="_blank"><img src="https://github.com/user-attachments/assets/b5ce6b19-df87-4486-ac71-3cf06776452e" width=100 height=80 /></a><br />
      EDA
    </td>
    <td>
      <a href="https://github.com/user-attachments/assets/34d401b2-ad79-49e3-b07f-f6fb61418ea1" target="_balnk"><img src="https://github.com/user-attachments/assets/34d401b2-ad79-49e3-b07f-f6fb61418ea1" width=100 height=80 /></a><br />
      overall retraining
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/user-attachments/assets/560aa7a6-c7ad-4dce-97e8-b1f8cee8edae" target="_blank"><img src="https://github.com/user-attachments/assets/a1aa13c1-2401-4527-b5eb-a8dc2ddca195" width=100 height=80 /></a><br />
      hyperparameter tuning
    </td>
    <td>
      <a href="https://github.com/user-attachments/assets/bf0e0e3f-a442-415d-bb79-104afba3f519" target="_blank"><img src="https://github.com/user-attachments/assets/bf0e0e3f-a442-415d-bb79-104afba3f519" width=100 height=80 /></a><br />
     key artifacts
    </td>
    <td>
      <a href="https://github.com/user-attachments/assets/d6d6c645-be5a-4b3b-9abf-339e0b034703" target="_blank"><img src="https://github.com/user-attachments/assets/35ddeb91-81c8-4caa-b17f-6704aae22410" width=100 height=80 /></a><br />
      pipeline DAG
    </td>
  </tr>
  <tr>
    <td colspan="3">
      <em><small>click thumbnails to enlarge</small></em>
    </td>
  </tr>
</table>
Browse a live example for yourself <a href="https://retrain-pipelines.w3spaces.com/html-custom-2d5ac4812402cf8726619e81d8cc6c8f0ba94c24.html" target="_blank">here on W3Schools Spaces</a>
(click "continue" on the W3Schools landing page)
<hr width=60% />
</details>

<details>
  <summary>Third-parties integration&nbsp;🔽</summary>
TensorBoard, PyTorch Profiler, Weights and Biases, Hugging Face. <b>retrain-pipelines</b> aims at making centrally available to ML engineers the information they care for.

  <details>
  <summary>illustration with <code>WandB</code> in the <code>LightGBM_hp_cv_WandB</code> sample pipeline&nbsp;🔽</summary>
  In the example of the <code>LightGBM_hp_cv_WandB</code> sample pipeline for instance, you can find information on how to view details on logging performed during the different <code>training_job</code> steps of a given run. Follow the guidance from the below video&nbsp;:<br />

  [wandb_integration.webm](https://github.com/user-attachments/assets/730bc695-0768-484b-8e6e-2dbf0db08d68)
  </details>
  <hr width=60% />
</details>

<details>
  <summary>customizability&nbsp;🔽</summary>
  As alluded to <a href="#customizability--adaptability">above</a>, a lot of room is given to ML engineers for them to customize <b>retrain-pipelines</b> workflows.<br />
  For staters, the sample pipelines are freely modifiable themselves. But, it goes far beyond that. One can go deep into customization with the defaults for <code>preprocessing</code> and for <code>pipeline_card</code> being fully amendable as well.

  <details>
    <summary>illustration with the <code>LightGBM_hp_cv_WandB</code> sample pipeline&nbsp;🔽</summary>
    Start by getting the default which you'd like to customize (any combination of the below 3 you'd like) :
    <ul>
      <li><code>reprocessing.py</code> module</li>
      <li><code>pipeline_card.py</code> module</li>
      <li><code>template.html</code> html template</li>
    </ul>

  ```shell
  cd sample_pipelines/LightGBM_hp_cv_WandB/
  ```
  ```python
  from retraining_pipeline import LightGbmHpCvWandbFlow

  LightGbmHpCvWandbFlow.copy_default_preprocess_module(".", exists_ok=True)
  LightGbmHpCvWandbFlow.copy_default_pipeline_card_module(".", exists_ok=True)
  LightGbmHpCvWandbFlow.copy_default_pipeline_card_html_template(".", exists_ok=True)
  ```
  Once you updated any of them, you can launch a <b>retrain-pipelines</b> run so it uses those :
  ```python
  %retrain_pipelines_local retraining_pipeline.py run \
    --pipeline_card_artifacts_path "." \
    --preprocess_artifacts_path "."
  ```
  </details>
  <hr width=60% />
</details>


## retrain-pipelines inspectors

Inspectors are convenience methods that abstract away some of the logic to get access to artifacts logged during <b>retrain-pipelines</b> runs.

For instance&nbsp;:
<ul>
  <li>
  <details>
    <summary><code>browse_local_pipeline_card</code>&nbsp;🔽</summary>
    With this convenience method, programmatically open a <code>pipeline_card</code> without the need to browse and click a ML-framework UI&nbsp;:<br />

  ```python
  from retrain_pipelines.inspectors import browse_local_pipeline_card
  
  browse_local_pipeline_card(mf_flow_name)
  ```
  This opens the <code>pipeline_card</code> in a web-browser tab, so you don't have to look for it.
  It's ideal for quick ideation during the drafting phase&nbsp;:
  developers can now <code>run/resume</code> &amp; <code>browse</code> in a chain of instructions.
  <hr width=60% />
  </details>
  </li>
  <li>
  <details>
    <summary><code>get_execution_source_code</code>&nbsp;🔽</summary>
    With this convenience method, programmatically access the versioned source code that was used for a particular <b>retrain-pipelines</b> run. This comes together with the <b>WandB integration</b>&nbsp;:<br />

  ```python
  from retrain_pipelines.inspectors import get_execution_source_code
  
  for source_code_artifact in get_execution_source_code(mf_run_id=<your_flow_id>):
    print(f" - {source_code_artifact.name} {source_code_artifact.url}")
  ```
  You can even have those artifacts downloaded on the go&nbsp;:

  ```python
  from retrain_pipelines.inspectors import explore_source_code
  # download and open file explorer
  explore_source_code(mf_run_id=<your_flow_id>)
  ```
  <hr width=60% />
  </details>
  </li>
  <li>
  <details>
    <summary><code>plot_run_all_cv_tasks</code>&nbsp;🔽</summary>
  Specific to <b>retrain-pipelines</b> runs that involve data-parallelism,
  this <b>inspector</b> method plots each individual hyperparameter-tuning cross-validation training job, showing details for every data-parallel worker.<br />
  For example, for executions of the <code><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/LightGBM_hp_cv_WandB" target="_blank">LightGbmHpCvWandbFlow</a></code> sample pipeline (which employs <b>Dask</b> for data-parallel training), this gives&nbsp;:<br />

  ```python
  from retrain_pipelines.inspectors.hp_cv_inspector import plot_run_all_cv_tasks
  
  plot_run_all_cv_tasks(mf_run_id=<your_flow_id>)
  ```
  with results looking like below for a run with 6 different sets of hp values, 2 cross-validation folds and with 4 Dask data-parallel workers&nbsp;:<br />
  <a href="https://github.com/user-attachments/assets/f3c03b06-a086-4be5-9815-73d1a887179d" target="_blank"><img src="https://github.com/user-attachments/assets/f3c03b06-a086-4be5-9815-73d1a887179d" width=400/></a>
  <hr width=60% />
  </details>
  </li>
  <li>
    and more.
  </li>
</ul>

# launch tests
    pytest -s tests

# build from source
    python -m build --verbose pkg_src
# install from source (dev mode)
    pip install -e pkg_src
# install from remote source
    pip install git+https://github.com/aurelienmorgan/retrain-pipelines.git@master#subdirectory=pkg_src

# PyPi
find us @ https://pypi.org/project/retrain-pipelines/
<br />
<hr />

Drop us a star&nbsp;!&nbsp;⭐&nbsp;&nbsp;&nbsp;&nbsp;[![GitHub Stars](https://img.shields.io/github/stars/aurelienmorgan/retrain-pipelines.svg?style=social&label=-%C2%A0retrain-pipelines%C2%A0-&maxAge=172800)](https://github.com/aurelienmorgan/retrain-pipelines/stargazers)<br /> 
Follow us on <b>Hugging&nbsp;Face</b>&nbsp;!&nbsp;[![GitHub Stars](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Forganizations%2Fretrain-pipelines%2Foverview&query=%24.numFollowers&logo=huggingface&label=followers)](https://hf.co/retrain-pipelines)
<br />
<hr />
