
![PyPI - Downloads](https://img.shields.io/pypi/dm/retrain-pipelines)
![GitHub - License](https://img.shields.io/github/license/aurelienmorgan/retrain-pipelines?logo=github&style=flat&color=green)

![uder_construction](https://github.com/user-attachments/assets/2ab16d54-c565-409b-b00c-fd3ad20d59df)
<center>This README is nowhere near ready yet.</center>

![logo_large](https://github.com/user-attachments/assets/19725866-13f9-48c1-b958-35c2e014351a)

<b>retrain-pipelines</b> simplifies the creation and management of machine learning retraining pipelines. 
The package is designed to remove the complexity of building end-to-end ML retraining pipelines, allowing users to focus on their data and model-architecture. 
With pre-built, highly adaptable pipeline examples that work out of the box, users can easily integrate their own data and begin retraining models with minimal-to-no setup. 

### Key features of retrain-pipelines include:
- **Model version blessing**: Automatically compare the performance of retrained models against previous best versions to ensure only superior models are deployed.
- **Infrastructure validation**: Each retraining pipeline includes inference pipeline packaging, local Docker container deployment, and request/response validation to ensure that models are production-ready.
- **Comprehensive documentation**: Every retraining pipeline is fully documented with sections covering Exploratory Data Analysis (EDA), hyperparameter tuning, retraining steps, model performance metrics, and key commands for retrieving training artifacts. 
  Additionally, DAG information for the retraining process is readily available for pipeline transparency and debugging.

In essence, <b>retrain-pipelines</b> offers a seamless solution: "Come with your data, and it works," with the added benefit of flexibility for more advanced users to adjust and extend pipelines as needed.

### Customizability & Adaptability
<b>retrain-pipelines</b> offers a high degree of flexibility, allowing users to tailor the pre-shipped pipelines to their specific needs:
- **Custom Preprocessing Functions**: Users can provide their own Python functions for custom data preprocessing. For example, some built-in pipelines for tabular data allow optional bucketization of numerical features by name, but you can easily modify or extend these preprocessing steps to suit your dataset and feature requirements.
- **Custom Pipeline Card Generation**: You can specify custom Python functions to generate pipeline cards, such as including specific performance charts or metrics relevant to your use case.
- **Custom HTML Templates**: For further personalization, retrain-pipelines supports customizable HTML templates, enabling you to adjust formatting, insert specific charts, change page colors, or even add your company's logo to documentation pages. 

<b>retrain-pipelines</b> doesn't just streamline the retraining process, it empowers teams to innovate faster, iterate smarter, and deploy more robust models with confidence. Whether you're looking for an out-of-the-box solution or a highly customizable pipeline, <b>retrain-pipelines</b> is your ultimate companion for continuous model improvement.


## Getting Started

You can trigger a <b>retrain-pipelines</b> launch from many different places.

[local_launcher.webm](https://github.com/user-attachments/assets/4164abfd-4cd6-4e8a-a720-07267241b9f6)


## Sample pipelines

the <b>retrain-pipelines</b> package comes with off-the-shelf pipelines. Find them at <code>/sample_pipelines</code>. For instance&nbsp;:

| framework | modality | task | model lib | Serving |  |
|----------|----------|----------|----------|----------|--|
| <a href="https://metaflow.org/" target="_blank">Metaflow</a> <img src="https://github.com/user-attachments/assets/30f4f382-3032-4bf7-b697-f6dbcab35fd7" height=20px /> | Tabular   | regression   | <a href="https://www.dask.org/" target="_blank">Dask</a> <img src="https://github.com/user-attachments/assets/a94807e7-cc67-4415-9a9e-da1ed4755cb1" width=20px /> / <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank">LightGBM</a> <img src="https://github.com/user-attachments/assets/92ac0b53-17f8-470d-9c73-619657db42bd" width=20px />   | <a href="https://www.seldon.io/solutions/seldon-mlserver" target="_blank">ML Server</a> <img src="https://github.com/user-attachments/assets/69c57bce-cd38-4f8c-8730-e5171e842d13" width=20px /> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/LightGBM_hp_cv_WandB" target="_blank">starter-kit</a></b> |
| <a href="https://metaflow.org/" target="_blank">Metaflow</a> <img src="https://github.com/user-attachments/assets/30f4f382-3032-4bf7-b697-f6dbcab35fd7" height=20px /> | Tabular   | classification | <a href="https://pytorch.org/" target="_blank">Pytorch</a> <img src="https://github.com/user-attachments/assets/bfa9b38e-e9b3-41ff-8370-e64a0a0a4a93" width=20px /> / <a href="https://github.com/dreamquark-ai/tabnet/tree/develop" target="_blank">TabNet</a> | <a href="https://pytorch.org/serve/" target="_blank">TorchServe</a> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/TabNet_hp_cv_WandB" target="_blank">starter-kit</a></b> |

You can simply give it your data and it just runs. The only manual change you need to do is regarding the endpoint request &amp; serving signatures, since it is purposely hard-coded.<br />
Indeed, the <code>infra_validator</code> step is here to ensure that <u>your inference pipeline</u> (the one you're working on building a continuous-retraining automation for) keeps adhering to the schema expected by consumers of the inference endpoint. So, if you break the format of the input data, you need to create a somehow new retraining pipeline and assign it a unique name. This is to ensure that any interface disruption between the inference endpoint and its consumer(s) is intentional.

## some important markers

One of the things that make <b>retrain-pipelines</b> stand is its focus on strong MLOps fundamentals.

<details>
  <summary>model blessing</summary>
<b>retrain-pipelines</b> cares for the newly-retrained model version to be evaluated against the previous model version from that retraining pipeline. We indeed ensure that no lesser-performing model ever gets into production.<br />
Default sample pipelines each come with certain built-in evaluation criteria but, you can customize those per your own requirement. You can for instance choose to include evaluation of model performance on a particular sub-population, so as to serve as a gateway against potential incoming biases.
<hr width=60% />
</details>

<details>
  <summary>infrastructure validation</summary>
<b>retrain-pipelines</b> cares for the inference endpoint to be tested prior to deployment. We pack the preprocessing engine together with the newly retrained (and blessed) model version with the ML-server of choice and deploy it locally. We then send an inference request to that temp endpoint and check for a <code>200 http-ok</code> response with a valid payload format.
<hr width=60% />
</details>

<details>
  <summary>pipeline cards</summary>
<b>retrain-pipelines</b> is strongly opinionated around ease of quick-access to information ML-engineers care for when it comes to retraining and serving.<br />
This comes with a central place and minimal amounts of clicks to navigate efficiently.
<hr width=60% />
</details>

<details>
  <summary>Third-parties integration</summary>
TensorBoard, PyTorch Profiler, Weights and Biases. <b>retrain-pipelines</b> aims at making centrally available to ML engineers the information they care for.

<details>
  <summary>illustration with <code>WandB</code> in the <code>LightGBM_hp_cv_WandB</code> sample pipeline</summary>
  In the example of the <code>LightGBM_hp_cv_WandB</code> sample pipeline for instance, you can find information on how to view details on logging performed during the different <code>training_job</code> steps of a given run. Follow the guidance from the below video&nbsp;:

  [wandb_unedited.webm](https://github.com/user-attachments/assets/23b8f8a1-d303-4f54-bf3e-b994e4c41adf)
</details>
<hr width=60% />
</details>

<details>
  <summary>customizability</summary>
As eluded to <a href="#customizability--adaptability">above<a>, a lot of room is given to ML engineers for them to customize <b>retrain-pipelines</b> workflows.<br />
For staters, the sample pipelines are freely modifiable themselves. But, it goes far beyond that. One can go deep into customization with defaults for <code>preprocessing</code> and for <code>pipeline_card generation</code> to be fully amendable as well.

<details>
  <summary>illustration with the <code>LightGBM_hp_cv_WandB</code> sample pipeline</summary>
  Start by getting the default which you'd like to customize (any combinaison of the below 3 you'd like) :
  <ul>
    <li>reprocessing.py module</li>
    <li>pipeline_card.py module</li>
    <li>template.html html template</li>
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
  retrain_pipelines_local retraining_pipeline.py run \
    --pipeline_card_artifacts_path "." \
    --preprocess_artifacts_path "."
  ```
</details>
<hr width=60% />
</details>


## retrain-pipelines inspectors

Inspectors are convenience methods that abstract away some of the logic to get access to Metaflow pipeline-run artifacts.

[insert here example of Dask data-paralellism]

## --  DRAFT  --

    - Say you use custom "preprocessing.py", "pipeline_card.py" and/or "template.html".
      If you chose to log the run on WandB, you can retrieve the versionned artifacts there afterwards via the WandB inspector "name_here" retrain-pipelines offers.

    - incl. link to pypi here https://pypi.org/project/retrain-pipelines/

    - all is fine to track your draft pipelines as you iterate on developping them, but keeping tracks of the artifacts generated during those dry runs on the other hand has no value. To address that and all the "..." that come with it, we propose private sandboxing.
      Stateful yet ephemeral. Once your happy with a given ML retraining pipeline advancement, you're free to drop all the draft artifacts.


# launch tests
    pytest -s tests

# build from source
    cd pkg_src && python -m build --verbose
# install from source (dev mode)
    pip install -e pkg_src
# install from remote source
    pip install git+https://github.com/aurelienmorgan/retrain-pipelines.git@master#subdirectory=pkg_src
