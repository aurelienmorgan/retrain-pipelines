
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

|  | framework | modality | task | model lib | Serving |
|--|----------|----------|----------|----------|----------|
|  | <a href="https://metaflow.org/" target="_blank">Metaflow</a> <img src="https://github.com/user-attachments/assets/30f4f382-3032-4bf7-b697-f6dbcab35fd7" height=20px /> | Tabular   | regression   | <a href="https://www.dask.org/" target="_blank">Dask</a> <img src="https://github.com/user-attachments/assets/a94807e7-cc67-4415-9a9e-da1ed4755cb1" width=20px /> / <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank">LightGBM</a> <img src="https://github.com/user-attachments/assets/92ac0b53-17f8-470d-9c73-619657db42bd" width=20px />   | <a href="https://www.seldon.io/solutions/seldon-mlserver" target="_blank">ML Server</a> <img src="https://github.com/user-attachments/assets/69c57bce-cd38-4f8c-8730-e5171e842d13" width=20px /> |
|  | <a href="https://metaflow.org/" target="_blank">Metaflow</a> <img src="https://github.com/user-attachments/assets/30f4f382-3032-4bf7-b697-f6dbcab35fd7" height=20px /> | Tabular   | classification | <a href="https://pytorch.org/" target="_blank">Pytorch</a> <img src="https://github.com/user-attachments/assets/bfa9b38e-e9b3-41ff-8370-e64a0a0a4a93" width=20px /> / <a href="https://github.com/dreamquark-ai/tabnet/tree/develop" target="_blank">TabNet</a> | <a href="https://pytorch.org/serve/" target="_blank">TorchServe</a> |

You can simply give it your data and it just runs. The only manual change you need to do is regarding the endpoint request &amp; serving signatures, since it is purposely hard-coded.<br />
Indeed, the <code>infra_validator</code> step is here to ensure that <u>your inference pipeline</u> (the one you're working on building a continuous-retraining automation for) keeps adhering to the schema expected by consumers of the inference endpoint. So, if you break the format of the input data, you need to create a somehow new retraining pipeline and assign it a unique name. This is to ensure that any interface disruption between the inference endpoint and its consumer(s) is intentional.

--  DRAFT  --

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
