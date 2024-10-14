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


--  DRAFT  --

    - Say you use custom "preprocessing.py", "pipeline_card.py" and/or "template.html".
      If you chose to log the run on WandB, you can retrieve the versionned artifacts there afterwards via the WandB inspector "name_here" retrain-pipelines offers.

    - incl. link to pypi here https://pypi.org/project/retrain-pipelines/

    - all is fine to track your draft pipelines as you iterate on developping them, but keeping tracks of the artifacts generated during those dry runs on the other hand has no value. To address that and all the "..." that come with it, we propose sandboxing.
      Stateful yet ephemeral. Once your happy with a given ML retraining pipeline advancement, you're free to drop all the draft artifacts.


# launch tests
    pytest -s tests

# build from source
    cd pkg_src && python -m build --verbose
# install from source (dev mode) via :
    pip install -e pkg_src
