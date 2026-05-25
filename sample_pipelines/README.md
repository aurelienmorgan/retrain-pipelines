
# example workflows
<b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/dag_engine" target="_blank">starter-kit</a></b>
  - `example_wf` - Simple sub-DAGing with context.
  - `example_wf_1` - 1-level deep nested sub-DAGing.
  - `example_wf_2` - Simple taskgroup.
  - `example_wf_3` - 2 taskgroups in series.
  - `example_wf_4` - 1-level deep nested taskgroups.
  - `example_wf_5` - 2-levels deep nested taskgroups.
  - `example_wf_6` - 3 taskgroups in series.
  - `example_wf_7` - Taskgroup inside a sub-DAG.
  - `example_wf_11` - 1-level deep nested sub-DAGing with an inner taskgroup & inner inline tasks.
  - `example_wf_111` - 2-levels deep nested sub-DAGing.
  - `example_wf_1111` - Sub-DAGs in series, inside sub-DAG branches.

&nbsp;

# industry use-cases
| modality | task | model lib | Serving |   |
|----------|----------|----------|----------|--|
| text, NLP | function&nbsp;calling | <a href="https://unsloth.ai/" target="_blank">Unsloth</a> <img src="https://github.com/user-attachments/assets/3bb9244b-8c89-41fa-8b38-c4862763eea1" width=20px /> / <a href="https://github.com/dreamquark-ai/tabnet/tree/develop" target="_blank">Qwen&nbsp;2.5</a>&nbsp;<img src="https://github.com/user-attachments/assets/3067f88e-3064-470f-9c8e-2d80c40b3d5c" width=20px /> | <a href="https://lightning.ai/docs/litserve/home/" target="_blank">LitServe</a> <img src="https://github.com/user-attachments/assets/b5abcd66-9cb4-420c-ad2c-29bafb0f3b62" width=20px /> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/Unsloth_Qwen_FuncCall" target="_blank">starter-kit</a></b> |
| Tabular | regression | <a href="https://www.dask.org/" target="_blank">Dask</a> <img src="https://github.com/user-attachments/assets/a94807e7-cc67-4415-9a9e-da1ed4755cb1" width=20px /> / <a href="https://lightgbm.readthedocs.io/en/stable/" target="_blank">LightGBM</a> <img src="https://github.com/user-attachments/assets/92ac0b53-17f8-470d-9c73-619657db42bd" width=20px /> | <a href="https://www.seldon.io/solutions/seldon-mlserver" target="_blank">ML Server</a> <img src="https://github.com/user-attachments/assets/69c57bce-cd38-4f8c-8730-e5171e842d13" width=20px /> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/LightGBM_hp_cv_WandB" target="_blank">starter-kit</a></b> |
| Tabular | classification | <a href="https://pytorch.org/" target="_blank">Pytorch</a> <img src="https://github.com/user-attachments/assets/bfa9b38e-e9b3-41ff-8370-e64a0a0a4a93" width=20px /> / <a href="https://github.com/dreamquark-ai/tabnet/tree/develop" target="_blank">TabNet</a> | <a href="https://pytorch.org/serve/" target="_blank">TorchServe</a>&nbsp;<img src="https://github.com/user-attachments/assets/bfa9b38e-e9b3-41ff-8370-e64a0a0a4a93" width=20px /> | <b><a href="https://github.com/aurelienmorgan/retrain-pipelines/tree/master/sample_pipelines/TabNet_hp_cv_WandB" target="_blank">starter-kit</a></b> |

&nbsp;

