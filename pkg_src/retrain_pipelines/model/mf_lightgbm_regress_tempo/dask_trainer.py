
import pandas as pd
import numpy as np

import lightgbm as lgb
import dask.dataframe as dd
import dask.array as da
import dask

from metaflow import current

class MetricsCallback:
    def __init__(self, queue_name):
        """
        Class that logs evaluation metrics per epoch
        for all Dask workers involved
        in a given model training.
        """
        self.queue = dask.distributed.Queue(queue_name)

    def __call__(self, env: lgb.callback.CallbackEnv):
        # env.iteration
        history = {
            dask.distributed.get_worker().name: {
                f"{dataset}_{metric}": [value]
                for dataset, metric, value, _ in env.evaluation_result_list
            }
        }
        self.queue.put(history)

def dask_regressor_fit(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    npartitions: int,
    hp_dict: dict
) -> (lgb.LGBMRegressor, dict, dict):
    """
    Fits a lightgbm regressor
    relying on Dask multi-threading.

    Params:
        - X_train (pd.DataFrame):
        - y_train (np.ndarray):
        - X_val (pd.DataFrame):
        - y_val (np.ndarray):
        - npartitions (int):
        - hp_dict (dict):
            model hyperparameters
            to be passed through to
            the regressor's init argument.
    Results:
        - lgb.LGBMRegressor:
            the fitted regressor
        - dict:
            the aggregated training history
            of the fitted regressor
        - dict:
            a dictionnary of history
            from all workers involved
            in the training.
    """

    ############################
    # train set to Dask format #
    ############################
    dX_train = dd.from_pandas(
        X_train, npartitions=npartitions)
    # for i in range(npartitions):
    #     partition_size = len(dX_train.get_partition(i))
    #     print(f"Partition {i}: {partition_size} rows")
    dy_train = dd.from_pandas(
        y_train, npartitions=npartitions)
    ############################

    ############################
    # valid set to Dask format #
    ############################
    val_partition_size = \
        ((len(X_val)+npartitions+1) //npartitions)
    dX_val = dd.from_pandas(X_val, npartitions=npartitions)
    # for i in range(npartitions):
    #     partition_size = len(dX_val.get_partition(i))
    #     print(f"Partition {i}: {partition_size} rows")
    dy_val = dd.from_pandas(
        y_val, npartitions=npartitions)
    ############################


    # eval_set = [(dX_val, dy_val)]
    # Add eval_set and eval_names to track evaluation results
    eval_set = [(dX_train, dy_train), (dX_val, dy_val)]
    eval_names = ['Training', 'Validation']

    dask_distrib_client = dask.distributed.Client()

    lgb_reg = lgb.DaskLGBMRegressor(
                    # max_depth=5,
                    # min_child_samples=\
                        # min(250, len(y_val)),
                    # lambda_l1=0.1,
                    # lambda_l2=0.1,
                    # bagging_fraction=0.8,
                    # feature_fraction=0.8,
                    **hp_dict
    )

    # Set up a Dask queue to collect metrics from workers
    queue_name = f"lgb_history_queue_{current.task_id}"
    metrics_queue = dask.distributed.Queue(queue_name)

    lgb_history = MetricsCallback(queue_name=queue_name)

    def sk_rmse(y_true, y_pred):
        """
        For debugging purpose.
        Distributed RMSE vs. "to_local" model RMSE.

        @see lightgbm.readthedocs.io/en/latest/pythonapi/
             lightgbm.DaskLGBMRegressor.html
        """
        from sklearn.metrics import mean_squared_error

        eval_result = mean_squared_error(y_true, y_pred)
        return ("sk_rmse", eval_result, False)

    # Fit the model
    lgb_reg.fit(
        dX_train, dy_train,
        eval_set=eval_set, eval_names=eval_names,
        eval_metric=["rmse", sk_rmse],
        callbacks=[lgb_history]
    )

    # Collect the metrics from the queue after training
    workers_history = {}
    for _ in range(metrics_queue.qsize()):
        history = metrics_queue.get()
        for worker_name, metrics in history.items():
            if worker_name not in workers_history:
                workers_history[worker_name] = {}
            for metric, values in metrics.items():
                if metric not in workers_history[worker_name]:
                    workers_history[worker_name][metric] = []
                workers_history[worker_name][metric].extend(values)
    metrics_queue.close()

    print(f"distributed best_score_ : "+
          f"{lgb_reg.best_score_['Validation']['rmse']}")

    return lgb_reg.to_local(), \
           lgb_reg.evals_result_, \
           workers_history

