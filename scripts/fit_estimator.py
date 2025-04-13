import os
import argparse
from functools import partial

import joblib
import torch
from sklearn import set_config

from malearn.utils import (
    load_config,
    create_results_dir,
    save_yaml,
)
from malearn.fit_evaluate import fit_estimator, evaluate_estimator


def custom_type(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_alias", type=str, required=True)
    parser.add_argument("--estimator_alias", type=str, required=True)
    parser.add_argument("--fixed_param_names", nargs='+', type=str)
    parser.add_argument("--fixed_param_values", nargs='+', type=custom_type)
    parser.add_argument("--new_output_dir", action="store_true")
    args = parser.parse_args()

    if args.fixed_param_names and args.fixed_param_values:
        if len(args.fixed_param_values) != len(args.fixed_param_names):
            raise ValueError(
                "The number of parameter values must match the number of "
                "parameter names."
            )
        fixed_params = dict(zip(args.fixed_param_names, args.fixed_param_values))
    else:
        fixed_params = None

    config = load_config(args.config_path)

    alpha = config["estimators"][args.estimator_alias].get("alpha", None)
    if alpha is not None:
        if fixed_params is None:
            fixed_params = {"estimator__alpha": alpha}
        else:
            fixed_params["estimator__alpha"] = alpha

    if args.new_output_dir:
        results_dir, config = create_results_dir(
            config, update_config=True, suffix=args.dataset_alias
        )
        save_yaml(config, results_dir, "config")
    else:
        results_dir = os.path.join(config["base_dir"], config["results_dir"])

    get_path = partial(os.path.join, results_dir)

    set_config(enable_metadata_routing=True)

    if args.estimator_alias == "mgam":
        for setting in ["no", "ind", "aug"]:
            estimator_alias = f"mgam_{setting}"

            search = fit_estimator(
                config, args.dataset_alias, estimator_alias, fixed_params
            )

            scores = evaluate_estimator(
                config, args.dataset_alias, search.best_estimator_, estimator_alias
            )

            joblib.dump(search.best_estimator_, get_path(f"{estimator_alias}_estimator.pkl"))
            joblib.dump(search.cv_results_, get_path(f"{estimator_alias}_cv_results.pkl"))
            scores.to_csv(get_path(f"{estimator_alias}_scores.csv"), index=False)
    else:
        if torch.cuda.device_count() > 1:
            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client
            cluster = LocalCUDACluster()
            client = Client(cluster)
            search = fit_estimator(
                config, args.dataset_alias, args.estimator_alias, fixed_params
            )
            for worker in cluster.workers.values():
                process = worker.process.process
                if process.is_alive():
                    process.terminate()
            client.shutdown()
        else:
            search = fit_estimator(
                config, args.dataset_alias, args.estimator_alias, fixed_params
            )

        scores = evaluate_estimator(
            config, args.dataset_alias, search.best_estimator_, args.estimator_alias
        )

        joblib.dump(search.best_estimator_, get_path(f"{args.estimator_alias}_estimator.pkl"))
        joblib.dump(search.cv_results_, get_path(f"{args.estimator_alias}_cv_results.pkl"))
        scores.to_csv(get_path(f"{args.estimator_alias}_scores.csv"), index=False)

    set_config(enable_metadata_routing=False)
