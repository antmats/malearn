from os.path import join

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, _final_estimator_has
from sklearn.base import clone, _fit_context
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metaestimators import available_if
from sklearn.utils.metadata_routing import _routing_enabled, process_routing

from .data import get_data_handler
from .estimators import get_classifier, get_regressor


def get_pipeline(config, dataset_alias, estimator_alias):
    data_handler = get_data_handler(config, dataset_alias)

    if estimator_alias.startswith("mgam"):
        preprocessor = None
    else:
        preprocessor = data_handler.get_preprocessor(estimator_alias)

    kwargs = {"seed": config["experiment"]["seed"]}
    if config["estimators"][estimator_alias].get("is_net_estimator", False):
        kwargs["checkpoint_dir_path"] = join(
            config["base_dir"], config["results_dir"]
        )
        X_train, y_train, _M_train, _, _, _ = data_handler.split_data()
        input_dim = clone(preprocessor).fit_transform(X_train).shape[1]
        output_dim = (
            len(set(y_train))
            if config["experiment"]["task_type"] == "classification" else 1
        )
        kwargs["input_dim"] = input_dim
        kwargs["output_dim"] = output_dim

    if config["experiment"]["task_type"] == "classification":
        estimator = get_classifier(estimator_alias, **kwargs)
    else:
        estimator = get_regressor(estimator_alias, **kwargs)

    if config["cache_dir"] is not None:
        memory = join(config["base_dir"], config["cache_dir"])
    else:
        memory = None
    pipeline = CustomPipeline(
        [
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ],
        memory=memory,
        verbose=True,
    )
    pipeline.set_score_request(sample_weight=False)
    return pipeline


class CustomPipeline(Pipeline):
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("The input data must be a pandas DataFrame.")
        routed_params = self._check_method_params(method="fit", props=params)
        Xt = self._fit(X, y, routed_params)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_fit_params = routed_params[self.steps[-1][0]]["fit"]
                M = last_step_fit_params.get("M", None)
                if M is not None and self.named_steps["preprocessor"] is not None:
                    last_step_fit_params["M"] = self._update_missingness_mask(
                        M, input_features=X.columns
                    )
                self._final_estimator.fit(Xt, y, **last_step_fit_params)
        return self

    def _update_missingness_mask(self, M, input_features):
        preprocessor = self.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out(input_features)
        feature_names = [f.split("__")[-1] for f in feature_names]
        feature_names = [f.removesuffix("_infrequent_sklearn") for f in feature_names]
        for i, f in enumerate(feature_names):
            f = f.removesuffix("_" + f.split("_")[-1])
            if f in M.columns:
                feature_names[i] = f
        M = [M[f].to_numpy() for f in feature_names]
        return np.array(M).T

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None, **params):
        Xt = X
        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt)
            score_params = {}
            if sample_weight is not None:
                score_params["sample_weight"] = sample_weight
            return self.steps[-1][1].score(Xt, y, **score_params)

        # Metadata routing is enabled.
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )

        last_step_score_params = routed_params[self.steps[-1][0]]["score"]
        M = last_step_score_params.get("M", None)
        if M is not None and self.named_steps["preprocessor"] is not None:
            last_step_score_params["M"] = self._update_missingness_mask(
                M, input_features=X.columns
            )

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt, **routed_params[name]["transform"])
        return self.steps[-1][1].score(Xt, y, **last_step_score_params)

    def compute_missingness_reliance(self, X, M):
        preprocessor, estimator = self.named_steps.values()
        if not hasattr(estimator, "compute_missingness_reliance"):
            raise NotImplementedError(
                "The estimator does not implement the "
                "compute_missingness_reliance method."
            )
        if preprocessor is not None:
            Xt = preprocessor.transform(X)
            M = self._update_missingness_mask(M, input_features=X.columns)
        else:
            Xt = X
        return estimator.compute_missingness_reliance(Xt, M)
