import warnings
import copy
from functools import partial

import torch
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from skorch.callbacks import Checkpoint

from .pipeline import get_pipeline
from .data import get_data_handler
from .search import GridSearchCV, RandomizedSearchCV
from .estimators import MARFClassifier
from .scoring import CustomScorer, get_best_candidate_index
from .estimators.mgam.utils import get_mgam_data


def fit_estimator(config, dataset_alias, estimator_alias, fixed_params=None):
    config = copy.deepcopy(config)

    data_handler = get_data_handler(config, dataset_alias)

    if estimator_alias.startswith("mgam"):
        estimator_alias, setting = estimator_alias.split("_")
        X_train, y_train, M_train, _, _, _ = get_mgam_data(data_handler, setting)
    else:
        X_train, y_train, M_train, _, _, _ = data_handler.split_data()

    estimator_config = config["estimators"][estimator_alias]

    pipeline = get_pipeline(config, dataset_alias, estimator_alias)

    hparams = estimator_config["hparams"]

    if estimator_config.get("include_ccp_alphas", False):
        assert hasattr(
            pipeline.named_steps["estimator"], "cost_complexity_pruning_path"
        )
        Xt_train = pipeline.named_steps["preprocessor"].fit_transform(X_train)
        Mt_train = pipeline._update_missingness_mask(
            M_train, input_features=X_train.columns
        )
        path = pipeline.named_steps["estimator"].cost_complexity_pruning_path(
            Xt_train, y_train, Mt_train
        )
        hparams["estimator__ccp_alpha"] = path.ccp_alphas[:-1]

    if fixed_params is not None:
        # Remove fixed parameters from the hyperparameter grid.
        if isinstance(hparams, list):
            for hparam_dict in hparams:
                for fixed_param_name in fixed_params:
                    hparam_dict.pop(fixed_param_name, None)
        elif isinstance(hparams, dict):
            for fixed_param_name in fixed_params:
                hparams.pop(fixed_param_name, None)

        pipeline.set_params(**fixed_params)

    scoring = {}
    scoring_metrics = config["model_selection"]["scoring"]
    if isinstance(scoring_metrics, str):
        scoring_metrics = [scoring_metrics]
    for metric in scoring_metrics:
        if metric == "missingness_reliance_score":
            estimator = pipeline.named_steps["estimator"]
            scoring["missingness_reliance_score"] = (
                CustomScorer(estimator=estimator, metric=metric)
                .set_score_request(M=True)
            )
        else:
            scoring[metric] = metric

    n_jobs = None if (
        estimator_config.get("is_net_estimator", False)
        # Prefer parallelizing the tree building process rather than the search
        # over hyperparameters.
        or isinstance(pipeline.named_steps["estimator"], MARFClassifier)
     ) else -1

    if config["model_selection"]["refit"] == "tradeoff":
        if not len(scoring_metrics) == 2:
            raise ValueError(
                "The trade-off refit strategy requires exactly two scoring metrics."
            )
        if not "gamma" in config["model_selection"]:
            raise ValueError(
                "The trade-off refit strategy requires a gamma value to be specified."
            )
        refit = partial(
            get_best_candidate_index,
            m1=scoring_metrics[0],
            m2=scoring_metrics[1],
            gamma=config["model_selection"]["gamma"],
        )
    else:
        refit = config["model_selection"]["refit"]

    cv_splitter = data_handler.get_cv_splitter(
        n_splits=config["model_selection"]["n_splits"],
        test_size=config["model_selection"]["test_size"],
        seed=config["model_selection"]["seed"],
    )

    parameters_to_index = None
    callbacks = getattr(pipeline.named_steps["estimator"], "callbacks", None)
    if callbacks is not None:
        checkpoint = next((c for c in callbacks if isinstance(c, Checkpoint)), None)
        if checkpoint is not None:
            parameters_to_index = {
                "estimator__callbacks__Checkpoint__fn_prefix": checkpoint.fn_prefix
            }

    search_strategy = estimator_config.get("search", config["model_selection"]["search"])

    if search_strategy == "exhaustive":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=hparams,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv_splitter,
            verbose=2,
            error_score="raise",
            parameters_to_index=parameters_to_index,
        )
    elif search_strategy == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=hparams,
            n_iter=config["model_selection"]["n_iter"],
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv_splitter,
            verbose=2,
            random_state=config["model_selection"]["seed"],
            error_score="raise",
            parameters_to_index=parameters_to_index,
        )
    else:
        raise ValueError(
            f"Invalid search method: {config['model_selection']['search']}."
        )

    estimator_router = pipeline.named_steps["estimator"].get_metadata_routing()

    fit_params = {}
    if hasattr(data_handler, "group"):
        fit_params["groups"] = data_handler.load_data().loc[
            X_train.index, data_handler.group
        ]
    if (
        "missingness_reliance_score" in scoring_metrics
        or (
            "M" in estimator_router.fit._serialize()
            and estimator_config.get("use_missingness_mask", False)
        )
    ):
        fit_params["M"] = M_train
    if (
        "sample_weight" in estimator_router.fit._serialize()
        and estimator_config.get("use_sample_weights", False)
    ):
        sample_weights = compute_sample_weight("balanced", y_train)
        fit_params["sample_weight"] = sample_weights

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if (
            estimator_config.get("is_net_estimator", False)
            and torch.cuda.device_count() > 1
        ):
            from joblib import parallel_backend
            with parallel_backend("dask"):
                search.fit(X_train, y_train, **fit_params)
        else:
            search.fit(X_train, y_train, **fit_params)

    return search


def evaluate_estimator(config, dataset_alias, estimator, estimator_alias):
    data_handler = get_data_handler(config, dataset_alias)

    if estimator_alias.startswith("mgam"):
        setting = estimator_alias.split("_")[-1]
        _, _, _, X_test, y_test, M_test = get_mgam_data(data_handler, setting)
    else:
        _, _, _, X_test, y_test, M_test = data_handler.split_data()

    scores = []
    for metric in config["experiment"]["evaluation_metrics"]:
        scores.append(
            (
                metric,
                estimator.score(X_test, y_test, metric=metric),
            ),
        )

    missingness_reliance = estimator.compute_missingness_reliance(X_test, M_test)
    scores.append(("missingness_reliance", missingness_reliance))

    return pd.DataFrame(scores, columns=["metric", "score"])
