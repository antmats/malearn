"""
This module contains custom search classes that enhance the functionality of
scikit-learn's GridSearchCV and RandomizedSearchCV. Specifically, these classes
accept a parameters_to_index dictionary, which defines parameters whose values
should be updated with the candidate's index during parallel search. This
feature is useful when the estimator saves parameters to a file during fitting,
as parameters_to_index ensures that the saved files are uniquely named for each
candidate.
"""

import time
from collections import defaultdict
from abc import abstractmethod
from itertools import product

import numpy as np

from sklearn.base import _fit_context, clone, is_classifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._validation import (
    _fit_and_score,
    _warn_or_raise_about_fit_failures,
    _insert_error_scores,
)
from sklearn.model_selection._split import check_cv
from sklearn.utils.validation import _check_method_params, indexable
from sklearn.utils.parallel import Parallel, delayed
from sklearn.metrics._scorer import _MultimetricScorer

from .pipeline import CustomPipeline


class CustomBaseSearchCV(BaseSearchCV):
    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=True,
        parameters_to_index=None,
    ):
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.parameters_to_index=parameters_to_index
 
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        estimator = self.estimator
        scorers, refit_metric = self._get_scorers()

        X, y = indexable(X, y)
        params = _check_method_params(X, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.estimator.fit,
            score_params=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits.".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=(
                            parameters
                            | (
                                {
                                    k: f"{cand_idx+1:02d}_{split_idx+1:02d}_{v}"
                                    for k, v in self.parameters_to_index.items()
                                }
                                if self.parameters_to_index is not None else {}
                            )
                        ),
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params),
                        enumerate(cv.split(X, y, **routed_params.splitter.split)),
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned inconsistent "
                        "results. Expected {} splits, got {}.".format(
                            n_splits, len(out) // n_candidates
                        )
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **clone(self.best_params_, safe=False)
            )

            # If the estimator is an MA tree pipeline, set
            # `compute_rho_per_node=True` to compute the missingness reliance
            # at each node.
            if (
                isinstance(self.best_estimator_, CustomPipeline)
                and hasattr(self.best_estimator_[-1], "compute_rho_per_node")
            ):
                self.best_estimator_[-1].compute_rho_per_node = True

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class GridSearchCV(CustomBaseSearchCV):
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        parameters_to_index=None,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
            parameters_to_index=parameters_to_index,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(ParameterGrid(self.param_grid))


class RandomizedSearchCV(CustomBaseSearchCV):
    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
        parameters_to_index=None,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
            parameters_to_index=parameters_to_index,
        )

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions,
                self.n_iter,
                random_state=self.random_state,
            )
        )
