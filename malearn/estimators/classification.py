from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import torch
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import (
    EpochTimer,
    PrintLog,
    PassthroughScoring,
    EarlyStopping,
    Checkpoint
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.ensemble._gb import (
    BaseGradientBoosting,
    set_huber_delta,
    _update_terminal_regions,
)
from sklearn.ensemble._forest import (
    ForestClassifier,
    _get_n_samples_bootstrap,
    _parallel_build_trees,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree._tree import DTYPE
from sklearn.base import BaseEstimator, _fit_context
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils.metadata_routing import _MetadataRequester
from sklearn.utils._param_validation import Interval, StrOptions, HasMethods
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
    _check_sample_weight,
    check_X_y,
    check_array,
)
from sklearn._loss.loss import HuberLoss, AbsoluteError, PinballLoss

from .base import ClassifierMixin, BaseMADT, MALassoMixin
from .missingness_utils import (
    check_missingness_mask,
    patch_missingness_mask,
    get_dt_missingness_reliance,
    get_ensemble_missingness_reliance,
    get_lm_missingness_reliance,
)
from .regression import MADTRegressor
from .modules import NeuMissMLP, collate_fn
from .criterion import info_gain_scorer, gini_scorer, entropy, gini_impurity
from ..utils import seed_torch
from .minty.minty import MintyClassifier

CRITERIA_CLF = {"info_gain": info_gain_scorer, "gini": gini_scorer}

__all__ = [
    "get_classifier",
    "LRClassifier",
    "DTClassifier",
    "RFClassifier",
    "MGAMClassifier",
    "XGBoostClassifier",
    "NeuMissClassifier",
    "MALassoClassifier",
    "MADTClassifier",
    "MARFClassifier",
    "MAGBTClassifier",
]


def get_classifier(
    estimator_alias,
    checkpoint_dir_path=None,
    input_dim=None,
    output_dim=None,
    seed=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if estimator_alias == "lasso":
        return (
            LRClassifier(penalty="l1", solver="liblinear", random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "dt":
        return (
            DTClassifier(random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "rf":
        return (
            RFClassifier(random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias.startswith("mgam"):
        return (
            MGAMClassifier()
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "xgboost":
        return (
            XGBoostClassifier(booster="gbtree", random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "neumiss":
        return (
            NeuMissClassifier(
                seed=seed,
                module__n_features=input_dim,
                module__output_dim=output_dim,
                module__neumiss_depth=3,
                module__mlp_depth=1,
                module__mlp_width=128,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                optimizer__lr=0.001,
                max_epochs=100,
                batch_size=32,
                iterator_train__shuffle=True,
                iterator_train__collate_fn=collate_fn,
                iterator_valid__collate_fn=collate_fn,
                # TODO: We should use GroupKFold when there are groups in the data.
                train_split=ValidSplit(5),
                callbacks=[
                    EarlyStopping(patience=10),
                    Checkpoint(
                        f_optimizer=None,
                        f_criterion=None,
                        dirname=checkpoint_dir_path,
                        load_best=True,
                    ),
                ],
                verbose=1,
                device=device,
            )
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "minty":
        return (
            MintyClassifier()
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "malasso":
        return (
            MALassoClassifier(random_state=seed)
            .set_fit_request(M=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "madt":
        return (
            MADTClassifier(random_state=seed)
            .set_fit_request(M=True, sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "marf":
        return (
            MARFClassifier(n_jobs=-1, verbose=2, random_state=seed)
            .set_fit_request(M=True, sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "magbt":
        return (
            MAGBTClassifier(verbose=2, random_state=seed)
            .set_fit_request(M=True, sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    else:
        raise ValueError(f"Unknown estimator: {estimator_alias}.")


# =============================================================================
# == scikit-learn classifiers =================================================
# =============================================================================

class LRClassifier(ClassifierMixin, LogisticRegression):
    """Logistic regression classifier from scikit-learn."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_lm_missingness_reliance(self, X, M)


class DTClassifier(ClassifierMixin, DecisionTreeClassifier):
    """Decision tree classifier from scikit-learn."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_dt_missingness_reliance(self, X, M)


class RFClassifier(ClassifierMixin, RandomForestClassifier):
    """Random forest classifier from scikit-learn."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)


# =============================================================================
# == M-GAM classifier =========================================================
# =============================================================================

class MGAMClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, lambda_0=1.0, max_support_size=100):
        self.lambda_0 = lambda_0
        self.max_support_size = max_support_size

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        labels = unique_labels(y)
        if not len(labels) == 2:
            raise ValueError("M-GAM supports only binary classification.")
        try:
            import fastsparsegams
        except ImportError:
            raise ModuleNotFoundError("fastsparsegams is not installed.")
        else:
            self.model_ = fastsparsegams.fit(
                X,
                y,
                lambda_grid=[[self.lambda_0]],
                loss="Exponential",
                algorithm="CDPSI",
                num_lambda=None,
                num_gamma=None,
                max_support_size=self.max_support_size,
            )
        self.classes_ = labels
        self.n_classes_ = len(labels)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        yp = self.model_.predict(X).ravel()
        return np.column_stack((1 - yp, yp))

    def predict(self, X):
        yp = self.predict_proba(X)
        return self.classes_[np.argmax(yp, axis=1)]

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            raise ValueError("`X` ust be a pandas DataFrame.")
        if not isinstance(M, pd.DataFrame):
            raise ValueError("`M` ust be a pandas DataFrame.")

        indices = [i for i, c in enumerate(X.columns) if c in M.columns]
        beta = self.model_.coeff(include_intercept=False)
        assert beta.shape[1] == 1, "The model must contain a single solution."
        beta = beta.toarray().ravel()[indices]

        return np.max(M * (beta != 0).astype(int), axis=1).mean()


# =============================================================================
# == XGBoost classifier =======================================================
# =============================================================================

# Ensure ClassifierMixin is last in the MRO to avoid conflicts with
# XGBClassifier.get_params.
class XGBoostClassifier(XGBClassifier, ClassifierMixin):
    """XGBoost classifier."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)

    def score(
        self,
        X,
        y,
        M=None,
        sample_weight=None,
        metric="accuracy",
        **predict_params,
    ):
        # This is a workaround to avoid conflicts with XGBClassifier.get_params
        # The penultimate parent class in the MRO is the custom ClassifierMixin
        # class, which implements the score method that we want to use.
        return self.__class__.__mro__[-2].score(
            self,
            X,
            y,
            M=M,
            sample_weight=sample_weight,
            metric=metric,
            **predict_params,
        )


# =============================================================================
# == Neural network classifier ================================================
# =============================================================================

class NNClassifier(ClassifierMixin, NeuralNetClassifier, _MetadataRequester):
    def __init__(self, *, seed=None, **kwargs):
        super().__init__(**kwargs)
        if seed is not None:
            seed_torch(seed)

    @property
    def _default_callbacks(self):
        return [
            ("epoch_timer", EpochTimer()),
            ("train_loss", PassthroughScoring(name="train_loss", on_train=True)),
            ("valid_loss", PassthroughScoring(name="valid_loss")),
            ("print_log", PrintLog())
        ]

    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)

        # Callback parameters are not returned by .get_params and need a
        # special treatment.
        params_cb = self._get_params_callbacks(deep=deep)
        params.update(params_cb)

        to_exclude = {"_modules", "_criteria", "_optimizers", "_metadata_request"}
        return {key: val for key, val in params.items() if key not in to_exclude}


# =============================================================================
# == NeuMiss classifier =======================================================
# =============================================================================

class NeuMissClassifier(NNClassifier):
    def __init__(self, **kwargs):
        kwargs.pop("module", None)
        super().__init__(module=NeuMissMLP, **kwargs)

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return np.max(M, axis=1).mean()


# =============================================================================
# == Missingness-avoiding Lasso classifier ====================================
# =============================================================================

class MALassoClassifier(MALassoMixin, ClassifierMixin, LogisticRegression):
    def __init__(self, alpha=1.0, beta=1.0, random_state=None, solver="liblinear"):
        C = 1 / (2 * alpha)
        super().__init__(
            penalty="l1", C=C, random_state=random_state, solver=solver
        )
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y, M=None):
        if M is not None:
            M = check_missingness_mask(M, X)
            X = self._transform_input(X, M)
        return super().fit(X, y)

    def set_params(self, **params):
        super().set_params(**params)
        self.C = 1 / (2 * self.alpha)


# =============================================================================
# == Missingness-avoiding decision tree classifier ============================
# =============================================================================

class MADTClassifier(ClassifierMixin, BaseMADT):
    """Missingness-avoiding decision tree classifier."""

    _parameter_constraints: dict = {
        **BaseMADT._parameter_constraints,
        "criterion": [StrOptions({"info_gain", "gini"})],
    }

    def __init__(
        self,
        criterion="gini",
        max_depth=3,
        random_state=None,
        alpha=1.0,
        compute_rho_per_node=False,
        ccp_alpha=0.0,
    ):
        super().__init__(
            max_depth, random_state, alpha, compute_rho_per_node, ccp_alpha
        )
        self.criterion = criterion

    def _get_node_value(self, y, sample_weight):
        node_value = np.zeros(self.n_classes_, dtype=float)
        for class_label, weight in zip(y, sample_weight):
            node_value[class_label] += weight
        node_value /= sum(sample_weight)
        return node_value

    def _get_node_impurity(self, node_value, y, sample_weight):
        return (
            gini_impurity(node_value)
            if self.criterion == "gini" else entropy(node_value)
        )

    def _is_homogeneous(self, node_value, y, sample_weight):
        return max(node_value) == 1.0

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None):
        return super()._fit(X, y, M, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        return self.predict(X, check_input, return_proba=True)

    def _best_split(self, X, y, feature, sample_weight, M=None):
        sorted_indices = np.argsort(X[:, feature])
        X_sorted = X[sorted_indices, feature]
        y_sorted = y[sorted_indices]
        sample_weight_sorted = sample_weight[sorted_indices]

        low_distr = np.zeros(self.n_classes_, dtype=float)
        high_distr = np.zeros(self.n_classes_, dtype=float)

        for class_label, weight in zip(y_sorted, sample_weight_sorted):
            high_distr[class_label] += weight

        n_low = 0
        n_high = np.sum(sample_weight)

        max_score = -np.inf
        i_max_score = None

        n = len(y)
        for i in range(n - 1):
            yi = y_sorted[i]
            wi = sample_weight_sorted[i]

            low_distr[yi] += wi
            high_distr[yi] -= wi

            n_low += wi
            n_high -= wi

            if n_low == 0:
                continue

            if n_high == 0:
                break

            # If the input is equal to the input at the next position, we will
            # not consider a split here.
            if np.isclose(X_sorted[i], X_sorted[i+1]):
                continue

            criterion_function = CRITERIA_CLF[self.criterion]
            score = criterion_function(n_low, low_distr, n_high, high_distr)

            if score > max_score:
                max_score = score
                i_max_score = i

        if i_max_score is None:
            return -np.inf, None, None

        if M is not None:
            max_score -= self.alpha * np.mean(sample_weight * M[:, feature])

        split_threshold = (
            0.5 * (X_sorted[i_max_score] + X_sorted[i_max_score + 1])
        )
        return max_score, feature, split_threshold


# =============================================================================
# == Missingness-avoiding random forest classifier ============================
# =============================================================================

class MARFClassifier(ClassifierMixin, ForestClassifier):
    """Missingness-avoiding random forest classifier."""

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **MADTClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=10,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        class_weight=None,
        max_samples=None,
        alpha=1.0,
        compute_rho_per_node=False,
    ):
        super().__init__(
            estimator=MADTClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion", "max_depth", "alpha", "compute_rho_per_node"
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=False,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.alpha = alpha
        self.compute_rho_per_node = compute_rho_per_node

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, M=None, sample_weight=None):
        X, y = self._validate_data(X, y, dtype=DTYPE)

        if M is not None:
            missing_values_in_feature_mask = check_missingness_mask(M, X)
        else:
            missing_values_in_feature_mask = None

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_samples` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set `max_samples=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError(
                "Out-of-bag estimation is only available if `bootstrap=True`."
            )

        random_state = check_random_state(self.random_state)

        trees = [
            self._make_estimator(append=False, random_state=random_state)
            for _ in range(self.n_estimators)
        ]

        self.estimators_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(
            delayed(_parallel_build_trees)(
                t,
                self.bootstrap,
                X,
                y,
                sample_weight,
                i,
                len(trees),
                verbose=self.verbose,
                class_weight=self.class_weight,
                n_samples_bootstrap=n_samples_bootstrap,
                missing_values_in_feature_mask=missing_values_in_feature_mask,
            )
            for i, t in enumerate(trees)
        )

        if self.oob_score and not hasattr(self, "oob_score_"):
            if callable(self.oob_score):
                self._set_oob_score_and_attributes(
                    X, y, scoring_function=self.oob_score
                )
            else:
                self._set_oob_score_and_attributes(X, y)

        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)


# =============================================================================
# == Missingness-avoiding gradient boosting classifier ========================
# =============================================================================

def predict_stage(estimators, stage, X, scale, out):
    """Python version of scikit-learn's Cython function with the same name."""
    return predict_stages(
        estimators=estimators[stage:stage + 1], X=X, scale=scale, out=out
    )


def predict_stages(estimators, X, scale, out):
    """Python version of scikit-learn's Cython function with the same name."""
    n_estimators = len(estimators)
    K = len(estimators[0])
    for i in range(n_estimators):
        for k in range(K):
            tree = estimators[i][k].tree_
            leaf_indices = tree.apply(X)
            # The value of a node is a 2D array of shape (n_outputs, n_classes).
            # We use this function for regression with n_outputs=1, so it is
            # safe to reshape the values to an array of shape (n_samples,).
            out[:, k] += scale * tree.value[leaf_indices].ravel()


# We define the base class here instead of in `base.py` to avoid circular imports.
class BaseMAGBT(BaseGradientBoosting, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        **MADTRegressor._parameter_constraints,
        "learning_rate": [Interval(Real, 0.0, None, closed="left")],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right")],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "validation_fraction": [Interval(Real, 0.0, 1.0, closed="neither")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "alpha": [Interval(Real, 0.0, None, closed="neither")],
    }

    @abstractmethod
    def __init__(
        self,
        *,
        loss,
        learning_rate,
        n_estimators,
        subsample,
        max_depth,
        max_features,
        init,
        random_state,
        quantile=0.9,  # Rename `alpha` to `quantile`
        verbose=0,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        alpha=1.0,
        compute_rho_per_node=False,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.max_features = max_features
        self.init = init
        self.random_state = random_state
        self.quantile = quantile
        self.verbose = verbose
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.alpha = alpha
        self.compute_rho_per_node = compute_rho_per_node

    def _init_state(self):
        self.init_ = self.init
        if self.init_ is None:
            if is_classifier(self):
                self.init_ = DummyClassifier(strategy="prior")
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                self.init_ = DummyRegressor(strategy="quantile", quantile=0.5)
            elif isinstance(self._loss, PinballLoss):
                self.init_ = DummyRegressor(strategy="quantile", quantile=self.quantile)
            else:
                self.init_ = DummyRegressor(strategy="mean")

        self.estimators_ = np.empty(
            (self.n_estimators, self.n_trees_per_iteration_), dtype=object
        )
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_scores_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_score_ = np.nan

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions

    def _staged_raw_predict(self, X, check_input=True):
        """Compute raw predictions of ``X`` for each iteration."""
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, order="C", reset=False)
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(self.estimators_, i, X, self.learning_rate, raw_predictions)
            yield raw_predictions.copy()

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        # Keeping the following arguments for compatibility with the base
        # class; sparse matrices are not supported.
        X_csc=None,
        X_csr=None,
    ):
        """Fit another stage of `n_trees_per_iteration_` trees."""

        original_y = y

        if isinstance(self._loss, HuberLoss):
            set_huber_delta(
                loss=self._loss,
                y_true=y,
                raw_prediction=raw_predictions,
                sample_weight=sample_weight,
            )

        neg_gradient = -self._loss.gradient(
            y_true=y,
            raw_prediction=raw_predictions,
            sample_weight=None,  # We pass sample_weights to the tree directly
        )
        # Create a 2D view of shape (n_samples, n_trees_per_iteration_) or
        # (n_samples, 1) of the neg_gradient to simplify the loop over
        # self.n_trees_per_iteration_.
        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient

        M = getattr(self, "_missingness_mask", None)

        for k in range(self.n_trees_per_iteration_):
            if self._loss.is_multiclass:
                y = np.array(original_y == k, dtype=np.float64)

            if M is not None:
                estimators = self.estimators_.ravel()
                last_estimator = next(
                    (x for x in np.flip(estimators) if x is not None),
                    None
                )
                if last_estimator is not None:
                    miss_reliance = (
                        last_estimator.compute_missingness_reliance(
                            X, M, sample_mask=sample_mask, reduce=False
                        )
                    )
                    # If we already rely on a feature with missing values, we
                    # should not penalize new trees for using it.
                    M = (M - miss_reliance).clip(min=0)

            # Induce regression tree on the negative gradient.
            tree = MADTRegressor(
                max_depth=self.max_depth,
                random_state=random_state,
                alpha=self.alpha,
                compute_rho_per_node=self.compute_rho_per_node,
            )

            if self.subsample < 1.0:
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            tree._fit(
                X,
                neg_g_view[:, k],
                M=M,
                sample_weight=sample_weight,
                check_input=False,
            )

            # Update tree leaves.
            _update_terminal_regions(
                self._loss,
                tree.tree_,
                X,
                y,
                neg_g_view[:, k],
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=self.learning_rate,
                k=k,
            )

            self.estimators_[i, k] = tree

        self._missingness_mask = M

        return raw_predictions


class MAGBTClassifier(ClassifierMixin, BaseMAGBT, GradientBoostingClassifier):
    """Missingness-avoiding gradient boosting classifier."""

    _parameter_constraints: dict = {
        **BaseGradientBoosting._parameter_constraints,
        "loss": [StrOptions({"log_loss", "exponential"})],
        "init": [StrOptions({"zero"}), None, HasMethods(["fit", "predict_proba"])],
    }

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        max_depth=3,
        init=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        alpha=1.0,
        compute_rho_per_node=False,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=None,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            alpha=alpha,
            compute_rho_per_node=compute_rho_per_node,
        )

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None, monitor=None):
        if M is not None and self.n_iter_no_change is not None:
            M, _ = train_test_split(
                M,
                random_state=self.random_state,
                test_size=self.validation_fraction,
                stratify=y,
            )
        with patch_missingness_mask(self, M):
            return super().fit(
                X, y, sample_weight=sample_weight, monitor=monitor
            )

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)
