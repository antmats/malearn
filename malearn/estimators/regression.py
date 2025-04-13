import torch
import numpy as np
from xgboost import XGBRegressor

from skorch import NeuralNet
from skorch.dataset import ValidSplit
from skorch.callbacks import (
    EpochTimer,
    PrintLog,
    PassthroughScoring,
    EarlyStopping,
    Checkpoint
)

from sklearn.base import _fit_context, BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metadata_routing import _MetadataRequester

from .base import BaseMADT, RegressorMixin, MALassoMixin
from .missingness_utils import (
    check_missingness_mask,
    get_dt_missingness_reliance,
    get_ensemble_missingness_reliance,
    get_lm_missingness_reliance,
)
from .minty.minty import MintyRegressor
from .modules import NeuMissMLP, collate_fn
from ..utils import seed_torch

VARIANCE_THRESHOLD = np.finfo("double").eps

__all__ = [
    "get_regressor",
    "LassoRegression",
    "DTRegressor",
    "RFRegressor",
    "XGBoostRegressor",
    "NeuMissRegressor",
    "MALasso",
    "MADTRegressor",
]


def get_regressor(
    estimator_alias,
    checkpoint_dir_path=None,
    input_dim=None,
    output_dim=None,
    seed=None,
):
    if estimator_alias == "lasso":
        return (
            LassoRegression(random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "dt":
        return (
            DTRegressor(random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "rf":
        return (
            RFRegressor(random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "xgboost":
        return (
            XGBoostRegressor(booster="gbtree", random_state=seed)
            .set_fit_request(sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "minty":
        return (
            MintyRegressor()
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "neumiss":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            NeuMissRegressor(
                seed=seed,
                module__n_features=input_dim,
                module__output_dim=output_dim,
                module__neumiss_depth=3,
                module__mlp_depth=1,
                module__mlp_width=128,
                criterion=torch.nn.MSELoss,
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

    elif estimator_alias == "malasso":
        return (
            MALasso(random_state=seed)
            .set_fit_request(M=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    elif estimator_alias == "madt":
        return (
            MADTRegressor(random_state=seed)
            .set_fit_request(M=True, sample_weight=True)
            .set_score_request(M=True, sample_weight=True, metric=True)
        )

    else:
        raise ValueError(f"Unknown regressor: {estimator_alias}.")


# =============================================================================
# == scikit-learn regressors ==================================================
# =============================================================================

class LassoRegression(RegressorMixin, Lasso):
    """Lasso regression."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_lm_missingness_reliance(self, X, M)


class DTRegressor(RegressorMixin, DecisionTreeRegressor):
    """Decision tree regressor from scikit-learn."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_dt_missingness_reliance(self, X, M)


class RFRegressor(RegressorMixin, RandomForestRegressor):
    """Random forest regressor from scikit-learn."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)


# =============================================================================
# == XGBoost regressor ========================================================
# =============================================================================

# Ensure RegressorMixin is last in the MRO to avoid conflicts with
# XGBRegressor.get_params.
class XGBoostRegressor(XGBRegressor, RegressorMixin):
    """XGBoost regressor."""

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)

    def score(
        self,
        X,
        y,
        M=None,
        sample_weight=None,
        metric="r2",
        **predict_params,
    ):
        # This is a workaround to avoid conflicts with XGBRegressor.get_params
        # The penultimate parent class in the MRO is the custom RegressorMixin
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
# == Neural network regressor =================================================
# =============================================================================

class NNRegressor(RegressorMixin, NeuralNet, _MetadataRequester):
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

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_pred = y_pred.squeeze()
        y_true = y_true.type(y_pred.dtype)
        return super().get_loss(y_pred, y_true, X, training)

    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)

        # Callback parameters are not returned by .get_params and need a
        # special treatment.
        params_cb = self._get_params_callbacks(deep=deep)
        params.update(params_cb)

        to_exclude = {"_modules", "_criteria", "_optimizers", "_metadata_request"}
        return {key: val for key, val in params.items() if key not in to_exclude}


# =============================================================================
# == NeuMiss regressor ========================================================
# =============================================================================

class NeuMissRegressor(NNRegressor):
    def __init__(self, **kwargs):
        kwargs.pop("module", None)
        super().__init__(module=NeuMissMLP, **kwargs)

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return np.max(M, axis=1).mean()


# =============================================================================
# == Missingness-avoiding Lasso ===============================================
# =============================================================================

class MALasso(MALassoMixin, RegressorMixin, Lasso):
    def __init__(self, alpha=1.0, beta=1.0, random_state=None):
        super().__init__(alpha=alpha, random_state=random_state)
        self.beta = beta

    def fit(self, X, y, M=None):
        if M is not None:
            M = check_missingness_mask(M, X)
            X = self._transform_input(X, M)
        return super().fit(X, y)


# =============================================================================
# == Missingness-avoiding decision tree regressor =============================
# =============================================================================

class MADTRegressor(RegressorMixin, BaseMADT):
    """Missingness-avoiding decision tree regressor."""

    criterion = "variance_reduction"

    def __init__(
        self,
        max_depth=3,
        random_state=None,
        alpha=1.0,
        compute_rho_per_node=False,
        ccp_alpha=0.0,
    ):
        super().__init__(
            max_depth, random_state, alpha, compute_rho_per_node, ccp_alpha
        )

    def _get_node_value(self, y, sample_weight):
        return np.average(y, weights=sample_weight)

    def _get_node_impurity(self, node_value, y, sample_weight):
        return np.var(y * sample_weight)

    def _is_homogeneous(self, node_value, y, sample_weight):
        return np.var(y * sample_weight) < VARIANCE_THRESHOLD

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None):
        return super()._fit(X, y, M, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        return self.predict(X, check_input)

    def _best_split(self, X, y, feature, sample_weight, M=None):
        sorted_indices = np.argsort(X[:, feature])
        X_sorted = X[sorted_indices, feature]
        y_sorted = y[sorted_indices]
        sample_weight_sorted = sample_weight[sorted_indices]

        y_sum_low = 0
        y2_sum_low = 0

        y_sum_high = np.sum(y_sorted * sample_weight_sorted)
        y2_sum_high = np.sum(y_sorted**2 * sample_weight_sorted)

        n_low = 0
        n_high = np.sum(sample_weight)

        max_score = -np.inf
        i_max_score = None

        n = len(y)
        for i in range(n - 1):
            yi = y_sorted[i]
            wi = sample_weight_sorted[i]

            n_low += wi
            y_sum_low += yi * wi
            y2_sum_low += yi**2 * wi

            n_high -= wi
            y_sum_high -= yi * wi
            y2_sum_high -= yi**2 * wi

            if n_low == 0:
                continue

            if n_high == 0:
                break

            # If the input is equal to the input at the next position, we will
            # not consider a split here.
            if np.isclose(X_sorted[i], X_sorted[i+1]):
                continue

            V_low = y2_sum_low/n_low - y_sum_low**2/n_low**2
            V_high = y2_sum_high/n_high - y_sum_high**2/n_high**2

            # Compute the variance reduction.
            score = -(n_high*V_high + n_low*V_low)

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
