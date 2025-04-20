import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils.validation import check_is_fitted, _check_feature_names_in
from sklearn.pipeline import _transform_one
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._missing import is_scalar_nan


def _custom_transform_one(transformer, X, y, weight, params=None):
    """
    Custom implementation of `_transform_one` (see sklearn/pipeline.py) that
    selects parameters for the transformer's `transform` method via the key
    "transform", instead of the transform attribute, as the latter is
    incompatible with the Dask joblib backend.
    """
    res = transformer.transform(X, **params["transform"])
    if weight is None:
        return res
    return res * weight


class CustomColumnTransformer(ColumnTransformer):
    def _call_func_on_transformers(self, X, y, func, column_as_labels, routed_params):
        if func is _transform_one:
            func = _custom_transform_one
        return super()._call_func_on_transformers(
            X, y, func, column_as_labels, routed_params
        )


class MissingnessAwareKBinsDiscretizer(TransformerMixin, BaseEstimator):
    def __init__(self, n_bins=5, strategy="quantile"):
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y=None, sample_weight=None):
        # Any sample weights are ignored for now.
        X = self._validate_data(X, dtype="numeric", force_all_finite="allow-nan")
        self.transformers_ = []
        for x in X.T:
            transformer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy=self.strategy,
                subsample=None,
            )
            x = x[~np.isnan(x)].reshape(-1, 1)
            transformer.fit(x)
            self.transformers_.append(transformer)
        return self

    def transform(self, X):
        if not hasattr(self, "transformers_"):
            raise NotFittedError("This transformer instance is not fitted yet.")
        X = self._validate_data(
            X, dtype=None, copy=True, reset=False, force_all_finite="allow-nan"
        )
        Xt = []
        for x, transformer in zip(X.T, self.transformers_):
            mask = ~np.isnan(x)
            xt = x[mask].reshape(-1, 1)
            xt = transformer.transform(xt)
            xt_nan = np.full_like(x, np.nan)
            xt_nan[mask] = xt.ravel()
            Xt.append(xt_nan)
        return np.stack(Xt, axis=-1)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "n_features_in_")
        input_features = _check_feature_names_in(self, input_features)
        # No encoding is performed, so the output feature names are the same as
        # the input features.
        return input_features


class MissingnessAwareOneHotEncoder(OneHotEncoder):
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        self._fit(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",
            # Missing values should not be considered infrequent.
            return_and_ignore_missing_for_infrequent=True,
        )
        new_categories = []
        for categories in self.categories_:
            if is_scalar_nan(categories[-1]):
                # The last category is reserved for missing values.
                new_categories.append(categories[:-1])
            else:
                new_categories.append(categories)
        self.categories_ = new_categories
        self._set_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        return self

    def transform(self, X):
        out = super().transform(X)
        # Preserve missing values.
        X_expanded = []
        for i, x in enumerate(X.T):
            cats = self.categories_[i]
            infrequent_cats = self.infrequent_categories_[i]
            drop_idx = self.drop_idx_[i] if self.drop_idx_ is not None else None
            if infrequent_cats is not None:
                n_repeats = cats.size - infrequent_cats.size + 1
            else:
                n_repeats = cats.size
            if drop_idx is not None:
                n_repeats -= 1
            X_expanded.append(np.repeat(x.reshape(1, -1), n_repeats, axis=0))
        X_expanded = np.concatenate(X_expanded, axis=0).T
        isnan = np.vectorize(lambda x: is_scalar_nan(x))
        M = isnan(X_expanded)
        out[M] = np.nan
        return out
