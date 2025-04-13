import numpy as np
from sklearn.utils.metadata_routing import (
    _MetadataRequester,
    MetadataRequest,
    _routing_enabled,
)


def get_best_candidate_index(cv_results, m1, m2, gamma=0.5):
    threshold = gamma * cv_results[f"mean_test_{m1}"].max()
    mask = cv_results[f"mean_test_{m1}"] >= threshold
    indices = np.where(mask)[0]
    return indices[cv_results[f"mean_test_{m2}"][mask].argmax()]


class CustomScorer(_MetadataRequester):
    def __init__(self, estimator, metric):
        self.estimator = estimator
        self.metric = metric

    def __call__(self, estimator, X, y, *, M=None, sample_weight=None):
        return estimator.score(
            X,
            y,
            M=M,
            sample_weight=sample_weight,
            metric=self.metric,
        )

    def set_score_request(self, **kwargs):
        if not _routing_enabled():
            raise RuntimeError(
                "This method is only available when metadata routing is enabled."
            )
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self
