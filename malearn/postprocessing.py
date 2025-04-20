import os
import re
import warnings
import glob
from numbers import Integral
from os.path import join, basename
from collections.abc import Iterable
from functools import partial

import joblib
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn._statistics import EstimateAggregator

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _criterion
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import validate_params, Interval, StrOptions
from sklearn.tree._export import _MPLTreeExporter, _color_brew
from sklearn.tree._reingold_tilford import Tree
from sklearn.base import is_classifier

from .estimators import MADTClassifier, MADTRegressor


class TreeExporter(_MPLTreeExporter):
    def __init__(
        self,
        *args,
        color_wrt_rho=False,
        node_ids_to_include=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.color_wrt_rho = color_wrt_rho
        self.node_ids_to_include = node_ids_to_include

    def _make_tree(self, node_id, et, criterion, depth=0):
        name = self.node_to_str(et, node_id, criterion=criterion)
        if (
            self.node_ids_to_include is not None
            and node_id not in self.node_ids_to_include
            and not self.node_ids
        ):
            name = self.node_to_str(et, node_id, criterion=criterion)
            left = et.children_left[node_id]
            right = et.children_right[node_id]
            if left != -1 and right != -1:
                splits = name.split("\n")
                splits[0] = "null"
                name = "\n".join(splits)
            return Tree(name, node_id)
        return super()._make_tree(node_id, et, criterion, depth)

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                if node.tree.label.startswith("null"):
                    kwargs["bbox"]["fc"] = "lightgrey"
                    label = node.tree.label.replace("null", "(...)")
                    ax.annotate(label, xy_parent, xy, **kwargs)
                else:
                    ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            if not node.tree.label.startswith("null"):
                for child in node.children:
                    self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "lightgrey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)

    def get_fill_color(self, tree, node_id):
        if "rgb" not in self.colors:
            # Initialize colors and bounds if required.
            self.colors["rgb"] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                self.colors["bounds"] = (-np.max(tree.impurity), -np.min(tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                # Find max and min values in leaf nodes for regression.
                self.colors["bounds"] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            if self.color_wrt_rho:
                node_val = tree.missingness_reliance[node_id]
            else:
                node_val = tree.value[node_id][0, :]
                if (
                    tree.n_classes[0] == 1
                    and isinstance(node_val, Iterable)
                    and self.colors["bounds"] is not None
                ):
                    # Unpack the float only for the regression tree case.
                    # Classification tree requires an Iterable in `get_color`.
                    node_val = node_val.item()
        else:
            # If multi-output color node by impurity.
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def get_color(self, value):
        if self.color_wrt_rho:
            cmap = sns.light_palette("tomato", as_cmap=True)
            color = cmap(value)
            color = [int(round(c * 255)) for c in color[:3]]
            return "#{:02x}{:02x}{:02x}".format(*color)
        return super().get_color(value)

    def node_to_str(self, tree, node_id, criterion):
        node_string = super().node_to_str(tree, node_id, criterion)
        missingness_reliance = (
            tree.missingness_reliance[node_id]
            if hasattr(tree, "missingness_reliance") else None
        )
        if missingness_reliance is not None:
            node_string += "\n" + f"rho = {missingness_reliance:.3f}"
        return node_string


@validate_params(
    {
        "decision_tree": [
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            MADTClassifier,
            MADTRegressor,
        ],
        "max_depth": [Interval(Integral, 0, None, closed="left"), None],
        "feature_names": ["array-like", None],
        "class_names": ["array-like", "boolean", None],
        "label": [StrOptions({"all", "root", "none"})],
        "filled": ["boolean"],
        "impurity": ["boolean"],
        "node_ids": ["boolean"],
        "proportion": ["boolean"],
        "rounded": ["boolean"],
        "precision": [Interval(Integral, 0, None, closed="left"), None],
        "ax": "no_validation",  # delegate validation to matplotlib
        "fontsize": [Interval(Integral, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def plot_tree(
    decision_tree,
    *,
    max_depth=None,
    feature_names=None,
    class_names=None,
    label="all",
    filled=False,
    impurity=True,
    color_wrt_rho=False,
    node_ids=False,
    proportion=False,
    rounded=False,
    precision=3,
    ax=None,
    fontsize=None,
    label_mapper=None,
    formatter=None,
    annotate_arrows=False,
    revert_true_false=False,
    inverse_transformer=None,
    node_ids_to_include=None,
):
    check_is_fitted(decision_tree)

    exporter = TreeExporter(
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=class_names,
        label=label,
        filled=filled,
        impurity=impurity,
        color_wrt_rho=color_wrt_rho,
        node_ids=node_ids,
        proportion=proportion,
        rounded=rounded,
        precision=precision,
        fontsize=fontsize,
        node_ids_to_include=node_ids_to_include,
    )
    annotations = exporter.export(decision_tree, ax=ax)

    if node_ids or label != "all":
        return annotations

    if not hasattr(decision_tree.tree_, "missingness_reliance"):
        return annotations

    if all([m is None for m in decision_tree.tree_.missingness_reliance]):
        return annotations

    if ax is None:
        ax = plt.gca()

    if label_mapper is None:
        label_mapper = {}

    criterion = decision_tree.criterion
    if isinstance(criterion, _criterion.FriedmanMSE):
        criterion = "friedman_mse"
    elif isinstance(criterion, _criterion.MSE) or criterion == "squared_error":
        criterion = "squared_error"
    elif not isinstance(criterion, str):
        criterion = "impurity"

    renderer = ax.figure.canvas.get_renderer()
    for annotation in annotations:
        text = annotation.get_text()
        if text.startswith(criterion) or text.startswith("samples"):
            # Leaf node
            if formatter is not None:
                if impurity:
                    i, s, v, r = text.split("\n")
                    i, s, v, r = formatter(i, s, v, r)
                    text = "\n".join([s, i, v, r])
                else:
                    s, v, r = text.split("\n")
                    s, v, r = formatter(s, v, r)
                    text = "\n".join([s, v, r])
        elif text.startswith("True"):
            text = text + "        " if not annotate_arrows else ""
        elif text.endswith("False"):
            text = "        " + text if not annotate_arrows else ""
        elif text.startswith("\n"):
            # (...)
            pass
        else:
            # Inner node
            l = text.split("\n")[0]
            if l in label_mapper:
                l = label_mapper[l]
            elif re.match(r".*?\s*<=\s*-?\w+", l):
                l1, l2 = l.split(" <= ")
                l2 = float(l2)
                if inverse_transformer is not None:
                    try:
                        l2 = inverse_transformer(l1, l2)
                    except ValueError:
                        pass
                l1 = label_mapper.get(l1, l1)
                l = l1 + r" $\leq$ " + "{:.{prec}f}".format(l2, prec=precision)
            if impurity:
                _l, i, s, v, r = text.split("\n")
                if formatter is not None:
                    i, s, v, r = formatter(i, s, v, r)
                text = "\n".join([l, i, s, v, r])
            else:
                _l, s, v, r = text.split("\n")
                if formatter is not None:
                    s, v, r = formatter(s, v, r)
                text = "\n".join([l, s, v, r])
        annotation.set_text(text)
        annotation.set(ha="center")
        annotation.draw(renderer)

    if annotate_arrows:
        kwargs = dict(va="center", fontsize=fontsize)
        x0, y0 = annotations[0].get_position()
        x1, y1 = annotations[1].get_position()
        t = "True" if not revert_true_false else "False"
        f = "False" if not revert_true_false else "True"
        ax.annotate(t, (x1 + (x0-x1) / 3, y0 - (y0-y1) / 2), ha="right", **kwargs)
        ax.annotate(f, (x0 + 2 * (x0-x1) / 3, y0 - (y0-y1) / 2), ha="left", **kwargs)


def collect_scores(experiment_dir):
    all_scores = []

    pfile_names = [x for x in os.listdir(experiment_dir) if x.startswith("parameters")]
    pfile_names.sort()

    index = 1

    for pfile_name in pfile_names:
        pfile_path = join(experiment_dir, pfile_name)
        pfile = pd.read_csv(pfile_path)
        pfile.index += index

        for i, parameters in pfile.iterrows():
            results_dir = join(experiment_dir, f"trial_{i:03d}")

            for scores_path in glob.glob(join(results_dir, "*_scores.csv")):
                scores = pd.read_csv(scores_path)
                estimator_alias = basename(scores_path).removesuffix("_scores.csv")
                scores["estimator_alias"] = estimator_alias
                scores = scores.assign(**parameters)
                all_scores.append(scores)

        index = i + 1

    scores = pd.concat(all_scores)

    required_columns = ["estimator_alias", "seed", "metric", "score"]
    assert set(required_columns).issubset(scores.columns)

    default_columns = [
        "numerical_imputation",
        "categorical_imputation",
        "missing_mechanism",
        "missing_features",
        "missing_rate",
        "alpha",
    ]

    scores = scores.reindex(
        columns=scores.columns.union(default_columns), fill_value=np.nan
    )

    scores = scores.fillna(
        dict(
            imputation="none",
            numerical_imputation="none",
            categorical_imputation="none",
            missing_mechanism="none",
            missing_features=0,
            missing_rate=0,
            alpha=-1,
        )
    )

    other_columns = set(scores.columns) - {"metric", "score"}

    scores = scores.pivot_table(
        index=other_columns,
        columns="metric",
        values="score",
        aggfunc="first",
    )

    scores = scores.reset_index()
    scores.columns.name = None

    return scores


def get_scoring_table(
    scores,
    metric,
    groupby="estimator_alias",
    include_cis=False,
):
    g = scores.groupby(groupby)

    agg = EstimateAggregator(np.mean, "ci", n_boot=1000, seed=0)

    table = g.apply(agg, var=metric)
    table = table * 100  # Convert to percentage

    if include_cis:
        f = lambda r: rf"{r[metric]:.1f} ({r[f'{metric}min']:.1f}, {r[f'{metric}max']:.1f})"
    else:
        f = lambda r: f"{r[metric]:.1f}"
    table[metric] = table[[metric, f"{metric}min", f"{metric}max"]].apply(f, axis=1)
    table = table.drop(columns=[f"{metric}min", f"{metric}max"])

    return table


def combine_scoring_tables(scores, metrics, **kwargs):
    tables = []
    for metric in metrics:
        table = get_scoring_table(scores, metric, **kwargs)
        tables.append(table)
    return pd.concat(tables, axis=1)


def collect_cv_results(experiment_dir, estimator_alias, exclude_params=None):
    results_dirs = [x for x in os.listdir(experiment_dir) if x.startswith("trial")]
    results_dirs.sort()

    if exclude_params is None:
        exclude_params = {}

    all_cv_results = []
    for results_dir in results_dirs:
        params = pd.read_csv(join(experiment_dir, results_dir, "parameters.csv"))
        skip_dir = False
        for p, v in exclude_params.items():
            if p in params and params[p].item() == v:
                skip_dir = True
        if skip_dir:
            continue
        for cv_results_path in glob.glob(join(experiment_dir, results_dir, "*_cv_results.pkl")):
            if basename(cv_results_path).removesuffix("_cv_results.pkl") == estimator_alias:
                cv_results = pd.DataFrame(joblib.load(cv_results_path))
                params = pd.concat([params] * len(cv_results), ignore_index=True)
                cv_results = pd.concat([cv_results, params], axis=1)
                all_cv_results.append(cv_results)

    return pd.concat([df for df in all_cv_results])


def collect_estimators(experiment_dir, estimator_alias, exclude_params=None):
    results_dirs = [x for x in os.listdir(experiment_dir) if x.startswith("trial")]
    results_dirs.sort()

    if exclude_params is None:
        exclude_params = {}

    all_estimators = []
    for results_dir in results_dirs:
        params = pd.read_csv(join(experiment_dir, results_dir, "parameters.csv"))
        skip_dir = False
        for p, v in exclude_params.items():
            if p in params and params[p].item() == v:
                skip_dir = True
        if skip_dir:
            continue
        for estimator_path in glob.glob(join(experiment_dir, results_dir, "*_estimator.pkl")):
            if basename(estimator_path).removesuffix("_estimator.pkl") == estimator_alias:
                estimator = joblib.load(estimator_path)
                all_estimators.append(estimator)

    return all_estimators


def inspect_hparams(
    experiment_dir,
    estimator_alias,
    metric="score",
    use_log_scale=None,
    exclude_params=None,
):
    results_dirs = [x for x in os.listdir(experiment_dir) if x.startswith("trial")]
    results_dirs.sort()

    if exclude_params is None:
        exclude_params = {}

    cv_results = []
    for results_dir in results_dirs:
        params = pd.read_csv(join(experiment_dir, results_dir, "parameters.csv"))
        skip_dir = False
        for p, v in exclude_params.items():
            if p in params and params[p].item() == v:
                skip_dir = True
        if skip_dir:
            continue
        for cv_results_path in glob.glob(join(experiment_dir, results_dir, "*_cv_results.pkl")):
            if basename(cv_results_path).removesuffix("_cv_results.pkl") == estimator_alias:
                cv_results.append(joblib.load(cv_results_path))

    cv_results = pd.concat([pd.DataFrame(d) for d in cv_results])

    if use_log_scale is None:
        use_log_scale = []

    for param in [c for c in cv_results.columns if c.startswith("param_")]:
        _fig, ax = plt.subplots()
        ax.scatter(
            cv_results[f"mean_test_{metric}"],
            cv_results[param],
        )
        if param in use_log_scale:
            ax.set_yscale("log")
        ax.set_title(param)


def get_results_table(
    experiment_dirs,
    metrics,
    *,
    exclude_estimators_from_experiment=None,
    replace={"alpha": {1.0e-9: 0}},
    experiment_replace=None,
    **kwargs,
):
    if isinstance(experiment_dirs, str):
        experiment_dirs = [experiment_dirs]

    all_scores = []
    for experiment_dir in experiment_dirs:
        experiment = os.path.basename(experiment_dir)
        scores = collect_scores(experiment_dir)
        scores["dataset_alias"] = experiment.split("_")[-1]
        if exclude_estimators_from_experiment is not None:
            estimators_to_exclude = (
                exclude_estimators_from_experiment.get(experiment, [])
            )
            scores = scores[
                ~scores.estimator_alias.isin(estimators_to_exclude)
            ]
        scores = scores.replace(replace)
        if experiment_replace is not None:
            scores = scores.replace(
                experiment_replace.get(experiment, {})
            )
        all_scores.append(scores)

    all_scores = pd.concat(all_scores)

    if (
        all_scores.dataset_alias.nunique() > 1
        and not "dataset_alias" in kwargs.get("groupby", [])
    ):
        warnings.warn(
            "Scores from multiple datasets were combined but `groupby` was "
            "not passed or does not contain 'dataset_alias'."
        )

    table = combine_scoring_tables(all_scores, metrics, **kwargs)

    return table, all_scores


def get_feature_names(preprocessor, categorical_feature_mapper={}):
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.split("__")[-1] for f in feature_names]
    for i, f in enumerate(feature_names):
        if f.endswith("_infrequent_sklearn"):
            feature_names[i] = f.removesuffix("_sklearn")
            continue
        t = f.removesuffix("_" + f.split("_")[-1])
        if t in categorical_feature_mapper:
            c = float(f.split("_")[-1])
            feature_names[i] = t + "_" + categorical_feature_mapper[t][c]
    return feature_names


def plot_ma_tree(
    madt_pipeline,
    data_handler,
    alpha=None,
    format_labels=None,
    invert_value=None,
    label_mapper=None,
    figsize=(9, 6),
    **kwargs,
):
    madt_preprocessor = madt_pipeline.named_steps["preprocessor"]
    madt_estimator = madt_pipeline.named_steps["estimator"]

    _, _, _, X_test, y_test, M_test = data_handler.split_data()
    if is_classifier(madt_estimator):
        score = madt_pipeline.score(X_test, y_test, metric="roc_auc_ovr")
    else:
        score = madt_pipeline.score(X_test, y_test, metric="r2")
    rho = madt_pipeline.compute_missingness_reliance(X_test, M_test)

    data_path = join(
        data_handler.config["base_dir"],
        data_handler.config["data_dir"],
        data_handler.file_name
    )
    data = data_handler._read(data_path)
    cat_features = (
        data[data_handler.features]
        .select_dtypes(include="object")
        .astype("category")
    )
    cat_feature_mapper = {}
    for cat_feature in cat_features:
        cat_feature_mapper[cat_feature] = dict(
            enumerate(cat_features[cat_feature].cat.categories)
        )

    if format_labels is not None:
        formatter = partial(
            format_labels, total_samples=madt_estimator.tree_.n_node_samples[0]
        )
    else:
        formatter = None

    if invert_value is not None:
        inverse_transformer = partial(
            invert_value,
            transformer=madt_preprocessor[0]["numerical_transformer"],
            feature_names=data_handler.numerical_features,
        )
    else:
        inverse_transformer = None

    feature_names = get_feature_names(madt_preprocessor, cat_feature_mapper)

    fig, ax = plt.subplots(figsize=figsize)

    plot_tree(
        madt_estimator,
        feature_names=feature_names,
        ax=ax,
        formatter=formatter,
        inverse_transformer=inverse_transformer,
        label_mapper=label_mapper,
        **kwargs,
    )

    if alpha is None:
        alpha = madt_estimator.alpha

    title = rf"MA-DT with $\alpha = {alpha}$"
    if is_classifier(madt_estimator):
        title += f" (AUC = {score:.2f}, " + r"$\hat{\rho}$ = " + f"{rho:.2f})"
    else:
        title += f" (R2 = {score:.2f}, " + r"$\hat{\rho}$ = " + f"{rho:.2f})"

    ax.set_title(title)

    return fig, ax
