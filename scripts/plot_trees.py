import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import set_config

from malearn.postprocessing import plot_ma_tree
from malearn.fit_evaluate import fit_estimator
from malearn.utils import load_config
from malearn.data import get_data_handler

RC_DEFAULT = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 14,
}

LABEL_MAPPER_LIFE = {
    "Region_European Union <= 0.5": r"Region $\neq$ European Union",
    "Region_Africa <= 0.5": r"Region $\neq$ Africa",
}

LABEL_MAPPERS = {
    "life": LABEL_MAPPER_LIFE,
    "adni": None,
}


def invert_value(feature, value, transformer, feature_names):
    X = pd.DataFrame(np.ones((1, len(feature_names))), columns=feature_names)
    X.at[0, feature] = value
    X_inv = transformer.inverse_transform(X)
    X_inv = pd.DataFrame(X_inv, columns=feature_names)
    return X_inv.at[0, feature]


def format_labels_adni(samples, value, rho, total_samples):
    samples = samples.split(" = ")[1]
    samples = int(samples)
    value = value.split(" = ")[1]
    value = value.split(",")[1].replace("]", "").strip()  # Positive class
    value = float(value)
    value = "Pr(Dx change) = {:.2f}".format(value / samples)
    samples = "Samples: {:.1f}$\\%$".format(100 * (samples / total_samples))
    rho = float(rho.split(" = ")[1])
    rho = r"$\hat{\rho}$ = " + "{:.2f}".format(rho)
    return samples, value, rho


def format_labels_life(samples, value, rho, total_samples):
    samples = samples.split(" = ")[1]
    samples = int(samples)
    value = value.split(" = ")[1]
    value = value.split(",")[1].replace("]", "").strip()  # Positive class
    value = float(value)
    value = "Pr(LE $>$ median) = {:.2f}".format(value / samples)
    samples = "Samples: {:.1f}$\\%$".format(100 * (samples / total_samples))
    rho = float(rho.split(" = ")[1])
    rho = r"$\hat{\rho}$ = " + "{:.2f}".format(rho)
    return samples, value, rho


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_alias", type=str, default="life")
    parser.add_argument("--output_dir", type=str, default=os.getcwd())
    args = parser.parse_args()

    plt.rcParams.update(RC_DEFAULT)
    plot_kwargs = {
        "precision": 2,
        "max_depth": 1,
        "impurity": False,
        "filled": True,
        "color_wrt_rho": True,
        "annotate_arrows": True,
        "label_mapper": LABEL_MAPPERS.get(args.dataset_alias, None),
        "format_labels": globals().get(f"format_labels_{args.dataset_alias}", None),
        "invert_value": invert_value,
    }
    savefig_kwargs = {
        "dpi": 300,
        "bbox_inches": "tight",
        "pad_inches": 0,
    }

    config = load_config(args.config_path)
    config["experiment"].update(
        {
            "missing_features": 0,
            "missing_rate": 0,
            "numerical_imputation": "zero",
            "categorical_imputation": "mode",
            "sample_size": None,
            "seed": 2024,
            "test_size": 0.2,
        },
    )
    config["model_selection"].update(
        {
            "scoring": ["roc_auc_ovr", "missingness_reliance_score"],
            "refit": "tradeoff",
            "gamma": 0.95,
            "n_splits": 3,
            "test_size": 0.2,
            "seed": 2024,

        },
    )
    config["estimators"]["madt"]["search"] = "exhaustive"
    config["estimators"]["madt"]["hparams"].update(
        {"estimator__compute_rho_per_node": [True]}
    )

    set_config(enable_metadata_routing=True)

    data_handler = get_data_handler(config, args.dataset_alias)

    fit_kwargs = {"dataset_alias": args.dataset_alias, "estimator_alias": "madt"}

    # \alpha = 0
    search = fit_estimator(
        config, fixed_params={"estimator__alpha": 0}, **fit_kwargs
    )
    fig, _ax = plot_ma_tree(
        search.best_estimator_,
        data_handler,
        alpha=0,
        **plot_kwargs,
    )
    filename = f"madt_{args.dataset_alias}_alpha_0.pdf"
    fig.savefig(os.path.join(args.output_dir, filename), **savefig_kwargs)
    plt.close(fig)

    # \alpha = \alpha^{*}
    search = fit_estimator(config, **fit_kwargs)
    fig, _ax = plot_ma_tree(
        search.best_estimator_,
        data_handler,
        alpha=r"\alpha^{*}",
        **plot_kwargs,
    )
    filename = f"madt_{args.dataset_alias}_alpha_*.pdf"
    fig.savefig(os.path.join(args.output_dir, filename), **savefig_kwargs)
    plt.close(fig)

    # \alpha = \infty
    config["model_selection"].update(
        {
            "scoring": ["missingness_reliance_score", "roc_auc_ovr"],
            "refit": "tradeoff",
            "gamma": 1,
        },
    )
    search = fit_estimator(config, **fit_kwargs)
    fig, _ax = plot_ma_tree(
        search.best_estimator_,
        data_handler,
        alpha=r"\infty",
        **plot_kwargs,
    )
    filename = f"madt_{args.dataset_alias}_alpha_inf.pdf"
    fig.savefig(os.path.join(args.output_dir, filename), **savefig_kwargs)
    plt.close(fig)

    set_config(enable_metadata_routing=False)
