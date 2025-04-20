import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from malearn.estimators import MADTClassifier
from malearn.postprocessing import plot_tree

SEED = 1337

N_SAMPLES = 1000
TEST_SIZE = 0.2

MIN_AGE = 50
MAX_AGE = 90
AGE_THRESHOLD = 65

P_OLD_POS_MMSE = 0.18
P_OLD_POS_MMSE_MRI_AVAIL = 1
P_OLD_POS_MMSE_MRI_AVAIL_POS_MRI = 0.82
P_OLD_NEG_MMSE_MRI_AVAIL = 0.13
P_OLD_NEG_MMSE_MRI_AVAIL_MRI_POS = 0.38
P_YOUNG_MRI_AVAIL = 0.10
P_YOUNG_MRI_AVAIL_MRI_POS = 0.03

P1_MRI_POS = 0.95
P1_MMSE_POS = 0.7
P1_DEFAULT = 0.1
P1_AGE_INCREMENT = 0.001

MAX_DEPTH = 5
CCP_ALPHA = 0.005

LABEL_MAPPER = {
    "mri <= 0.5": r"$V_h$ low",
    "mmse <= 0.5": "MMSE score low",
    "age <= 65.3": "Age > 65",
    "age <= 65.2": "Age > 65",
}


def format_labels(samples, value, rho):
    samples = samples.split(" = ")[1]
    samples = int(samples)
    value = value.split(" = ")[1]
    value = value.split(",")[1].replace("]", "").strip()  # Positive class
    value = float(value)
    value = "Pr(CI) = {:.2f}".format(value / samples)
    samples = "Samples: {:.1f}$\\%$".format(100 * (samples / N_SAMPLES))
    rho = float(rho.split(" = ")[1])
    rho = r"$\hat{\rho}$ = " + "{:.2f}".format(rho)
    return samples, value, rho


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=os.getcwd())
    parser.add_argument("alphas", type=float, nargs="*", default=[0, 1])
    args = parser.parse_args()

    random_state = np.random.RandomState(SEED)

    # Sample age.
    age = random_state.uniform(MIN_AGE, MAX_AGE, N_SAMPLES)

    is_young = age < AGE_THRESHOLD
    is_old = ~is_young

    indices_young = np.where(is_young)[0]
    indices_old = np.where(is_old)[0]

    n_young = len(indices_young)
    n_old = len(indices_old)

    # Sample MMSE result.
    #
    # An MMSE is performed for all patients over 65 years old.
    mmse_outcome = np.full_like(age, np.nan, dtype=float)
    mmse_outcome[is_old] = random_state.choice(
        [0, 1], size=n_old, p=[1-P_OLD_POS_MMSE, P_OLD_POS_MMSE]
    )

    # Sample MRI scan result.

    mri_outcome = np.full_like(age, np.nan, dtype=float)

    # For young patients, an MRI scan is performed with probability
    # P_YOUNG_MRI_AVAIL.
    has_mri_young = random_state.rand(n_young) < P_YOUNG_MRI_AVAIL
    indices_mri_young = indices_young[has_mri_young]
    # Default to 0.
    mri_outcome[indices_mri_young] = 0.
    # Update with positive cases.
    n_mri_young = len(indices_mri_young)
    has_pos_mri_young = random_state.rand(n_mri_young) < P_YOUNG_MRI_AVAIL_MRI_POS
    mri_outcome[indices_mri_young[has_pos_mri_young]] = 1.

    # For old patients with a positive MMSE result, an MRI is performed with
    # probability P_OLD_POS_MMSE_MRI_AVAIL.
    has_pos_mmse_old = mmse_outcome[is_old] == 1
    indices_pos_mmse_old = indices_old[has_pos_mmse_old]
    n_pos_mmse_old = len(indices_pos_mmse_old)
    has_mri_old_pos_mmse = (
        random_state.rand(n_pos_mmse_old) < P_OLD_POS_MMSE_MRI_AVAIL
    )
    indices_mri_old_pos_mmse = indices_pos_mmse_old[has_mri_old_pos_mmse]
    # Default to 0.
    mri_outcome[indices_mri_old_pos_mmse] = 0.
    # Update with positive cases.
    n_mri_old_pos_mmse = len(indices_mri_old_pos_mmse)
    has_pos_mri_old_pos_mmse = (
        random_state.rand(n_mri_old_pos_mmse) < P_OLD_POS_MMSE_MRI_AVAIL_POS_MRI
    )
    mri_outcome[indices_mri_old_pos_mmse[has_pos_mri_old_pos_mmse]] = 1.

    # For old patients with a negative MMSE result, an MRI is performed with
    # probability P_OLD_NEG_MMSE_MRI_AVAIL.
    has_neg_mmse_old = mmse_outcome[is_old] == 0
    indices_neg_mmse_old = indices_old[has_neg_mmse_old]
    n_neg_mmse_old = len(indices_neg_mmse_old)
    has_mri_old_neg_mmse = (
        random_state.rand(n_neg_mmse_old) < P_OLD_NEG_MMSE_MRI_AVAIL
    )
    indices_mri_old_neg_mmse = indices_neg_mmse_old[has_mri_old_neg_mmse]
    # Default to 0.
    mri_outcome[indices_mri_old_neg_mmse] = 0.
    # Update with positive cases.
    n_mri_old_neg_mmse = len(indices_mri_old_neg_mmse)
    has_pos_mri_old_neg_mmse = (
        random_state.rand(n_mri_old_neg_mmse) < P_OLD_NEG_MMSE_MRI_AVAIL_MRI_POS
    )
    mri_outcome[indices_mri_old_neg_mmse[has_pos_mri_old_neg_mmse]] = 1.

    # Sample outcome.

    def get_outcome(row):
        age = row["age"]
        mmse = row["mmse"]
        mri = row["mri"]
        r = random_state.rand()
        if mri == 1:
            return int(r < P1_MRI_POS)
        if mri == 0:
            return int(r < 1 - P1_MRI_POS)
        if mmse == 1:
            return int(r < P1_MMSE_POS)
        if mmse == 0:
            return int(r < 1 - P1_MMSE_POS)
        return int(r < P1_DEFAULT + P1_AGE_INCREMENT * (age - MIN_AGE))

    X = pd.DataFrame({"age": age, "mmse": mmse_outcome, "mri": mri_outcome})
    y = X.apply(get_outcome, axis=1)

    M = X.isna().astype(int).values

    X_train, X_test, y_train, y_test, M_train, M_test = train_test_split(
        X, y, M, test_size=0.2, random_state=random_state
    )

    imputer = SimpleImputer(strategy="constant", fill_value=0)
    Xt_train = imputer.fit_transform(X_train)
    Xt_test = imputer.transform(X_test)

    for alpha in args.alphas:
        madt = MADTClassifier(
            max_depth=MAX_DEPTH,
            random_state=random_state,
            alpha=alpha,
            compute_rho_per_node=True,
            ccp_alpha=CCP_ALPHA,
        )
        madt.fit(Xt_train, y_train, M_train)
    
        auc = madt.score(Xt_test, y_test, metric="roc_auc_ovr")
        rho = madt.compute_missingness_reliance(Xt_test, M_test)

        fig, ax = plt.subplots(figsize=(10, 8))

        plot_tree(
            madt,
            ax=ax,
            formatter=format_labels,
            precision=1,
            fontsize=14,
            impurity=False,
            feature_names=X.columns,
            filled=True,
            color_wrt_rho=True,
            annotate_arrows=True,
            label_mapper=LABEL_MAPPER,
            revert_true_false=True,
        )

        title = (
            rf"MA-DT w/ $\alpha$={alpha} | AUC = {auc:.2f} | "
            + r"$\hat{\rho}$ = "
            + f"{rho:.2f}"
        )
        ax.set_title(title, fontsize=16)

        filename = f"matree_alpha_{alpha}.pdf"
        fig.savefig(
            os.path.join(args.output_dir, filename),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
