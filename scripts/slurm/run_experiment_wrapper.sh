#!/bin/bash

n_seeds=5

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 config_file dataset experiment_type"
    exit 1
fi

config_file="$1"
dataset="$2"
experiment_type="$3"

# Default is to use both zero and MICE imputation for numerical features.
imputations=("zero:mode" "mice:mode")

# Default is not to apply any synthetic missingness.
missing_mechanisms=("none")
missing_features=(0)
missing_rates=(0)

if [[ "$experiment_type" == "all" ]]; then
    estimators_imp=("malasso" "madt" "marf" "magbt" "dt" "rf" "lasso")
    estimators_def=("neumiss" "minty" "xgboost" "mgam")

elif [[ "$experiment_type" == "ma" ]]; then
    estimators_imp=("malasso" "madt" "marf" "magbt")

elif [[ "$experiment_type" == "ma_alpha_0" ]]; then
    alphas1=(0)
    estimators_alpha1=("madt" "marf" "magbt")
    alphas2=(0.000000001)
    estimators_alpha2=("malasso")

elif [[ "$experiment_type" == "ma_alpha_sweep" ]]; then
    n_seeds=20
    imputations=("zero:mode")
    alphas1=(0 0.001 0.01 0.1 1 10 100)
    estimators_alpha1=("madt")
    alphas2=(0.000000001 1 10 100 1000 10000)
    estimators_alpha2=("malasso")

elif [[ "$experiment_type" == "lasso_miss" ]]; then
    n_seeds=10
    imputations=("zero:mode")
    missing_mechanisms=("mar" "mnar")
    missing_features=(0.2 0.4 0.6)
    missing_rates=(0.5)
    alphas2=(0.000000001 1 10 100 1000 10000)
    estimators_alpha2=("malasso")
    estimators_imp=("lasso")

else
    echo "Invalid experiment type. Use 'all', 'ma', 'ma_alpha_0', 'ma_alpha_sweep', or 'lasso_miss'."
    exit 1
fi

wait_for_next_minute() {
    current_second=$(date +"%S")
    seconds_to_wait=$((60 - current_second))
    sleep $seconds_to_wait
}

check_timestamp() {
    local expected_timestamp="$1"
    local current_timestamp=$(date +"%y%m%d_%H%M")

    if [[ "$expected_timestamp" == "$current_timestamp" ]]; then
        return 0
    else
        exit 1
    fi
}

wait_for_next_minute

timestamp=$(date +"%y%m%d_%H%M")

# Fit MA estimators with varying alpha (low range).
check_timestamp "$timestamp"
echo "Fitting MA estimators with varying alpha (low range)..."
parameters_file=$(mktemp)
echo "seed,numerical_imputation,categorical_imputation,alpha,missing_mechanism,missing_features,missing_rate" > "$parameters_file"
for seed in $(seq 0 $((n_seeds - 1))); do
    for imputation in "${imputations[@]}"; do
        IFS=":" read -r numerical_imputation categorical_imputation <<< "$imputation"
        for alpha in "${alphas1[@]}"; do
            for miss_mechanism in "${missing_mechanisms[@]}"; do
                for miss_features in "${missing_features[@]}"; do
                    for miss_rate in "${missing_rates[@]}"; do
                        echo "$seed,$numerical_imputation,$categorical_imputation,$alpha,$miss_mechanism,$miss_features,$miss_rate" >> "$parameters_file"
                    done
                done
            done
        done
    done
done
./scripts/slurm/run_experiment.sh "$parameters_file" "$config_file" "$dataset" "${estimators_alpha1[@]}"
rm "$parameters_file"

# Fit MA estimators with varying alpha (high range).
check_timestamp "$timestamp"
echo "Fitting MA estimators with varying alpha (high range)..."
parameters_file=$(mktemp)
echo "seed,numerical_imputation,categorical_imputation,alpha,missing_mechanism,missing_features,missing_rate" > "$parameters_file"
for seed in $(seq 0 $((n_seeds - 1))); do
    for imputation in "${imputations[@]}"; do
        IFS=":" read -r numerical_imputation categorical_imputation <<< "$imputation"
        for alpha in "${alphas2[@]}"; do
            for miss_mechanism in "${missing_mechanisms[@]}"; do
                for miss_features in "${missing_features[@]}"; do
                    for miss_rate in "${missing_rates[@]}"; do
                        echo "$seed,$numerical_imputation,$categorical_imputation,$alpha,$miss_mechanism,$miss_features,$miss_rate" >> "$parameters_file"
                    done
                done
            done
        done
    done
done
./scripts/slurm/run_experiment.sh "$parameters_file" "$config_file" "$dataset" "${estimators_alpha2[@]}"
rm "$parameters_file"

# Fit estimators with varying imputation.
check_timestamp "$timestamp"
echo "Fitting estimators with varying imputation..."
parameters_file=$(mktemp)
echo "seed,numerical_imputation,categorical_imputation,missing_mechanism,missing_features,missing_rate" > "$parameters_file"
for seed in $(seq 0 $((n_seeds - 1))); do
    for imputation in "${imputations[@]}"; do
        IFS=":" read -r numerical_imputation categorical_imputation <<< "$imputation"
        for miss_mechanism in "${missing_mechanisms[@]}"; do
            for miss_features in "${missing_features[@]}"; do
                for miss_rate in "${missing_rates[@]}"; do
                    echo "$seed,$numerical_imputation,$categorical_imputation,$miss_mechanism,$miss_features,$miss_rate" >> "$parameters_file"
                done
            done
        done
    done
done
./scripts/slurm/run_experiment.sh "$parameters_file" "$config_file" "$dataset" "${estimators_imp[@]}"
rm "$parameters_file"

# Fit estimators with default imputation.
check_timestamp "$timestamp"
echo "Fitting estimators with default imputation..."
parameters_file=$(mktemp)
echo "seed,missing_mechanism,missing_features,missing_rate" > "$parameters_file"
for seed in $(seq 0 $((n_seeds - 1))); do
    for miss_mechanism in "${missing_mechanisms[@]}"; do
        for miss_features in "${missing_features[@]}"; do
            for miss_rate in "${missing_rates[@]}"; do
                echo "$seed,$miss_mechanism,$miss_features,$miss_rate" >> "$parameters_file"
            done
        done
    done
done
./scripts/slurm/run_experiment.sh "$parameters_file" "$config_file" "$dataset" "${estimators_def[@]}"
rm "$parameters_file"
