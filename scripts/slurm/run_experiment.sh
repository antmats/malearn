#!/bin/bash

# Set Slurm parameters.
hostname_var=$(hostname)
if [[ $hostname_var == alvis* ]]; then
    account="NAISS2024-5-480"
    partition="alvis"
    base_dir="/mimer/NOBACKUP/groups/oovgen/malearn"
elif [[ $hostname_var == tetralith* ]]; then
    account="NAISS2024-22-285"
    partition="tetralith"
    base_dir="/proj/healthyai/malearn"
else
    echo "Unknown cluster."
    exit 1
fi

# Check if the correct number of arguments were passed in.
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 parameters_file default_config_file dataset estimator1 [estimator2 ...]"
    exit 1
fi

# Get the required input arguments.
parameters_file="$1"
default_config_file="$2"
dataset="$3"

# Shift arguments to get the list of estimators.
shift 3

# Extract the path to the root directory for the results.
results_dir=$(grep 'results_dir:' "$default_config_file" | awk -F': ' '{print $2}')
results_root_dir="${base_dir}/${results_dir}"

# Check that it was found.
if [ -z "$results_root_dir" ]; then
    echo "Could not extract a root directory for the results from $default_config_file."
    exit 1
fi

# Create a new experiment directory in the root directory for the results.
timestamp=$(date +"%y%m%d_%H%M")
experiment_dir="${results_root_dir}/${timestamp}_${dataset}"

if [[ -d "$experiment_dir" ]]; then
    trial_index=$(
        find "$experiment_dir" -type d -name "trial_*" | \
        grep -oP 'trial_\K[0-9]+' | \
        sort -n | \
        tail -1
    )
    trial_index=$((10#$trial_index))

    parameter_index=$(
        find "$experiment_dir" -type f -name "parameters_*.csv" | \
        grep -oP 'parameters_\K[0-9]+' | \
        sort -n | \
        tail -1
    )
    parameter_index=$((10#$parameter_index))
    parameter_index=$((parameter_index + 1))
    cp "$parameters_file" "${experiment_dir}/parameters_$(printf "%03d" "$parameter_index").csv"
else
    mkdir -p "$experiment_dir"

    cp "$default_config_file" "${experiment_dir}/default_config.yml"
    cp "$parameters_file" "${experiment_dir}/parameters_001.csv"

    trial_index=0

    # Create a directory for the log files.
    mkdir -p "${experiment_dir}/logs"
fi

# Define a function to update a parameter value in the config file.
update_config_file() {
    local config_file="$1"
    local parameter_name="$2"
    local new_parameter_value="$3"
    sed -ri "s/^(\s*)(${parameter_name}\s*:\s*)\S+.*/\1${parameter_name}: $new_parameter_value/" "$config_file"
}

i="$trial_index"
while IFS="," read -r -a parameter_values; do

    if [ $i -eq "$trial_index" ]; then
        # Extract the parameter names.
        parameter_names=("${parameter_values[@]}")

        i=$((i + 1))

        continue
    fi

    # Create a results directory for the current parameters.
    results_dir="${experiment_dir}/trial_$(printf "%03d" "$i")"
    mkdir -p "$results_dir"

    # Copy the default config file to the results directory.
    config_file="${results_dir}/config.yml"
    cp "$default_config_file" "$config_file"

    # Update the config file with the current parameters.
    for index in "${!parameter_names[@]}"; do
        parameter_name="${parameter_names[$index]}"
        parameter_value="${parameter_values[$index]}"
        update_config_file "$config_file" "$parameter_name" "$parameter_value"
    done

    # Update the path to the root directory for the project.
    awk -v repl="$base_dir" '
    BEGIN {OFS=FS=": "}
    $1 == "base_dir" {$2 = repl}
    1
    ' "$config_file" > temp.yml && mv temp.yml "$config_file"

    # Update the path to the root directory for the results.
    awk -v repl="${results_dir#$base_dir/}" '
    BEGIN {OFS=FS=": "}
    $1 == "results_dir" {$2 = repl}
    1
    ' "$config_file" > temp.yml && mv temp.yml "$config_file"

    # Save the current parameters to a file.
    echo "$(IFS=,; echo "${parameter_names[*]}")" > "${results_dir}/parameters.csv"
    echo "$(IFS=,; echo "${parameter_values[*]}")" >> "${results_dir}/parameters.csv"

    i=$((i + 1))

done < <(tr -d '\r' < "$parameters_file")

# Fit each estimator.
for estimator in "$@"; do
    # TODO: Should use the "is_net_estimator" field in the config file to
    # determine if the estimator requires a GPU.
    if [ "$partition" == "tetralith" ]; then
        nodes="--nodes=1 --exclusive"
        if [ "$estimator" == "neumiss" ]; then
            gpu_resources="--gpus-per-node=1"
        else
            gpu_resources=""
        fi
    elif [ "$partition" == "alvis" ]; then
        nodes="--nodes=1"
        if [ "$estimator" == "neumiss" ]; then
            gpu_resources="--gpus-per-node=T4:1"
        else
            gpu_resources="--constraint=NOGPU"
        fi
    else
        echo "Unknown partition: $partition."
        exit 1
    fi
    sbatch \
        --account="$account" \
        --partition="$partition" \
        $nodes \
        $gpu_resources \
        --output="${experiment_dir}/logs/%x_%A_%a.out" \
        --time="1-0:0" \
        --array=$((trial_index + 1))-$((i - 1)) \
        --job-name="fit_${estimator}" \
        "scripts/slurm/fit_estimator.sh" "${base_dir}/containers/ma_env.sif" "$experiment_dir" "$dataset" "$estimator"
done
