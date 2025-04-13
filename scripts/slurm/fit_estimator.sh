#!/bin/bash

if [[ "$#" -ne 4 ]]; then
    echo "Usage: $0 container_path config_or_experiment_dir_path dataset_alias estimator_alias"
    exit 1
fi

container_path="$1"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    config_path="$2"
else
    config_path="${2}/trial_$(printf "%03d" "$SLURM_ARRAY_TASK_ID")/config.yml"
fi

dataset_alias="$3"

estimator_alias="$4"

cd ~
rsync -r malearn "$TMPDIR" --exclude="*_env"
cd "${TMPDIR}/malearn"

if [[ "$SLURM_JOB_PARTITION" == "alvis" ]]; then
    bind="--bind ${TMPDIR}:/mnt"
elif [[ "$SLURM_JOB_PARTITION" == "tetralith" ]]; then
    bind="--bind /proj:/proj,${TMPDIR}:/mnt"
else
    echo "Unknown partition: $SLURM_JOB_PARTITION."
    exit 1
fi

apptainer exec $bind --nv "$container_path" python scripts/fit_estimator.py \
    --config_path "$config_path" \
    --dataset_alias "$dataset_alias" \
    --estimator_alias "$estimator_alias"

if [ "$SLURM_ARRAY_JOB_ID" ]
then
    log_path="${2}/logs/"
    log_path+="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    base_dir=$(grep 'base_dir:' "$config_path" | awk -F': ' '{print $2}')
    results_dir=$(grep 'results_dir:' "$config_path" | awk -F': ' '{print $2}')
    cp "$log_path" "${base_dir}/${results_dir}"
fi
