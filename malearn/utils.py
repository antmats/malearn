import os
import sys
import copy
import datetime
from pathlib import Path
from collections.abc import Mapping

import yaml
import torch
import skorch

__all__ = [
    "seed_torch",
    "to_tensor",
    "print_log",
    "load_config",
    "create_results_dir",
    "save_yaml",
]


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def to_tensor(X, device, dtype=None, accept_sparse=False):
    X = skorch.utils.to_tensor(X, device, accept_sparse)
    if dtype is not None:
        if isinstance(X, Mapping):
            X = {k: v.to(dtype) for k, v in X.items()}
        elif isinstance(X, torch.Tensor):
            X = X.to(dtype)
        else:
            raise ValueError(f"Got unexpected data type {type(X)}.")
    return X


def print_log(output):
    print(output)
    sys.stdout.flush()


def _change_to_local_paths(d, cluster_project_path, local_project_path):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            recursive = {
                k: _change_to_local_paths(
                    v,
                    cluster_project_path,
                    local_project_path,
                )
            }
            out.update(recursive)
        elif (
            isinstance(k, str) 
            and ("path" in k or "dir" in k)
            and isinstance(v, str)
        ):
            out[k] = v.replace(
                cluster_project_path,
                local_project_path,
            )
        else:
            out[k] = v
    return out


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    local_home_path = os.environ.get("LOCAL_HOME_PATH")
    cluster_project_path = os.environ.get("CLUSTER_PROJECT_PATH")
    local_project_path = os.environ.get("LOCAL_PROJECT_PATH")

    if (
        local_home_path == str(Path.home())
        and cluster_project_path is not None
        and local_project_path is not None
    ):
        config = _change_to_local_paths(
            config,
            cluster_project_path,
            local_project_path,
        )

    return config


def create_results_dir(
    config,
    suffix=None,
    update_config=False
):
    time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if suffix is not None:
        time_stamp += f"_{suffix}"

    results_dir = os.path.join(config["results_dir"], time_stamp)
    results_dir_path = os.path.join(config["base_dir"], results_dir)
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)

    if update_config:
        config = copy.deepcopy(config)
        config["results_dir"] = results_dir
        return results_dir_path, config
    else:
        return results_dir_path


def save_yaml(data, path, filename, **kwargs):
    Path(path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(path, filename + ".yml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, **kwargs)
