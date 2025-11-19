"""Utility functions for handling configuration files."""

import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def check_for_existing_config(cfg: DictConfig, run_dir: str, break_on_differences: bool = True) -> bool:
    """Check if a configuration file already exists in the run directory and compare it with the current configuration.

    Args:
        cfg: The current configuration to compare.
        run_dir: The directory where the configuration file is expected to be found.
        break_on_differences: If True, will raise an error if differences are found.

    Returns:
        bool: True if the configurations match, False otherwise.

    """
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        return False

    old_cfg = OmegaConf.load(config_path)
    assert isinstance(old_cfg, DictConfig), "Loaded configuration is not a DictConfig."
    differences = compare_configs(cfg, old_cfg)

    if differences:
        logger.warning(f"A previous run that created the run directory {run_dir} used a different config:")
        for k, v_new, v_old in differences:
            logger.warning(f"{k}: {v_old} -> {v_new}")

        if break_on_differences:
            raise ValueError("Configuration mismatch detected. Please check the differences above.")
        else:
            logger.info("Differences found but not breaking execution. Continuing with the current configuration.")
        return False

    return True


def compare_configs(config1: DictConfig, config2: DictConfig, path: str = "") -> List[Tuple[str, Any, Any]]:
    """Recursively compare two OmegaConf DictConfigs and return a list of differences.

    Args:
        config1: The first OmegaConf DictConfig to compare.
        config2: The second OmegaConf DictConfig to compare.
        path: Current path in the config structure (used for recursion)

    Returns:
        List of tuples containing (path, value1, value2) for each difference

    """
    differences = []

    # Get all keys from both configs. Conf.keys() includes ??? and interpolation keys.
    all_keys = set(config1.keys()) | set(config2.keys())

    for key in all_keys:
        key = str(key)
        # Construct the current path
        current_path = f"{path}.{key}" if path else key

        # Check if the key exists in both configs
        # Interpolations are fields that are not resolved yet. We can skip them since they are reported elsewhere.
        if (key in config1.keys() and OmegaConf.is_interpolation(config1, key)) or (
            key in config2.keys() and OmegaConf.is_interpolation(config2, key)
        ):
            pass
        # Missing keys are ??? fields. We can skip them since they are reported elsewhere.
        elif (key in config1.keys() and OmegaConf.is_missing(config1, key)) or (
            key in config2.keys() and OmegaConf.is_missing(config2, key)
        ):
            pass
        # If key is missing on one of the configs, it will be reported as a difference.
        elif key not in config1 and key in config2:
            differences.append((current_path, "<missing>", config2[key]))
        elif key not in config2 and key in config1:
            differences.append((current_path, config1[key], "<missing>"))

        else:
            value1 = config1[key]
            value2 = config2[key]

            # Recursively compare nested DictConfigs
            if isinstance(value1, DictConfig) and isinstance(value2, DictConfig):
                differences.extend(compare_configs(value1, value2, current_path))
            # Compare non-DictConfig values
            elif value1 != value2:
                differences.append((current_path, value1, value2))

    return differences


def instantiate_config(cfg: DictConfig):
    """Instantiate the configuration."""
    # Instantiate all objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    renderer = hydra.utils.instantiate(cfg.renderer)
    cfg.observation_dim = dataset.observation_dim
    cfg.action_dim = dataset.action_dim
    agent = hydra.utils.instantiate(cfg.agent)
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        agent=agent,
        dataset=dataset,
        renderer=renderer,
    )
    return trainer, dataset, renderer


def load_config(
    env_name: str,
    experiment_dir: str,
    overrides: list,
) -> DictConfig:
    """Load and merge configuration files based on environment name.

    This function initializes the Hydra configuration system, composes the configuration based on the provided
    environment name and overrides, and optionally saves the configuration to a file.

    Args:
        env_name (str): Name of the environment (e.g., "hopper-medium-v2").
        experiment_dir (str): Directory containing the configuration files.
        overrides (list): List of additional overrides to apply to the configuration.

    Returns:
        DictConfig: The composed configuration object.

    """
    overrides = [f"env={env_name}", *overrides]
    env_config, dataset_config = config_mapping(env_name, experiment_dir)

    if env_config is not None:
        overrides.append(f"+environments/{env_config}={env_config}")
    if dataset_config is not None:
        overrides.append(f"+environments/{env_config}/datasets={dataset_config}")

    # Compose configuration
    with initialize_config_dir(config_dir=os.path.abspath(experiment_dir), version_base="1.3"):
        cfg = compose(config_name="root", overrides=overrides)

    return cfg


def save_config(cfg: DictConfig, run_dir: str) -> None:
    """Save the configuration to a file.

    Args:
        cfg (DictConfig): The configuration object to save.
        run_dir (str): Directory where the configuration file will be saved, relative to the project root.

    """
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    logger.info(f"Configuration saved to {os.path.join(run_dir, 'config.yaml')}")


def config_mapping(env_name: str, base_dir: str) -> Tuple[str | None, str | None]:
    """Map from environment names to dataset configurations and environment-dataset pairs.

    This provides utility functions to map environment names to their corresponding dataset and environment
    configurations.

    Configurations for different experiments are assumed to follow a folder structure:
    # <config_dir/experiment_name> # e.g., "config/baselines/ajay_dd"
    # ├── environments
    # │   ├── <env_name>: # e.g., "locomotion", "maze2d", "kitchen"...
    # │   │   ├── datasets
    # │   │   │   ├── <dataset_name>.yaml: # e.g., "halfcheetah-medium-replay", "hopper-medium-expert"...
    # │   │   │   └── ... -> Dataset specific overrides. Overrides defaults and environment overrides.
    # │   │   └── <env_name>.yaml -> Environment specific overrides. Overrides defaults.
    # │   └── ...
    # ├── defaults.yaml -> Default configuration file that is used for all experiments. May be overridden.
    # └── root.yaml -> Base configuration file for the experiment. Contains placeholders and globals.

    This function then returns the names of the environment and dataset configurations based on the provided environment
    name.

    Args:
        env_name (str): Name of the environment.
        base_dir (str): Base directory for the configuration files relative to project root.

    Returns:
        Tuple[str, str]: Dataset configuration name and environment-dataset string.

    """
    env_config = _environment_mapping(env_name, base_dir)

    if env_config is None:
        return None, None
    dataset_config = _dataset_mapping(env_name, env_config, base_dir)

    return env_config, dataset_config


def _environment_mapping(env_name, base_dir: str):
    """Map environment names to environment configurations.

    Args:
        env_name (str): Name of the environment.
        base_dir (str): Base directory for the configuration files.

    Returns:
        str: Environment configuration name.

    """
    # env dir: project_root/base_dir/environments
    env_dir = os.path.join(os.path.abspath(base_dir), "environments")

    if any([s in env_name for s in ["hopper", "halfcheetah", "walker"]]):
        env_config = "locomotion"
    elif "maze2d" in env_name:
        env_config = "maze2d"
    elif "kitchen" in env_name:
        env_config = "kitchen"
    else:
        env_config = None

    if env_config is not None and os.path.exists(os.path.join(env_dir, env_config)):
        return env_config
    else:
        logger.warning(f"Environment '{env_name}' not recognized or configuration not found in {env_dir}")
        return None


def _dataset_mapping(env_name: str, env_config: str, base_dir: str) -> str | None:
    """Map environment names to specific environment-dataset configurations.

    Args:
        env_name (str): Name of the environment.
        env_config (str): Name of the environment configuration.
        base_dir (str): Base directory for the configuration files.

    Returns:
        str or None: The matched environment-dataset string, or None if not found.

    """
    dataset_dir = os.path.join(base_dir, "environments", env_config, "datasets")

    # Locomotion datasets
    if env_config == "locomotion":
        for env in ["halfcheetah", "hopper", "walker2d"]:
            for dataset in ["medium-replay", "medium-expert", "medium"]:  # medium goes last
                if f"{env}-{dataset}" in env_name:
                    dataset_config = f"{env}-{dataset}"
                    break

    # Maze2D datasets
    elif env_config == "maze2d":
        for dataset in ["umaze", "medium", "large"]:
            if dataset in env_name:
                dataset_config = dataset

    # Kitchen datasets
    elif env_config == "kitchen":
        for dataset in ["mixed", "partial", "complete"]:
            if dataset in env_name:
                dataset_config = dataset

    # If no specific dataset is found, return None
    else:
        dataset_config = None

    if dataset_config is not None and os.path.exists(os.path.join(dataset_dir, f"{dataset_config}.yaml")):
        return dataset_config
    else:
        logger.info(f"Dataset configuration '{dataset_config}' not found in {dataset_dir}. Proceeding with defaults.")
        return None
