"""Classifier-Free Guidance Experiment Script.

This script is designed to run experiments using classifier-free guidance for offline reinforcement learning tasks.
It supports training, testing, and optimization of sampling parameters using Optuna. The script is configurable via
command-line arguments and configuration files.
"""

import argparse
import logging

import lightning  # noqa: F401, prevents double logging
import numpy as np
import optuna
import optuna_distributed
import optuna_distributed.trial
import torch
from omegaconf import DictConfig, OmegaConf
from optuna.storages.journal import JournalFileBackend

from tgdp.utils.configs import check_for_existing_config, instantiate_config, load_config, save_config

# Set up logging.
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Register custom resolvers for OmegaConf.
OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))

# Set up torch to be fast.
torch.set_float32_matmul_precision("medium")
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True


def train(cfg: DictConfig):
    """Train the model using the provided configuration."""
    trainer, dataset, renderer = instantiate_config(cfg)
    trainer.train()


def test(cfg: DictConfig):
    """Test the trained model using the provided configuration and render the first rollout episode."""
    cfg.trainer.load_ckpt = True
    trainer, dataset, renderer = instantiate_config(cfg)
    episodes = trainer.rollout_episodes(100)
    returns = [np.sum(ep["rewards"]) for ep in episodes]
    print(f"Return mean: {np.mean(returns)}, std: {np.std(returns)}")
    norm_env = dataset.make_env()
    if hasattr(norm_env, "get_normalized_score"):
        # If the environment has a method to get normalized scores, use it.
        norm_returns = 100 * norm_env.get_normalized_score(np.array(returns))
        logger.info(
            f"Normalized return mean: {np.mean(norm_returns)}, std: {np.std(norm_returns)}, "
            f"ste: {np.std(norm_returns) / np.sqrt(len(norm_returns))}"
        )
    renderer.render_rollout(episodes[0])


def optim(cfg: DictConfig, use_optuna_distributed: bool = True):
    """Optimize sampling parameters using Optuna for the given configuration."""
    # Setup the study.
    study = optuna.create_study(
        direction="maximize",
        storage=optuna.storages.JournalStorage(JournalFileBackend(file_path="./optim.log")),
        sampler=optuna.samplers.TPESampler(n_startup_trials=40),
        study_name=f"{cfg.experiment_name}",
        load_if_exists=True,
    )
    if use_optuna_distributed:
        study = optuna_distributed.from_study(study, client=None)

    # Set the Ajay parameters as a starting point.
    study.enqueue_trial(
        {
            "target_value": cfg.guides.trajectory_return.cfg_target_value,
            "guide_scale": cfg.diffusion_model.guide_scale,
        }
    )
    study.optimize(
        lambda trial: _optimize_sampling_params(trial, cfg),
        n_trials=100 - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),  # Run 100 trials.
        n_jobs=4,
    )


def _optimize_sampling_params(
    trial: optuna.Trial | optuna_distributed.trial.DistributedTrial, cfg: DictConfig
) -> float:
    # Copy the configuration.
    cfg_cp = cfg.copy()
    cfg_cp.trainer.load_ckpt = True
    # Search space.
    target_value = trial.suggest_float("target_value", 0.5, 1.2)
    guide_scale = trial.suggest_float("guide_scale", 0.5, 6.0)
    cfg_cp.diffusion_model.guide_scale = guide_scale
    cfg_cp.guides.trajectory_return.cfg_target_value = target_value
    # Run the inference.
    trainer, dataset, renderer = instantiate_config(cfg_cp)
    episodes = trainer.rollout_episodes(64)
    returns = [np.sum(ep["rewards"]) for ep in episodes]
    return float(np.mean(returns))


def main(mode, cfg: DictConfig) -> None:
    """Run the specified mode ('train', 'test', or 'optim') with the given configuration.

    Args:
        mode (str): The operation mode to run ('train', 'test', or 'optim').
        cfg (DictConfig): The configuration object for the experiment.

    """
    if mode == "train":
        train(cfg)
    elif mode == "test":
        test(cfg)
    elif mode == "optim":
        optim(cfg)


if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment name (e.g., hopper-medium-v2)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "optim"],
        default="train",
    )
    args, overrides = parser.parse_known_args()

    # Load configuration.
    config = load_config(args.env, "config/experiments/classifier_free_guidance", overrides)

    # Check and log configurations if any.
    check_for_existing_config(config, config.run_dir, break_on_differences=False)

    # Save the configuration to the run directory.
    if args.mode == "train":
        save_config(config, config.run_dir)

    # Run the main function with the specified mode and configuration.
    main(args.mode, config)
