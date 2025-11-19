"""Trainer class for training an agent on a dataset."""

import logging
import sys
from collections import OrderedDict
from os import path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.utils
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger as LightningLogger

# import lightning.pytorch.callbacks as plc
from omegaconf.dictconfig import DictConfig
from tensordict import TensorDict
from tqdm import tqdm

from ..agent.base_agent import BaseAgent
from ..datasets import BaseDataset
from ..rendering.base_renderer import BaseRenderer
from .base_trainer import BaseTrainer
from .callbacks import (
    BasicStatsMonitorCallback,
    GradientNormLoggerCallback,
    ModelSummaryCallback,
    SimpleTQDMProgressBarCallback,
    TestAgentCallback,
    UpdateEMACallback,
    WeightHistogramLoggerCallback,
)

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """Trainer class for training an agent on a dataset."""

    def __init__(
        self,
        # Experiment name
        experiment_name: str,
        # Components
        agent: BaseAgent,
        dataset: BaseDataset,
        renderer: BaseRenderer,
        # Training parameters
        train_batch_size: int = 128,
        n_train_epochs: int = -1,
        n_train_steps_per_epoch: int = 1000,
        accumulate_grad_batches: int = 1,
        # SWA and clipping parameters
        gradient_clip_algorithm: Optional[str] = None,  # [None, "norm", "value"]
        gradient_clip_value: Optional[float] = None,  # Only used if gradient_clip_algorithm is "value"
        # EMA parameters
        ema_decay: float = 0.99,
        step_start_ema: int = 1000,
        update_ema_every: int = 10,
        # Testing parameters
        n_test_episodes: int = 16,
        test_every_n_epochs: Optional[int] = None,
        start_test_step: int = 0,
        # Loading of previous runs
        load_ckpt: bool = True,
        resume_training: bool = False,
        load_ckpt_folder: Optional[str] = None,
        load_ckpt_step: Union[int, str, dict] = "latest",
        # Checkpointing parameters
        save_freq: int = 1000,
        label_freq: int = 100000,
        results_dir: str = "results/",
        # Additional logging parameters.
        lightning_logger: Optional[LightningLogger] = None,
        log_device_stats: bool = False,
        log_dataset_stats: bool = False,
        log_gradient_norms: bool = False,
        log_weight_histograms: bool = False,
        log_learning_rate: bool = False,
        profiler: Optional[str] = None,
        device: Union[torch.device, str] = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the trainer.

        Args:
            experiment_name (str): The name of the experiment.
            agent (BaseAgent): The agent to train.
            dataset (BaseDataset): The dataset to train on.
            renderer (BaseRenderer): The renderer to use for rendering.
            train_batch_size (int): The batch size for training.
            n_train_epochs (int): The number of training epochs. If -1, the trainer will train indefinitely.
            n_train_steps_per_epoch (int): The number of training steps per epoch.
            accumulate_grad_batches (int): The number of batches to accumulate gradients over.
            gradient_clip_algorithm (Optional[str]): The algorithm to use for gradient clipping. [None, "norm", "value"]
            gradient_clip_value (Optional[float]): The value to use for gradient clipping if gradient_clip_algorithm is
                "value".
            ema_decay (float): The decay value for the EMA.
            step_start_ema (int): The step at which to start the EMA.
            update_ema_every (int): The number of steps between EMA updates.
            n_test_episodes (int): The number of episodes to test the agent on.
            test_every_n_epochs (Optional[int]): The number of epochs between testing the agent. If None, the agent is
                not tested. Note that we already get test data when we do data collection.
            start_test_step (int): The step at which to start testing the agent.
            load_ckpt (bool): Whether to load a checkpoint.
            resume_training (bool): Whether to resume training from the checkpoint. If False, the trainer will start
                from scratch; the model can still be loaded.
            load_ckpt_folder (Optional[str]): The folder of the ckpt to load. If None, the results_dir/experiment_name
                folder is used.
            load_ckpt_step (Optional[Union[int, str]]): The step of the checkpoint to load. If 'latest', the latest
                checkpoint is loaded.
            save_freq (int): The frequency at which to save the model.
            label_freq (int): The frequency at which to save the model without overwriting.
            lightning_logger (Optional[LightningLogger]): The logger to use for logging
            results_dir (str): The directory to save the results to.
            log_device_stats (bool): Whether to log device statistics.
            log_dataset_stats (bool): Whether to log dataset statistics.
            log_gradient_norms (bool): Whether to log gradient norms.
            log_weight_histograms (bool): Whether to log weight histograms.
            log_learning_rate (bool): Whether to log the learning rate.
            profiler (Optional[BaseProfiler]): The profiler to use for performance profiling.
            device (torch.device): The device to use for training.
            dtype (torch.dtype): The data type to use for training.

        """
        self.dataset = dataset
        self.renderer = renderer
        self.agent = agent

        # Loading checkpoint.
        def _load_weights(ckpt_path: str, model: torch.nn.Module, key: Optional[str] = None):
            """Load the weights from a checkpoint."""
            key_msg = f"for key {key} " if key else ""
            logger.debug(f"Loading weights {key_msg}from {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=str(device), weights_only=False)
                state_dict = checkpoint["state_dict"]
                if key is not None:
                    state_dict = OrderedDict({k: v for k, v in state_dict.items() if k.startswith(key)})
                model.load_state_dict(state_dict, strict=key is None)
            except FileNotFoundError:
                logger.critical(f"The checkpoint at {ckpt_path} could not be loaded. Abborting run.")
                sys.exit(1)
            logger.info(f"Loaded checkpoint from {ckpt_path} {key_msg}at step {checkpoint['global_step']}.")

        # If we want to load a checkpoint, we need to specify the folder and the step.
        self.resume_next_fit_from = None
        if load_ckpt:
            # If no ckpt folder is specified, we use the one from a previous run.
            if load_ckpt_folder is None:
                load_ckpt_folder = path.join(results_dir, experiment_name)
            # If we want to load a specific checkpoint, we need to specify the step else we load the latest one.
            if load_ckpt_step == "latest" or load_ckpt_step is None or isinstance(load_ckpt_step, DictConfig):
                ckpt_file = "model-latest.ckpt"
            else:
                ckpt_file = f"model-step={load_ckpt_step}.ckpt"
            ckpt_path = path.join(load_ckpt_folder, "checkpoints", ckpt_file)
            # Try loading the checkpoint from the specified location.
            _load_weights(ckpt_path, self.agent)
            # If we have specified different steps of different parts of the model, we need to load them separately.
            if isinstance(load_ckpt_step, DictConfig):
                for k, v in load_ckpt_step.items():
                    key_ckpt_path = path.join(load_ckpt_folder, "checkpoints", f"model-step={v}.ckpt")
                    _load_weights(key_ckpt_path, self.agent, key=str(k))
            # We might also want to recover the state of optimizers and the global step.
            if resume_training:
                self.resume_next_fit_from = ckpt_path
        else:
            logger.info("No checkpoint loaded. Starting from scratch.")
            if resume_training:
                logger.warning(
                    "Resume training is set to True but no checkpoint is loaded. The trainer will start from scratch."
                )

        # Put the agent on the device
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.agent.to(device=self.device, dtype=self.dtype)

        # Compute the number of batches per epoch.
        n_batches_per_epoch = n_train_steps_per_epoch * accumulate_grad_batches
        n_samples_per_epoch = n_batches_per_epoch * train_batch_size

        # Dataloaders.
        sampler = torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=n_samples_per_epoch)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            num_workers=8,
            pin_memory=self.device == torch.device("cuda"),
            collate_fn=TensorDict.stack,
        )

        # Initialization of the lighting trainer. Here we specify all Callbacks.
        super().__init__(
            accelerator="cpu" if self.device == torch.device("cpu") else "gpu",
            devices=1,
            logger=lightning_logger,
            # Has to be set to true. We use our own checkpoints but Lighting is not very consistent with their API here.
            enable_checkpointing=True,
            # Has to be set to false. We use our own progress bar.
            enable_progress_bar=False,
            # Has to be set to false. We use our own model summary.
            enable_model_summary=False,
            # We accumulate gradients over multiple batches.
            limit_train_batches=n_batches_per_epoch,
            max_epochs=n_train_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_value,
            gradient_clip_algorithm=gradient_clip_algorithm,
            profiler=profiler,
            callbacks=[
                # Logging/Monitoring callbacks
                SimpleTQDMProgressBarCallback(),
                ModelSummaryCallback(depth=2, report_zero_param=False),
                *([LearningRateMonitor(logging_interval="epoch")] if log_learning_rate else []),
                *([BasicStatsMonitorCallback(log_every_n_steps=1000)] if log_device_stats else []),
                *(
                    [
                        TestAgentCallback(
                            num_episodes=n_test_episodes,
                            test_every_n_epochs=test_every_n_epochs,
                            start_test_step=start_test_step,
                        )
                    ]
                    if n_test_episodes > 0 and test_every_n_epochs is not None
                    else []
                ),
                *([GradientNormLoggerCallback(log_every_n_steps=1000, depth=4)] if log_gradient_norms else []),
                *([WeightHistogramLoggerCallback(log_every_n_steps=5000, depth=4)] if log_weight_histograms else []),
                # Functional callbacks
                ## Update EMA
                UpdateEMACallback(
                    ema_decay=ema_decay, step_start_ema=step_start_ema, update_ema_every=update_ema_every
                ),
                ## Checkpointing the latest model
                *(
                    [
                        ModelCheckpoint(
                            dirpath=f"{results_dir}/{experiment_name}/checkpoints",
                            filename="model-latest",
                            save_top_k=1,
                            every_n_epochs=save_freq // n_train_steps_per_epoch,
                            enable_version_counter=False,
                        )
                    ]
                    if save_freq > 0
                    else []
                ),
                ## Checkpointing without overwriting
                *(
                    [
                        ModelCheckpoint(
                            dirpath=f"{results_dir}/{experiment_name}/checkpoints",
                            filename="model-{step}",
                            save_top_k=-1,  # Keep checkpoints
                            every_n_epochs=label_freq // n_train_steps_per_epoch,
                            enable_version_counter=False,
                        )
                    ]
                    if label_freq is not None
                    else []
                ),
            ],
        )

    def train(self):
        """Train the agent."""
        # Set model to training mode
        self.agent.train()

        # The ckpt_path tells the trainer to resume training from the checkpoint.
        # If the ckpt_path is None, the trainer will start from scratch (we still might load model parameters in init).
        # If we use this in a notebook, we only want to load the state once so we set self.resume_training_from to None
        # after the first training loop.
        resume_next_fit_from = self.resume_next_fit_from
        self.resume_next_fit_from = None
        super().fit(
            self.agent,
            self.dataloader,
            ckpt_path=resume_next_fit_from,
        )

    def rollout_episodes(
        self,
        n_episodes: int = 1,
        max_episode_length: Optional[int] = None,
        return_plans: bool = False,
        env_kwargs: Dict[str, Any] = {},
    ) -> List[Dict[str, Any]]:
        """Rollout episodes using the agent's ema policy.

        Args:
            n_episodes (int, optional): The number of episodes to rollout. Defaults to 1.
            max_episode_length (int, optional): The maximum length of the episodes. If None, we use the value from the
                dataset. Defaults to None.
            return_plans (bool, optional): Whether to return the plans. Defaults to False.
            env_kwargs (Dict[str, Any], optional): Additional keyword arguments to pass to the environment constructor.

        Returns:
            TensorDict(str, Any): The collected episodes.

        """
        # Set the agent to evaluation mode
        self.agent.eval()

        # If no max_episode_length is specified, we use the dataset's default.
        if max_episode_length is None:
            max_episode_length = self.dataset.max_episode_length

        logger.debug(f"Rolling out {n_episodes} episodes with a maximum length of {max_episode_length} steps.")

        # Initialize envs.
        envs = [self.dataset.make_env(env_kwargs=env_kwargs) for _ in range(n_episodes)]

        # Initialize the episode dicts.
        obs_shape = envs[0].observation_space.shape
        act_shape = envs[0].action_space.shape
        if obs_shape is None:
            raise ValueError("Environment(s) not properly initialized or observation_space.shape is None.")
        if act_shape is None:
            raise ValueError("Environment(s) not properly initialized or action_space.shape is None.")
        # Observations tensor is one step longer as we consider the observation in the goal state as final observation
        episodes = [
            {
                "observations": np.zeros((max_episode_length + 1, *obs_shape), dtype=np.float32),
                "next_observations": np.zeros((max_episode_length, *obs_shape), dtype=np.float32),
                "actions": np.zeros((max_episode_length, *act_shape), dtype=np.float32),
                "rewards": np.zeros((max_episode_length, 1), dtype=np.float32),
                "terminals": np.zeros((max_episode_length, 1), dtype=bool),
                "truncateds": np.zeros((max_episode_length, 1), dtype=bool),
                "infos": [{} for _ in range(max_episode_length)],
            }
            for _ in range(n_episodes)
        ]

        # If we are recording plans, we also store the plans and the steps taken
        if return_plans:
            for eps in episodes:
                eps["plans"] = [{} for _ in range(max_episode_length)]

        # Keep track of which episodes are still running.
        episodes_running = [True] * n_episodes

        # Reset all environments and collect the initial observations.
        observations = []
        infos = []
        for env in envs:
            obs, inf = env.reset()
            observations.append(obs)
            infos.append(inf)
        observations = np.stack(observations)
        rewards = np.zeros(n_episodes)

        # Extract goals if they are present in the infos.
        if "goal" in infos[0]:
            goal = np.stack([inf["goal"] for inf in infos])
            goal = self.dataset.normalize(goal, "goal")
        else:
            goal = None

        for step in tqdm(
            range(max_episode_length),
            desc=f"Rolling out {n_episodes} episodes",
            leave=False,
            bar_format="{desc:<20}{percentage:3.0f}%|{bar}{r_bar}",
        ):
            logger.debug(f"Rollout step {step}")

            # Since we only choose actions for environments that are not done, we need a mapping from the environment
            # index to the action index
            mapping = np.arange(n_episodes)[episodes_running]

            # the agent works in normalized space, we store non-normalized observations and actions
            observations_normed = self.dataset.normalize(observations, "observations")
            actions_normed = (
                self.agent.act(
                    observation=torch.from_numpy(observations_normed).to(device=self.device, dtype=self.dtype),
                )
                .cpu()
                .numpy()
            )
            actions = self.dataset.unnormalize(actions_normed, "actions")

            # Environment step for all environments
            next_observations, rewards, terminals, truncateds, infos = map(
                np.array, zip(*[envs[mapping[i]].step(actions[i]) for i in range(len(observations))])
            )

            # Update the episode data for each environment
            for i in range(len(observations)):
                episodes[mapping[i]]["observations"][step] = observations[i]
                episodes[mapping[i]]["next_observations"][step] = next_observations[i]
                episodes[mapping[i]]["actions"][step] = actions[i]
                episodes[mapping[i]]["rewards"][step] = rewards[i]
                episodes[mapping[i]]["terminals"][step] = terminals[i]
                episodes[mapping[i]]["truncateds"][step] = truncateds[i]
                episodes[mapping[i]]["infos"][step] = infos[i]

                # Store the plan if we are recording them.
                if return_plans:
                    plan = self.agent.get_plan(i).cpu().numpy()
                    for k, v in plan.items():
                        if k in self.dataset.normalizers:
                            v = self.dataset.unnormalize(v, k)
                        episodes[mapping[i]]["plans"][step][k] = v

                # If an episode is done, we finalize it
                if terminals[i] or truncateds[i]:
                    logger.debug(f"Episode {mapping[i]} is done.")
                    episodes_running[mapping[i]] = False
                    episodes[mapping[i]]["observations"][step + 1] = next_observations[i]
                    for k, v in episodes[mapping[i]].items():
                        if k == "observations":
                            episodes[mapping[i]][k] = v[: step + 2]
                        else:
                            episodes[mapping[i]][k] = v[: step + 1]

            # Delete finished episodes from the agent.
            if any(terminals) or any(truncateds):
                self.agent.delete_episode(idx=np.logical_or(terminals, truncateds).nonzero()[0])

            # If all episodes are done, we break the loop.
            if not np.any(episodes_running):
                self.agent.delete_episode()
                break

            # In the next step, we only feed back observations from environments that are not done.
            observations = np.stack(
                [next_observations[i] for i in range(len(observations)) if not terminals[i] and not truncateds[i]]
            )
            if goal is not None:
                goal = np.stack([goal[i] for i in range(len(observations))])

        # Reset the agent and unstash the episodes.
        self.agent.delete_episode()

        return episodes
