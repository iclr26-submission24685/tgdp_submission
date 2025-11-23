"""D4RL Dataset Class."""

import logging
from typing import Dict, Optional, Type

import d4rl  # noqa: F401
import gym
import gymnasium
import numpy as np
from gym import vector as gym_vector

from ..environments import (
    KITCHEN_ENVS,
    LOCOMOTION_ENVS,
    MAZE_ENVS,
    D4RL2GymnasiumCompatWrapper,
    D4RLMazeWrapper,
    Gymnasium2D4RLKitchenWrapper,
    Gymnasium2D4RLMazeWrapper,
    VectorD4RL2GymnasiumCompatWrapper,
    d4rl_env_mapping,
    kitchen_tasks_to_complete_mapping,
)
from .base_dataset import BaseDataset
from .datastore import BaseDatastore, NumpyDatastore
from .normalizers import BaseNormalizer

logger = logging.getLogger(__name__)

# This can be set to True to use gymnasium environments instead of the original D4RL ones. This may change performance.
DEFAULT_USE_GYMNASIUM = False


class D4RLDataset(BaseDataset):
    """A dataset class that loads data from a d4rl dataset and provides an interface to sample sequences from it.

    This class is a subclass of BaseDataset and provides methods to create environments and load datasets from d4rl.
    It supports both gym and gymnasium environments, and can create single or vectorized environments.
    It also handles the loading of datasets from d4rl environments, converting them into a format compatible with the
    BaseDataset class.

    """

    def make_env(self, env_kwargs: Dict = {}, gymnasium_env: bool = DEFAULT_USE_GYMNASIUM) -> gymnasium.Env:
        """Create and return a single environment.

        Create and return a single environment. Note that D4RL envs already include a TimeLimit wrapper.
        Per default, this will return the original gym environment from d4rl, but if gymnasium_env is True,
        it will return a gymnasium environment.

        Args:
            env_kwargs (Dict): Keyword arguments to pass to the environment.
            gymnasium_env (bool): If True, use gymnasium environments, otherwise use gym environments.

        Returns:
            gymnasium.Env: A single environment.

        """
        # If gymnasium_env is False, we use the original gym environment from d4rl with a compatibility wrapper.
        if not gymnasium_env:
            env = D4RL2GymnasiumCompatWrapper(gym.make(self.env_name, **env_kwargs))  # type: ignore[no-untyped-call]
            if d4rl_env_mapping(self.env_name) in MAZE_ENVS:
                env = D4RLMazeWrapper(env)  # type: ignore[no-untyped-call]
            return env  # type: ignore[no-untyped-call]

        # If gymnasium_env is True, we use the gymnasium environment.
        gymnasium_env_name = d4rl_env_mapping(self.env_name)

        # Maze2D environments.
        if gymnasium_env_name in MAZE_ENVS:
            env = Gymnasium2D4RLMazeWrapper(
                gymnasium.make(
                    gymnasium_env_name, max_episode_steps=self.max_episode_length, continuing_task=False, **env_kwargs
                )
            )

        # Franka Kitchen environments.
        elif gymnasium_env_name in KITCHEN_ENVS:
            env = Gymnasium2D4RLKitchenWrapper(
                gymnasium.make(
                    gymnasium_env_name,
                    max_episode_steps=self.max_episode_length,
                    tasks_to_complete=kitchen_tasks_to_complete_mapping(self.env_name),
                    **env_kwargs,
                )
            )

        # MuJoCo Locomotion environments.
        elif gymnasium_env_name in LOCOMOTION_ENVS:
            env = gymnasium.make(gymnasium_env_name, max_episode_steps=self.max_episode_length)

        # If the environment is not recognized, raise an error.
        else:
            raise ValueError(
                f"Gymnasium environment {gymnasium_env_name} not implemented. Please use a supported environment:\
                             {MAZE_ENVS + KITCHEN_ENVS + LOCOMOTION_ENVS} or use the D4RL gym version."
            )

        return env

    def make_vector_env(
        self, num_envs: int, env_kwargs: Dict = {}, gymnasium_env: bool = DEFAULT_USE_GYMNASIUM
    ) -> gymnasium.vector.VectorEnv:
        """Create and return a vector env with num_envs sub-environments.

        Create and return a vector env with num_envs sub-environments. Note that D4RL envs already include a TimeLimit
        wrapper. If gymnasium_env is True, it will return a gymnasium environment, otherwise it will return a gym
        environment.

        Args:
            num_envs (int): The number of sub-environments.
            env_kwargs (Dict): Keyword arguments to pass to the environment.
            gymnasium_env (bool): If True, use gymnasium environments, otherwise use gym environments.

        Returns:
            gymnasium.vector.VectorEnv: A vector environment with num_envs sub-environments.

        """
        # If gymnasium_env is False, we use the original gym environment from d4rl with a compatibility wrapper.
        if not gymnasium_env:
            env = VectorD4RL2GymnasiumCompatWrapper(
                gym_vector.make(
                    self.env_name,
                    num_envs=num_envs,
                    **env_kwargs,
                )
            )  # type: ignore[no-untyped-call], ignore missmatch of gym and gymnasium types
            if self.env_name in MAZE_ENVS:
                env = D4RLMazeWrapper(env)  # type: ignore[no-untyped-call], ignore missmatch of gym and gymnasium types
            return env  # type: ignore[no-untyped-call], ignore missmatch of gym and gymnasium types

        # If gymnasium_env is True, we use the gymnasium environment.
        gymnasium_env_name = d4rl_env_mapping(self.env_name)

        # Maze2D environments.
        if gymnasium_env_name in MAZE_ENVS:
            env = gymnasium.make_vec(
                gymnasium_env_name,
                num_envs=num_envs,
                wrappers=[
                    lambda env: gymnasium.wrappers.Autoreset(env),
                    lambda env: Gymnasium2D4RLMazeWrapper(env),
                ],
                continuing_task=False,
                **env_kwargs,
            )

        # Franka Kitchen environments.
        elif gymnasium_env_name in KITCHEN_ENVS:
            env = gymnasium.make_vec(
                gymnasium_env_name,
                num_envs=num_envs,
                wrappers=[
                    lambda env: gymnasium.wrappers.Autoreset(env),
                    lambda env: Gymnasium2D4RLKitchenWrapper(env),
                ],
                tasks_to_complete=kitchen_tasks_to_complete_mapping(self.env_name),
                **env_kwargs,
            )

        # MuJoCo Locomotion environments.
        elif gymnasium_env_name in LOCOMOTION_ENVS:
            env = gymnasium.make_vec(
                gymnasium_env_name,
                num_envs=num_envs,
                wrappers=[
                    lambda env: gymnasium.wrappers.Autoreset(env),
                ],
                **env_kwargs,
            )
        else:
            raise ValueError(
                f"Gymnasium environment {gymnasium_env_name} not implemented. Please use a supported environment:\
                             {MAZE_ENVS + KITCHEN_ENVS + LOCOMOTION_ENVS} or use the D4RL gym version."
            )

        return env

    def _load_dataset(
        self,
        env_name: str,
    ) -> None:
        """Load the dataset from a d4rl environment.

        This loads the dataset from a D4RL environment. This is different from the BaseDataset method, since we need
        to load D4RL using gym, not gymnasium.

        Args:
            env_name (str): The name of the d4rl environment to load.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the dataset.

        """
        env = gym.make(env_name)
        dataset = env.get_dataset()  # type: ignore[no-untyped-call]

        # Every Dataset should contain the following fields:
        keys = [
            ("observations", "observations"),
            ("actions", "actions"),
            ("terminals", "terminations"),
            ("timeouts", "truncations"),
            ("rewards", "rewards"),
        ]

        self._append_data_from_dict(
            {key: dataset[original_key] for original_key, key in keys},
            update_normalizer_statistics=False,
        )


class D4RLMazeDataset(D4RLDataset):
    """A dataset class for D4RL Maze environments.

    This class is a subclass of D4RLDataset and provides methods to create environments and load datasets from d4rl
    Maze environments. We largely follow the dataset preprocessing from Diffusion Veteran (Lu et al., 2025).

    """

    def __init__(
        self,
        # Environment name.
        env_name: str,
        # Normalizers.
        normalizers: Dict[str, BaseNormalizer],
        # Datastore.
        datastore: Type[BaseDatastore] = NumpyDatastore,
        max_n_episodes: int = 10000,
        max_episode_length: int = 1000,
        # Reward/MC-return parameters.
        dv_data_augmentation: bool = True,
        iql: bool = False,
        non_finishing_penalty: float = 0.0,
        compute_mc_return: bool = False,
        discount: Optional[float] = 0.99,
        repeat_rewards_at_terminations: bool = False,
        # Sampling parameters.
        horizon: int = 1,
        jump_step_stride: int = 1,
        padding_terminations: Optional[str] = None,  # [None, 'zero', 'last']
        padding_truncations: Optional[str] = None,  # [None, 'zero', 'last']
    ):
        """Initialize the D4RL Maze Dataset.

        Args:
            env_name (str): The name of the environment to load.
            normalizers (Dict[str, BaseNormalizer]): A dictionary of normalizers for the dataset fields.
            datastore (BaseDatastore): The datastore to use.
            max_n_episodes (int): The maximum number of episodes to store in the dataset.
            max_episode_length (int): The maximum length of an episode.
            dv_data_augmentation (bool): If True, use data augmentation as in Diffusion Veteran.
            include_truncated_episodes (bool): If True, include episodes that were truncated (timed out) when
                preprocessing and constructing datasets.
            iql (bool): If True, apply IQL-style reward adjustments or preprocessing when preparing the dataset.
            non_finishing_penalty (float): The penalty to apply to the last step of a non-finishing episode.
            compute_mc_return (bool): If True, compute Monte Carlo returns for each episode.
                This is only possible if episodes contain rewards.
            discount (float): The discount factor for the Monte Carlo returns.
            goal_sampling_strategy (Optional[Union[Dict[str, float], str]]): The goal sampling strategy to use if
                we sample goals from the observations at runtime.
                Can be None, a single strategy, or a mixture of strategies. If not None, a goal is sampled from the
                observations. There still might be a goal included in the data.
                Strategies are: 'final' (final observation of sample), 'sample' (random observation from sample),
                'dataset' (random observation from dataset), 'episode' (random observation from episode).
            observation_normalize_goal (bool): If True, normalize the goal, using the observation normalizer.
            goal_indices (Optional[List[int]]): The indices of the goal in the observation. If None, the goal is assumed
                to be the full observation.
            horizon (int): The number of steps to include in each sequence.
            jump_step_stride (int): The stride to use when sampling sequences from the dataset.
            sample_unfinished_episodes (bool): If True, sample sequences that are not yet finished.
            padding_terminations (Optional[str]): The padding mode to use for terminated sequences. Can be None, 'zero',
                or 'last'.
            padding_truncations (Optional[str]): The padding mode to use for truncated sequences. Can be None, 'zero',
                or 'last'.
            repeat_rewards_at_terminations (bool): If True, repeat the last reward at the termination up until the
                maximum episode length. This effectively implements a terminal bonus/penalty in the mc returns. (it is
                used in the cleandiffuser implementations of kitchen and maze2d).
            include_terminal_states (bool): If True, include the terminal states in the sequences.
            run_dir (Optional[str]): The directory to save new episodes to and load episodes from. If None, no episodes
                are saved to disk.
            load_episodes_from_previous_run (bool): If True, load episodes from a previous run from disk.
            seed (Optional[int]): The random seed to use.

        """
        self.dv_data_augmentation = dv_data_augmentation
        self.non_finishing_penalty = non_finishing_penalty
        self.iql = iql
        super().__init__(
            env_name=env_name,
            normalizers=normalizers,
            datastore=datastore,
            max_n_episodes=max_n_episodes,
            max_episode_length=max_episode_length,
            terminal_penalty=0.0,  # No terminal penalty for Maze environments.
            compute_mc_return=compute_mc_return,
            discount=discount,
            horizon=horizon,
            jump_step_stride=jump_step_stride,
            padding_terminations=padding_terminations,
            padding_truncations=padding_truncations,
            repeat_rewards_at_terminations=repeat_rewards_at_terminations,
        )

    def _load_dataset(
        self,
        env_name: str,
    ) -> None:
        """Load the dataset from a d4rl environment.

        This loads the dataset from a D4RL environment. This is different from the BaseDataset method, since we need to
        load D4RL using gym, not gymnasium.

        Args:
            env_name (str): The name of the d4rl environment to load.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the dataset.

        """
        env = gym.make(env_name)
        dataset = env.get_dataset()  # type: ignore[no-untyped-call]

        # Every Dataset should contain at least the following fields:
        keys = [
            ("observations", "observations"),
            ("actions", "actions"),
            ("terminals", "terminations"),
            ("timeouts", "truncations"),
        ]
        # In addition, we check for the presence of the following fields:
        if "rewards" in dataset:
            keys.append(("rewards", "rewards"))
        if "infos/goal" in dataset:
            keys.append(("infos/goal", "goals"))
        dataset = {key: dataset[original_key] for original_key, key in keys}

        # Preprocess the dataset depending on the environment type. This reproduces the dataset from Lu et al.
        if d4rl_env_mapping(self.env_name) in MAZE_ENVS and self.dv_data_augmentation:
            dataset = self._shorten_successful_episodes_maze2d(
                dataset, max_episode_length=self.max_episode_length, subtract_rewards_per_step=1.0 * float(self.iql)
            )
        elif self.dv_data_augmentation:
            raise ValueError(f"Environment {self.env_name} not supported for D4RLMazeDataset.")

        # Append the new dataset to the datastore.
        self._append_data_from_dict(dataset, update_normalizer_statistics=False)

    def _shorten_successful_episodes_maze2d(
        self,
        dataset: Dict[str, np.ndarray],
        subtract_rewards_per_step: float = 0.0,
        max_episode_length: int = 800,
    ) -> Dict[str, np.ndarray]:
        """Shorten episodes so that they are no longer than max_episode_length.

        This is a helper function to shorten episodes that are longer than max_episode_length by removing the steps at
        the beginning of the episode until the episode is no longer than max_episode_length. This can also reassign the
        rewards by subtracting values for all steps, which can implement IQL. This is aligned with the dataset
        preprocessing of the D4RL Maze2D environments in the CleanDiffuser paper.

        Args:
            dataset (Dict[str, np.ndarray]): The dataset to preprocess.
            subtract_rewards_per_step (float): The value to subtract from each reward at each step. This can be used to
                encourage shorter paths to the goal. CleanDiffuser uses 1.0 for Maze2D (to implement IQL).
            max_episode_length (int): The maximum length of an episode.

        Returns:
            Dict[str, np.ndarray]: The preprocessed dataset with shortened episodes.

        """
        new_dataset = {key: [] for key in dataset.keys()}
        end = -1
        i = len(dataset["rewards"]) - 1
        while i >= 0:
            if (
                (dataset["rewards"][i - 1] == 1.0 and dataset["rewards"][i] == 0.0)  # Reward goes from 1->0,
                or end - i + 1 >= max_episode_length  # Max episode length exceeded
                or i == 0  # Start of the dataset.
            ):
                # We found the start of an episode.
                start = i
                if end > start:
                    for key in dataset.keys():
                        if key == "rewards":
                            new_dataset["rewards"].extend(
                                dataset["rewards"][start : end + 1] - subtract_rewards_per_step
                            )
                        elif key == "terminations":
                            new_dataset[key].extend([False] * (end - start))  # Set other termination signals to False.
                            new_dataset[key].append(True)  # Set the final termination signal to True.
                        elif key == "truncations":
                            new_dataset[key].extend([False] * (end - start + 1))  # Set all truncations to False.
                        else:
                            new_dataset[key].extend(dataset[key][start : end + 1])
                if end - i > max_episode_length:
                    # If the episode is longer than max_episode_length, we skip the rest of the episode.
                    while i > 0:
                        if dataset["rewards"][i - 1] == 1.0 and dataset["rewards"][i] == 0.0:
                            # If the reward steps from 1->0, we found the start of the original episode.
                            break
                        i -= 1
                end = -1  # Do not add data until a new end is found.

            elif dataset["rewards"][i - 1] == 0.0 and dataset["rewards"][i] == 1.0:
                # If the reward steps from 0->1, we found the end of an episode.
                end = i

            i -= 1

        # Cast to numpy arrays and return.
        return {key: np.array(value, dtype=np.float32) for key, value in new_dataset.items()}
