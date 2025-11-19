"""Base class for the Dataset classes."""

import logging
from typing import Dict, Optional, Type

import gymnasium
import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

from .datastore import BaseDatastore, NumpyDatastore
from .normalizers import BaseNormalizer

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """A dataset class that loads data from a dataset and provides an interface to sample sequences from it.

    A dataset should contain at least the following fields:
        - observations: The observations of the environment.
        - actions: The actions taken by the agent.
        - terminations: A boolean array indicating whether the episode was terminated at each step.
        - truncations: A boolean array indicating whether the episode was truncated at each step.
    Optional fields are:
        - rewards: The rewards received by the agent at each step (optional).
        - mc_returns: The Monte Carlo returns for each step (optional, computed if compute_mc_return is True).
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
        terminal_penalty: float = 0.0,
        compute_mc_return: bool = False,
        discount: Optional[float] = 0.99,
        repeat_rewards_at_terminations: bool = False,
        # Sampling parameters.
        horizon: int = 1,
        jump_step_stride: int = 1,
        padding_terminations: Optional[str] = None,  # [None, 'zero', 'last']
        padding_truncations: Optional[str] = None,  # [None, 'zero', 'last']
    ):
        """Initialize the BaseDataset.

        Args:
            env_name (str): The name of the environment to load.
            normalizers (Dict[str, BaseNormalizer]): A dictionary of normalizers for the dataset fields.
            datastore (BaseDatastore): The datastore to use.
            max_n_episodes (int): The maximum number of episodes to store in the dataset.
            max_episode_length (int): The maximum length of an episode.
            terminal_penalty (float): The penalty to apply to the last step of a terminated episode.
            compute_mc_return (bool): If True, compute Monte Carlo returns for each episode.
                This is only possible if episodes contain rewards.
            discount (float): The discount factor for the Monte Carlo returns.
            repeat_rewards_at_terminations (bool): If True, repeat the last reward at the termination up until the
                maximum episode length. This effectively implements a terminal bonus/penalty in the mc returns. (it is
                used in the cleandiffuser implementations of kitchen and maze2d).
            horizon (int): The number of steps to include in each sequence.
            jump_step_stride (int): The stride for the jump step sampling. If n, only every n-th step is sampled.
                The padding is adjusted accordingly.
            padding_terminations (Optional[str]): The padding mode to use for terminated sequences. Can be None, 'zero',
                or 'last'.
            padding_truncations (Optional[str]): The padding mode to use for truncated sequences. Can be None, 'zero',
                or 'last'.

        """
        super().__init__()
        # Check the input parameters.
        assert padding_terminations in [None, "zero", "last"], f"Invalid padding mode: {padding_terminations}"
        assert padding_truncations in [None, "zero", "last"], f"Invalid padding mode: {padding_truncations}"

        # Dataset parameters
        self.env_name = env_name
        self.max_n_episodes = max_n_episodes
        self.max_episode_length = max_episode_length

        # Monte Carlo return parameters
        self.terminal_penalty = terminal_penalty
        self.compute_mc_return = compute_mc_return
        self.discount = discount
        self.repeat_rewards_at_terminations = repeat_rewards_at_terminations

        # Sampling parameters
        self.horizon = horizon
        self.jump_step_stride = jump_step_stride
        self.padding_terminations = padding_terminations
        self.padding_truncations = padding_truncations

        # Indices that map indexes to sequences in memory
        self.indices = []

        # Initialize the datastore
        pad = self.padding_terminations is not None or self.padding_truncations is not None
        max_path_length = self.max_episode_length + ((self.horizon - 1) * self.jump_step_stride) * int(pad)
        self.data = datastore(max_path_length, max_n_episodes)  # Datastore should be partial

        # Load the dataset
        logger.debug(f"Trying to load dataset from environment {env_name}.")
        self._load_dataset(env_name)
        logger.info(f"Dataset loaded. Number of episodes: {len(self.data)}.Number of unique samples: {len(self)}.")

        # Compute statistics for all normalizers. The normalizers are instantiated in the config file.
        self.normalizers = normalizers
        for k in self.normalizers.keys():
            self.normalizers[k].set_statistics(self.data.get_field(k, flatten=True).astype(np.float32))

        # Action and observation dimensions
        self.observation_dim = self.data.get_field("observations").shape[-1]
        self.action_dim = self.data.get_field("actions").shape[-1]

    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Return a sequence from the dataset.

        Args:
            idx (int): The index of the sequence to return. This is an index into the indices list.

        Returns:
            TensorDict: A TensorDict containing the data for the sequence. The keys are the fields of the dataset.

        """
        # Get the data from the datastore.
        path_idx, start, end = self.indices[idx]
        data = self.data.get_episode_data(path_idx, slice(start, end, self.jump_step_stride))

        # Add information about where padding occurs. Padding starts with the last observation.
        data["padding"] = np.zeros((data["observations"].shape[0],), dtype=bool)
        data["padding"][self.data.get_episode_length(path_idx) - start :] = True

        # Normalize the data.
        for k in self.normalizers:
            data[k] = self.normalizers[k].normalize(data[k])

        return TensorDict.from_dict(data, batch_size=torch.Size([]))

    def get_field(self, key: str, flatten: bool = False):
        """Return the field with the given key.

        Args:
            key (str): The key of the field.
            flatten (bool): If True, flatten the field.

        Returns:
            np.ndarray: The field.

        """
        return self.data.get_field(key, flatten)

    def make_env(self, env_kwargs: Dict = {}) -> gymnasium.Env:
        """Create and return a single environment.

        Args:
            env_kwargs (Dict): Additional keyword arguments for the environment.

        Returns:
            gymnasium.Env: A single environment.

        """
        raise NotImplementedError

    def make_vector_env(self, num_envs: int, env_kwargs: Dict = {}) -> gymnasium.vector.VectorEnv:
        """Create and return a vector env with num_envs sub-environments.

        Args:
            num_envs (int): The number of sub-environments.
            env_kwargs (Dict): Additional keyword arguments for the environment.

        Returns:
            gymnasium.vector.VectorEnv: A vector environment with num_envs sub-environments.

        """
        raise NotImplementedError

    def normalize(
        self,
        X: np.ndarray,
        key: str,
        vector: bool = False,
    ) -> np.ndarray:
        """Normalize the data X for the given key.

        Args:
            X (np.ndarray): The data to normalize. (batch_size, [timesteps,] dim)
            key (str): The key of the normalizer to use.
            vector (bool): If True, normalize the data as a vector.

        Returns:
            np.ndarray: The normalized data. (batch_size, [timesteps,] dim)

        """
        # Normalize for the given key.
        if key in self.normalizers.keys():
            return self.normalizers[key].normalize(X, vector=vector)

        # If the key is not found, we return the data unchanged.
        logger.warning(f"Normalizer for key {key} not found. Returning data unchanged.")
        return X

    def unnormalize(
        self,
        X: np.ndarray,
        key: str,
        vector: bool = False,
    ) -> np.ndarray:
        """Unnormalize the data X for the given key.

        Args:
            X (np.ndarray): The data to unnormalize.
            key (str): The key of the normalizer to use.
            vector (bool): If True, normalize the data as a vector.

        Returns:
            np.ndarray: The unnormalized data.

        """
        # Unnormalize for the given key.
        if key in self.normalizers.keys():
            return self.normalizers[key].unnormalize(X, vector=vector)

        # If the key is not found, we return the data unchanged.
        logger.debug(f"Normalizer for key {key} not found. Returning data unchanged.")
        return X

    def get_new_episode_idx(self):
        """Return the index of the next episode. This is used to add new episodes to the dataset.

        Returns:
            int: The index of the next episode.

        """
        return self.data.reserve_new_episode_idx()

    def add_episode(
        self,
        episode_idx: int,
        data: Dict[str, np.ndarray],
    ):
        """Add an episode to the dataset. This is a shorthand to add all data and finalize the episode.

        This method adds a full episode with the given index, finalizes the episode, and updates the
        normalization statistics (if not turned off).

        Args:
            episode_idx (int): The index of the episode to add.
            data (Dict): The data to add. This has to be a full episode, the final step should be terminal or truncated.

        """
        self.append_episode_data(episode_idx, data)
        self._finalize_episode(episode_idx)
        logger.debug(f"Added episode {episode_idx} to the dataset. Length: {self.data.get_episode_length(episode_idx)}")

    def append_episode_data(
        self,
        episode_idx: int,
        data: Dict[str, np.ndarray],
    ):
        """Append data to the episode with the given index.

        This method appends data to an episode for the given index. It checks if the episode is done, and if so,
        it finalizes the episode.

        Args:
            episode_idx (int): The index of the episode to add the data to.
            data (Dict): The data to append.

        """
        # We create a copy of the data to avoid modifying the original data.
        data = data.copy()

        # Add the data to the datastore
        self.data.append_episode_data(episode_idx, data)

    def _add_indices(
        self,
        episode_idx: int,
        min_t_start: int,
        max_t_start: int,
    ):
        """Add indices if they are not yet added.

        This method adds indices for the given episode index and start indices range. It creates tuples of the form
        (episode_idx, start, start + horizon) for each start index in the range [min_t_start, max_t_start].

        Args:
            episode_idx (int): The index of the episode to add indices to.
            min_t_start (int): The minimum start index.
            max_t_start (int): The maximum start index.

        """
        new_indices = [
            (episode_idx, start, start + (self.horizon - 1) * self.jump_step_stride + 1)
            for start in range(min_t_start, max_t_start + 1)
        ]
        self.indices.extend(new_indices)

    def _finalize_episode(self, episode_idx: int):
        """Finalize the episode with the given index.

        This method finalizes the episode by computing the Monte Carlo return, adding padding if necessary, and
        marking the episode as done.

        Args:
            episode_idx (int): The index of the episode to finalize.

        """
        # The last index of the episode that contains data (not padding).
        path_length = self.data.get_episode_length(episode_idx)

        # Terminal and truncation flags.
        terminated = self.data.get_episode_data(episode_idx, path_length - 1, "terminations")
        truncated = self.data.get_episode_data(episode_idx, path_length - 1, "truncations")
        pad_episode = (self.padding_terminations is not None and terminated) or (
            self.padding_truncations is not None and truncated
        )

        # Add padding. We here pad observations according to the mode. Actions are implicitly zero-padded.
        if pad_episode:
            padding_mode = self.padding_terminations if terminated else self.padding_truncations

            if padding_mode == "zero":
                self.data.set_episode_data(
                    episode_idx,
                    slice(path_length, None),
                    "observations",
                    np.array(0.0),
                )

            elif padding_mode == "last":
                self.data.set_episode_data(
                    episode_idx,
                    slice(path_length, None),
                    "observations",
                    self.data.get_episode_data(episode_idx, path_length - 1, "observations"),
                )
            else:
                logger.warning(
                    f"Invalid padding mode {padding_mode} for episode {episode_idx}. "
                    "No padding will be applied to the observations."
                )

            # Rewards are padded according to the padding mode.
            if "rewards" in self.data.keys():
                if self.repeat_rewards_at_terminations:
                    # Repeat the last reward at the padding steps.
                    self.data.set_episode_data(
                        episode_idx,
                        slice(path_length, None),
                        "rewards",
                        self.data.get_episode_data(episode_idx, path_length - 1, "rewards"),
                    )

        # Apply the terminal penalty if the episode is terminated (but not truncated).
        if "rewards" in self.data.keys() and terminated and not truncated and self.terminal_penalty is not None:
            self.data.set_episode_data(
                episode_idx,
                path_length - 1,
                "rewards",
                self.data.get_episode_data(episode_idx, path_length - 1, "rewards") + self.terminal_penalty,
            )

        # Compute Monte Carlo return.
        if self.compute_mc_return and "rewards" in self.data.keys():
            mc_returns = self.data.get_episode_data(episode_idx, slice(0, None), "rewards").copy()
            for i in reversed(range(self.max_episode_length - 1)):  # No reward pad -> mc_returns 0 beyond path_length
                mc_returns[i] += self.discount * mc_returns[i + 1]
            self.data.set_episode_data(episode_idx, slice(0, None), "mc_returns", mc_returns)

        # Add indices.
        min_t_start = 0
        if pad_episode:
            max_t_start = path_length - 1
        else:
            max_t_start = path_length - ((self.horizon - 1) * self.jump_step_stride) - 1
        if min_t_start <= max_t_start:
            self._add_indices(episode_idx, min_t_start, max_t_start)

    def _load_dataset(self, env_name: str) -> None:
        """Load the dataset from the environment.

        Args:
            env_name (str): The name of the environment.

        Returns:
            Dict[str, np.ndarray]: The dataset.

        """
        raise NotImplementedError

    def _append_data_from_dict(self, data: Dict[str, np.ndarray], update_normalizer_statistics: bool = True):
        """Append data from a numpy array to the dataset.

        This method appends data from a dictionary to the dataset. The dictionary should contain the keys
        'observations', 'actions', 'terminations', and 'truncations'. If the key 'rewards' is present, it will also
        be added to the dataset. The data is chunked into episodes based on the termination and truncation flags.
        This method is used for D4RL datasets, which are dictionaries containing the data in numpy arrays.

        Args:
            data (dict): The data to load. Should at least contain the keys 'observations', 'actions', 'terminations',
                'truncations'.
            update_normalizer_statistics (bool): If True, update the normalization statistics.

        """
        # Keys to add to the dataset.
        keys = ["observations", "actions", "terminations", "truncations"]
        if "rewards" in data.keys():
            keys.append("rewards")
        if "goals" in data.keys():
            keys.append("goals")
        assert np.all([k in data for k in keys]), "Data does not contain all necessary fields."

        # Chunk the loaded dataset into episodes.
        n_eps = 0
        i_eps_start = 0
        for i in range(data["truncations"].shape[0]):
            if data["truncations"][i] or data["terminations"][i]:
                self.add_episode(
                    self.get_new_episode_idx(),
                    {k: data[k][i_eps_start : i + 1] for k in keys},
                )
                i_eps_start = i + 1
                n_eps += 1
        logger.debug(f"Loaded {n_eps} episodes.")
