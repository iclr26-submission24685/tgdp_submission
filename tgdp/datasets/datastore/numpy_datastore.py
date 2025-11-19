"""Datastore that stores data in numpy arrays."""

import logging
from typing import Dict, Optional, Union, overload

import numpy as np
from tgdp.datasets.datastore.base_datastore import BaseDatastore

logger = logging.getLogger(__name__)


class NumpyDatastore(BaseDatastore):
    """Datastore implementation that stores episodic data in preallocated numpy arrays.

    This class manages storage, retrieval, and manipulation of episode-based data
    using numpy arrays for efficient access and memory usage.
    """

    def __init__(self, max_path_length: int, max_n_episodes: int, float_dtype: np.dtype = np.dtype(np.float32)):
        """Initialize the data store.

        Args:
            max_path_length (int): The maximum length of a path.
            max_n_episodes (int): The maximum number of episodes.
            float_dtype (np.dtype, optional): The numpy data type to use for floating point arrays.

        """
        super().__init__(max_path_length, max_n_episodes)

        # Initialize data store.
        self.data = {}

        # Length of the dataset/first unused episode index.
        self.dataset_length = 0

        # Store the current episode lengths.
        self.episode_lengths = np.zeros(self.max_n_episodes, dtype=np.int32)

        # Store the float dtype.
        self.float_dtype = float_dtype

    def __len__(self):
        """Return the number of episodes currently stored in the datastore.

        Returns:
            int: The number of episodes in the datastore.

        """
        return self.dataset_length

    def keys(self):
        """Return the keys of the data store.

        Returns:
            np.ndarray: The keys of the data store.

        """
        return self.data.keys()

    def items(self):
        """Return the items of the data store.

        Returns:
            np.ndarray: The items of the data store.

        """
        return self.data.items()

    def values(self):
        """Return the values of the data store.

        Returns:
            np.ndarray: The values of the data store.

        """
        return self.data.values()

    def reserve_new_episode_idx(self) -> int:
        """Return the index of the next episode. This returns an ID which can then be used to access the episode data.

        Returns:
            int: The index of the next episode.

        """
        new_index = self.dataset_length
        self.dataset_length += 1
        if new_index >= self.max_n_episodes:
            logger.critical("A new episode index was requested, but the dataset is full. Raising an exception.")
            raise RuntimeError("The dataset is full. Cannot reserve a new episode index.")
        logger.debug(f"New episode index: {new_index}")
        return new_index

    @overload
    def get_episode_length(self, episode_idx: int) -> int: ...
    @overload
    def get_episode_length(self, episode_idx: None = None) -> np.ndarray: ...
    def get_episode_length(self, episode_idx: Optional[int] = None) -> Union[np.ndarray, int]:
        """Get the length of an episode. If episode_idx is None, return the lengths of all episodes.

        Args:
            episode_idx (int): The index of the episode.

        Returns:
            Union[int, np.ndarray]: The length of the episode.

        """
        if episode_idx is None:
            return self.episode_lengths
        return self.episode_lengths[episode_idx]

    def get_field(self, key, flatten: bool = False) -> np.ndarray:
        """Return the field with the given key.

        Args:
            key (str): The key of the field.
            flatten (bool): If True, flatten the field.

        Returns:
            np.ndarray: The field.

        """
        if key not in self.data.keys():
            logger.warning(f"Field {key} not found. Returning empty array.")
            # Return an empty array with shape (0,) and dtype float32 by default
            return np.empty((0,), dtype=self.float_dtype)
        if flatten:
            return np.concatenate([self.data[key][i, :length] for i, length in enumerate(self.episode_lengths)])
        return self.data[key][: self.dataset_length]

    @overload
    def get_episode_data(self, episode_idx: int, step_idx: Union[int, slice], key: str) -> np.ndarray: ...
    @overload
    def get_episode_data(
        self, episode_idx: int, step_idx: Union[int, slice], key: None = None
    ) -> Dict[str, np.ndarray]: ...
    def get_episode_data(
        self, episode_idx: int, step_idx: Union[int, slice] = slice(None), key: Optional[str] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Get data from the data store. If field is None, return all fields.

        Args:
            episode_idx (int): The index of the episode.
            step_idx (int, slice): The index of the step. If slice, return the data for the slice.
            key (str): The key of the field to get.

        Returns:
            Any: The data.

        """
        # If field is None, return data for all fields.
        if key is None:
            episode_slice = {}
            for k in self.data.keys():
                episode_slice[k] = self.data[k][episode_idx, step_idx]
            return episode_slice

        # If field is not None, return the data of the field.
        return self.data[key][episode_idx, step_idx]

    def append_episode_data(self, episode_idx: int, episode_data: Dict) -> None:
        """Append data to the data store for a single episode.

        This method appends data for a single episode to the data store. The episode data is expected to contain at
        least the "observations" field, which is used to determine the length of the episode. The method will also
        update the episode length for the episode and appends the data for each field in the episode data dictionary.
        If a field does not exist in the data store, it will be created with the appropriate shape and data type.

        Args:
            episode_idx (int): The index of the episode.
            episode_data (dict): A dictionary containing the data for the episode, where each key is a field name and
                each value is a numpy array or list of data for that field.

        """
        # Determine the start index.
        start = self.episode_lengths[episode_idx]

        # Update episode length.
        self.episode_lengths[episode_idx] += len(episode_data["observations"])
        assert self.episode_lengths[episode_idx] <= self.max_path_length, (
            f"Path length {self.episode_lengths[episode_idx]} exceeds maximum episode length {self.max_path_length}."
        )

        # For all fields in the episode data.
        for k, v in episode_data.items():
            # Cast to numpy array
            if not isinstance(v, np.ndarray):
                v = np.array(v)

            # Add the field if it doesn't exist.
            if k not in self.data.keys():
                logger.debug(f"Field {k} not found in data store. Adding it.")
                self._add_field(k, v.shape[1:], self.float_dtype if "float" in str(v.dtype) else v.dtype)

            # Check if new data does not exceed the maximum path length.
            if start + len(v) > self.max_path_length:
                logger.warning(
                    f"Setting data for field {k} exceeds max_path_length. Index {start + len(v) - 1} cannot be set in "
                    f"array of length {self.max_path_length}. Truncating the data. Consider increasing max_path_length."
                )

            # Append the data.
            self.data[k][episode_idx, start : start + len(v)] = v

    def set_episode_data(self, episode_idx: int, step_idx: Union[int, slice], key: str, data: np.ndarray) -> None:
        """Set the data for a specific episode and step.

        This method sets the data for a specific episode and step in the data store. This does not update the path
        length. If the field does not exist, it will create it with the appropriate shape and data type.

        Args:
            episode_idx (int): The index of the episode.
            step_idx (int, slice): The index of the step.
            key (str): The field to set.
            data (np.ndarray): The value to set.

        """
        # Add the field if it doesn't exist.
        if key not in self.data.keys():
            logger.debug(f"Field {key} not found in data store. Adding it.")
            self._add_field(key, data.shape[1:], self.float_dtype if "float" in str(data.dtype) else data.dtype)

        # Check if new data does not exceed the maximum path length.
        if isinstance(step_idx, int) or isinstance(step_idx, np.integer):
            highest_index = step_idx
        elif isinstance(step_idx, slice):
            if step_idx.stop is None:
                highest_index = self.max_path_length - 1
            else:
                # If the slice has a stop value, use it to determine the highest index.
                highest_index = step_idx.stop - 1
        if highest_index > self.max_path_length - 1:
            logger.warning(
                f"Setting data for field {key} exceeds max_path_length. Index {highest_index} cannot be set in array "
                f"of length {self.max_path_length}. Truncating the data. Consider increasing max_path_length."
            )

        # Set the field.
        self.data[key][episode_idx, step_idx] = data

    def _add_field(self, key, shape, dtype):
        """Add a field to the data store.

        This method adds a new field to the data store with the specified key, shape, and data type. The field is
        initialized with zeros, and its shape is determined by the maximum number of episodes and the maximum path
        length.

        Args:
            key (str): The key of the field.
            shape (tuple): The shape of the field.
            dtype (np.dtype): The data type of the field.

        """
        self.data[key] = np.zeros((self.max_n_episodes, self.max_path_length, *shape), dtype=dtype)
