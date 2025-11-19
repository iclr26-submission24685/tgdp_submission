"""Base class for a data store. Data stores store episode data in a structured way."""

from typing import Dict, Optional, Union, overload

import numpy as np


class BaseDatastore:
    """Base class for a data store that stores episode data in a structured way."""

    def __init__(self, max_path_length: int, max_n_episodes: int):
        """Initialize the data store.

        Args:
            max_path_length (int): The maximum length of a path.
            max_n_episodes (int): The maximum number of episodes.

        """
        # Dataset dimensions
        self.max_path_length = max_path_length
        self.max_n_episodes = max_n_episodes

    def __len__(self) -> int:
        """Return the length of the data store.

        Returns:
            int: The length of the data store.

        """
        raise NotImplementedError

    def keys(self):
        """Return the keys of the data store.

        Returns:
            np.ndarray: The keys.

        """
        raise NotImplementedError

    def items(self):
        """Return the items of the data store.

        Returns:
            np.ndarray: The items.

        """
        raise NotImplementedError

    def values(self):
        """Return the values of the data store.

        Returns:
            np.ndarray: The values.

        """
        raise NotImplementedError

    def reserve_new_episode_idx(self) -> int:
        """Return the index of the next episode. This is used to add new episodes to the dataset.

        Returns:
            int: The index of the next episode.

        """
        raise NotImplementedError

    def get_field(self, key: str, flatten: bool = False) -> np.ndarray:
        """Return the field with the given key.

        Args:
            key (str): The key of the field.
            flatten (bool): If True, flatten the field.

        Returns:
            np.ndarray: The field.

        """
        raise NotImplementedError

    @overload
    def get_episode_length(self, episode_idx: int) -> int: ...
    @overload
    def get_episode_length(self, episode_idx: None = None) -> np.ndarray: ...
    def get_episode_length(self, episode_idx: Optional[int] = None) -> Union[int, np.ndarray]:
        """Get the length of an episode. If episode_idx is None, return the lengths of all episodes.

        This returns the number of steps in the episode, which is the length of the episode. If we use padding,
        the length is the number of steps before padding occurs, i.e. the number of actually encountered observations.

        Args:
            episode_idx (int): The index of the episode.

        Returns:
            int: The length of the episode.

        """
        raise NotImplementedError

    @overload
    def get_episode_data(self, episode_idx: int, step_idx: Union[int, slice], key: str) -> np.ndarray: ...
    @overload
    def get_episode_data(
        self, episode_idx: int, step_idx: Union[int, slice], key: None = None
    ) -> Dict[str, np.ndarray]: ...
    def get_episode_data(
        self,
        episode_idx: int,
        step_idx: Union[int, slice],
        key: Optional[str] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Get data from the data store.

        Args:
            episode_idx (int): The index of the episode.
            step_idx (int, slice): The index of the step. If slice, return the data for the slice.
            key (str, optional): The field to get. If None, return all fields.

        Returns:
            Any: The data.

        """
        raise NotImplementedError

    def append_episode_data(self, episode_idx: int, episode_data: Dict) -> None:
        """Append data to the data store.

        Args:
            episode_idx (int): The index of the episode.
            episode_data (Dict): The data to append for the episode.

        """
        raise NotImplementedError

    def set_episode_data(self, episode_idx: int, step_idx: Union[int, slice], key: str, data: np.ndarray) -> None:
        """Set the data for a specific episode and step.

        Args:
            episode_idx (int): The index of the episode.
            step_idx (int, slice): The index of the step.
            key (str): The field to set.
            data (np.ndarray): The value to set.

        """
        raise NotImplementedError
