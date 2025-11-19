"""Normalizers for datasets."""

# Adapted from Janner et. al. - Planning with Diffusion for Flexible BEhavior Synthesis
# https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/normalization.py

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class BaseNormalizer(ABC):
    """Abstract base class for normalizers that compute and apply normalization statistics to datasets.

    Subclasses should implement methods to compute statistics, normalize, and unnormalize data.
    """

    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        decay: float = 1.0,
        eps: float = 1e-12,
    ):
        """Initialize the normalization class.

        Args:
            X (Optional[np.ndarray]): Optional initial data array to set statistics.
            decay (float): Decay factor for normalization, must be in the range [0, 1].
            eps (float): Small value to avoid division by zero.

        """
        super().__init__()
        assert decay >= 0 and decay <= 1, "Decay must be in [0, 1]"
        self.decay = decay
        self.eps = eps
        if X is not None:
            self.set_statistics(X)

        # We cache copies of the statistics to avoid repeated fancy indexing.
        self.cache = {}

    def _slice_statistics_from_cache(self, indices) -> Dict[str, np.ndarray]:
        """Get the statistics from the cache.

        Since we repeatedly normalize single datapoints at runtime, and fancy indexing is expensive, we cache the
        statistics for a set of indices. If statistics are cached, we return the cache, if not, we get the statistics
        and add to the cache. When the statistics are updated, the cache is cleared.

        Args:
            indices (List[int]): The indices to get the statistics for.

        Returns:
            Dict[str, np.ndarray]: The cached statistics.

        """
        # If no indices are provided, we return the full statistics, no caching needed.
        if indices is None:
            return self.statistics
        # If the statistics for the indices are not cached, we get them and add to the cache.
        if tuple(indices) not in self.cache:
            self.cache[tuple(indices)] = {k: np.take(v, indices) for k, v in self.statistics.items()}
        return self.cache[tuple(indices)]

    def set_statistics(
        self,
        X: np.ndarray,
    ):
        """Set the statistics of the normalizer to the statistics of the data X.

        Args:
            X (np.ndarray): The data to set the statistics to.

        """
        self.statistics = self.compute_statistics(X)
        self.cache = {}

    def update_statistics(self, X):
        """Update the statistics of the normalizer with new data X.

        Args:
            X (np.ndarray): The new data to update the statistics with.

        """
        if self.decay < 1:
            new_statistics = self.compute_statistics(X)
            # Check that the keys and shapes match.
            assert new_statistics.keys() == self.statistics.keys()
            for k, v in new_statistics.items():
                assert v.shape == self.statistics[k].shape
            # Update the statistics
            for key in self.statistics:
                self.statistics[key] = self.decay * self.statistics[key] + (1 - self.decay) * new_statistics[key]
        # Clear the cache.
        self.cache = {}

    @abstractmethod
    def compute_statistics(self, X) -> Dict[str, np.ndarray]:
        """Compute and return normalization statistics for the given data X.

        Args:
            X (np.ndarray): The data to compute statistics from.

        Returns:
            Dict[str, np.ndarray]: The computed statistics.

        """
        raise NotImplementedError

    @abstractmethod
    def normalize(self, x, vector=False, indices=None):
        """Normalize the data x.

        Args:
            x (np.ndarray): The data to normalize.
            vector (bool): If true, normalize the data as a vector.
            indices (Optional[List[int]]): If we want to normalize only some dimensions, we specify the indices.
                If None, normalize all dimensions.

        Returns:
            np.ndarray: The normalized data.

        """
        raise NotImplementedError()

    @abstractmethod
    def unnormalize(self, x, vector=False, indices=None):
        """Unnormalize the data x.

        Args:
            x (np.ndarray): The data to unnormalize.
            vector (bool): If true, unnormalize the data as a vector.
            indices (Optional[List[int]]): If we want to unnormalize only some dimensions, we specify the indices.
                If None, unnormalize all dimensions.

        Returns:
            np.ndarray: The unnormalized data.

        """
        raise NotImplementedError()


class IdentityNormalizer(BaseNormalizer):
    """Identity normalizer, does not normalize the data."""

    def compute_statistics(self, X) -> Dict[str, np.ndarray]:
        """Compute the statistics of the data X.

        Args:
            X (np.ndarray): The data to compute the statistics of.

        Returns:
            Dict[str, np.ndarray]: The computed statistics.

        """
        return {}

    def normalize(self, x, vector=False, indices=None):
        """Normalize the data x.

        Args:
            x (np.ndarray): The data to normalize.
            vector (bool): If true, normalize the data as a vector.
            indices (Optional[List[int]]): If we want to normalize only some dimensions, we specify the indices.
                If None, normalize all dimensions.

        Returns:
            np.ndarray: The normalized data.

        """
        return x

    def unnormalize(self, x, vector=False, indices=None):
        """Unnormalize the data x.

        Args:
            x (np.ndarray): The data to unnormalize.
            vector (bool): If true, unnormalize the data as a vector.
            indices (Optional[List[int]]): If we want to unnormalize only some dimensions, we specify the indices.
                If None, unnormalize all dimensions.

        Returns:
            np.ndarray: The unnormalized data.

        """
        return x


class GaussianNormalizer(BaseNormalizer):
    """Normalizes to zero mean and unit variance."""

    def compute_statistics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute the mean and standard deviation of the data X.

        This computes the mean and standard deviation of the data X, which are used to normalize the data.
        If the standard deviation is zero for a dimension, it is set to 1 to avoid division by zero.

        Args:
            X (np.ndarray): The data to compute the statistics of.

        Returns:
            Dict[str, np.ndarray]: The computed statistics.

        """
        # We use np.float128 to avoid overflow/underflow issues when computing mean and std.
        means = np.mean(X, axis=0, dtype=np.float128, keepdims=True).astype(X.dtype)
        stds = np.std(X, axis=0, dtype=np.float128, keepdims=True).astype(X.dtype)
        stds[stds == 0.0] = 1.0  # For constant dims, we set the std to 1 to avoid division by zero.
        return {"means": means, "stds": stds}

    def normalize(self, x: np.ndarray, vector: bool = False, indices: Optional[List[int]] = None) -> np.ndarray:
        """Normalize the data x.

        Args:
            x (np.ndarray): The data to normalize.
            vector (bool): If true, normalize the data as a vector.
            indices (Optional[List[int]]): If we want to normalize only some dimensions, we specify the indices.
                If None, normalize all dimensions.

        Returns:
            np.ndarray: The normalized data.

        """
        statistics = self._slice_statistics_from_cache(indices)
        return (x - statistics["means"] * (1 - vector)) / statistics["stds"]

    def unnormalize(self, x: np.ndarray, vector: bool = False, indices: Optional[List[int]] = None) -> np.ndarray:
        """Unnormalize the data x.

        Args:
            x (np.ndarray): The data to unnormalize.
            vector (bool): If true, unnormalize the data as a vector.
            indices (Optional[List[int]]): If we want to unnormalize only some dimensions, we specify the indices.
                If None, unnormalize all dimensions.

        Returns:
            np.ndarray: The unnormalized data.

        """
        statistics = self._slice_statistics_from_cache(indices)
        return x * statistics["stds"] + statistics["means"] * (1 - vector)


class LimitsNormalizer(BaseNormalizer):
    """Normalizes to the range specified range."""

    def __init__(self, min_value: float = -1.0, max_value: float = 1.0, decay: float = 1.0, eps: float = 1e-6):
        """Initialize the normalization class.

        Args:
            min_value (float): The minimum value of the range.
            max_value (float): The maximum value of the range.
            decay (float): Decay factor for normalization, must be in the range [0, 1].
            eps (float): Small value to avoid division by zero.

        """
        super().__init__(decay=decay, eps=eps)
        self.norm_min = min_value
        self.norm_max = max_value

    def compute_statistics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute the min and max of the data X.

        This computes the minimum and maximum of the data X, which are used to normalize the data.
        If the maximum and minimum are equal for a dimension, we set them to -min/2 and -max/2 respectively to avoid
        division by zero.

        Args:
            X (np.ndarray): The data to compute the statistics of.

        Returns:
            Dict[str, np.ndarray]: The computed statistics.

        """
        mins = np.min(X, axis=0, keepdims=True)
        maxs = np.max(X, axis=0, keepdims=True)
        degenerate_dims = np.where(maxs == mins)[0]
        mins[degenerate_dims] = -mins[degenerate_dims] / 2  # Avoid division by zero
        maxs[degenerate_dims] = -maxs[degenerate_dims] / 2
        return {"mins": mins, "maxs": maxs}

    def normalize(self, x: np.ndarray, vector: bool = False, indices: Optional[List[int]] = None) -> np.ndarray:
        """Normalize the data x to the range [-1, 1].

        Args:
            x (np.ndarray): The data to normalize.
            vector (bool): If true, normalize the data as a vector.
            indices (Optional[List[int]]): If we want to normalize only some dimensions, we specify the indices.
                If None, normalize all dimensions.

        Returns:
            np.ndarray: The normalized data.

        """
        statistics = self._slice_statistics_from_cache(indices)
        m = self.norm_max - self.norm_min
        b = statistics["maxs"] * self.norm_min - statistics["mins"] * self.norm_max
        return (m * x + b * (1 - vector)) / (statistics["maxs"] - statistics["mins"])

    def unnormalize(self, x: np.ndarray, vector: bool = False, indices: Optional[List[int]] = None) -> np.ndarray:
        """Unnormalize the data x from the range [-1, 1] to the original range.

        Args:
            x (np.ndarray): The data to unnormalize.
            vector (bool): If true, unnormalize the data as a vector.
            indices (Optional[List[int]]): If we want to unnormalize only some dimensions, we specify the indices.
                If None, unnormalize all dimensions.

        Returns:
            np.ndarray: The unnormalized data.

        """
        statistics = self._slice_statistics_from_cache(indices)
        m = statistics["maxs"] - statistics["mins"]
        b = (statistics["mins"] * self.norm_max - statistics["maxs"] * self.norm_min) / (self.norm_max - self.norm_min)
        return m * x + b * (1 - vector)


class FixedValueNormalizer(BaseNormalizer):
    """Normalizes by a fixed factor."""

    def __init__(self, value: float = 1.0, decay: float = 1.0, eps: float = 1e-6):
        """Initialize the normalization class.

        Args:
            value (float): The value to normalize by.
            decay (float): Decay factor for normalization, must be in the range [0, 1].
            eps (float): Small value to avoid division by zero.

        """
        super().__init__(decay=decay, eps=eps)
        self.value = value

    def compute_statistics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """We don't need to compute any statistics for a fixed value normalizer.

        Args:
            X (np.ndarray): The data to compute the statistics of.

        Returns:
            Dict[str, np.ndarray]: The computed statistics.

        """
        return {}

    def normalize(self, x: np.ndarray, vector: bool = False, indices: Optional[List[int]] = None) -> np.ndarray:
        """Normalize the data x by the value.

        Args:
            x (np.ndarray): The data to normalize.
            vector (bool): If true, normalize the data as a vector.
            indices (Optional[List[int]]): If we want to normalize only some dimensions, we specify the indices.
                If None, normalize all dimensions.

        Returns:
            np.ndarray: The normalized data.

        """
        return x / self.value

    def unnormalize(self, x: np.ndarray, vector: bool = False, indices: Optional[List[int]] = None) -> np.ndarray:
        """Unnormalize the data x by the value.

        Args:
            x (np.ndarray): The data to unnormalize.
            vector (bool): If true, unnormalize the data as a vector.
            indices (Optional[List[int]]): If we want to unnormalize only some dimensions, we specify the indices.
                If None, unnormalize all dimensions.

        Returns:
            np.ndarray: The unnormalized data.

        """
        return x * self.value
