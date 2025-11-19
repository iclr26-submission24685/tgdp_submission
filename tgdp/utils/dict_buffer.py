"""A dictionary-like buffer that registers key-value pairs as buffers in PyTorch models.

This allows the dictionary to be saved and loaded with the model, ensuring that the key-value pairs
are treated as part of the model's state.
"""

import torch
import torch.nn as nn


class DictBuffer(nn.Module):
    """A dict that registers key-value pairs as buffers in order to be saved and loaded with the model."""

    def __init__(self, initial_dict=None):
        """Initialize the DictBuffer with an optional initial dictionary.

        Args:
            initial_dict (dict, optional): A dictionary of key-value pairs to initialize the buffer.

        """
        super().__init__()
        self._keys = []
        if initial_dict:
            for key, value in initial_dict.items():
                self[key] = value

    def __setitem__(self, key, value):
        """Set a key-value pair in the buffer, registering the value as a buffer if necessary."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.register_buffer(str(key), value)
        if key not in self._keys:
            self._keys.append(key)

    def __getitem__(self, key):
        """Retrieve the value associated with the given key."""
        return getattr(self, str(key))

    def __contains__(self, key):
        """Check if the buffer contains the given key."""
        return hasattr(self, str(key))

    def __len__(self):
        """Return the number of key-value pairs in the buffer."""
        return len(self._keys)

    def __iter__(self):
        """Return an iterator over the keys in the buffer."""
        return iter(self._keys)

    def items(self):
        """Return a list of (key, value) pairs in the buffer."""
        return [(key, self[key]) for key in self._keys]

    def keys(self):
        """Return a list of keys in the buffer."""
        return self._keys

    def values(self):
        """Return a list of values in the buffer."""
        return [self[key] for key in self._keys]
