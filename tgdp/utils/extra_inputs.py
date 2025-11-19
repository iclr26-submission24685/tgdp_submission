"""Utility functions for handling extra inputs to models."""

from typing import Optional

import tensordict


def concat_extra_inputs(extra_inputs: Optional[tensordict.TensorDict]) -> tensordict.TensorDict:
    """Concatenate each entry of the extra inputs to a single tensor.

    This method concatenates the entries of the extra inputs. The input should be a TensorDict that contains
    two TensorDicts: `local_condition` and `global_condition`. Each of these should contain tensors
    that can be concatenated along the last dimension. If any of these entries are empty, they will be set to `None`.
    If `extra_inputs` is `None`, it returns a TensorDict with both entries set to `None`.

    Args:
        extra_inputs (TensorDict): The extra inputs.

    Returns:
        TensorDict: The concatenated extra inputs.

    """
    if extra_inputs is None:
        out = tensordict.TensorDict({})
        out["local_condition"] = None
        out["global_condition"] = None
        return out
    out = tensordict.TensorDict({})
    for k, v in extra_inputs.items():
        assert isinstance(v, tensordict.TensorDict), f"Expected TensorDict, got {type(v)} for key {k}"
        if v.is_empty():
            out[k] = None
        else:
            out[k] = v.cat_from_tensordict(dim=-1, sorted=True)

    return out
