"""Conditioning utilities for state/action inpainting."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

ConditionType = Dict[Tuple[Optional[int], Optional[int], Optional[int], Optional[int]], torch.Tensor]


def apply_conditions(
    sample: torch.Tensor,
    conditions: Optional[ConditionType],
    scale: Optional[torch.Tensor] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply conditioning to the sample.

    Args:
        sample: The unconditioned sample. (batch_size, time, features)
        conditions: Dictionary of observation conditions of all samples of the batch.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.
        scale: We scale the sample by this factor. (batch_size, 1, 1) or None.
        inplace: If True, the sample is modified in place. If False, a new tensor is returned.

    Returns:
        sample: Conditioned sample. (batch_size, time, features)

    """
    if inplace:
        out = sample.clone()
    else:
        out = sample
    if conditions is None:
        return out
    apply_conditions_(out, conditions, scale)
    return out


def apply_conditions_(
    sample: torch.Tensor,
    conditions: Optional[ConditionType],
    scale: Optional[torch.Tensor] = None,
):
    """Apply conditioning to the sample in place.

    Args:
        sample: The unconditioned sample. (batch_size, time, features)
        conditions: Dictionary of observation conditions of all samples of the batch.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.
        scale: We scale the sample by this factor. (batch_size, 1, 1) or None.

    """
    if conditions is None:
        return
    assert (
        scale is None
        or isinstance(scale, float)
        or scale.shape == torch.Size([len(sample), 1, 1])
        or scale.shape == torch.Size([])
    ), "Scale must be of shape (batch_size, 1, 1) or ()."
    assert len(conditions.keys()) == 0 or sample.shape[0] == conditions[list(conditions.keys())[0]].shape[0], (
        "Batch size of sample and observation condition must be the same."
    )
    for (t_start, t_end, start_idx, end_idx), condition in conditions.items():
        assert sample.shape[0] == condition.shape[0], "Batch size of sample and observation condition must be the same."
        sample[:, t_start:t_end, start_idx:end_idx] = scale * condition if scale is not None else condition


def generate_history_condition_from_t_list(
    action_history: Optional[List[torch.Tensor]] = None,
    observation_history: Optional[List[torch.Tensor]] = None,
    conditions: Optional[ConditionType] = None,
) -> ConditionType:
    """Generate conditions for the given history.

    This generates the conditions for the history. The history here is a list of tensors, where every list entry
    corresponds to the history at a specific time index. The history may be a list of actions, observations, or both.
    The tensors in the list are shape (batch_size, condition_dim).

    Args:
        action_history: The action history. Set to None for no conditioning on action history.
            List(history_length, torch.Tensor(batch_size, action_dimensions)) or None.
        observation_history: The observation history. Set to None for no conditioning on observation history.
            List(history_length, torch.Tensor(batch_size, observation_dimensions)) or None.
        conditions: The existing conditions dictionary to add to. Input None to create a new dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    Returns:
        conditions: Condition dict. {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    """
    # If no conditions are given, create an empty dictionary.
    if conditions is None:
        conditions = {}

    # Determine history length and dimensionality of the conditions.
    if action_history is not None:
        assert observation_history is None or len(action_history) == len(observation_history), (
            "If both action and observation history are given, they must have the same length."
        )
        history_length = len(action_history)
        start_idx = 0
        end_idx = action_history[0].shape[1] if observation_history is None else None
    elif observation_history is not None:
        history_length = len(observation_history)
        start_idx = -observation_history[0].shape[1]
        end_idx = None
    else:
        return conditions  # No new conditions.

    # If the history length is 0, we return the conditions as is.
    if history_length == 0:
        return conditions

    # Generate the condition. For this we go from the list of tensors of shape (batch_size, condition_dim) to a tensor
    # of shape (batch_size, history_length, condition_dim).
    condition = torch.cat(
        (
            torch.stack(action_history) if action_history is not None else torch.empty(0),
            torch.stack(observation_history) if observation_history is not None else torch.empty(0),
        ),
        dim=2,
    ).swapaxes(0, 1)
    conditions[(0, history_length, start_idx, end_idx)] = condition

    return conditions


def generate_history_condition_from_tensor(
    action_history: Optional[torch.Tensor] = None,
    observation_history: Optional[torch.Tensor] = None,
    conditions: Optional[ConditionType] = None,
) -> ConditionType:
    """Generate conditions for the given history.

    This generates the conditions for the history given the action and observation history tensors.
    The history is a tensor of shape (batch_size, history_length, condition_dim).

    Args:
        action_history: The action history. Set to None for no conditioning on action history.
            (batch_size, history_length, action_dimensions) or None.
        observation_history: The observation history. Set to None for no conditioning on observation history.
            (batch_size, history_length, observation_dimensions) or None.
        conditions: The existing conditions dictionary to add to. Input None to create a new dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    Returns:
        conditions: Condition dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    """
    # If no conditions are given, create an empty dictionary.
    if conditions is None:
        conditions = {}

    # Determine history length and dimensionality of the conditions.
    if action_history is not None:
        assert observation_history is None or action_history.shape[1] == observation_history.shape[1], (
            "If both action and observation history are given, they must have the same length."
        )
        history_length = action_history.shape[1]
        start_idx = 0
        end_idx = action_history.shape[2] if observation_history is None else None
    elif observation_history is not None:
        history_length = observation_history.shape[1]
        start_idx = -observation_history.shape[2]
        end_idx = None
    else:
        return conditions
    # If the history length is 0, we return the conditions as is.
    if history_length == 0:
        return conditions
    # Generate the condition. For this we go from the tensors of shape (batch_size, history_length, obs/act_dim) to a
    # tensor of shape (batch_size, history_length, condition_dim).
    condition = torch.cat(
        (
            action_history if action_history is not None else torch.empty(0, history_length, 0),
            observation_history if observation_history is not None else torch.empty(0, history_length, 0),
        ),
        dim=2,
    )
    conditions[(0, history_length, start_idx, end_idx)] = condition
    return conditions


def generate_observation_condition(
    t_idx: int,
    observation: torch.Tensor,
    conditions: Optional[ConditionType] = None,
) -> ConditionType:
    """Generate conditions for the given observation.

    Args:
        t_idx: The time index.
        observation: The observation tensor. (batch_size, observation_dimensions)
        conditions: The existing conditions dictionary to add to. Input None to create a new dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    Returns:
        conditions: Condition dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    """
    if conditions is None:
        conditions = {}
    conditions[(t_idx, t_idx + 1, -observation.shape[1], None)] = observation.unsqueeze(1)
    return conditions


def generate_action_condition(
    t_idx: int,
    action: torch.Tensor,
    conditions: Optional[ConditionType] = None,
) -> ConditionType:
    """Generate conditions for the given action.

    Args:
        t_idx: The time index.
        action: The action tensor. (batch_size, action_dimensions)
        conditions: The existing conditions dictionary to add to. Input None to create a new dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    Returns:
        conditions: Condition dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    """
    if conditions is None:
        conditions = {}
    conditions[(t_idx, t_idx + 1, 0, action.shape[1])] = action.unsqueeze(1)
    return conditions


def generate_goal_condition(
    goal: torch.Tensor,
    goal_idxs: Optional[Sequence[int]] = None,
    observation_dim: Optional[int] = None,
    conditions: Optional[ConditionType] = None,
) -> ConditionType:
    """Generate conditions for the given goal.

    Args:
        goal: The goal tensor. (batch_size, goal_dimensions)
        goal_idxs: The indices of the sample that are part of the goal. If None, we assume that the goal is a full
            observation. List of integers or None.
        observation_dim: The dimension of the observation. If None, we assume that the goal is a full observation.
            Integer or None.
        conditions: The existing conditions dictionary to add to. Input None to create a new dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    Returns:
        conditions: Condition dictionary.
            {(t_start, t_end, start_idx, end_idx): condition[batch_size, time, condition_dim]}.

    """
    if conditions is None:
        conditions = {}
    if goal_idxs is None:
        # If no goal index is given, we assume that the goal is the final part of the observation.
        conditions[(-1, None, -goal.shape[1], None)] = goal.unsqueeze(1)
    else:
        # We batch consequtive goal indices together to generate as few conditions as possible.
        assert observation_dim is not None, "If goal_idxs is given, observation_dim must be given as well."
        i = 0
        i_start = 0
        start = -observation_dim + goal_idxs[0]
        while i <= len(goal_idxs) - 1:
            if i == len(goal_idxs) - 1 or goal_idxs[i + 1] != goal_idxs[i] + 1:
                end = -observation_dim + goal_idxs[i] + 1
                if end == 0:
                    end = None
                i_end = i + 1
                conditions[(-1, None, start, end)] = goal[:, i_start:i_end].unsqueeze(1)
                if i < len(goal_idxs) - 1:
                    start = -observation_dim + goal_idxs[i + 1]
                    i_start = i + 1
            i += 1

    return conditions
