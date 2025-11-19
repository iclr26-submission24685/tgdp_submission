"""Components for using the D4RL environments."""

import logging
from typing import Dict, List, Tuple

import gym
import gym.vector as gym_vector
import gymnasium
import numpy as np
from gymnasium import Wrapper, spaces

LOCOMOTION_ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]
MAZE_ENVS = ["PointMaze_Open-v3", "PointMaze_UMaze-v3", "PointMaze_Medium-v3", "PointMaze_Large-v3"]
KITCHEN_ENVS = ["FrankaKitchen-v1"]

logger = logging.getLogger(__name__)


def kitchen_tasks_to_complete_mapping(env_name: str) -> list:
    """Return the tasks to complete for the Kitchen environments.

    This function maps the kitchen environment names to their corresponding set of tasks.
    """
    if "kitchen-mixed" in env_name:
        return ["light switch", "bottom burner", "microwave", "kettle"]
    elif "kitchen-partial" in env_name:
        return ["kettle", "slide cabinet", "light switch", "microwave"]
    elif "kitchen-complete" in env_name:
        return ["kettle", "slide cabinet", "light switch", "microwave"]
    else:
        raise ValueError(f"Unknown kitchen environment: {env_name}. Supported: {KITCHEN_ENVS}")


def d4rl_env_mapping(env_name: str, visualization: bool = False) -> str:
    """Map D4RL dataset names to correct gymnasium environment names.

    Map D4RL dataset names to correct gymnasium environment names.
    This mostly means that we do not use the latest versions.
    """
    if "halfcheetah" in env_name:
        return "HalfCheetah-v5" if visualization else "HalfCheetah-v2"
    elif "hopper" in env_name:
        return "Hopper-v5" if visualization else "Hopper-v2"
    elif "walker2d" in env_name:
        return "Walker2d-v5" if visualization else "Walker2d-v2"
    elif "maze2d-open" in env_name:
        return "PointMaze_Open-v3"
    elif "maze2d-umaze" in env_name:
        return "PointMaze_UMaze-v3"
    elif "maze2d-medium" in env_name:
        return "PointMaze_Medium-v3"
    elif "maze2d-large" in env_name:
        return "PointMaze_Large-v3"
    elif "kitchen" in env_name:
        return "FrankaKitchen-v1"
    else:
        return env_name


class D4RL2GymnasiumCompatWrapper(gym.Wrapper):
    """Wrapper to make gym environments compatible with the gymnasium interface.

    This wrapper splits the done flag into terminated and truncated flags, as required by the gymnasium interface. It
    also ensures that the reset method returns the initial observation and an empty info dictionary.
    """

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:  # type: ignore[override]
        """Take a step in the environment using the given action.

        This method takes an action in the environment and returns the new observation, reward, terminated flag,
        truncated flag, and info dictionary. It splits the done flag into terminated and truncated flags.

        Args:
            action (np.ndarray): The action to take in the environment.

        Returns:
            obs (np.ndarray): The new observation of the environment.
            reward (float): The reward received after taking the action.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            info (dict): An info dictionary containing additional information about the step.

        """
        obs, reward, done, info = self.env.step(action)
        truncated = info.get("TimeLimit.truncated", False)
        terminated = done and not truncated

        return obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation and info dictionary.

        The reset method returns the initial observation and an empty info dictionary.
        This is needed to ensure compatibility with the gymnasium interface.

        Args:
            **kwargs: Additional keyword arguments to pass to the reset method.

        Returns:
            obs (np.ndarray): The initial observation of the environment.
            info (dict): An empty info dictionary.

        """
        obs = self.env.reset(**kwargs)
        info = {}
        return obs, info


class D4RLMazeWrapper(gym.Wrapper):
    """Wrapper to make D4RL Maze2D environments. This replaces the reward signal.

    This wrapper replaces the original reward signal of the D4RL Maze2D environments with so taht the
    reward is 1.0 always after reaching the goal once (defined through a single non-zero reward). This is in line
    with the implementation in the CleanDiffuser repo/paper.
    Even though the environment should be a gym environment, we assume that the step and reset method follow the
    gymnasium interface, i.e., they return a tuple of (obs, reward, terminated, truncated, info). This can be achieved
    by wrapping first in a D4RL2GymnasiumCompatWrapper.
    """

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and return the initial observation and info dictionary.

        This method resets the environment and returns the initial observation and an empty info dictionary.

        Returns:
            obs (np.ndarray): The initial observation of the environment.
            info (dict): An empty info dictionary.

        """
        self.is_done = False

        obs, info = self.env.reset()
        info = self._add_goal_to_info(info)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment using the given action.

        This method takes an action in the environment and returns the new observation, reward, terminated flag,
        truncated flag, and info dictionary. The reward is set to 1.0 if the goal is reached.

        Args:
            action (np.ndarray): The action to take in the environment.

        Returns:
            obs (np.ndarray): The new observation of the environment.
            reward (float): The reward received after taking the action.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            info (dict): An info dictionary containing additional information about the step.

        """
        obs, reward, terminated, truncated, info = self.env.step(action)  # type: ignore[no-untyped-call]
        self.is_done |= reward == 1.0

        # Add the goal to the info dictionary if not already present.
        info = self._add_goal_to_info(info)

        return obs, float(self.is_done), terminated, truncated, info  # type: ignore[no-untyped-call]

    def _add_goal_to_info(self, info: Dict) -> Dict:
        """Add the goal to the info dictionary if not already present.

        This method checks if the goal is already in the info dictionary and adds it if not.
        It is used to ensure that the goal is always available in the info dictionary.

        Args:
            info (dict): The info dictionary to check and modify.

        Returns:
            dict: The modified info dictionary with the goal added if it was not already present.

        """
        if "goal" not in info and hasattr(self.env, "get_target"):
            info["goal"] = np.array(self.env.get_target())  # type: ignore[no-untyped-call]
        elif hasattr(self.env.unwrapped, "_goal"):
            info["goal"] = np.array(self.env.unwrapped._goal)  # type: ignore[no-untyped-call]
        return info


class Gymnasium2D4RLKitchenWrapper(Wrapper):
    """Wrapper for the Gymnasium Kitchen environments to make them compatible with D4RL datasets.

    This wrapper processes the observation dictionary of Gymnasium Kitchen environments to be an observation array and
    a goal vector that is compatible with the D4RL format. The Gymnasium Kitchen environments' observation space is a
    Dict including the observation (including position and velocity) and the goal. The D4RL Kitchen environments
    observation space is a single array that includes the position of the robot and the object, as well as the goal
    positions of all objects and the robot.
    """

    def __init__(self, env: gymnasium.Env):
        """Initialize the Gymnasium2D4RLKitchenWrapper.

        Args:
            env: The Gymnasium Kitchen environment to wrap.

        """
        super().__init__(env)

        # Modify the observation space to exclude goal information.
        env.observation_space = spaces.Box(low=-8, high=8, shape=(60,))

        # Dimensions of observation space parts.
        self._robot_qpos_dim = 9
        self._robot_qvel_dim = 9
        self._obj_qpos_dim = 21
        self._obj_qvel_dim = 20

        # Mapping from keywords to goal indices.
        self.goal_indices = {
            "robot": (0, 9),  # Robot position and velocity (9 dim)
            "bottom right burner": (9, 11),  # Bottom right burner position (bottom right burner, 2 dim)
            "bottom burner": (11, 13),  # Bottom burner position (bottom left burner, 2 dim)
            "top right burner": (13, 15),  # Top right burner position (top right burner, 2 dim)
            "top burner": (15, 17),  # Top burner position (top left burner, 2 dim)
            "light switch": (17, 19),  # Light switch position (2 dim)
            "slide cabinet": (19, 20),  # Slide cabinet position (1 dim)
            "hinge cabinet": (20, 22),  # Hinge cabinet position (2 dim)
            "microwave": (22, 23),  # Microwave position (1 dim)
            "kettle": (23, 30),  # Kettle position (7 dim)
        }

    def _gymnasium_to_d4rl_observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Process the observation dictionary of Gymnasium Kitchen environments.

        This processes the observation from the Gymansium Kitchen environment to the D4RL format.

        Args:
            obs (dict): The observation dictionary from the Gymnasium Kitchen environment.

        Returns:
            np.ndarray: The observation in D4RL format, which includes the robot and object positions and the goal.

        """
        # Extract the observation and goal from the dictionary
        gymnasium_obs = obs["observation"]
        gymnasium_goal = obs["desired_goal"]

        # D4RL observation are the position of the robot and the object, while gymnasium observations are the position
        # and velocities.
        robot_qp = gymnasium_obs[: self._robot_qpos_dim]
        obj_qp = gymnasium_obs[
            self._robot_qpos_dim + self._robot_qvel_dim : self._robot_qpos_dim
            + self._robot_qvel_dim
            + self._obj_qpos_dim
        ]

        # The D4RL observation features goal positions of all objects and the robot.
        goal = np.zeros(30)
        for key, (start, end) in self.goal_indices.items():
            if key in gymnasium_goal:
                assert end - start == gymnasium_goal[key].shape[0], f"Goal {key} has incorrect shape."
                goal[start:end] = gymnasium_goal[key]

        # Concatenate the robot and object positions with the goal.
        d4rl_obs = np.concatenate([robot_qp, obj_qp, goal])

        return d4rl_obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment using the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: A tuple containing the new observation, reward, terminated flag, truncated flag, and info dictionary.

        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(obs, np.ndarray):  # Vectorized environment
            new_obs = zip(*[self._gymnasium_to_d4rl_observation(o) for o in obs])
            new_obs = np.array(new_obs)
        else:  # Single environment
            new_obs = self._gymnasium_to_d4rl_observation(obs)

        return new_obs, float(reward), terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation and info dictionary.

        Args:
            **kwargs: Additional keyword arguments to pass to the reset method.

        Returns:
            tuple: A tuple containing the initial observation in D4RL format and the info dictionary.

        """
        obs, info = self.env.reset(**kwargs)

        return self._gymnasium_to_d4rl_observation(obs), info


class VectorD4RL2GymnasiumCompatWrapper(gym_vector.VectorEnvWrapper):
    """Wrapper for vector environments to make them compatible with Gymnasium's API.

    This wrapper splits the `dones` array into `terminated` and `truncated` flags and ensures
    that `reset()` returns a tuple of `(obs, infos)`.
    """

    def reset(self, **kwargs) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments and returns batched observations and infos.

        Returns:
            obs (np.ndarray): Batched initial observations.
            infos (List[Dict]): List of info dictionaries (empty by default).

        """
        obs = self.env.reset(**kwargs)
        # Create empty infos if the base env doesn't provide them
        infos = getattr(self.env, "reset_infos", [{} for _ in range(self.num_envs)])
        return obs, infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step through all environments and returns Gymnasium-compliant outputs.

        Args:
            actions (np.ndarray): Batched actions for all environments.

        Returns:
            obs (np.ndarray): Batched new observations.
            rewards (np.ndarray): Batched rewards.
            terminated (np.ndarray): Boolean array indicating termination per sub-env.
            truncated (np.ndarray): Boolean array indicating truncation per sub-env.
            infos (List[Dict]): List of info dictionaries per sub-env.

        """
        obs, rewards, dones, infos = self.env.step(actions)

        terminated = np.zeros_like(dones, dtype=bool)
        truncated = np.zeros_like(dones, dtype=bool)

        for i in range(self.num_envs):
            truncated[i] = infos[i].get("TimeLimit.truncated", False)
            terminated[i] = dones[i] and not truncated[i]
            # Remove "TimeLimit.truncated" from infos to avoid redundancy
            infos[i].pop("TimeLimit.truncated", None)

        return obs, rewards, terminated, truncated, infos


class Gymnasium2D4RLMazeWrapper(Wrapper):
    """Wrapper to make Gymnasium Maze environments compatible with D4RL datsets.

    This wrapper ensures compatibility between Gymnasium Maze environments and D4RL. The Maze environments' observation
    space is a Dict inluding the observation and the goal. This wrapper extracts the observation from the dict and puts
    the goal into the infos. Additionally, it transforms the coordinates to match the D4RL coordinate system and adapts
    the spawn and goal variance to match D4RL's sampling method.
    """

    def __init__(self, env: gymnasium.Env):
        """Initialize the Gymnasium2D4RLMazeWrapper.

        Args:
            env: The Gymnasium Maze environment to wrap.

        """
        super().__init__(env)
        # Gymnasium Maze envs have (0,0) at the center. D4RL envs have (0,0) at the center of the bottom left tile.
        # Also, gymnasium uses transposed map layouts of the D4RL ones. This effectively flips the x and y axes and
        # inverts the D4RL x-axis.
        y_dim, x_dim = np.shape(env.spec.kwargs["maze_map"])  # type: ignore[no-untyped-call]
        self._offset = np.array([(y_dim - 1) / 2, (x_dim - 1) / 2, 0.0, 0.0])
        self.spawn_radius = 0.1  # D4RL uses a spawn radius of 0.1, while gymnasium uses 0.25.

        # Modify the observation space to exclude goal information
        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space = env.observation_space.spaces["observation"]
        else:
            self.observation_space = env.observation_space

    def _gymnasium_to_d4rl_observation(self, obs):
        """Process the observation dictionary of Gymnasium Maze environments to be an observation and a goal array.

        This processes the observation from the Gymnasium Maze environment to the D4RL format. It extracts the
        observation and goal from the dictionary, transforms the coordinates to match the D4RL coordinate system, and
        returns the observation and goal as separate arrays.

        """
        new_obs = self._transform_coordinates(obs["observation"])
        goal = obs.get("desired_goal", None)

        if goal is not None:
            goal = self._transform_coordinates(goal)

        return new_obs, goal

    def _d4rl_to_gymnasium_action(self, action):
        """Transform the action to match the D4RL coordinate system.

        This transforms the action from the D4RL coordinate system to the Gymnasium coordinate system.

        """
        return np.array([action[1], -action[0]])

    def _transform_coordinates(self, obs):
        """Transform the observation/goal to match the D4RL coordinate system."""
        if len(obs) == 4:  # Observation
            # Switch x and y coordinates and add offset
            return np.array([-obs[1], obs[0], -obs[3], obs[2]]) + self._offset
        elif len(obs) == 2:  # Goal
            return np.array([-obs[1], obs[0]]) + self._offset[:2]

    def _adapt_spawn_and_goal_variance(self):
        """Adapt the spawn and goal variance to match D4RL's sampling method.

        This adjusts the spawn and goal positions to match the D4RL sampling method. In D4RL, the spawn and goal
        positions are sampled with a variance of 0.1 around tile centers. In gymnasium environments with variance 0.25.
        """
        # Modify agent position
        noise = np.random.uniform(-self.spawn_radius, self.spawn_radius, size=2)
        noise_free_qpos = np.round(self.env.unwrapped.data.qpos[:2])  # type: ignore[no-untyped-call]
        self.env.unwrapped.data.qpos[:2] = noise_free_qpos + noise  # type: ignore[no-untyped-call]

        # Modify goal position
        noise = np.random.uniform(-self.spawn_radius, self.spawn_radius, size=2)
        noise_free_goal = np.round(self.env.unwrapped.goal)  # type: ignore[no-untyped-call]
        self.env.unwrapped.goal = noise_free_goal + noise  # type: ignore[no-untyped-call]

        # Return the new observation
        return self.env.unwrapped._get_obs(  # type: ignore[no-untyped-call]
            np.concatenate(
                [self.env.unwrapped.data.qpos, self.env.unwrapped.data.qvel]  # type: ignore[no-untyped-call]
            )
        )

    def step(self, action):
        """Take a step in the environment using the given action.

        This method transforms the action to match the D4RL coordinate system, steps the environment,
        processes the observation to match the D4RL format, and updates the info dictionary with the goal.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: A tuple containing the new observation, reward, terminated flag, truncated flag, and info dictionary.

        """
        action = self._d4rl_to_gymnasium_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Transform goal and observation to D4RL format.
        new_obs, goal = self._gymnasium_to_d4rl_observation(obs)
        info["goal"] = goal

        return new_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and return the initial observation and info dictionary.

        Args:
            **kwargs: Additional keyword arguments to pass to the reset method.

        Returns:
            tuple: A tuple containing the initial observation in D4RL format and the info dictionary.

        """
        obs, info = self.env.reset(**kwargs)
        obs = self._adapt_spawn_and_goal_variance()

        # Transform goal and observation to D4RL format.
        new_obs, goal = self._gymnasium_to_d4rl_observation(obs)
        info["goal"] = goal

        return new_obs, info
