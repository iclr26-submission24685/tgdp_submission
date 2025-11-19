"""Rendering utilities for MuJoCo-based environments."""

# import gym
import logging
import os
from os import path
from typing import Dict

import gymnasium
import gymnasium_robotics
import gymnasium_robotics.envs
import gymnasium_robotics.envs.franka_kitchen
import gymnasium_robotics.envs.maze
import numpy as np

from ..environments import d4rl_env_mapping
from .base_renderer import BaseRenderer
from .utils import save_video

os.environ["MUJOCO_GL"] = "egl"
logger = logging.getLogger(__name__)


class MuJoCoRenderer(BaseRenderer):
    """Base renderer class for MuJoCo-based environments.

    This class provides a framework for rendering frames and videos from MuJoCo environments.
    It handles environment creation, state setting, and rendering frames from a sequence of states.
    Subclasses should implement environment-specific logic for state manipulation and camera parameters.

    Args:
        env_name (str): Name of the environment to render.
        run_dir (str): Directory to save rendered videos.
        **render_kwargs: Additional keyword arguments for rendering.

    """

    def __init__(
        self,
        env_name: str,
        run_dir: str,
        **render_kwargs,
    ):
        """Initialize the MuJoCoRenderer.

        Args:
            env_name (str): Name of the environment to render.
            run_dir (str): Directory to save rendered videos.
            **render_kwargs: Additional keyword arguments for rendering.

        """
        super().__init__(run_dir)
        self.env_name = d4rl_env_mapping(env_name, visualization=True)
        self.render_env = gymnasium.make(self.env_name, render_mode="rgb_array").unwrapped

        # Get dimensions of the environment state.
        self.qpos_dim = self.render_env.model.nq  # type: ignore[no-untyped-call]
        self.qvel_dim = self.render_env.model.nv  # type: ignore[no-untyped-call]
        self.render_kwargs = render_kwargs

    def _set_render_env_state(self, state: np.ndarray):
        """Set the state of the rendering environment.

        Args:
            state (np.ndarray): State vector to set in the environment.

        Subclasses must implement this method to set the environment state appropriately.

        """
        raise NotImplementedError

    def render_frames_from_states(
        self,
        states: np.ndarray,
    ) -> np.ndarray:
        """Render a sequence of frames from a sequence of environment states.

        Args:
            states (np.ndarray): Array of environment states to render.

        Returns:
            np.ndarray: Array of rendered frames (images).

        """
        self._set_camera_params(**self.render_kwargs)
        frames = []
        for s in states:
            self._set_render_env_state(s)
            frames.append(self.render_env.render())
        return np.array(frames)

    def _set_camera_params(self, **render_kwargs):
        """Set camera parameters for rendering.

        Subclasses can override this method to set camera parameters such as lookat, distance, etc.

        Args:
            **render_kwargs: Camera parameters to set.

        """
        pass

    def render_rollout(
        self,
        rollout: Dict[str, np.ndarray],
        file_name: str = "rollout.mp4",
    ):
        """Render a rollout and save it as a video.

        Args:
            rollout (Dict[str, np.ndarray]): Rollout dictionary containing at least "observations".
            file_name (str): Name of the output video file (default: "rollout.mp4").

        """
        states = self._get_states_from_rollout(rollout)
        frames = self.render_frames_from_states(states)
        save_video(frames, path.join(self.render_dir, file_name), fps=60)

    def _get_states_from_rollout(self, rollout: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract states from a rollout dictionary.

        Subclasses can override this method if the state extraction logic differs.

        Args:
            rollout (Dict[str, np.ndarray]): Rollout dictionary.

        Returns:
            np.ndarray: Array of states to render.

        """
        return rollout["observations"]

    def render_plan(self, plan: np.ndarray, file_name: str = "plan.mp4"):
        """Render a plan (sequence of states or actions).

        Subclasses must implement this method.

        Args:
            plan (np.ndarray): Plan to render.
            file_name (str): Name of the output video file (default: "plan.mp4").

        """
        raise NotImplementedError

    def render_diffusion(self, diffusion: np.ndarray):
        """Render a diffusion process.

        Subclasses must implement this method.

        Args:
            diffusion (np.ndarray): Diffusion data to render.

        """
        raise NotImplementedError

    def render_plan_and_rollout(self, plan: np.ndarray, rollout: np.ndarray):
        """Render both a plan and a rollout for comparison.

        Subclasses must implement this method.

        Args:
            plan (np.ndarray): Plan to render.
            rollout (np.ndarray): Rollout to render.

        """
        raise NotImplementedError


class MuJoCoLocomotionRenderer(MuJoCoRenderer):
    """Renderer for MuJoCo locomotion environments.

    This subclass implements environment-specific logic for locomotion tasks, including state setting and
    camera parameter handling.
    """

    def _set_render_env_state(self, state: np.ndarray):
        """Set the state of the locomotion environment.

        Args:
            state (np.ndarray): State vector containing qpos and qvel.

        """
        qpos = state[: self.qpos_dim]
        qvel = state[self.qpos_dim :]
        self.render_env.set_state(qpos, qvel)  # type: ignore[no-untyped-call]

    def _set_camera_params(self, **render_kwargs):
        """Set camera parameters for the locomotion environment viewer.

        Args:
            **render_kwargs: Camera parameters to set (e.g., lookat, distance).

        """
        if hasattr(self.render_env, "viewer") and hasattr(self.render_env.viewer, "cam"):  # type: ignore[no-untyped-call]
            for key, val in render_kwargs.items():
                if key == "lookat":
                    self.render_env.viewer.cam.lookat[:] = val[:]  # type: ignore[no-untyped-call]
                elif hasattr(self.render_env.viewer.cam, key):  # type: ignore[no-untyped-call]
                    setattr(self.render_env.viewer.cam, key, val)  # type: ignore[no-untyped-call]
                else:
                    logger.warning(f"Camera has no attribute '{key}' to set.")

    def _get_states_from_rollout(self, rollout: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract states from a locomotion rollout.

        If "x_position" is present in rollout infos, prepend it to the observations.

        Args:
            rollout (Dict[str, np.ndarray]): Rollout dictionary.

        Returns:
            np.ndarray: Array of states for rendering.

        """
        observations = rollout["observations"]
        if "x_position" in rollout["infos"][0]:
            x_pos = np.array([step_info["x_position"] for step_info in rollout["infos"]])
        else:
            x_pos = np.zeros(len(observations))
        states = np.concatenate([x_pos[:, None], observations[: len(x_pos)]], axis=1)
        return states


class MuJoCoKitchenRenderer(MuJoCoRenderer):
    """Renderer for MuJoCo kitchen environments.

    This subclass implements environment-specific logic for kitchen tasks, including state setting.
    """

    render_env: gymnasium_robotics.envs.franka_kitchen.KitchenEnv

    def _set_render_env_state(self, state: np.ndarray):
        """Set the state of the kitchen environment.

        Args:
            state (np.ndarray): State vector containing qpos.

        """
        qpos = state[: self.qpos_dim]
        qvel = np.zeros(self.qvel_dim)
        self.render_env.robot_env.set_state(qpos, qvel)


class MuJoCoMaze2DRenderer(MuJoCoRenderer):
    """Renderer for MuJoCo Maze2D environments.

    This subclass implements environment-specific logic for Maze2D tasks, including state setting.
    """

    render_env: gymnasium_robotics.envs.maze.PointMazeEnv

    def __init__(self, env_name: str, run_dir: str, **render_kwargs):
        """Initialize the Maze2D renderer.

        Args:
            env_name (str): Name of the Maze2D environment to render.
            run_dir (str): Directory to save rendered videos.
            **render_kwargs: Additional keyword arguments for rendering.

        """
        super().__init__(env_name, run_dir, **render_kwargs)
        y_dim, x_dim = np.shape(self.render_env.spec.kwargs["maze_map"])  # type: ignore[no-untyped-call]
        self._offset = -np.array([(y_dim - 1) / 2 - 0.15, (x_dim - 1) / 2 - 0.15, 0.0, 0.0])

    def _set_render_env_state(self, state: np.ndarray):
        """Set the state of the Maze2D environment.

        Args:
            state (np.ndarray): State vector containing qpos and qvel.

        """
        state_transformed = self._transform_coordinates(state)
        qpos = state_transformed[: self.qpos_dim]
        qvel = state_transformed[self.qpos_dim :]
        self.render_env.point_env.set_state(qpos, qvel)

    def _set_camera_params(self, **render_kwargs):
        """Set camera parameters for the Maze2D environment viewer.

        Args:
            **render_kwargs: Camera parameters to set (e.g., lookat, distance).

        """
        if hasattr(self.render_env.point_env, "mujoco_renderer") and hasattr(
            self.render_env.point_env.mujoco_renderer, "_get_viewer"
        ):
            viewer = self.render_env.point_env.mujoco_renderer._get_viewer("rgb_array")
            for key, val in render_kwargs.items():
                if key == "lookat":
                    viewer.cam.lookat[:] = val[:]
                else:
                    setattr(viewer.cam, key, val)

    def _get_states_from_rollout(self, rollout: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract states from a Maze2D rollout.

        Args:
            rollout (Dict[str, np.ndarray]): Rollout dictionary.

        Returns:
            np.ndarray: Array of states for rendering.

        """
        return rollout["observations"]

    def _transform_coordinates(self, obs: np.ndarray) -> np.ndarray:
        """Transform the observation/goal to match the D4RL coordinate system."""
        if len(obs) == 4:  # Observation
            # Switch x and y coordinates and add offset
            return np.array([obs[1], obs[0], obs[3], -obs[2]]) + self._offset
        elif len(obs) == 2:  # Goal
            return np.array([obs[1], obs[0]]) + self._offset[:2]
        else:
            raise ValueError(f"Unexpected observation length: {len(obs)}. Expected 2 or 4.")
