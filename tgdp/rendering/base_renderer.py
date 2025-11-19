"""Base class for rendering."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class BaseRenderer(ABC):
    """Abstract base class for rendering different types of visualizations such as rollouts, plans, and diffusions.

    Subclasses should implement the rendering methods for specific visualization outputs.
    """

    def __init__(
        self,
        run_dir: str,
    ):
        """Initialize the BaseRenderer with a directory for rendering outputs.

        Args:
            run_dir (str): The base directory where rendered outputs will be saved.

        """
        self.render_dir = os.path.join(run_dir, "render")
        if not os.path.exists(self.render_dir):
            os.makedirs(self.render_dir)

    @abstractmethod
    def render_rollout(
        self,
        rollout: Dict[str, np.ndarray],
        file_name: str = "rollout.mp4",
        **kwargs,
    ):
        """Render a rollout visualization and save it to a file.

        Args:
            rollout (Dict[str, np.ndarray]): The rollout data to visualize.
            file_name (str, optional): The name of the output file. Defaults to "rollout.mp4".
            **kwargs: Additional keyword arguments for rendering.

        """
        raise NotImplementedError

    @abstractmethod
    def render_plan(
        self,
        plan: np.ndarray,
        *,
        file_name: str,
    ):
        """Render a plan visualization and save it to a file.

        Args:
            plan (np.ndarray): The plan data to visualize.
            file_name (str, optional): The name of the output file. Defaults to "plan.mp4".

        """
        raise NotImplementedError

    @abstractmethod
    def render_diffusion(
        self,
        diffusion: np.ndarray,
        file_name: str = "diffusion.mp4",
        **kwargs,
    ):
        """Render a diffusion visualization and save it to a file.

        Args:
            diffusion (np.ndarray): The diffusion data to visualize.
            file_name (str, optional): The name of the output file. Defaults to "diffusion.mp4".
            **kwargs: Additional keyword arguments for rendering.

        """
        raise NotImplementedError

    @abstractmethod
    def render_plan_and_rollout(
        self,
        plan: np.ndarray,
        rollout: np.ndarray,
        file_name: str = "plan_and_rollout.mp4",
        **kwargs,
    ):
        """Render a combined visualization of a plan and a rollout and save it to a file.

        Args:
            plan (np.ndarray): The plan data to visualize.
            rollout (np.ndarray): The rollout data to visualize.
            file_name (str, optional): The name of the output file. Defaults to "plan_and_rollout.mp4".
            **kwargs: Additional keyword arguments for rendering.

        """
        raise NotImplementedError
