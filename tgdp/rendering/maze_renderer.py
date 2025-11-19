"""Rendering for the Maze Environments (Maze2D)."""

from os import path
from typing import Callable, Dict, List, Optional, Tuple, Union

import gymnasium
import gymnasium_robotics  # noqa: F401, required for environment registration
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import ndimage

from ..environments import d4rl_env_mapping
from .base_renderer import BaseRenderer
from .utils import plt_plot_to_np, save_image, save_video

# Flag to indicate if the data is from a gym environment, which have an offset in the position.
DATA_IS_GYM = True


class Maze2DRenderer(BaseRenderer):
    """Renderer for Maze2D environemtns.

    Renderer for Maze2D environments, providing visualization tools for rollouts, plans, value fields, gradient fields,
    and diffusion processes in maze-based RL tasks. This is not using the MuJoCo rendering capabilities but implements
    its own rendering logic using matplotlib.
    """

    def __init__(self, env_name: str, run_dir: str, **kwargs):
        """Initialize the Maze2DRenderer.

        Args:
            env_name (str): Name of the environment to render.
            run_dir (str): Directory to save rendered outputs.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(run_dir)
        self.render_env = gymnasium.make(d4rl_env_mapping(env_name), render_mode="rgb_array").unwrapped
        self._background = np.array(self.render_env.spec.kwargs["maze_map"])  # type: ignore[reportArgumentType]
        self._extent = (0.0, self._background.shape[0] - 1.0, 0.0, self._background.shape[1] - 1.0)
        self._background = np.kron(self._background, np.ones((2, 2), dtype=self._background.dtype))[1:-1, 1:-1]
        self._histogram = None
        self._pos_offset = np.array([0.2, 0.2]) if DATA_IS_GYM else np.array([0.0, 0.0])

        # Video frame skip.
        self.video_frame_skip = kwargs.get("video_frame_skip", 1)

    def _prepare_plot(self, render_data_histogram: bool = False):
        plt.clf()
        fig, ax = plt.subplots(figsize=(self._extent[1], self._extent[3]), dpi=200)

        # Plot grid-lines
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=1, alpha=0.5)
        ax.grid(which="major", color="gray", linestyle="--", linewidth=0.2, alpha=0.5)

        # Plot ticks
        ax.axis("on")
        ax.set_xticks(np.arange(0, self._extent[1], 1))
        ax.set_yticks(np.arange(0, self._extent[3], 1))

        # Set axis limits
        ax.set_xlim(self._extent[0], self._extent[1])
        ax.set_ylim(self._extent[2], self._extent[3])

        # Plot background. The background has to be transposed and flipped to match the coordinate system of the plot.
        ax.imshow(
            0.5 * self._background.T,
            extent=self._extent,
            alpha=1.0 * (self._background.T > 0.5),
            cmap="binary",
            vmin=0,
            vmax=1,
            origin="lower",
        )

        # Plot data histogram. The histogram has to be transposed and flipped to match the plot's coordinate system.
        if render_data_histogram and self._histogram is not None:
            ax.imshow(self._histogram.T, cmap="jet", alpha=0.5 * self._histogram.T, extent=self._extent, origin="lower")

        return fig, ax

    def _plot_trajectory_from_states(
        self,
        states: np.ndarray,
        alpha: float = 1.0,
        colors: Optional[Union[str, List[str]]] = None,
        size: int = 8,
        label: Optional[str] = None,
        colorbar: bool = False,
        ax: Optional[Axes] = None,
    ):
        """Plot the colored trajectory with scatter points."""
        # States should be 2D. If multiple trajectories are passed, only plot the first one.
        if states.ndim == 3:
            states = states[0]
        if colors is None:
            _colors = colormaps.get_cmap("jet")(np.linspace(0, 1, len(states)))
        else:
            _colors = colors
        if ax is None:
            ax = plt.gca()

        # Transform states to the plot's coordinate system.
        states = self._transform_coordinates(states)

        # Plot the trajectory as a scatter plot.
        scatter = ax.scatter(states[:, 0], states[:, 1], c=_colors, alpha=alpha, s=size, zorder=10, label=label)
        # add a small colorbar that indicates the environment step
        if colorbar:
            gca = plt.gca()
            sm = plt.cm.ScalarMappable(cmap="jet", norm=Normalize(vmin=0, vmax=len(states)))
            cax = inset_axes(ax, width="2.5%", height="15%", loc="upper right", borderpad=0.2)
            cbar = plt.colorbar(sm, orientation="vertical", cax=cax)  # , fraction=0.1, pad=0.05)
            cbar.ax.tick_params(labelsize=6, pad=0.1)
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.set_label("step", fontsize=6, labelpad=0.1)
            cbar.ax.yaxis.set_label_coords(-0.1, -0.1)
            cbar.ax.yaxis.label.set_rotation(0)
            plt.sca(gca)  # Reset gca()
        return scatter

    def _transform_coordinates(self, state: np.ndarray) -> np.ndarray:
        return state[..., :2] + self._pos_offset

    def _plot_targets(self, targets: Dict[str, np.ndarray], ax: Optional[Axes] = None):
        if ax is None:
            ax = plt.gca()
        markerstyles = ["*", "v", "+", "^", "<", ">", "p", "h", "H", "|", "_"]
        colors = colormaps.get_cmap("jet")(np.linspace(0, 1, len(targets[list(targets.keys())[0]])))
        markers = []
        for i, (key, value) in enumerate(targets.items()):
            markers.append(
                ax.scatter(value[:, 0], value[:, 1], s=8, c=colors, alpha=0.5, marker=markerstyles[i], label=key)
            )
        return markers

    def _plot_gradients(self, start: np.ndarray, gradients: Dict[str, np.ndarray], ax: Optional[Axes] = None):
        # Gradients should be 2D. If multiple trajectories are passed, only plot the first one.
        if start.ndim == 3:
            start = start[0]
        if ax is None:
            ax = plt.gca()
        colors = colormaps.get_cmap("jet")(np.linspace(0, 1, len(start)))
        linestyles = ["-", "--", "-.", ":"]
        quivers = []
        for key, value in gradients.items():
            if value.ndim == 3:
                value = value[0]
            quivers.append(
                ax.quiver(
                    start[:, 0],
                    start[:, 1],
                    value[:, 0],
                    value[:, 1],
                    color=colors,
                    linestyle=linestyles.pop(0),
                    scale=1,
                    scale_units="xy",
                    angles="xy",
                    headwidth=2,
                    width=0.003,
                    headlength=2,
                    headaxislength=2,
                    label=key,
                )
            )
            start = start + value
        return quivers

    def _plot_goal(self, goals: np.ndarray, ax: Optional[Axes] = None):
        if ax is None:
            ax = plt.gca()
        goals = self._transform_coordinates(goals)
        return ax.scatter(goals[0], goals[1], s=8, c="red", marker="*", label="Goal")

    def _plot_state(self, state: np.ndarray, color: str = "red", ax: Optional[Axes] = None):
        if ax is None:
            ax = plt.gca()
        state = self._transform_coordinates(state)
        return ax.scatter(state[0], state[1], s=8, c=color, marker="o", label="State")

    def render_rollout_image(
        self,
        rollout: Dict[str, np.ndarray],
        cut_on_reward: bool = True,
        file_name: str = "rollout.png",
        **kwargs,
    ):
        """Render a single rollout.

        Renders a single rollout from the environment, including the trajectory, goal, and optionally a data histogram.
        The resulting image will be saved under the specified name in the run_folder/render.

        Args:
            rollout: The rollout to render.
            cut_on_reward: If True, cut the rendering on the first positive reward signal.
            file_name: The name of the file to save the image to.
            **kwargs: Additional keyword arguments, such as goal, render_data_histogram, colorbar.

        """
        if "goal" in kwargs:
            goal = kwargs["goal"]
        elif "goal" in rollout["infos"][0]:
            goal = rollout["infos"][0]["goal"]
        else:
            goal = None

        # Get data histogram and colorbar.
        render_data_histogram = kwargs.get("render_data_histogram", False)
        colorbar = kwargs.get("colorbar", False)

        # Extract states from the rollout.
        states = rollout["observations"]
        if cut_on_reward and "rewards" in rollout:
            # Cut the rollout on the first positive reward signal.
            for i, reward in enumerate(rollout["rewards"]):
                if reward > 0:
                    states = states[: i + 1]
                    break

        # Plot background
        fig, ax = self._prepare_plot(render_data_histogram)

        # Plot trajectory
        self._plot_trajectory_from_states(states, colorbar=colorbar, ax=ax)

        # Plot goal
        if goal is not None:
            goal = self._plot_goal(goal, ax=ax)

        # Render the plot to an image
        img = plt_plot_to_np(fig)

        # Save the image
        save_image(img, path.join(self.render_dir, file_name))

    def render_rollout_video(
        self,
        rollout: Dict[str, np.ndarray],
        goal: Optional[np.ndarray] = None,
        plans: Optional[List[Dict[str, np.ndarray]]] = None,
        render_data_histogram: bool = False,
        cut_on_reward: bool = True,
        file_name: str = "rollout.mp4",
    ):
        """Render a video of the rollout.

        Renders a video of the rollout from the environment, including the trajectory, goal, and optionally plans and a
        data histogram.

        Args:
            rollout: The rollout to render.
            goal: The goal state.
            plans: The plans to render.
            render_data_histogram: If True, render the data histogram.
            cut_on_reward: If True, cut the video on the first positive reward signal.
            file_name: The name of the file to save the image to.

        """
        frames = []
        fig, ax = self._prepare_plot(render_data_histogram)
        if goal is not None:
            self._plot_goal(goal, ax=ax)

        # Render frames.
        for i in range(0, len(rollout["observations"]), self.video_frame_skip):
            render_plan = "plans" in rollout and "observations" in rollout["plans"][0] and i < len(rollout["plans"])
            if i > 0:
                past = self._plot_trajectory_from_states(
                    rollout["observations"][:i], colors="gray", size=4, alpha=1, ax=ax
                )
            else:
                past = None
            current = self._plot_state(rollout["observations"][i], color="gray", ax=ax)
            if render_plan:
                planned_obs = rollout["plans"][i]["observations"]
                plan = self._plot_trajectory_from_states(
                    planned_obs,
                    size=4,
                    alpha=0.25 + 0.75 * (np.arange(len(planned_obs)) > rollout["plans"][i]["step"]),
                    ax=ax,
                )
            else:
                plan = None
            img = plt_plot_to_np(fig)
            if past is not None:
                past.remove()
            current.remove()
            if plan is not None:
                plan.remove()
            frames.append(img)

            if cut_on_reward and i < len(rollout["rewards"]) and rollout["rewards"][i] > 0:
                break

        plt.close(fig)

        # Save the video
        save_video(np.array(frames), path.join(self.render_dir, file_name), fps=10)

    def render_rollout(self, rollout: Dict[str, np.ndarray], **kwargs):
        """Render a rollout and save it.

        Args:
            rollout (Dict[str, np.ndarray]): Rollout dictionary containing at least "observations".
            **kwargs: Additional keyword arguments for rendering, such as goal, render_data_histogram, colorbar.

        """
        self.render_rollout_image(rollout, **kwargs)
        self.render_rollout_video(rollout, **kwargs)

    def render_plan(
        self,
        plan: np.ndarray,
        goal: Optional[np.ndarray] = None,
        render_data_histogram: bool = False,
        colorbar: bool = False,
        file_name: str = "plan.png",
    ):
        """Render a single plan.

        Args:
            plan: The plan to render. [batch, timesteps, transition_dim]
            goal: The goal state. [batch, transition_dim]
            render_data_histogram: If True, render the data histogram.
            colorbar: If True, render the colorbar indicating the environment step.
            file_name: The name of the file to save the image to. The image will be saved under the
                specified name in the run_folder/render.

        """
        # Plot background
        fig, ax = self._prepare_plot(render_data_histogram)

        # Plot trajectory
        self._plot_trajectory_from_states(plan, colorbar=colorbar, ax=ax)

        # Plot goal
        if goal is not None:
            self._plot_goal(goal, ax=ax)

        # Render the plot to an image
        img = plt_plot_to_np(fig)
        plt.close(fig)

        # Save the image
        save_image(img, path.join(self.render_dir, file_name))

    def render_multiple_plans(
        self,
        plans: List[np.ndarray],
        labels: Optional[List[str]] = None,
        goal: Optional[np.ndarray] = None,
        render_data_histogram: bool = False,
        file_name: str = "plans.png",
    ):
        """Render multiple plans.

        Args:
            plans: The plans to render. [batch, timesteps, transition_dim]
            labels: The labels for the plans.
            goal: The goal state. [batch, transition_dim]
            render_data_histogram: If True, render the data histogram.
            colorbar: If True, render the colorbar indicating the environment step.
            file_name: The name of the file to save the image to. The image will be saved under the
                specified name in the run_folder/render.

        """
        # Plot background
        fig, ax = self._prepare_plot(render_data_histogram)

        # Plot trajectories
        colors = ["blue", "green", "orange", "purple", "brown", "pink", "gray", "black"]
        for i, plan in enumerate(plans):
            self._plot_trajectory_from_states(
                plan,
                colors=colors[i % len(colors)],
                ax=ax,
                label=labels[i] if labels is not None and i < len(labels) else None,
            )

        # Legend
        ax.legend(loc="lower right", fontsize=6)

        # Plot goal
        if goal is not None:
            self._plot_goal(goal, ax=ax)

        # Render the plot to an image
        img = plt_plot_to_np(fig)
        plt.close(fig)

        # Save the image
        save_image(img, path.join(self.render_dir, file_name))

    def compute_data_histogram(
        self,
        observations: np.ndarray,
        bins_per_tile: int = 20,
        smoothing_sigma: float = 0.0,
        process_fn: Optional[Callable] = None,
    ):
        """Compute the histogram of the observations and save it for later rendering.

        Args:
            observations: The observations to compute the histogram for.
            bins_per_tile: The number of bins per tile in the histogram.
            smoothing_sigma: The sigma for the gaussian smoothing of the histogram.
            process_fn: A function to process the observations before computing the histogram.

        """
        # Compute the raw histogram.
        histogram = np.zeros((bins_per_tile * int(self._extent[1]), bins_per_tile * int(self._extent[3])))
        for obs in observations:
            x = int(obs[0] * bins_per_tile)
            y = int(obs[1] * bins_per_tile)
            histogram[x, y] += 1

        # Smooth the histogram with a gaussian filter.
        if smoothing_sigma > 0:
            histogram = ndimage.gaussian_filter(histogram, sigma=smoothing_sigma)

        # Apply the process function.
        if process_fn is not None:
            for i in range(histogram.shape[0]):
                for j in range(histogram.shape[1]):
                    histogram[i, j] = process_fn(histogram[i, j])

        # Normalize the histogram.
        histogram = histogram / np.max(histogram)

        self._histogram = histogram

    def compute_gradient_field(
        self,
        gradient_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        norm_fn: Callable[[np.ndarray, str, bool], np.ndarray],
        unnorm_fn: Callable[[np.ndarray, str, bool], np.ndarray],
        action_dim: int = 0,
        observation_dim: int = 4,
        sigma: float = 0.002,
        horizon: int = 4,
        grid_len: int = 100,
        low: float = 0.5,
        high: float = 5.5,
        color: str = "magnitude",
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Compute the gradient field and save it for later rendering.

        Args:
            gradient_fn: The function that computes the gradient at a given state. It should be a mapping from the
                noisy sample and the sigma value to the value and gradient field:
                (torch.Tensor, torch.Tensor) -> (Tuple[torch.Tensor, torch.Tensor])
            norm_fn: The function that normalizes the state. (unnormalized, key, vector) -> normalized
            unnorm_fn: The function that unnormalizes the state. (normalized, key, vector) -> unnormalized
            action_dim: The dimension of the action space.
            observation_dim: The dimension of the observation space.
            sigma: The noise to add to the state.
            horizon: The number of steps to compute the gradient for.
            grid_len: The number of grid points in each dimension.
            low: The lower bound of the grid.
            high: The upper bound of the grid.
            color: The color of the gradient field. Can be "magnitude" or "value".
            dtype: The data type of the tensors.
            device: The device to use for computation (CPU or GPU).

        """
        assert color in ["magnitude", "value"]
        assert observation_dim >= 2

        # Compute the gradient field
        self.grads = np.zeros((grid_len, grid_len, 2))
        self.values = np.zeros((grid_len, grid_len))
        for i, x in enumerate(np.linspace(low, high, grid_len)):
            for j, y in enumerate(np.linspace(low, high, grid_len)):
                traj = np.concatenate(
                    [
                        np.random.randn(horizon, action_dim),
                        np.tile([x, y], (horizon, 1)),
                        np.random.randn(horizon, observation_dim - 2),
                    ],
                    axis=1,
                )[None, ...]
                norm_traj = np.concatenate(
                    (
                        norm_fn(traj[:, :, :action_dim], "actions", False)
                        if action_dim > 0
                        else traj[:, :, :action_dim],
                        norm_fn(traj[:, :, -observation_dim:], "observations", False),
                    ),
                    axis=2,
                )
                value, norm_grad = gradient_fn(
                    torch.tensor(norm_traj, device=device, dtype=dtype),
                    torch.tensor(sigma, device=device, dtype=dtype),
                )
                norm_grad = norm_grad.detach().cpu().numpy()
                value = value.detach().cpu().numpy()
                grad = np.concatenate(
                    (
                        unnorm_fn(
                            norm_grad[:, :, :action_dim],
                            "actions",
                            True,
                        )
                        if action_dim > 0
                        else norm_grad[:, :, :action_dim],
                        unnorm_fn(
                            norm_grad[:, :, -observation_dim:],
                            "observations",
                            True,
                        ),
                    ),
                    axis=2,
                )[0]
                grad = grad.max(axis=0)
                value = value / horizon
                self.grads[i, j] = grad[-observation_dim:]
                self.values[i, j] = value.max()

    def compute_value_field(
        self,
        value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        norm_fn: Callable[[np.ndarray, str, bool], np.ndarray],
        unnorm_fn: Callable[[np.ndarray, str, bool], np.ndarray],
        action_dim: int = 0,
        observation_dim: int = 4,
        sigma: Optional[float] = 0.002,
        horizon: int = 4,
        grid_len: int = 100,
        low: float = 0.5,
        high: float = 5.5,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Compute the value field and save it for later rendering.

        Args:
            value_fn: The function that computes the value at a given state.
            norm_fn: The function that normalizes the state.
            unnorm_fn: The function that unnormalizes the state.
            action_dim: The dimension of the action space.
            observation_dim: The dimension of the observation space.
            sigma: The noise level.
            horizon: The number of steps to compute the value for.
            grid_len: The number of grid points in each dimension.
            low: The lower bound of the grid.
            high: The upper bound of the grid.
            dtype: The data type of the tensors.
            device: The device to use for computation (CPU or GPU).

        """
        assert observation_dim >= 2

        self.values = np.zeros((grid_len, grid_len))
        for i, x in enumerate(np.linspace(low, high, grid_len)):
            for j, y in enumerate(np.linspace(low, high, grid_len)):
                # We only care about the x and y dimenstion. All other dims, including actions are sampled randomly.
                traj = np.concatenate(
                    [
                        np.random.rand(horizon, action_dim),
                        np.tile(np.array([x, y]), (horizon, 1)),
                        np.random.rand(horizon, observation_dim - 2),
                    ],
                    axis=1,
                )[None, ...]
                norm_traj = np.concatenate(
                    (
                        norm_fn(traj[:, :, :action_dim], "actions", False)
                        if action_dim > 0
                        else traj[:, :, :action_dim],
                        norm_fn(traj[:, :, -observation_dim:], "observations", False),
                    ),
                    axis=2,
                )
                value = value_fn(
                    torch.tensor(norm_traj, device=device, dtype=dtype),
                    torch.tensor(sigma, device=device, dtype=dtype),
                )
                value = value.detach().cpu().numpy()
                # The value is the sum of values over the horizon.
                self.values[i, j] = value.item() / horizon

    def render_value_field(
        self,
        low: float = 0.5,
        high: float = 5.5,
        log_color_scale: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        file_name: str = "values.png",
    ):
        """Render the value field.

        Args:
            low: The lower bound of the grid.
            high: The upper bound of the grid.
            log_color_scale: If True, use a logarithmic color scale for the value field.
            min_value: Minimum value for color normalization. If None, uses the minimum of the value field.
            max_value: Maximum value for color normalization. If None, uses the maximum of the value field.
            file_name: The name of the file to save the image to.

        """
        assert hasattr(self, "values")
        grid_len = self.values.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Compute the colors
        col_values = self.values
        max_value = float(np.max(col_values)) if max_value is None else max_value
        min_value = float(np.min(col_values)) if min_value is None else min_value

        # plot the values
        fig, ax = self._prepare_plot(render_data_histogram=False)
        cmap = colormaps.get_cmap("jet")
        if log_color_scale:
            norm = LogNorm(vmin=min_value, vmax=max_value)
        else:
            norm = Normalize(vmin=min_value, vmax=max_value)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # plt.grid(True)
        plt.xlim(low, high)
        plt.ylim(low, high)

        # plot the values
        for i, x in enumerate(torch.linspace(low, high, grid_len)):
            for j, y in enumerate(torch.linspace(low, high, grid_len)):
                value = col_values[i, j]
                ax.scatter(x, y, color=mappable.to_rgba(value), s=0.4)

        # Add colorbar.
        fig.colorbar(mappable, orientation="vertical", cax=fig.add_axes((0.85, 0.1, 0.02, 0.8)))

        img = plt_plot_to_np(fig)
        save_image(img, path.join(self.render_dir, file_name))

    def render_gradient_field(
        self,
        low: float = 0.5,
        high: float = 5.5,
        color: str = "magnitude",
        log_color_scale: bool = False,
        file_name: str = "gradients.png",
    ):
        """Render the gradient field.

        Args:
            low: The lower bound of the grid.
            high: The upper bound of the grid.
            color: The color of the gradient field. Can be "magnitude" or "value".
            log_color_scale: If True, use a log color scale.
            file_name: The name of the file to save the image to.

        """
        assert color in ["magnitude", "value"]
        assert hasattr(self, "grads")
        assert hasattr(self, "values")
        grid_len = self.grads.shape[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Compute the colors
        magnitudes = np.linalg.norm(self.grads, axis=2)
        if color == "magnitude":
            col_values: np.ndarray = magnitudes
            max_value = float(np.mean(col_values) + np.std(col_values))
            min_value = max(0.0, float(np.mean(col_values) - np.std(col_values)))
        elif color == "value":
            col_values = self.values
            max_value = float(np.max(col_values))
            min_value = float(np.min(col_values))
        else:
            raise ValueError(f"Unknown color: {color}. Must be 'magnitude' or 'value'.")

        # plot the gradients
        fig, ax = self._prepare_plot(render_data_histogram=False)
        cmap = colormaps.get_cmap("jet")
        if log_color_scale:
            norm = LogNorm(vmin=min_value, vmax=max_value)
        else:
            norm = Normalize(vmin=min_value, vmax=max_value)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        # plt.grid(True)
        plt.xlim(low, high)
        plt.ylim(low, high)

        #
        arrows = []
        for i, x in enumerate(np.linspace(low, high, grid_len)):
            for j, y in enumerate(np.linspace(low, high, grid_len)):
                magnitude = magnitudes[i, j]
                col_value = col_values[i, j]
                grad = self.grads[i, j]
                direction = 0.001 * grad / magnitude
                if magnitude > 1e-3:
                    direction = 0.001 * grad / magnitude
                    arrows.append(
                        ax.arrow(
                            x,
                            y,
                            direction[0],
                            direction[1],
                            head_width=2.0 / grid_len,
                            color=mappable.to_rgba(col_value),
                            linewidth=0.1,
                        )
                    )
                else:
                    arrows.append(ax.scatter(x, y, color=mappable.to_rgba(col_value), s=0.4))

        # Add colorbar.
        fig.colorbar(mappable, orientation="vertical", cax=fig.add_axes((0.85, 0.1, 0.02, 0.8)))

        img = plt_plot_to_np(fig)
        save_image(img, path.join(self.render_dir, file_name))

    def render_diffusion(
        self,
        steps: List[np.ndarray],
        gradients: Optional[Dict[str, np.ndarray]] = None,
        targets: Optional[Dict[str, np.ndarray]] = None,
        render_data_histogram: bool = False,
        file_name: str = "diffusion.mp4",
    ):
        """Render a video of the diffusion process.

        Args:
            steps: List of states at each diffusion step.
            gradients: Dict of gradients at each diffusion step.
            targets: Dict of targets at each diffusion step.
            render_data_histogram: If True, render the data histogram.
            file_name: Name of the video file

        """
        frames = []
        fig, ax = self._prepare_plot(render_data_histogram)
        for i in range(len(steps)):
            sample = self._plot_trajectory_from_states(steps[i], ax=ax)
            if targets is not None and i != len(steps) - 1:
                target_samples = self._plot_targets({k: v[i] for k, v in targets.items()}, ax=ax)
            if gradients is not None and i != len(steps) - 1:
                quivers = self._plot_gradients(steps[i], {k: v[i] for k, v in gradients.items()}, ax=ax)
            if ax.get_legend_handles_labels()[1]:
                ax.legend(loc="upper right", fontsize=6)
            img = plt_plot_to_np(fig)
            sample.remove()
            if gradients is not None and i != len(steps) - 1:
                for quiver in quivers:
                    quiver.remove()
            if targets is not None and i != len(steps) - 1:
                for target_sample in target_samples:
                    target_sample.remove()
            frames.append(img)
        plt.close(fig)

        # Save the video
        save_video(np.array(frames), file_path=path.join(self.render_dir, file_name), fps=1)

    def render_plan_and_rollout(self, plan, rollout):
        """Render both a plan and a rollout on the maze.

        Args:
            plan: The planned trajectory to render.
            rollout: The actual rollout trajectory to render.

        """
        raise NotImplementedError
