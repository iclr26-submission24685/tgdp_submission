"""Utility functions for rendering images and videos."""

import logging
from typing import List, Optional

import cv2

# import skvideo.io
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def tile_images(
    images: List[np.ndarray],
    rows: Optional[int],
    cols: Optional[int],
) -> np.ndarray:
    """Tile a list of images into a single image.

    Args:
        images: A list of numpy arrays of shape (H, W, C) containing the images.
        rows: The number of rows in the tiled image.
        cols: The number of columns in the tiled image.

    Returns:
        A numpy array containing the tiled image.

    """
    if rows is None and cols is None:
        rows = 1
        cols = len(images)
    elif rows is None and cols is not None:
        rows = len(images) // cols
    elif cols is None and rows is not None:
        cols = len(images) // rows
    else:
        assert rows is not None and cols is not None
    if len(images) != rows * cols:
        raise ValueError("Number of images does not match rows * cols")

    # Get the shape of the images
    img_height, img_width, img_channels = images[0].shape

    # Create an empty array to hold the tiled image
    tiled_image = np.zeros((rows * img_height, cols * img_width, img_channels), dtype=images[0].dtype)

    # Fill the tiled image with the individual images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        tiled_image[row * img_height : (row + 1) * img_height, col * img_width : (col + 1) * img_width, :] = img

    return tiled_image


def tile_videos(
    video_frames: List[np.ndarray],
    rows: Optional[int],
    cols: Optional[int],
) -> np.ndarray:
    """Tile a list of videos into a single video.

    Args:
        video_frames: A list of numpy arrays of shape (T, H, W, C) containing the video frames.
        rows: The number of rows in the tiled video.
        cols: The number of columns in the tiled video.

    Returns:
        A numpy array containing the tiled video.

    """
    # Get the number of frames in the video
    num_frames = video_frames[0].shape[0]

    # Initialize a list to hold the tiled frames
    tiled_frames = []

    # Tile each frame across all videos
    for frame_idx in range(num_frames):
        frames = [video[frame_idx] for video in video_frames]
        tiled_frame = tile_images(frames, rows, cols)
        tiled_frames.append(tiled_frame)

    # Stack the tiled frames into a single numpy array
    tiled_video = np.stack(tiled_frames, axis=0)

    return tiled_video


def plt_plot_to_np(fig: Figure, remove_margins: bool = True):
    """Convert a matplotlib plot to a numpy array.

    Args:
        fig: The matplotlib figure to convert.
        remove_margins: Whether to remove the margins from the plot.

    """
    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(img_as_string, dtype="uint8").reshape((height, width, 4))


def save_image(
    image: np.ndarray,
    file_path: str,
):
    """Save an image to disk.

    Args:
        image: A numpy array of shape (H, W, C) containing the image.
        file_path: The path to save the image to.

    """
    logger.debug(f"Saving image to {file_path}.")

    # Ensure the array is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # OpenCV expects images in BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save the image as PNG using OpenCV
    cv2.imwrite(file_path, image)


def save_video(
    video_frames: np.ndarray,
    file_path: str,
    fps: int = 60,
    video_format: str = "mp4",  # 'mp4' or 'avi'
):
    """Save a video to disk.

    Args:
        video_frames: A numpy array of shape (T, H, W, C) containing the video frames.
        file_path: The path to save the video to.
        fps: The frames per second of the video.
        video_format: The format of the video.

    """
    logger.debug(f"Saving video to {file_path}.")

    # Ensure the array is in uint8 format
    if video_frames.dtype != np.uint8:
        video_frames = (video_frames * 255).astype(np.uint8)

    # Determine the codec based on the video format
    if video_format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # type: ignore
    elif video_format == "avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore
    else:
        raise ValueError(f"Unsupported video format: {video_format}")

    # Get the size of the frames
    height, width = video_frames.shape[1:3]

    # Initialize the video writer
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in video_frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    # Release the video writer
    out.release()
