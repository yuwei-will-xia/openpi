"""Visualization utilities for training and dataset validation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Tuple, List
import os
import logging
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.figure import Figure
import plotly.graph_objects as go
from tqdm import tqdm
import jax.numpy as jnp

from openpi.training.config import TrainConfig
import openpi.models.model as _model


@dataclass
class VisualizationCache:
    """Cache for storing visualization data to reuse across steps."""
    max_trajectories: int = 100
    _cache: Dict[str, Any] = field(default_factory=OrderedDict)
    
    def cache_trajectory(self, key: str, data: Any) -> None:
        """Cache trajectory data with LRU eviction policy."""
        if len(self._cache) >= self.max_trajectories:
            # Remove oldest entry
            self._cache.popitem(last=False)
        self._cache[key] = data
        
    def get_trajectory(self, key: str) -> Optional[Any]:
        """Retrieve cached trajectory data."""
        return self._cache.get(key)


def compute_trajectory_metrics(pred_actions: np.ndarray, true_actions: np.ndarray) -> Dict[str, float]:
    """Compute metrics between predicted and true actions.
    
    Args:
        pred_actions: Predicted actions of shape (B, action_dim) or (B, T, action_dim)
        true_actions: Ground truth actions of shape (B, action_dim) or (B, T, action_dim)
        
    Returns:
        Dictionary of metrics including MSE for each dimension using only the first action
    """
    # Take only the first action if we have a trajectory
    if pred_actions.ndim == 3:
        pred_actions = pred_actions[:, 0]  # (B, action_dim)
    if true_actions.ndim == 3:
        true_actions = true_actions[:, 0]  # (B, action_dim)
    
    # Ensure we're working with numpy arrays
    pred_actions = convert_to_numpy(pred_actions)
    true_actions = convert_to_numpy(true_actions)
    
    # Action dimension labels
    action_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    # Compute MSE for each dimension
    metrics = {}
    for i, label in enumerate(action_labels):
        dim_mse = float(np.mean((pred_actions[:, i] - true_actions[:, i]) ** 2))
        metrics[f"{label}_mse"] = dim_mse
        # logging.info(f"{label.upper()} MSE: {dim_mse}")
    
    return metrics


def create_trajectory_video(images: np.ndarray, fps: int = 10) -> wandb.Video:
    """Create a video from a sequence of images.
    
    Args:
        images: Array of shape (T, H, W, C) containing image sequence
        fps: Frames per second for the video
        
    Returns:
        wandb.Video object for logging
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4D array (T,H,W,C), got shape {images.shape}")
    
    # Normalize images if needed
    if images.dtype != np.uint8:
        images = (normalize_image_for_display(images) * 255).astype(np.uint8)
        
    return wandb.Video(
        images.transpose(0, 3, 1, 2),  # Convert to (T,C,H,W) for wandb
        fps=fps
    )


def plot_interactive_trajectory(
    actions: np.ndarray,
    state: Optional[np.ndarray] = None,
    pred_actions: Optional[np.ndarray] = None
) -> go.Figure:
    """Create an interactive Plotly figure for trajectory visualization.
    
    Args:
        actions: Ground truth actions of shape (T, action_dim)
        state: Optional state trajectory of shape (T, state_dim)
        pred_actions: Optional predicted actions of shape (T, action_dim)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Ensure actions is 2D
    actions = ensure_2d(actions)
    # logging.info(f"Actions shape in interactive plot: {actions.shape}")
    
    # Action dimension labels and groups
    action_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
    action_units = ['m', 'm', 'm', 'rad', 'rad', 'rad', '']
    
    # Create subplots for position, orientation, and gripper
    fig = go.Figure()
    
    # Plot ground truth actions
    # Position (XYZ)
    for i in range(3):
        fig.add_trace(go.Scatter(
            y=actions[:, i],
            name=f"{action_labels[i]} (true)",
            mode='lines',
            line=dict(width=2),
            yaxis="y1"
        ))
    
    # Orientation (RPY)
    for i in range(3, 6):
        fig.add_trace(go.Scatter(
            y=actions[:, i],
            name=f"{action_labels[i]} (true)",
            mode='lines',
            line=dict(width=2),
            yaxis="y2"
        ))
    
    # Gripper
    fig.add_trace(go.Scatter(
        y=actions[:, 6],
        name="Gripper (true)",
        mode='lines',
        line=dict(width=2),
        yaxis="y3"
    ))
        
    # Plot predicted actions if available
    if pred_actions is not None:
        pred_actions = ensure_2d(pred_actions)
        # logging.info(f"Predicted actions shape in interactive plot: {pred_actions.shape}")
        
        # Position (XYZ)
        for i in range(3):
            fig.add_trace(go.Scatter(
                y=pred_actions[:, i],
                name=f"{action_labels[i]} (pred)",
                mode='lines',
                line=dict(dash='dash', width=2),
                yaxis="y1"
            ))
        
        # Orientation (RPY)
        for i in range(3, 6):
            fig.add_trace(go.Scatter(
                y=pred_actions[:, i],
                name=f"{action_labels[i]} (pred)",
                mode='lines',
                line=dict(dash='dash', width=2),
                yaxis="y2"
            ))
        
        # Gripper
        fig.add_trace(go.Scatter(
            y=pred_actions[:, 6],
            name="Gripper (pred)",
            mode='lines',
            line=dict(dash='dash', width=2),
            yaxis="y3"
        ))
    
    # Update layout with three y-axes
    fig.update_layout(
        title="Robot Trajectory",
        xaxis=dict(title="Time Step"),
        yaxis=dict(
            title="Position (m)",
            side="left",
            domain=[0.7, 1.0]
        ),
        yaxis2=dict(
            title="Orientation (rad)",
            side="left",
            domain=[0.35, 0.65]
        ),
        yaxis3=dict(
            title="Gripper",
            side="left",
            domain=[0, 0.3]
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def convert_to_numpy(x: Any) -> np.ndarray:
    """Convert JAX arrays or other types to numpy arrays."""
    if isinstance(x, jnp.ndarray):
        return np.array(x)
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure array is 2D by adding a time dimension if needed."""
    if x.ndim == 1:
        return x[np.newaxis, :]  # Add time dimension
    return x


def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize image data for display.
    
    Args:
        image: Input image array, can be in range [-1, 1] or [0, 255]
        
    Returns:
        Normalized image in range [0, 1]
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    else:
        # If image is in [-1, 1] range, convert to [0, 1]
        if image.min() < 0:
            image = (image + 1.0) / 2.0
        return np.clip(image, 0, 1)


def plot_mse_over_time(step: int, metrics: Dict[str, float]) -> Figure:
    """Plot MSE values over training steps.
    
    Args:
        step: Current training step
        metrics: Dictionary of MSE values for each dimension
        
    Returns:
        Matplotlib figure showing MSE trends
    """
    action_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot individual MSE values
    for i, label in enumerate(action_labels):
        plt.subplot(3, 3, i + 1)
        mse_value = metrics[f"{label}_mse"]
        plt.scatter(step, mse_value, alpha=0.6)
        plt.title(f"{label.upper()} MSE")
        plt.xlabel("Training Step")
        plt.ylabel("MSE")
        plt.grid(True)
        
        # Use log scale if values are very small
        if mse_value < 1e-3:
            plt.yscale('log')
    
    # Plot grouped MSE values
    plt.subplot(3, 3, 8)
    for group in ['position', 'orientation', 'gripper']:
        plt.scatter(step, metrics[f"{group}_mse"], label=group, alpha=0.6)
    plt.title("Grouped MSE")
    plt.xlabel("Training Step")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    
    # Plot overall MSE
    plt.subplot(3, 3, 9)
    plt.scatter(step, metrics["overall_mse"], alpha=0.6)
    plt.title("Overall MSE")
    plt.xlabel("Training Step")
    plt.ylabel("MSE")
    plt.grid(True)
    
    plt.tight_layout()
    return fig


def plot_easo_trajectory(
    observation: _model.Observation,
    actions: Union[np.ndarray, jnp.ndarray],
    pred_actions: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    max_frames: int = 5,
    step: Optional[int] = None,
) -> Dict[str, Union[Tuple[wandb.Image, Figure], wandb.Video]]:
    """Creates visualizations for EASO trajectory data.
    
    Args:
        observation: OpenPI Observation object
        actions: Ground truth actions
        pred_actions: Optional predicted actions for comparison
        max_frames: Maximum number of frames to show in visualization
        step: Current training step
    """
    visualizations = {}
    
    # Compute metrics if we have predicted actions
    if pred_actions is not None:
        metrics = compute_trajectory_metrics(pred_actions, actions)
        visualizations["metrics"] = metrics

    # Convert and ensure actions are 2D
    actions = ensure_2d(convert_to_numpy(actions))
    
    # Plot action comparisons if we have predictions
    if pred_actions is not None:
        pred_actions = ensure_2d(convert_to_numpy(pred_actions))
        action_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']
        action_units = ['m', 'm', 'm', 'rad', 'rad', 'rad', '']
        
        fig_actions = plt.figure(figsize=(20, 12))
        for i in range(7):
            plt.subplot(3, 3, i + 1)
            plt.plot(actions[:, i], label='Ground Truth', color='blue', alpha=0.7)
            plt.plot(pred_actions[:, i], label='Predicted', color='red', alpha=0.7, linestyle='--')
            plt.title(f"{action_labels[i]}")
            plt.xlabel("Time Step")
            plt.ylabel(f"Value ({action_units[i]})")
            plt.grid(True)
            if i == 0:  # Only show legend once
                plt.legend()
            
            # Add MSE value to the title
            mse = float(np.mean((pred_actions[:, i] - actions[:, i]) ** 2))
            plt.title(f"{action_labels[i]}\nMSE: {mse:.6f}")
        
        plt.tight_layout()
        visualizations["action_comparison"] = (wandb.Image(fig_actions), fig_actions)
    
    # Plot wrist camera images if available
    if observation.images is not None and "right_wrist_0_rgb" in observation.images:
        images = convert_to_numpy(observation.images["right_wrist_0_rgb"])
        # logging.info(f"Image shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Normalize images
        if images.dtype != np.uint8:
            images = normalize_image_for_display(images)
            
        if len(images.shape) == 4:  # Sequence of images
            # Create video
            video_images = (images * 255).astype(np.uint8)
            visualizations["wrist_camera_video"] = create_trajectory_video(video_images)
            
            # Create static visualization of key frames
            indices = np.linspace(0, len(images) - 1, min(max_frames, len(images))).astype(int)
            fig_images = plt.figure(figsize=(4 * len(indices), 4))
            for i, idx in enumerate(indices):
                plt.subplot(1, len(indices), i + 1)
                plt.imshow(images[idx])
                plt.axis("off")
                plt.title(f"Frame {idx}")
        else:  # Single image
            fig_images = plt.figure(figsize=(6, 6))
            plt.imshow(images)
            plt.axis("off")
            plt.title("Wrist Camera View")
        
        visualizations["wrist_camera"] = (wandb.Image(fig_images), fig_images)

    return visualizations


@dataclass
class VisualizationCallback:
    """Callback for visualizing training progress and dataset validation."""
    
    config: TrainConfig
    cache: VisualizationCache = field(default_factory=lambda: VisualizationCache())
    
    def process_train_step(
        self,
        step: int,
        info: Dict[str, Any],
        batch: tuple[_model.Observation, _model.Actions],
        model_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process training step and generate visualizations."""
        if not self.config.visualization_enabled:
            return info

        if step % self.config.visualization_interval != 0:
            return info

        figures_to_save = []
        
        # Extract observation and actions from batch
        observation, actions = batch
        
        # Get predicted actions from model outputs
        pred_actions = model_outputs.get("predicted_actions")
        if pred_actions is None:
            logging.warning("No predicted actions found in model outputs")
            return info
            
        # For batched data, visualize up to max_trajectories
        batch_size = actions.shape[0]
        for i in range(min(self.config.max_trajectories_to_visualize, batch_size)):
            # Extract single example from batch
            single_obs = _model.Observation(
                images={k: v[i] for k, v in observation.images.items()} if observation.images is not None else None,
                image_masks={k: v[i] for k, v in observation.image_masks.items()} if observation.image_masks is not None else None,
                state=observation.state[i] if observation.state is not None else None,
                tokenized_prompt=observation.tokenized_prompt[i] if observation.tokenized_prompt is not None else None,
                tokenized_prompt_mask=observation.tokenized_prompt_mask[i] if observation.tokenized_prompt_mask is not None else None,
            )
            
            single_actions = actions[i]
            single_pred_actions = pred_actions[i] if pred_actions is not None else None
            
            # Generate visualizations
            traj_viz = plot_easo_trajectory(
                single_obs,
                single_actions,
                pred_actions=single_pred_actions,
                max_frames=self.config.max_frames_per_trajectory,
                step=step,
            )
            
            # Log visualizations directly to wandb
            for key, viz_obj in traj_viz.items():
                viz_key = f"trajectory_{i}/{key}"
                
                if isinstance(viz_obj, tuple):  # wandb.Image and matplotlib Figure
                    wandb_img, fig = viz_obj
                    wandb.log({viz_key: wandb_img}, step=step)
                    figures_to_save.append((viz_key, fig))
                elif isinstance(viz_obj, wandb.Video):  # Video
                    wandb.log({viz_key: viz_obj}, step=step)
                elif isinstance(viz_obj, dict):  # Metrics
                    # Log each MSE value as a separate line in wandb
                    for metric_name, value in viz_obj.items():
                        if metric_name.endswith('_mse'):
                            wandb.log({f"mse/{metric_name}": value}, step=step)
        
        # Clean up figures
        for _, fig in figures_to_save:
            plt.close(fig)
            
        return info


def create_visualization_callback(config: TrainConfig) -> VisualizationCallback:
    """Creates a visualization callback for training.
    
    Args:
        config: Training configuration
        
    Returns:
        VisualizationCallback instance
    """
    return VisualizationCallback(config=config)


def process_train_step(
    model_outputs: dict,
    batch: dict,
    step: int,
    *,
    max_frames: int = 5,
    log_prefix: str = "train",
) -> dict:
    """Process a single training step and generate visualizations.

    Args:
        model_outputs: Dictionary containing model outputs including:
            - predicted_actions: Model's predicted actions
            - loss: Loss values
        batch: Dictionary containing training data including:
            - observation: OpenPI Observation object
            - actions: Ground truth actions
        step: Current training step
        max_frames: Maximum number of frames to show in visualization
        log_prefix: Prefix for wandb logging

    Returns:
        Dictionary of wandb logs
    """
    # Extract predicted actions from model outputs
    # For pi0 model, predicted actions are the output of sample_actions
    # For pi0-fast model, they need to be decoded from the output tokens
    pred_actions = None
    if "predicted_actions" in model_outputs:
        pred_actions = model_outputs["predicted_actions"]
    elif "actions" in model_outputs:
        # For backwards compatibility
        pred_actions = model_outputs["actions"]
    
    # Log warning if no predicted actions found
    if pred_actions is None:
        logging.warning("No predicted actions found in model outputs. Keys: %s", list(model_outputs.keys()))
        return {}

    # Convert and ensure actions are 2D numpy arrays
    actions = ensure_2d(convert_to_numpy(batch["actions"]))
    pred_actions = ensure_2d(convert_to_numpy(pred_actions))
    
    # Log shapes and first few values for debugging
    # logging.info(f"Actions shape: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")
    # logging.info(f"Predicted actions shape: {pred_actions.shape}, range: [{pred_actions.min():.3f}, {pred_actions.max():.3f}]")
    # logging.info(f"First true action: {actions[0, :7]}")
    # logging.info(f"First predicted action: {pred_actions[0, :7]}")

    # Compute average absolute difference between predicted and true actions
    avg_diff = np.mean(np.abs(pred_actions - actions))
    # logging.info(f"Average absolute difference between predicted and true actions: {avg_diff:.3f}")

    # Generate visualizations
    visualizations = plot_easo_trajectory(
        observation=batch["observation"],
        actions=actions,
        pred_actions=pred_actions,
        max_frames=max_frames,
        step=step,
    )

    # Create wandb logs
    logs = {}
    for name, viz in visualizations.items():
        logs[f"{log_prefix}/{name}"] = viz

    # Add metrics
    metrics = compute_trajectory_metrics(pred_actions, actions)
    for name, value in metrics.items():
        logs[f"{log_prefix}/metrics/{name}"] = value

    return logs 