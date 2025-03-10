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
import jax
import flax.nnx as nnx

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
        pred_actions: Predicted actions of shape (B, T, action_dim) or (B, T, horizon, action_dim)
        true_actions: Ground truth actions of shape (B, T, action_dim)
        
    Returns:
        Dictionary of metrics including MSE for each dimension
    """
    # Convert to numpy arrays
    pred_actions = convert_to_numpy(pred_actions)
    true_actions = convert_to_numpy(true_actions)
    
    # If pred_actions has horizon dimension, take first action from each horizon
    if pred_actions.ndim == 4:  # Shape (B, T, horizon, action_dim)
        pred_actions = pred_actions[:, :, 0]  # -> (B, T, action_dim)
    
    # Ensure both arrays have same shape (B, T, action_dim)
    if pred_actions.ndim != true_actions.ndim or pred_actions.shape != true_actions.shape:
        raise ValueError(
            f"Shape mismatch: pred_actions shape {pred_actions.shape} != "
            f"true_actions shape {true_actions.shape}"
        )
    
    # Action dimension labels
    action_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    # Compute MSE for each dimension
    metrics = {}
    for i, label in enumerate(action_labels):
        dim_mse = float(np.mean((pred_actions[..., i] - true_actions[..., i]) ** 2))
        metrics[f"{label}_mse"] = dim_mse
    
    # Add overall MSE
    metrics["overall_mse"] = float(np.mean((pred_actions - true_actions) ** 2))
    
    # Add position, orientation, and gripper group MSEs
    metrics["position_mse"] = float(np.mean((pred_actions[..., :3] - true_actions[..., :3]) ** 2))
    metrics["orientation_mse"] = float(np.mean((pred_actions[..., 3:6] - true_actions[..., 3:6]) ** 2))
    metrics["gripper_mse"] = float(np.mean((pred_actions[..., 6:] - true_actions[..., 6:]) ** 2))
    
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


def plot_trajectory_comparison(
    actions: Union[np.ndarray, jnp.ndarray],
    pred_actions: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    title: str = "Action Comparison"
) -> Figure:
    """Plot comparison of ground truth vs predicted actions over a trajectory.
    
    Args:
        actions: Ground truth actions of shape (T, action_dim) where T is trajectory length
        pred_actions: Predicted actions of shape (T, action_dim)
        title: Title for the overall plot
    """
    # Convert to numpy and ensure 2D
    actions = convert_to_numpy(actions)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to be 2D array (T, action_dim), got shape {actions.shape}")
    
    T = len(actions)  # Full trajectory length
    action_dim = actions.shape[1]
    
    logging.info(f"Plotting trajectory comparison with T={T} timesteps")
    
    if pred_actions is not None:
        pred_actions = convert_to_numpy(pred_actions)
        if pred_actions.shape != actions.shape:
            raise ValueError(
                f"Predicted actions shape {pred_actions.shape} does not match "
                f"ground truth actions shape {actions.shape}"
            )
    
    # Create figure with subplots for each action dimension
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(title, fontsize=14, y=0.95)
    
    # Create timesteps array for x-axis
    timesteps = np.arange(T)
    
    # Create separate axes for each component group
    ax1 = plt.subplot(4, 1, 1)  # Position
    ax2 = plt.subplot(4, 1, 2)  # Orientation
    ax3 = plt.subplot(4, 1, 3)  # Gripper
    
    # Plot position dimensions (X, Y, Z)
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax1.plot(timesteps, actions[:, i], '-', label=f'Ground Truth {label}', alpha=0.7)
        if pred_actions is not None:
            ax1.plot(timesteps, pred_actions[:, i], '--', label=f'Predicted {label}', alpha=0.7)
    ax1.set_title('Position Components')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot orientation dimensions (Roll, Pitch, Yaw)
    for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
        ax2.plot(timesteps, actions[:, i+3], '-', label=f'Ground Truth {label}', alpha=0.7)
        if pred_actions is not None:
            ax2.plot(timesteps, pred_actions[:, i+3], '--', label=f'Predicted {label}', alpha=0.7)
    ax2.set_title('Orientation Components')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Angle (rad)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot gripper
    ax3.plot(timesteps, actions[:, 6], '-', label='Ground Truth Gripper', alpha=0.7)
    if pred_actions is not None:
        ax3.plot(timesteps, pred_actions[:, 6], '--', label='Predicted Gripper', alpha=0.7)
    ax3.set_title('Gripper State')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Gripper Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
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
    prefix: str = ""
) -> Dict[str, Union[Tuple[Any, Figure], wandb.Video]]:
    """Creates visualizations for EASO trajectory data.
    
    Args:
        observation: Observation object containing images and state
        actions: Ground truth actions of shape (T, action_dim) where T is trajectory length
        pred_actions: Predicted actions of shape (T, horizon, action_dim) or (T, action_dim)
        max_frames: Maximum number of frames to show in image visualization
        step: Current training step
        prefix: Prefix for visualization names
    """
    visualizations = {}
    
    # Convert and ensure actions are 2D numpy arrays
    actions = convert_to_numpy(actions)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions to be 2D array (T, action_dim), got shape {actions.shape}")
    
    # Plot action comparisons if we have predictions
    if pred_actions is not None:
        pred_actions = convert_to_numpy(pred_actions)
        
        # For each timestep t, take the first predicted action from the horizon
        # This gives us the model's best guess for what action to take at each timestep
        if pred_actions.ndim == 3:  # Shape (T, horizon, action_dim)
            pred_actions = pred_actions[:, 0]  # Take first action from each horizon -> (T, action_dim)
        elif pred_actions.ndim != 2:
            raise ValueError(f"Expected pred_actions to be 2D or 3D array, got shape {pred_actions.shape}")
        
        if pred_actions.shape != actions.shape:
            raise ValueError(
                f"Predicted actions shape {pred_actions.shape} does not match "
                f"ground truth actions shape {actions.shape}"
            )
        
        # Now both actions and pred_actions should be (T, action_dim)
        # Add batch dimension for metrics computation
        metrics = compute_trajectory_metrics(pred_actions[None], actions[None])
        visualizations[f"{prefix}_metrics" if prefix else "metrics"] = metrics

        # Create trajectory comparison plot
        fig_actions = plot_trajectory_comparison(
            actions, 
            pred_actions=pred_actions,
            title=f"{prefix.capitalize()} Trajectory Comparison" if prefix else "Trajectory Comparison"
        )
        visualizations[f"{prefix}_action_comparison" if prefix else "action_comparison"] = (wandb.Image(fig_actions), fig_actions)
    
    # Plot wrist camera images if available
    if observation.images is not None and "right_wrist_0_rgb" in observation.images:
        images = convert_to_numpy(observation.images["right_wrist_0_rgb"])
        
        # Normalize images
        if images.dtype != np.uint8:
            images = normalize_image_for_display(images)
            
        if len(images.shape) == 4:  # Sequence of images
            # Create video
            video_images = (images * 255).astype(np.uint8)
            visualizations[f"{prefix}_wrist_camera_video" if prefix else "wrist_camera_video"] = create_trajectory_video(video_images)
            
            # Create static visualization of key frames
            indices = np.linspace(0, len(images) - 1, min(max_frames, len(images))).astype(int)
            fig_images = plt.figure(figsize=(4 * len(indices), 4))
            for i, idx in enumerate(indices):
                plt.subplot(1, len(indices), i + 1)
                plt.imshow(images[idx])
                plt.axis("off")
                plt.title(f"{prefix.capitalize() + ' ' if prefix else ''}Frame {idx}")
            
            visualizations[f"{prefix}_wrist_camera" if prefix else "wrist_camera"] = (wandb.Image(fig_images), fig_images)
        else:  # Single image
            fig_images = plt.figure(figsize=(6, 6))
            plt.imshow(images)
            plt.axis("off")
            plt.title(f"{prefix.capitalize() + ' ' if prefix else ''}Wrist Camera View")
            visualizations[f"{prefix}_wrist_camera" if prefix else "wrist_camera"] = (wandb.Image(fig_images), fig_images)

    return visualizations


@dataclass
class VisualizationCallback:
    """Callback for visualizing training progress and dataset validation."""
    
    config: TrainConfig
    cache: VisualizationCache = field(default_factory=lambda: VisualizationCache())
    
    def infer_full_trajectory(
        self,
        model: nnx.Module,
        observation: _model.Observation,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Infer the complete trajectory using the current model state.
        
        Args:
            model: Current model in eval mode
            observation: Initial observation
            actions: Ground truth actions for trajectory length reference
            
        Returns:
            Predicted action sequence for the full trajectory
        """
        # Get trajectory length from ground truth actions
        traj_length = actions.shape[0]
        
        # Set model to evaluation mode
        model.eval()
        
        # Add batch dimension to observation components and ensure tokenized inputs exist
        batched_obs = _model.Observation(
            images={k: v[None] if v is not None else None for k, v in observation.images.items()} if observation.images is not None else None,
            state=observation.state[None] if observation.state is not None else None,
            tokenized_prompt=jnp.zeros((1, model.max_token_len), dtype=jnp.int32),  # Empty prompt
            tokenized_prompt_mask=jnp.ones((1, model.max_token_len), dtype=jnp.bool_),  # All tokens valid
            image_masks={k: jnp.ones((1,), dtype=jnp.bool_) for k in observation.images.keys()} if observation.images is not None else None
        )
        
        # Add batch dimension to actions
        batched_actions = actions[None]
        
        # Use the model's sample_actions method to get the full trajectory prediction
        predicted_actions = model.sample_actions(
            jax.random.PRNGKey(0),  # Fixed seed for deterministic inference
            batched_obs,
            num_steps=10  # Number of denoising steps
        )
        
        # Remove batch dimension and return
        return np.array(predicted_actions[0])

    def _update_state(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Update the state based on the predicted action.
        State contains: [joint_angles (7), eef_pose (6)]
        Action contains: [x, y, z, roll, pitch, yaw, gripper]
        """
        new_state = current_state.copy()
        # Update end-effector pose (positions and orientations) based on action
        new_state[7:13] += action[:6]  # Update eef_pose with xyz and rpy deltas
        # Note: We don't update joint angles as they would require inverse kinematics
        return new_state

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

        matplotlib_figures = []
        
        # Extract observation and actions from batch
        observation, actions = batch
        
        # Get model from train state
        model = nnx.merge(info["model_def"], info["params"])
        
        # For batched data, visualize up to max_trajectories
        batch_size = actions.shape[0]
        for i in range(min(self.config.max_trajectories_to_visualize, batch_size)):
            # Extract single example from batch
            single_obs = _model.Observation(
                images={k: v[i] for k, v in observation.images.items()} if observation.images is not None else None,
                state=observation.state[i] if observation.state is not None else None,
                tokenized_prompt=observation.tokenized_prompt[i] if observation.tokenized_prompt is not None else None,
                image_masks={k: v[i] for k, v in observation.image_masks.items()} if observation.image_masks is not None else None
            )
            
            # Extract ground truth actions for this example
            single_actions = convert_to_numpy(actions[i])
            
            # Get full trajectory prediction using current model state
            predicted_trajectory = self.infer_full_trajectory(model, single_obs, single_actions)
            
            # Create comparison plots - separate into position, orientation, and gripper
            fig = plt.figure(figsize=(15, 12))
            plt.suptitle(f'Full Trajectory Comparison (Step {step})')
            
            # Plot position (x,y,z)
            ax1 = plt.subplot(3, 1, 1)
            for j, label in enumerate(['X', 'Y', 'Z']):
                ax1.plot(single_actions[:, j], label=f'GT {label}', linestyle='-')
                ax1.plot(predicted_trajectory[:, j], label=f'Pred {label}', linestyle='--')
            ax1.set_title('End-Effector Position')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Position (m)')
            ax1.legend()
            ax1.grid(True)
            
            # Plot orientation (roll,pitch,yaw)
            ax2 = plt.subplot(3, 1, 2)
            for j, label in enumerate(['Roll', 'Pitch', 'Yaw']):
                ax2.plot(single_actions[:, j+3], label=f'GT {label}', linestyle='-')
                ax2.plot(predicted_trajectory[:, j+3], label=f'Pred {label}', linestyle='--')
            ax2.set_title('End-Effector Orientation')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Angle (rad)')
            ax2.legend()
            ax2.grid(True)
            
            # Plot gripper
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(single_actions[:, 6], label='GT Gripper', linestyle='-')
            ax3.plot(predicted_trajectory[:, 6], label='Pred Gripper', linestyle='--')
            ax3.set_title('Gripper State')
            ax3.set_xlabel('Timestep')
            ax3.set_ylabel('Gripper Value')
            ax3.legend()
            ax3.grid(True)
            
            plt.tight_layout()
            
            # Calculate metrics for each component
            position_mse = np.mean((single_actions[:, :3] - predicted_trajectory[:, :3]) ** 2)
            orientation_mse = np.mean((single_actions[:, 3:6] - predicted_trajectory[:, 3:6]) ** 2)
            gripper_mse = np.mean((single_actions[:, 6] - predicted_trajectory[:, 6]) ** 2)
            
            # Log to wandb
            wandb.log({
                f'trajectory_{i}/full_comparison': wandb.Image(fig),
                f'trajectory_{i}/position_mse': position_mse,
                f'trajectory_{i}/orientation_mse': orientation_mse,
                f'trajectory_{i}/gripper_mse': gripper_mse,
                f'trajectory_{i}/total_mse': np.mean((single_actions - predicted_trajectory) ** 2),
            }, step=step)
            
            matplotlib_figures.append(fig)
            
            # Also generate original visualizations
            traj_viz = plot_easo_trajectory(
                single_obs,
                single_actions,
                pred_actions=predicted_trajectory,
                max_frames=self.config.max_frames_per_trajectory,
                step=step,
                prefix="train"
            )
            
            # Log original visualizations
            for key, viz_obj in traj_viz.items():
                viz_key = f"trajectory_{i}/{key}"
                
                if isinstance(viz_obj, tuple):
                    wandb_obj, fig = viz_obj
                    wandb.log({viz_key: wandb_obj}, step=step)
                    if isinstance(fig, plt.Figure):
                        matplotlib_figures.append(fig)
                elif isinstance(viz_obj, wandb.Video):
                    wandb.log({viz_key: viz_obj}, step=step)
                elif isinstance(viz_obj, dict):
                    for metric_name, value in viz_obj.items():
                        if metric_name.endswith('_mse'):
                            wandb.log({f"train_mse/{metric_name}": value}, step=step)
        
        # Clean up matplotlib figures
        for fig in matplotlib_figures:
            plt.close(fig)
            
        return info

    def process_validation_step(
        self,
        step: int,
        batch: tuple[_model.Observation, _model.Actions],
        model_outputs: Dict[str, Any],
    ) -> None:
        """Process validation step and generate visualizations."""
        if not self.config.visualization_enabled:
            return

        if step % self.config.visualization_interval != 0:
            return

        matplotlib_figures = []
        
        # Extract observation and actions from batch
        observation, actions = batch
        
        # Get predicted actions from model outputs
        pred_actions = model_outputs.get("predicted_actions")
        if pred_actions is None:
            logging.warning("No predicted actions found in validation outputs")
            return
            
        # Log shapes for debugging
        logging.info(f"Raw validation actions shape: {actions.shape}")
        logging.info(f"Raw validation predicted actions shape: {pred_actions.shape}")
        
        # For batched data, visualize up to max_trajectories
        batch_size = actions.shape[0]
        for i in range(min(self.config.max_trajectories_to_visualize, batch_size)):
            # Extract single example from batch
            single_obs = _model.Observation(
                images={k: v[i] for k, v in observation.images.items()} if observation.images is not None else None,
                state=observation.state[i] if observation.state is not None else None,
                tokenized_prompt=observation.tokenized_prompt[i] if observation.tokenized_prompt is not None else None,
                image_masks={k: v[i] for k, v in observation.image_masks.items()} if observation.image_masks is not None else None
            )
            
            # Extract full trajectory for this example
            # For ground truth actions, reshape from (B, T, action_dim) to (T, action_dim)
            single_actions = convert_to_numpy(actions[i])  # Shape (T, action_dim)
            
            # For predicted actions, if shape is (B, T, horizon, action_dim), 
            # take first action from each horizon and reshape to (T, action_dim)
            single_pred_actions = convert_to_numpy(pred_actions[i])
            if single_pred_actions.ndim == 3:  # Shape (T, horizon, action_dim)
                single_pred_actions = single_pred_actions[:, 0]  # Take first action -> (T, action_dim)
            
            # Log shapes for debugging
            logging.info(f"Single validation actions shape: {single_actions.shape}")
            logging.info(f"Single validation predicted actions shape: {single_pred_actions.shape}")
            
            # Generate visualizations with val prefix
            traj_viz = plot_easo_trajectory(
                single_obs,
                single_actions,
                pred_actions=single_pred_actions,
                max_frames=self.config.max_frames_per_trajectory,
                step=step,
                prefix="val"  # Add prefix for validation visualizations
            )
            
            # Log visualizations directly to wandb
            for key, viz_obj in traj_viz.items():
                viz_key = f"val_trajectory_{i}/{key}"
                
                if isinstance(viz_obj, tuple):  # (wandb object, figure)
                    wandb_obj, fig = viz_obj
                    wandb.log({viz_key: wandb_obj}, step=step)
                    if isinstance(fig, plt.Figure):  # Only append matplotlib figures
                        matplotlib_figures.append(fig)
                elif isinstance(viz_obj, wandb.Video):  # Video
                    wandb.log({viz_key: viz_obj}, step=step)
                elif isinstance(viz_obj, dict):  # Metrics
                    # Log each MSE value as a separate line in wandb with val prefix
                    for metric_name, value in viz_obj.items():
                        if metric_name.endswith('_mse'):
                            wandb.log({f"val_mse/{metric_name}": value}, step=step)
        
        # Clean up only matplotlib figures
        for fig in matplotlib_figures:
            plt.close(fig)


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