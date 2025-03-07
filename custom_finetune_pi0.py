#!/usr/bin/env python3
"""
Script to finetune the pi0 model using the specific lerobot dataset at /home/willx/data/try1.

This script:
1. Sets up a custom policy configuration for the specific dataset
2. Creates a training configuration
3. Computes normalization statistics
4. Runs the finetuning process

Usage:
uv run custom_finetune_pi0.py --data_dir /home/willx/data/try1 --model_type pi0 --exp_name my_experiment

Arguments:
--data_dir: Path to your local lerobot dataset
--model_type: Type of model to finetune (pi0 or pi0_fast)
--exp_name: Name of the experiment
--batch_size: Batch size for training (default: 32)
--num_train_steps: Number of training steps (default: 30000)
--overwrite: Whether to overwrite existing checkpoints
"""

import dataclasses
import os
import pathlib
import subprocess
import sys
from typing import Literal

import einops
import etils.epath as epath
import flax.nnx as nnx
import numpy as np
import tyro

# Add the repository to the Python path
sys.path.insert(0, os.path.abspath("."))

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.shared.download as _download
import openpi.training.config as _config
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class CustomInputs(_transforms.DataTransformFn):
    """
    Custom input transformation for the dataset at /home/willx/data/try1.
    
    Based on the dataset structure, it has the following keys:
    - observation.joint_angles: 7-dimensional joint angles
    - observation.eef_pose: 6-dimensional end-effector pose
    - observation.target_eef_pose: 6-dimensional target end-effector pose
    - observation.images.wrist_camera_right: Wrist camera image
    - action: 10-dimensional action
    """
    # The action dimension of the model
    action_dim: int
    
    # Determines which model will be used
    model_type: _model.ModelType = _model.ModelType.PI0
    
    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0
        
        # Create a state vector by concatenating joint angles and end-effector pose
        # Modify this based on what you want to include in the state
        joint_angles_key = "observation.joint_angles"
        eef_pose_key = "observation.eef_pose"
        
        if joint_angles_key in data and eef_pose_key in data:
            # Concatenate joint angles and end-effector pose
            state = np.concatenate([
                data[joint_angles_key],
                data[eef_pose_key]
            ])
        elif joint_angles_key in data:
            state = data[joint_angles_key]
        elif eef_pose_key in data:
            state = data[eef_pose_key]
        else:
            # Fallback to empty state if neither is available
            state = np.zeros(self.action_dim)
        
        # Pad the state to the action dimension of the model
        state = _transforms.pad_to_dim(state, self.action_dim)
        
        # Handle the wrist camera image
        wrist_image_key = "observation.images.wrist_camera_right"
        if wrist_image_key in data:
            wrist_image = self._parse_image(data[wrist_image_key])
        else:
            wrist_image = None
        
        # Create inputs dict with the expected keys for the model
        inputs = {
            "state": state,
            "image": {
                # For this dataset, we only have a wrist camera image
                # We'll use it as the base image and leave the other images as zeros
                "base_0_rgb": wrist_image if wrist_image is not None else np.zeros((224, 224, 3), dtype=np.uint8),
                "left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                "right_wrist_0_rgb": wrist_image if wrist_image is not None else np.zeros((224, 224, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": np.True_ if wrist_image is not None else (np.False_ if mask_padding else np.True_),
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.True_ if wrist_image is not None else (np.False_ if mask_padding else np.True_),
            },
        }
        
        # Handle actions (only available during training)
        action_key = "action"
        if action_key in data:
            # The action dimension in the dataset is 10, but we need to pad or truncate to the model's action_dim
            actions = _transforms.pad_to_dim(data[action_key], self.action_dim)
            inputs["actions"] = actions
        
        # Handle prompt/language instruction
        # In this dataset, the task is "robot_task" as seen in tasks.jsonl
        # You can set a more specific prompt here if needed
        inputs["prompt"] = "robot_task"
        
        return inputs
    
    def _parse_image(self, image):
        """Parse image to the correct format."""
        if image is None:
            return None
        
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:  # If image is in CHW format, convert to HWC
            image = einops.rearrange(image, "c h w -> h w c")
        
        # Resize to 224x224 which is the expected input size for the model
        from PIL import Image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((224, 224), Image.LANCZOS)
        return np.array(pil_image)


@dataclasses.dataclass(frozen=True)
class CustomOutputs(_transforms.DataTransformFn):
    """
    Custom output transformation for the dataset at /home/willx/data/try1.
    
    The action dimension in the dataset is 10.
    """
    # The actual action dimension of your robot
    action_dim: int = 10
    
    def __call__(self, data: dict) -> dict:
        # Get the actions from the model output
        actions = np.asarray(data["actions"])
        
        # The model outputs actions with shape [batch_size, sequence_length, action_dim]
        # We need to reshape it to match the expected format for the robot
        # For this dataset, we need to return the first 10 dimensions of the action
        # If the model's action_dim is less than 10, we'll pad with zeros
        if actions.shape[-1] < self.action_dim:
            padded_actions = np.zeros((actions.shape[0], self.action_dim))
            padded_actions[:, :actions.shape[-1]] = actions[:, :actions.shape[-1]]
            return {"action": padded_actions}
        else:
            return {"action": actions[:, :self.action_dim]}


@dataclasses.dataclass(frozen=True)
class CustomDataConfig(_config.DataConfigFactory):
    """
    Custom data configuration for the dataset at /home/willx/data/try1.
    """
    # The action dimension of your robot
    action_dim: int = 10
    
    # If provided, will be injected into the input data if the "prompt" key is not present
    default_prompt: str | None = "robot_task"
    
    # Action keys that will be used to read the action sequence from the dataset
    action_sequence_keys: tuple[str, ...] = ("action",)
    
    @dataclasses.override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # Create data transforms
        data_transforms = _transforms.Group(
            inputs=[CustomInputs(action_dim=self.action_dim, model_type=model_config.model_type)],
            outputs=[CustomOutputs(action_dim=self.action_dim)],
        )
        
        # Create model transforms
        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            # Use quantile normalization for pi0-FAST
            use_quantile_norm=model_config.model_type == _model.ModelType.PI0_FAST,
            # Enable local files only since we're using local datasets
            local_files_only=True,
        )


def create_train_config(
    repo_id: str,
    model_type: Literal["pi0", "pi0_fast"],
    exp_name: str,
    action_dim: int = 10,
    batch_size: int = 32,
    num_train_steps: int = 30000,
    overwrite: bool = False,
) -> _config.TrainConfig:
    """
    Create a training configuration for finetuning.
    
    Args:
        repo_id: ID of your LeRobot dataset
        model_type: Type of model to finetune (pi0 or pi0_fast)
        exp_name: Name of the experiment
        action_dim: Dimension of your robot's action space
        batch_size: Batch size for training
        num_train_steps: Number of training steps
        overwrite: Whether to overwrite existing checkpoints
        
    Returns:
        Training configuration
    """
    # Set up model config based on model type
    if model_type == "pi0":
        model_config = pi0.Pi0Config(
            action_dim=action_dim,
            action_horizon=1,
            max_token_len=128,
        )
        checkpoint_path = "s3://openpi-assets/checkpoints/pi0_base"
    else:  # pi0_fast
        model_config = pi0_fast.Pi0FastConfig(
            action_dim=action_dim,
            action_horizon=1,
            max_token_len=128,
        )
        checkpoint_path = "s3://openpi-assets/checkpoints/pi0_fast_base"
    
    # Set up weight loader to load the base model weights
    weight_loader = _weight_loaders.CheckpointWeightLoader(
        checkpoint_dir=_download.maybe_download(checkpoint_path),
    )
    
    # Create data config
    data_config = CustomDataConfig(
        repo_id=repo_id,
        action_dim=action_dim,
    )
    
    # Create training config
    return _config.TrainConfig(
        name=f"custom_{model_type}_{exp_name}",
        project_name="openpi",
        exp_name=exp_name,
        model=model_config,
        weight_loader=weight_loader,
        data=data_config,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        overwrite=overwrite,
        # Freeze all parameters except for the last few layers for efficient finetuning
        # You can adjust this based on your needs
        freeze_filter=nnx.PathRegex("^(?!.*/(output|action|final)).*$"),
    )


def main(
    data_dir: str = "/home/willx/data/try1",
    model_type: Literal["pi0", "pi0_fast"] = "pi0",
    exp_name: str = "custom_finetuning",
    action_dim: int = 10,  # Based on the dataset's action dimension
    batch_size: int = 32,
    num_train_steps: int = 30000,
    overwrite: bool = False,
):
    """
    Main function to finetune the pi0 model.
    
    Args:
        data_dir: Path to your local lerobot dataset
        model_type: Type of model to finetune (pi0 or pi0_fast)
        exp_name: Name of the experiment
        action_dim: Dimension of your robot's action space
        batch_size: Batch size for training
        num_train_steps: Number of training steps
        overwrite: Whether to overwrite existing checkpoints
    """
    # Get the dataset repo ID from the data directory
    data_path = pathlib.Path(data_dir)
    repo_id = data_path.name
    
    # Create the training config
    config_name = f"custom_{model_type}_{exp_name}"
    train_config = create_train_config(
        repo_id=repo_id,
        model_type=model_type,
        exp_name=exp_name,
        action_dim=action_dim,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        overwrite=overwrite,
    )
    
    # Register the config
    _config._CONFIGS[config_name] = train_config
    
    # Compute normalization statistics
    print("Computing normalization statistics...")
    subprocess.run(
        ["uv", "run", "scripts/compute_norm_stats.py", "--config-name", config_name],
        check=True,
    )
    
    # Run training
    print(f"Starting finetuning of {model_type} model...")
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # Allow JAX to use more GPU memory
    subprocess.run(
        [
            "uv", "run", "scripts/train.py", config_name,
            "--exp-name", exp_name,
            "--overwrite" if overwrite else "",
        ],
        env=env,
        check=True,
    )
    
    print(f"Finetuning completed! Checkpoints saved to: ./checkpoints/{config_name}/{exp_name}")
    print("\nTo run inference with your finetuned model, use:")
    print(f"uv run scripts/serve_policy.py policy:checkpoint --policy.config={config_name} --policy.dir=checkpoints/{config_name}/{exp_name}/[STEP]")


if __name__ == "__main__":
    tyro.cli(main) 