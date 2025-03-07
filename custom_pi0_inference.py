#!/usr/bin/env python3
"""
Script for inference with the pi0 model using the specific lerobot dataset structure.

This script contains customized input and output transformations for the dataset at /home/willx/data/try1.
"""

import dataclasses
import os
import pathlib
import sys
from typing import Dict, Any, Literal

import numpy as np
import einops

# Add the repository to the Python path
sys.path.insert(0, os.path.abspath("."))

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.transforms as _transforms
from openpi.policies.policy import Policy
from openpi.policies import policy_config
from openpi.training import config
from openpi.shared import download, normalize


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


def create_custom_config(model_type: Literal["pi0", "pi0_fast"], action_dim: int = 10):
    """
    Create a custom config for inference.
    
    Args:
        model_type: Type of model to use (pi0 or pi0_fast)
        action_dim: Dimension of the action space
        
    Returns:
        Custom training config
    """
    # Set up model config based on model type
    if model_type == "pi0":
        model_config = pi0.Pi0Config(
            action_dim=action_dim,
            action_horizon=1,
            max_token_len=128,
        )
    else:  # pi0_fast
        model_config = pi0_fast.Pi0FastConfig(
            action_dim=action_dim,
            action_horizon=1,
            max_token_len=128,
        )
    
    # Create a custom data config
    data_config = config.DataConfig(
        repo_id="try1",
        use_quantile_norm=model_type == "pi0_fast",
        local_files_only=True,
    )
    
    # Create a custom train config
    return config.TrainConfig(
        name=f"custom_{model_type}_inference",
        project_name="openpi",
        exp_name="inference",
        model=model_config,
        data=data_config,
    )


def create_inference_policy(model_type: Literal["pi0", "pi0_fast"] = "pi0", checkpoint_dir=None, action_dim: int = 10) -> Policy:
    """
    Create a trained policy for inference.
    
    Args:
        model_type: Type of model to use (pi0 or pi0_fast)
        checkpoint_dir: Path to the checkpoint directory
        action_dim: Dimension of the action space
        
    Returns:
        Trained policy
    """
    try:
        # Create a custom config
        train_config = create_custom_config(model_type, action_dim)
        
        # If no checkpoint is specified, use the pre-trained model
        if checkpoint_dir is None:
            if model_type == "pi0":
                checkpoint_path = "s3://openpi-assets/checkpoints/pi0_base"
            else:  # pi0_fast
                checkpoint_path = "s3://openpi-assets/checkpoints/pi0_fast_base"
            
            # Download the checkpoint
            print(f"Downloading checkpoint from {checkpoint_path}...")
            checkpoint_dir = download.maybe_download(checkpoint_path)
        
        print(f"Using checkpoint directory: {checkpoint_dir}")
        
        # Create a random number generator
        import jax
        rng = jax.random.key(0)  # Use a fixed seed for reproducibility
        
        # Create the model
        print(f"Creating {model_type} model...")
        model = train_config.model.create(rng)
        
        # Load the model parameters
        params_path = pathlib.Path(checkpoint_dir) / "params"
        print(f"Loading model parameters from {params_path}...")
        
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters directory not found: {params_path}")
        
        model_params = _model.restore_params(params_path, dtype=np.float32)
        model = train_config.model.load(model_params)
        
        # Create empty norm stats (we'll handle normalization in our custom transforms)
        norm_stats = {"state": normalize.NormStats(mean=np.zeros(action_dim), std=np.ones(action_dim))}
        
        # Create the policy with our custom transforms
        print("Creating policy with custom transforms...")
        policy = Policy(
            model=model,
            transforms=[CustomInputs(action_dim=action_dim, model_type=train_config.model.model_type)],
            output_transforms=[CustomOutputs(action_dim=action_dim)],
        )
        
        return policy
        
    except Exception as e:
        print(f"Error creating inference policy: {e}")
        print("Creating a dummy policy instead.")
        
        # Create a dummy policy that returns random actions
        class DummyPolicy:
            def infer(self, data):
                print("Using dummy policy (returns random actions)")
                return {"action": np.random.rand(1, action_dim)}
        
        return DummyPolicy()


def run_inference_example(policy, data):
    """
    Run inference on an example.
    
    Args:
        policy: Trained policy
        data: Input data
        
    Returns:
        Action
    """
    # Run inference
    result = policy.infer(data)
    return result


def main():
    """
    Main function to demonstrate how to use the custom input and output transformations.
    """
    # Create a trained policy
    policy = create_inference_policy(model_type="pi0", action_dim=10)
    
    # Create a dummy example
    example = {
        "observation.joint_angles": np.random.rand(7),
        "observation.eef_pose": np.random.rand(6),
        "observation.images.wrist_camera_right": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }
    
    # Run inference
    result = run_inference_example(policy, example)
    print("Inference result:", result)


if __name__ == "__main__":
    main() 