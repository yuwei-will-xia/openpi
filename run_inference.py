#!/usr/bin/env python3
"""
Script to run inference with the pi0 model on the dataset at /home/willx/data/try1.

This script loads a pre-trained pi0 model and runs inference on a sample from the dataset.

Usage:
uv run run_inference.py --model_type pi0 --checkpoint_dir /path/to/checkpoint
"""

import argparse
import os
import sys
import pathlib
from typing import Literal, Optional

import numpy as np
import pandas as pd
from PIL import Image

# Add the repository to the Python path
sys.path.insert(0, os.path.abspath("."))

# Import the custom inference policy
from custom_pi0_inference import create_inference_policy, run_inference_example


def load_sample_from_dataset(data_dir: str = "/home/willx/data/try1"):
    """
    Load a sample from the dataset.
    
    Args:
        data_dir: Path to the dataset
        
    Returns:
        Sample data
    """
    try:
        # Load the parquet file
        data_path = pathlib.Path(data_dir) / "data" / "chunk-000" / "episode_000000.parquet"
        print(f"Loading data from: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
            
        df = pd.read_parquet(data_path)
        print(f"Loaded dataset with {len(df)} samples")
        
        # Get a random sample
        sample_idx = np.random.randint(0, len(df))
        print(f"Selected sample index: {sample_idx}")
        sample = df.iloc[sample_idx]
        
        # Print available columns
        print(f"Available columns in the dataset: {list(sample.keys())}")
        
        # Convert to dictionary
        sample_dict = {}
        
        # Extract joint angles
        if "observation.joint_angles" in sample:
            sample_dict["observation.joint_angles"] = np.array(sample["observation.joint_angles"])
        
        # Extract end-effector pose
        if "observation.eef_pose" in sample:
            sample_dict["observation.eef_pose"] = np.array(sample["observation.eef_pose"])
        
        # Extract target end-effector pose
        if "observation.target_eef_pose" in sample:
            sample_dict["observation.target_eef_pose"] = np.array(sample["observation.target_eef_pose"])
        
        # Extract wrist camera image
        if "observation.images.wrist_camera_right" in sample:
            # The image is stored as bytes, we need to convert it to a numpy array
            image_bytes = sample["observation.images.wrist_camera_right"]
            if isinstance(image_bytes, bytes):
                # Convert bytes to image
                from io import BytesIO
                image = Image.open(BytesIO(image_bytes))
                sample_dict["observation.images.wrist_camera_right"] = np.array(image)
            else:
                print(f"Warning: wrist_camera_right is not bytes, but {type(image_bytes)}")
        
        # If no data was extracted, create a dummy sample
        if not sample_dict:
            print("Warning: No data extracted from the dataset. Creating a dummy sample.")
            sample_dict = {
                "observation.joint_angles": np.random.rand(7),
                "observation.eef_pose": np.random.rand(6),
                "observation.images.wrist_camera_right": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            }
        
        return sample_dict
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating a dummy sample instead.")
        
        # Create a dummy sample
        return {
            "observation.joint_angles": np.random.rand(7),
            "observation.eef_pose": np.random.rand(6),
            "observation.images.wrist_camera_right": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }


def main(model_type: Literal["pi0", "pi0_fast"] = "pi0", 
         checkpoint_dir: Optional[str] = None, 
         data_dir: str = "/home/willx/data/try1",
         action_dim: int = 10,
         save_visualization: bool = False):
    """
    Main function to run inference with the pi0 model.
    
    Args:
        model_type: Type of model to use (pi0 or pi0_fast)
        checkpoint_dir: Path to the checkpoint directory (if None, use pre-trained model)
        data_dir: Path to the dataset
        action_dim: Dimension of the action space
        save_visualization: Whether to save a visualization of the inference
    """
    print(f"Running inference with {model_type} model...")
    
    # Create the policy
    policy = create_inference_policy(
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        action_dim=action_dim,
    )
    
    # Load a sample from the dataset
    sample = load_sample_from_dataset(data_dir)
    
    # Print sample info
    print("\nSample information:")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Run inference
    print("\nRunning inference...")
    result = run_inference_example(policy, sample)
    
    # Print result
    print("\nInference result:")
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"    values: {value.flatten()[:5]}...")
        else:
            print(f"  {key}: {value}")
    
    # Save visualization if requested
    if save_visualization and "observation.images.wrist_camera_right" in sample:
        print("\nSaving visualization...")
        image = sample["observation.images.wrist_camera_right"]
        
        # Create a visualization with the image and predicted action
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Show the image
        ax.imshow(image)
        
        # Add the predicted action as text
        action_text = f"Predicted action: {result['action'].flatten()[:5]}..."
        ax.text(10, 30, action_text, color='white', fontsize=12, 
                bbox=dict(facecolor='black', alpha=0.7))
        
        # Save the visualization
        output_dir = pathlib.Path("inference_results")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{model_type}_inference.png")
        print(f"Visualization saved to {output_dir / f'{model_type}_inference.png'}")
    
    print("\nInference completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the pi0 model.")
    parser.add_argument("--model_type", type=str, choices=["pi0", "pi0_fast"], default="pi0",
                        help="Type of model to use (pi0 or pi0_fast)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to the checkpoint directory (if None, use pre-trained model)")
    parser.add_argument("--data_dir", type=str, default="/home/willx/data/try1",
                        help="Path to the dataset")
    parser.add_argument("--action_dim", type=int, default=10,
                        help="Dimension of the action space")
    parser.add_argument("--save_visualization", action="store_true",
                        help="Whether to save a visualization of the inference")
    
    args = parser.parse_args()
    
    main(
        model_type=args.model_type,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        action_dim=args.action_dim,
        save_visualization=args.save_visualization,
    ) 