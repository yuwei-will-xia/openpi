#!/usr/bin/env python3
"""
Script to convert local data to LeRobot format.

This script provides a template for converting your local dataset to the LeRobot format
required for finetuning the pi0 model. You'll need to modify this script based on your
specific data format.

Usage:
uv run convert_to_lerobot.py --data_dir /path/to/your/data --output_name your_dataset_name

Requirements:
- lerobot package: `uv pip install lerobot`
"""

import os
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def convert_data(
    data_dir: str,
    output_name: str,
    robot_type: str = "custom",
    fps: int = 10,
    image_size: Tuple[int, int] = (256, 256),
    state_dim: int = 7,
    action_dim: int = 7,
    push_to_hub: bool = False,
):
    """
    Convert local data to LeRobot format.
    
    Args:
        data_dir: Path to your local data
        output_name: Name for the output dataset
        robot_type: Type of robot (used for metadata)
        fps: Frames per second of the data
        image_size: Size of the images (height, width)
        state_dim: Dimension of the robot state
        action_dim: Dimension of the robot actions
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
    # Clean up any existing dataset in the output directory
    repo_id = output_name
    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create LeRobot dataset with appropriate features
    # Modify the features based on your data
    features = {
        "image": {
            "dtype": "image",
            "shape": (*image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }
    
    # Add wrist image if your data has it
    if has_wrist_camera(data_dir):
        features["wrist_image"] = {
            "dtype": "image",
            "shape": (*image_size, 3),
            "names": ["height", "width", "channel"],
        }
    
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Process your data and add it to the LeRobot dataset
    # This is where you need to modify the code based on your data format
    process_data(data_dir, dataset)
    
    # Consolidate the dataset
    dataset.consolidate(run_compute_stats=False)
    
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=[robot_type, "custom"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )
    
    print(f"Dataset converted and saved to: {output_path}")
    return output_path


def has_wrist_camera(data_dir: str) -> bool:
    """
    Check if the data has wrist camera images.
    
    Args:
        data_dir: Path to your local data
        
    Returns:
        Whether the data has wrist camera images
    """
    # Modify this function based on your data format
    # This is just a placeholder implementation
    return False


def process_data(data_dir: str, dataset: LeRobotDataset):
    """
    Process your data and add it to the LeRobot dataset.
    
    Args:
        data_dir: Path to your local data
        dataset: LeRobot dataset to add data to
    """
    # Modify this function based on your data format
    # This is just a placeholder implementation
    data_path = pathlib.Path(data_dir)
    
    # Example: Process each episode directory
    for episode_dir in sorted(data_path.glob("episode_*")):
        process_episode(episode_dir, dataset)


def process_episode(episode_dir: pathlib.Path, dataset: LeRobotDataset):
    """
    Process a single episode and add it to the LeRobot dataset.
    
    Args:
        episode_dir: Path to the episode directory
        dataset: LeRobot dataset to add data to
    """
    # Modify this function based on your data format
    # This is just a placeholder implementation
    
    # Example: Read episode metadata
    metadata_file = episode_dir / "metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        task = metadata.get("task", "unknown task")
    else:
        task = "unknown task"
    
    # Example: Process each frame in the episode
    frames_dir = episode_dir / "frames"
    if not frames_dir.exists():
        print(f"Warning: No frames directory found in {episode_dir}")
        return
    
    # Example: Read state and action data
    states_file = episode_dir / "states.npy"
    actions_file = episode_dir / "actions.npy"
    
    if states_file.exists() and actions_file.exists():
        states = np.load(states_file)
        actions = np.load(actions_file)
    else:
        print(f"Warning: Missing state or action data in {episode_dir}")
        return
    
    # Example: Process each frame
    for i, frame_file in enumerate(sorted(frames_dir.glob("*.png"))):
        if i >= len(states) or i >= len(actions):
            break
        
        # Read the image
        from PIL import Image
        image = np.array(Image.open(frame_file))
        
        # Create frame data
        frame_data = {
            "image": image,
            "state": states[i],
            "actions": actions[i],
        }
        
        # Add wrist image if available
        wrist_frame_file = episode_dir / "wrist_frames" / frame_file.name
        if wrist_frame_file.exists():
            wrist_image = np.array(Image.open(wrist_frame_file))
            frame_data["wrist_image"] = wrist_image
        
        # Add the frame to the dataset
        dataset.add_frame(frame_data)
    
    # Save the episode with the task as prompt
    dataset.save_episode(task=task)


def main(
    data_dir: str,
    output_name: str,
    robot_type: str = "custom",
    fps: int = 10,
    state_dim: int = 7,
    action_dim: int = 7,
    push_to_hub: bool = False,
):
    """
    Main function to convert local data to LeRobot format.
    
    Args:
        data_dir: Path to your local data
        output_name: Name for the output dataset
        robot_type: Type of robot (used for metadata)
        fps: Frames per second of the data
        state_dim: Dimension of the robot state
        action_dim: Dimension of the robot actions
        push_to_hub: Whether to push the dataset to Hugging Face Hub
    """
    convert_data(
        data_dir=data_dir,
        output_name=output_name,
        robot_type=robot_type,
        fps=fps,
        state_dim=state_dim,
        action_dim=action_dim,
        push_to_hub=push_to_hub,
    )


if __name__ == "__main__":
    tyro.cli(main) 