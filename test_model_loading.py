#!/usr/bin/env python3
"""
Simple script to test model loading without running the full inference pipeline.

This script attempts to load the pi0 model and prints debugging information.

Usage:
uv run test_model_loading.py
"""

import os
import sys
import pathlib
from typing import Literal

import numpy as np
import jax

# Add the repository to the Python path
sys.path.insert(0, os.path.abspath("."))

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
from openpi.shared import download


def test_model_loading(model_type: Literal["pi0", "pi0_fast"] = "pi0", action_dim: int = 10):
    """
    Test loading the model.
    
    Args:
        model_type: Type of model to use (pi0 or pi0_fast)
        action_dim: Dimension of the action space
    """
    print(f"Testing {model_type} model loading...")
    
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
    
    try:
        # Download the checkpoint
        print(f"Downloading checkpoint from {checkpoint_path}...")
        checkpoint_dir = download.maybe_download(checkpoint_path)
        print(f"Checkpoint downloaded to: {checkpoint_dir}")
        
        # List the contents of the checkpoint directory
        print("\nContents of checkpoint directory:")
        for item in pathlib.Path(checkpoint_dir).iterdir():
            print(f"  {item.name}")
        
        # Check if params directory exists
        params_path = pathlib.Path(checkpoint_dir) / "params"
        if params_path.exists():
            print(f"\nParams directory exists: {params_path}")
            
            # List the contents of the params directory
            print("Contents of params directory:")
            for item in params_path.iterdir():
                print(f"  {item.name}")
        else:
            print(f"\nParams directory does not exist: {params_path}")
        
        # Create a random number generator
        rng = jax.random.key(0)  # Use a fixed seed for reproducibility
        
        # Create the model
        print("\nCreating model...")
        model = model_config.create(rng)
        print(f"Model created successfully: {type(model)}")
        
        # Try to load the model parameters
        print("\nLoading model parameters...")
        try:
            model_params = _model.restore_params(params_path, dtype=np.float32)
            print("Model parameters loaded successfully")
            
            # Load the parameters into the model
            model = model_config.load(model_params)
            print("Parameters loaded into model successfully")
            
            print("\nModel loading test completed successfully!")
            return True
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            return False
    
    except Exception as e:
        print(f"Error during model loading test: {e}")
        return False


if __name__ == "__main__":
    # Test pi0 model loading
    success_pi0 = test_model_loading(model_type="pi0")
    
    # Test pi0_fast model loading
    # success_pi0_fast = test_model_loading(model_type="pi0_fast")
    
    # Print summary
    print("\n=== Model Loading Test Summary ===")
    print(f"pi0 model: {'SUCCESS' if success_pi0 else 'FAILED'}")
    # print(f"pi0_fast model: {'SUCCESS' if success_pi0_fast else 'FAILED'}") 