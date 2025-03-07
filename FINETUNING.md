# Finetuning pi0 Models with Your Own Data

This guide explains how to finetune the pi0 or pi0-FAST models using your own robot data. The process involves:

1. Converting your data to LeRobot format (if not already in that format)
2. Finetuning the model on your data
3. Running inference with the finetuned model

## Requirements

- NVIDIA GPU with at least 22.5GB of memory for LoRA finetuning (RTX 4090 or better)
- Ubuntu 22.04 (other operating systems are not officially supported)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

1. Clone the openpi repository and update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi
```

2. Install dependencies using uv:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

3. Install additional dependencies for data conversion:

```bash
uv pip install lerobot pillow
```

## Step 1: Convert Your Data to LeRobot Format

If your data is not already in LeRobot format, you need to convert it first. We provide a template script `convert_to_lerobot.py` that you can modify based on your data format.

1. Examine your data structure and modify the `process_data` and `process_episode` functions in `convert_to_lerobot.py` to match your data format.

2. Run the conversion script:

```bash
uv run convert_to_lerobot.py --data_dir /path/to/your/data --output_name your_dataset_name --state_dim 7 --action_dim 7
```

Replace the parameters with your own:
- `data_dir`: Path to your raw data
- `output_name`: Name for the converted dataset
- `state_dim`: Dimension of your robot state
- `action_dim`: Dimension of your robot actions

The converted dataset will be saved to `~/.cache/lerobot/your_dataset_name`.

## Step 2: Finetune the Model

Once your data is in LeRobot format, you can finetune the pi0 or pi0-FAST model using the `finetune_pi0.py` script.

1. Examine the `CustomInputs` and `CustomOutputs` classes in `finetune_pi0.py` and modify them if needed to match your data structure.

2. Run the finetuning script:

```bash
uv run finetune_pi0.py --data_dir ~/.cache/lerobot/your_dataset_name --model_type pi0 --exp_name my_experiment --action_dim 7
```

Replace the parameters with your own:
- `data_dir`: Path to your LeRobot dataset
- `model_type`: Type of model to finetune (`pi0` or `pi0_fast`)
- `exp_name`: Name for your experiment
- `action_dim`: Dimension of your robot's action space

Additional options:
- `--batch_size`: Batch size for training (default: 32)
- `--num_train_steps`: Number of training steps (default: 30000)
- `--overwrite`: Whether to overwrite existing checkpoints

The finetuning process consists of:
1. Computing normalization statistics for your data
2. Loading the base model weights
3. Finetuning the model on your data
4. Saving checkpoints at regular intervals

Checkpoints will be saved to `./checkpoints/custom_{model_type}_{exp_name}/`.

## Step 3: Run Inference with Your Finetuned Model

After finetuning, you can run inference with your model using the policy server:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=custom_{model_type}_{exp_name} --policy.dir=checkpoints/custom_{model_type}_{exp_name}/{step}
```

Replace:
- `{model_type}`: Type of model you finetuned (`pi0` or `pi0_fast`)
- `{exp_name}`: Name of your experiment
- `{step}`: Checkpoint step to use (e.g., 20000)

This will start a policy server that listens on port 8000 and waits for observations to be sent to it.

## Customizing the Finetuning Process

### Freezing Layers

By default, the finetuning script freezes most of the model parameters and only trains the final layers. This is efficient and prevents overfitting when you have limited data. You can modify the `freeze_filter` parameter in the `create_train_config` function to change which layers are frozen.

### LoRA Finetuning

For more efficient finetuning with limited GPU memory, you can implement LoRA (Low-Rank Adaptation) by modifying the training configuration. This requires additional changes to the code.

### Hyperparameters

You can adjust various hyperparameters in the `create_train_config` function, such as learning rate, batch size, and number of training steps.

## Troubleshooting

- **Out of memory errors**: Reduce the batch size or use a GPU with more memory.
- **Slow training**: Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use more GPU memory.
- **Data loading errors**: Check that your data is correctly formatted and the paths are correct.
- **Model not learning**: Check your data quality, adjust learning rate, or modify which layers are being finetuned.

For more detailed information, refer to the [openpi documentation](https://github.com/Physical-Intelligence/openpi). 