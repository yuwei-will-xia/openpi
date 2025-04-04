import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_easo_example() -> dict:
    """Creates a random input example for the EASO policy."""
    return {
        "observation.joint_angles": np.random.rand(7),
        "observation.eef_pose": np.random.rand(6),
        "observation.target_eef_pose": np.random.rand(6),
        "observation.images.wrist_camera_right": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "actions": np.random.rand(8),
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class EasoJsInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs from the EASO robot data format to the model's expected format.
    The EASO data format includes joint angles, end effector poses, and camera images from multiple views.
    """

    # The actions dimension of the model. Will be used to pad state and actions for pi0 model.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0

        # Combine joint angles and eef pose as the state input
        state = data["observation.joint_angles"]
        state = transforms.pad_to_dim(state, self.action_dim)


        # Parse images from both wrist cameras
        # Note: Images should already be in (H,W,C) format from the LeRobot dataset
        right_wrist_image = _parse_image(data["observation.images.wrist_camera_right"])
        empty_image = np.zeros_like(right_wrist_image)

        # Create inputs dict with the model's expected format
        inputs = {
            "state": state,
            "image": {
                # Use wrist camera left as base view since it's our primary camera
                "base_0_rgb": empty_image,
                "left_wrist_0_rgb": empty_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_ if mask_padding else np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Handle actions if present (during training)
        if "actions" in data:
            # use absolute joint angles and gripper action as action input 
            actions = np.concatenate([data["observation.joint_angles"], data["actions"][6]])
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions

        # Add target pose information as part of the prompt
        # This helps the model understand the desired end effector position
        target_str = f"insert vials into the tray"
        inputs["prompt"] = target_str

        return inputs


@dataclasses.dataclass(frozen=True)
class EasoJsOutputs(transforms.DataTransformFn):
    """
    Converts model outputs back to the EASO-specific format.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
