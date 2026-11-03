#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor import PolicyAction, RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """Log missing and unexpected keys when loading a model.

    Args:
        missing_keys (list[str]): Keys that were expected but not found.
        unexpected_keys (list[str]): Keys that were found but not expected.
    """
    if missing_keys:
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")


# TODO(Steven): Move this function to a proper preprocessor step
def prepare_observation_for_inference(
    observation: dict[str, np.ndarray],
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """Converts observation data to model-ready PyTorch tensors.

    This function takes a dictionary of NumPy arrays, performs necessary
    preprocessing, and prepares it for model inference. The steps include:
    1. Converting NumPy arrays to PyTorch tensors.
    2. Normalizing and permuting image data (if any).
    3. Adding a batch dimension to each tensor.
    4. Moving all tensors to the specified compute device.
    5. Adding task and robot type information to the dictionary.

    Args:
        observation: A dictionary mapping observation names (str) to NumPy
            array data. For images, the format is expected to be (H, W, C).
        device: The PyTorch device (e.g., 'cpu' or 'cuda') to which the
            tensors will be moved.
        task: An optional string identifier for the current task.
        robot_type: An optional string identifier for the robot being used.

    Returns:
        A dictionary where values are PyTorch tensors preprocessed for
        inference, residing on the target device. Image tensors are reshaped
        to (C, H, W) and normalized to a [0, 1] range.
    """
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    return observation


def build_inference_frame(
    observation: dict[str, Any],
    device: torch.device,
    ds_features: dict[str, dict],
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """Constructs a model-ready observation tensor dict from a raw observation.

    This utility function orchestrates the process of converting a raw,
    unstructured observation from an environment into a structured,
    tensor-based format suitable for passing to a policy model.

    Args:
        observation: The raw observation dictionary, which may contain
            superfluous keys.
        device: The target PyTorch device for the final tensors.
        ds_features: A configuration dictionary that specifies which features
            to extract from the raw observation.
        task: An optional string identifier for the current task.
        robot_type: An optional string identifier for the robot being used.

    Returns:
        A dictionary of preprocessed tensors ready for model inference.
    """
    # Extracts the correct keys from the incoming raw observation
    observation = build_dataset_frame(ds_features, observation, prefix=OBS_STR)

    # Performs the necessary conversions to the observation
    observation = prepare_observation_for_inference(observation, device, task, robot_type)

    return observation


def make_robot_action(action_tensor: PolicyAction, ds_features: dict[str, dict]) -> RobotAction:
    """Converts a policy's output tensor into a dictionary of named actions.

    This function translates the numerical output from a policy model into a
    human-readable and robot-consumable format, where each dimension of the
    action tensor is mapped to a named motor or actuator command.

    Args:
        action_tensor: A PyTorch tensor representing the policy's action,
            typically with a batch dimension (e.g., shape [1, action_dim]).
        ds_features: A configuration dictionary containing metadata, including
            the names corresponding to each index of the action tensor.

    Returns:
        A dictionary mapping action names (e.g., "joint_1_motor") to their
        corresponding floating-point values, ready to be sent to a robot
        controller.
    """
    # TODO(Steven): Check if these steps are already in all postprocessor policies
    action_tensor = action_tensor.squeeze(0)
    action_tensor = action_tensor.to("cpu")

    action_names = ds_features[ACTION]["names"]
    act_processed_policy: RobotAction = {
        f"{name}": float(action_tensor[i]) for i, name in enumerate(action_names)
    }
    return act_processed_policy


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    Raises a standardized ValueError for feature mismatches between dataset/environment and policy config.
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): provide a dynamic rename map suggestion to the user.
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def validate_visual_features_consistency(
    cfg: PreTrainedConfig,
    features: dict[str, PolicyFeature],
) -> None:
    """
    Validates visual feature consistency between a policy config and provided dataset/environment features.

    Args:
        cfg (PreTrainedConfig): The model or policy configuration containing input_features and type.
        features (Dict[str, PolicyFeature]): A mapping of feature names to PolicyFeature objects.
    """
    expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}
    if not provided_visuals.issubset(expected_visuals):
        raise_feature_mismatch_error(provided_visuals, expected_visuals)


@torch.no_grad()
def load_safetensors_into_model(
    path: str | Path,
    model: nn.Module,
    *,
    map_key: Callable[[str], str | None],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """safetensors をテンソル単位で読み、モデル側の dtype に変換してからパラメータ / バッファへ直接写す。

    ``load_state_dict`` に全体を渡す読み方だと、float32 で保存した重みは (1) CPU 上で全体が展開され、
    (2) bfloat16 のモデル(GPU 上)へ写すときに GPU 側で float32 の一時テンソルを経由して丸められる。
    π0.5 の float32 保存(14.5 GB)は (2) で 12 GB 級 GPU の読み込み中に CUDA OOM になる。
    ここでは 1 本ずつ読み、同名のパラメータ / バッファの dtype に CPU 上で変換してから ``copy_`` で
    写すので、GPU の一時領域は要らず、CPU の常駐もテンソル 1 本分で済む。丸めは ``load_state_dict`` が
    行う dtype 変換と同じ最近接偶数なので、載る値は変わらない。

    ``map_key(name)`` はファイル内の名前をモデルの state_dict のキーに写す(捨てるなら None)。
    戻り値は ``load_state_dict`` と同じ (missing_keys, unexpected_keys)。``strict`` のときに
    どちらかがあれば、写し終えてから RuntimeError を投げる(``load_state_dict`` と同じ順)。
    """
    from safetensors import safe_open

    targets = model.state_dict()          # パラメータ / バッファと記憶域を共有する(copy_ で書ける)
    loaded: set[str] = set()
    unexpected: list[str] = []
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for name in f.keys():
            key = map_key(name)
            if key is None:
                continue
            dst = targets.get(key)
            if dst is None:
                unexpected.append(key)
                continue
            src = f.get_tensor(name)
            if tuple(src.shape) != tuple(dst.shape):
                raise RuntimeError(
                    f"size mismatch for {key}: copying a param with shape {tuple(src.shape)} from checkpoint, "
                    f"the shape in current model is {tuple(dst.shape)}."
                )
            if src.is_floating_point() and dst.is_floating_point() and src.dtype != dst.dtype:
                src = src.to(dtype=dst.dtype)   # CPU で丸める(load_state_dict と同じ丸め)
            dst.copy_(src)
            loaded.add(key)
    missing = [key for key in targets if key not in loaded]
    if strict and (missing or unexpected):
        lines = []
        if missing:
            lines.append("Missing key(s) in state_dict: " + ", ".join(f'"{k}"' for k in missing) + ". ")
        if unexpected:
            lines.append("Unexpected key(s) in state_dict: " + ", ".join(f'"{k}"' for k in unexpected) + ". ")
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t" + "\n\t".join(lines)
        )
    return missing, unexpected
