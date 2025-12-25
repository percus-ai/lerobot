#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
MuJoCo Simulator Robot Configuration

This module defines the configuration for the MuJoCo simulation robot,
which allows running LeRobot policies in a simulated environment.
"""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("mujoco_sim")
@dataclass
class MuJoCoSimConfig(RobotConfig):
    """
    Configuration for MuJoCo Simulation Robot.

    This robot type allows running policies in a MuJoCo simulation environment,
    providing the same interface as physical robots.
    """

    # MuJoCo model file path
    mjcf: str = "scripts/simulation/mjcf/so_arm100/scene.xml"

    # Simulation settings
    sim_hz: int = 600  # Physics simulation frequency
    render_fps: int = 60  # Viewer rendering FPS

    # Camera settings
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30

    # Cube position preset (JSON file with pos/quat arrays)
    cube_positions: str | None = None

    # Viewer settings
    show_viewer: bool = True
    cam_distance: float = 0.95
    cam_azimuth: float = 25.0
    cam_elevation: float = -35.0

    # Grasp assist settings
    enable_grasp_assist: bool = False
    grasp_dist_thresh: float = 0.02
    grasp_close_u_thresh: float = 0.35
    grasp_open_u_thresh: float = 0.25

    # Use v1 compatible normalization (-100 to 100 for joints, 0-100 for gripper)
    use_degrees: bool = True

    # Joint names for SO-101 arm
    joint_names: tuple[str, ...] = field(default_factory=lambda: (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ))

    # Camera names in MJCF (used for rendering)
    camera_names: tuple[str, ...] = field(default_factory=lambda: (
        "cam_overhead",
        "cam_wrist",
    ))

    # Output observation key names (must match policy input features)
    # Maps to camera_names in order: camera1->cam_overhead, camera2->cam_wrist
    obs_camera_names: tuple[str, ...] = field(default_factory=lambda: (
        "camera1",
        "camera2",
    ))

    # Number of empty cameras (for policies trained with empty_camera placeholders)
    # Set to 0 to match real robot behavior (policy handles empty cameras internally)
    num_empty_cameras: int = 0
