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
MuJoCo Simulation Robot

This module implements a LeRobot-compatible robot that runs in MuJoCo simulation.
It provides the same interface as physical robots, allowing policies to be
evaluated in simulation with full visualization support.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .config_mujoco_sim import MuJoCoSimConfig

logger = logging.getLogger(__name__)

# Encoder resolution for SO-101 motors
ENCODER_RESOLUTION = 4096

# MuJoCo gripper joint range (from so101_new_calib.xml)
GRIPPER_QMIN = -0.17453
GRIPPER_QMAX = 1.74533


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _rot180(img: np.ndarray) -> np.ndarray:
    """Rotate image 180 degrees."""
    return np.ascontiguousarray(np.rot90(img, 2))


def _to_gray3(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale, replicated to 3 channels."""
    gray = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


class MuJoCoSim(Robot):
    """
    MuJoCo Simulation Robot.

    This robot runs policies in a MuJoCo simulation environment, providing
    camera observations and accepting normalized joint commands.
    """

    config_class = MuJoCoSimConfig
    name = "mujoco_sim"

    def __init__(self, config: MuJoCoSimConfig):
        super().__init__(config)
        self.config = config

        # State
        self._connected = False
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.viewer: mujoco.viewer.Handle | None = None
        self.renderer: mujoco.Renderer | None = None

        # Cameras - empty dict since MuJoCo renders cameras internally
        # This is required by lerobot-record for calculating image writer threads
        self.cameras: dict = {}

        # Joint mapping
        self._joint_id: dict[str, int] = {}
        self._qpos_adr: dict[str, int] = {}
        self._joint_to_act: dict[str, int] = {}

        # Calibration data (loaded from file)
        self._calib: dict[str, dict] = {}
        self._joint_centers: dict[str, int] = {}
        self._joint_sign: dict[str, float] = {}
        self._joint_range: dict[str, tuple[int, int]] = {}

        # Grasp assist state
        self._attached = False
        self._rel_pos = np.zeros(3, dtype=np.float64)
        self._rel_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # IDs for grasp assist
        self._grasp_site_id: int | None = None
        self._cube_site_id: int | None = None
        self._gripper_body_id: int | None = None
        self._cube_body_id: int | None = None
        self._cube_qpos_adr: int | None = None
        self._cube_dof_adr: int | None = None
        self._cube_geom_id: int | None = None
        self._cube_contype_orig: int | None = None
        self._cube_conaff_orig: int | None = None

        # Camera IDs
        self._cam_ids: dict[str, int] = {}

        # Timing
        self._last_step_time = 0.0
        self._substeps = 1

        # Cube positions
        self._cube_positions: list[dict] = []
        self._cube_pos_idx = 0
        self._episode_count = 0  # Total episode counter for position cycling

        # Place target (dish) IDs
        self._place_target_body_id: int | None = None
        self._place_target_site_id: int | None = None
        self._place_target_geom_ids: list[int] = []  # All geom IDs of place target for contact detection

    @property
    def observation_features(self) -> dict:
        """Return observation feature specifications."""
        features = {}

        # Joint positions (normalized to -100 to 100, gripper 0-100)
        for name in self.config.joint_names:
            features[f"{name}.pos"] = float

        # Camera images - use obs_camera_names (without observation.images. prefix)
        for obs_cam_name in self.config.obs_camera_names:
            features[obs_cam_name] = (
                self.config.cam_height,
                self.config.cam_width,
                3,
            )

        # Empty cameras
        for i in range(self.config.num_empty_cameras):
            features[f"empty_camera_{i}"] = (
                self.config.cam_height,
                self.config.cam_width,
                3,
            )

        return features

    @property
    def action_features(self) -> dict:
        """Return action feature specifications."""
        features = {}
        for name in self.config.joint_names:
            features[f"{name}.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        # Simulation is always "calibrated"
        return True

    def calibrate(self) -> None:
        # No calibration needed for simulation
        pass

    def connect(self, calibrate: bool = True) -> None:
        """Initialize MuJoCo simulation."""
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Load MuJoCo model
        mjcf_path = Path(self.config.mjcf).expanduser()
        if not mjcf_path.is_absolute():
            # Try relative to project root
            mjcf_path = Path.cwd() / mjcf_path
        if not mjcf_path.exists():
            raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")

        logger.info(f"Loading MuJoCo model from {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.model.opt.timestep = 1.0 / float(self.config.sim_hz)
        self.data = mujoco.MjData(self.model)

        # Calculate substeps per frame
        self._substeps = max(1, int(round(float(self.config.sim_hz) / float(self.config.render_fps))))

        # Build joint mappings
        self._build_joint_mappings()

        # Load calibration
        self._load_calibration_data()

        # Setup grasp assist
        self._setup_grasp_assist()

        # Setup place target IDs
        self._setup_place_target()

        # Setup cameras
        self._setup_cameras()

        # Load cube positions if specified
        if self.config.cube_positions:
            self._load_cube_positions()

        # Apply keyframe if defined (for initial pose)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            logger.info(f"Applied keyframe 'home' as initial pose")

        # Initialize simulation
        mujoco.mj_forward(self.model, self.data)

        # Set initial cube position if loaded
        if self._cube_positions:
            self.reset_cube_position(0)
            mujoco.mj_forward(self.model, self.data)

        # Debug: log initial state
        initial_state = self._normalize_state()
        state_vals = [initial_state.get(f"{n}.pos", 0.0) for n in self.config.joint_names]
        logger.info(f"Initial state (normalized): {[f'{v:.2f}' for v in state_vals]}")

        # Start viewer if requested
        if self.config.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            if self.viewer:
                self.viewer.cam.distance = self.config.cam_distance
                self.viewer.cam.azimuth = self.config.cam_azimuth
                self.viewer.cam.elevation = self.config.cam_elevation
                # Initial sync to render the scene
                self.viewer.sync()

        self._connected = True
        self._last_step_time = time.time()
        logger.info(f"{self} connected.")

    def _build_joint_mappings(self) -> None:
        """Build mappings from joint names to MuJoCo IDs."""
        for name in self.config.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint not found in MJCF: {name}")
            self._joint_id[name] = int(jid)
            self._qpos_adr[name] = int(self.model.jnt_qposadr[jid])

        # Build actuator mapping
        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name in self.config.joint_names:
                self._joint_to_act[act_name] = i

    def _load_calibration_data(self) -> None:
        """Load calibration from file."""
        if not self.calibration_fpath.is_file():
            logger.warning(f"Calibration file not found: {self.calibration_fpath}")
            logger.warning("Using DEFAULT calibration values: center=2048, sign=1.0, range=(0,4095)")
            logger.warning("If training data used custom calibration, actions will be WRONG!")
            # Use defaults
            for name in self.config.joint_names:
                self._joint_centers[name] = 2048
                self._joint_sign[name] = 1.0
                self._joint_range[name] = (0, 4095)
            return

        logger.info(f"Loading calibration from {self.calibration_fpath}")
        with open(self.calibration_fpath) as f:
            self._calib = json.load(f)

        for name in self.config.joint_names:
            if name not in self._calib:
                self._joint_centers[name] = 2048
                self._joint_sign[name] = 1.0
                self._joint_range[name] = (0, 4095)
                continue

            c = self._calib[name]
            rmin = int(c.get("range_min", 0))
            rmax = int(c.get("range_max", 4095))
            homing = int(c.get("homing_offset", 0))
            drive_mode = int(c.get("drive_mode", 0))

            center = int(round((rmin + rmax) / 2 + homing))
            self._joint_centers[name] = center
            self._joint_sign[name] = -1.0 if drive_mode == 1 else 1.0
            self._joint_range[name] = (rmin, rmax)

    def _setup_grasp_assist(self) -> None:
        """Setup grasp assist (weld) functionality."""
        def try_get_id(obj_type, name):
            id_ = mujoco.mj_name2id(self.model, obj_type, name)
            return int(id_) if id_ >= 0 else None

        self._grasp_site_id = try_get_id(mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self._cube_site_id = try_get_id(mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self._gripper_body_id = try_get_id(mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._cube_body_id = try_get_id(mujoco.mjtObj.mjOBJ_BODY, "cube")

        cube_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        if cube_jid >= 0:
            self._cube_qpos_adr = int(self.model.jnt_qposadr[cube_jid])
            self._cube_dof_adr = int(self.model.jnt_dofadr[cube_jid])

        self._cube_geom_id = try_get_id(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        if self._cube_geom_id is not None:
            self._cube_contype_orig = int(self.model.geom_contype[self._cube_geom_id])
            self._cube_conaff_orig = int(self.model.geom_conaffinity[self._cube_geom_id])

    def _setup_place_target(self) -> None:
        """Setup place target (dish) IDs for evaluation."""
        def try_get_id(obj_type, name):
            id_ = mujoco.mj_name2id(self.model, obj_type, name)
            return int(id_) if id_ >= 0 else None

        self._place_target_body_id = try_get_id(mujoco.mjtObj.mjOBJ_BODY, "place_target")
        self._place_target_site_id = try_get_id(mujoco.mjtObj.mjOBJ_SITE, "place_target_site")

        # Find all geom IDs belonging to place_target body for contact detection
        self._place_target_geom_ids = []
        if self._place_target_body_id is not None:
            # Check all geoms to find those attached to place_target body
            for gid in range(self.model.ngeom):
                if int(self.model.geom_bodyid[gid]) == self._place_target_body_id:
                    self._place_target_geom_ids.append(int(gid))
            logger.info(f"Place target found with {len(self._place_target_geom_ids)} geoms: {self._place_target_geom_ids}")

    def _setup_cameras(self) -> None:
        """Setup camera rendering."""
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.cam_height,
            width=self.config.cam_width,
        )

        for cam_name in self.config.camera_names:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cid >= 0:
                self._cam_ids[cam_name] = int(cid)
            else:
                logger.warning(f"Camera not found: {cam_name}")

    def _load_cube_positions(self) -> None:
        """Load cube position presets from JSON."""
        path = Path(self.config.cube_positions).expanduser()
        if not path.exists():
            logger.warning(f"Cube positions file not found: {path}")
            return

        with open(path) as f:
            self._cube_positions = json.load(f)
        logger.info(f"Loaded {len(self._cube_positions)} cube positions")

    def configure(self) -> None:
        """Apply configuration (no-op for simulation)."""
        pass

    def _render_camera(self, cam_name: str) -> np.ndarray | None:
        """Render a single camera view."""
        if self.renderer is None or cam_name not in self._cam_ids:
            return None

        cam_id = self._cam_ids[cam_name]
        self.renderer.update_scene(self.data, camera=cam_id)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
        return rgb

    def _get_gripper_u(self) -> float:
        """Get gripper openness (0=closed, 1=open)."""
        if "gripper" not in self._qpos_adr:
            return 0.0
        q = float(self.data.qpos[self._qpos_adr["gripper"]])
        jid = self._joint_id["gripper"]
        qmin = float(self.model.jnt_range[jid, 0])
        qmax = float(self.model.jnt_range[jid, 1])
        if qmax == qmin:
            return 0.0
        return _clip01((q - qmin) / (qmax - qmin))

    def _normalize_state(self) -> dict[str, float]:
        """
        Get normalized joint positions.

        Uses the same normalization as convert_sim_to_lerobot.py:
        - Regular joints: -100 to 100
        - Gripper: 0 to 100
        """
        state = {}

        for name in self.config.joint_names:
            if name not in self._qpos_adr:
                state[f"{name}.pos"] = 0.0
                continue

            rad = float(self.data.qpos[self._qpos_adr[name]])

            if name == "gripper":
                # Gripper: qpos -> openness u -> 0-100
                u = self._get_gripper_u()
                state[f"{name}.pos"] = u * 100.0
            else:
                # Regular joint: rad -> tick -> -100 to 100
                c = self._calib.get(name, {})
                rmin = float(c.get("range_min", 0))
                rmax = float(c.get("range_max", 4095))
                homing_offset = int(c.get("homing_offset", 0))
                drive_mode = int(c.get("drive_mode", 0))

                center = (rmin + rmax) / 2.0 + homing_offset
                center = round(center)
                sign = -1.0 if drive_mode == 1 else 1.0

                tick = center + sign * (rad * ENCODER_RESOLUTION / (2.0 * math.pi))
                tick = max(rmin, min(rmax, tick))

                norm = ((tick - rmin) / (rmax - rmin + 1e-9)) * 200.0 - 100.0
                if drive_mode == 1:
                    norm = -norm

                state[f"{name}.pos"] = norm

        return state

    def _denormalize_action(self, action: dict[str, Any]) -> dict[str, float]:
        """
        Convert normalized action to radians.

        Inverse of _normalize_state.
        """
        rad_action = {}

        for name in self.config.joint_names:
            key = f"{name}.pos"
            if key not in action:
                continue

            norm = float(action[key])

            if name == "gripper":
                # Gripper: 0-100 -> openness u -> qpos
                u = _clip01(norm / 100.0)
                jid = self._joint_id["gripper"]
                qmin = float(self.model.jnt_range[jid, 0])
                qmax = float(self.model.jnt_range[jid, 1])
                rad_action[name] = qmin + u * (qmax - qmin)
            else:
                # Regular joint: -100 to 100 -> tick -> rad
                c = self._calib.get(name, {})
                rmin = float(c.get("range_min", 0))
                rmax = float(c.get("range_max", 4095))
                homing_offset = int(c.get("homing_offset", 0))
                drive_mode = int(c.get("drive_mode", 0))

                if drive_mode == 1:
                    norm = -norm

                # norm -> tick
                tick = (norm + 100.0) / 200.0 * (rmax - rmin) + rmin

                # tick -> rad
                center = (rmin + rmax) / 2.0 + homing_offset
                center = round(center)
                sign = -1.0 if drive_mode == 1 else 1.0

                rad = sign * (tick - center) * (2.0 * math.pi / ENCODER_RESOLUTION)
                rad_action[name] = rad

        return rad_action

    def _update_grasp_assist(self) -> None:
        """Update grasp assist (attach/detach cube)."""
        if not self.config.enable_grasp_assist:
            return

        if (
            self._cube_qpos_adr is None
            or self._grasp_site_id is None
            or self._cube_site_id is None
            or self._gripper_body_id is None
            or self._cube_body_id is None
        ):
            return

        gpos = np.array(self.data.site_xpos[self._grasp_site_id], dtype=np.float64)
        cpos_site = np.array(self.data.site_xpos[self._cube_site_id], dtype=np.float64)
        dist = float(np.linalg.norm(gpos - cpos_site))
        u = self._get_gripper_u()

        want_attach = (u >= self.config.grasp_close_u_thresh) and (dist <= self.config.grasp_dist_thresh)
        want_detach = (u <= self.config.grasp_open_u_thresh)

        if self._attached:
            if want_detach:
                self._attached = False
                self._set_cube_collision(True)
                return

            # Update cube position relative to gripper
            gquat = np.array(self.data.xquat[self._gripper_body_id], dtype=np.float64)
            cquat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(cquat, gquat, self._rel_quat)

            dpos = np.empty(3, dtype=np.float64)
            mujoco.mju_rotVecQuat(dpos, self._rel_pos, gquat)

            cpos_new = gpos + dpos
            self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = cpos_new
            self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = cquat
            if self._cube_dof_adr is not None:
                self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            return

        if want_attach:
            gquat = np.array(self.data.xquat[self._gripper_body_id], dtype=np.float64)
            cquat_body = np.array(self.data.xquat[self._cube_body_id], dtype=np.float64)
            cpos_body = np.array(self.data.xpos[self._cube_body_id], dtype=np.float64)

            gquat_conj = _quat_conj(gquat)
            mujoco.mju_mulQuat(self._rel_quat, gquat_conj, cquat_body)
            mujoco.mju_rotVecQuat(self._rel_pos, cpos_body - gpos, gquat_conj)

            self._attached = True
            self._set_cube_collision(False)

    def _set_cube_collision(self, enabled: bool) -> None:
        """Enable/disable cube collision."""
        if self._cube_geom_id is None:
            return
        if enabled:
            self.model.geom_contype[self._cube_geom_id] = self._cube_contype_orig
            self.model.geom_conaffinity[self._cube_geom_id] = self._cube_conaff_orig
        else:
            self.model.geom_contype[self._cube_geom_id] = 0
            self.model.geom_conaffinity[self._cube_geom_id] = 0

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from simulation.

        Returns:
            dict containing:
            - {joint}.pos: Normalized joint positions
            - {camera_name}: Camera images (without observation.images. prefix)
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs = {}

        # Get normalized joint state
        state = self._normalize_state()
        obs.update(state)

        # Render cameras - use obs_camera_names for output keys (without prefix)
        for i, mjcf_cam_name in enumerate(self.config.camera_names):
            if i < len(self.config.obs_camera_names):
                obs_key = self.config.obs_camera_names[i]
            else:
                obs_key = mjcf_cam_name

            rgb = self._render_camera(mjcf_cam_name)
            if rgb is not None:
                # Apply camera2 preprocessing if enabled (to match training data)
                if obs_key == "camera2" and self.config.camera2_preprocess:
                    rgb = _rot180(rgb)
                    rgb = _to_gray3(rgb)
                obs[obs_key] = rgb

        # Add empty cameras (black images) if configured
        for i in range(self.config.num_empty_cameras):
            empty_img = np.zeros(
                (self.config.cam_height, self.config.cam_width, 3),
                dtype=np.uint8
            )
            obs[f"empty_camera_{i}"] = empty_img

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action to simulation.

        Args:
            action: Dictionary with normalized joint targets (e.g., "shoulder_pan.pos": 50.0)

        Returns:
            The action that was actually applied.
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Debug: Log received action (first few frames only to avoid spam)
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1

        action_vals = [action.get(f"{n}.pos", 0.0) for n in self.config.joint_names]

        if self._debug_frame_count <= 5:
            logger.debug(f"[send_action #{self._debug_frame_count}] Received action keys: {list(action.keys())}")
            logger.debug(f"[send_action #{self._debug_frame_count}] Normalized action values: {[f'{v:.2f}' for v in action_vals]}")

        # Check if actions are in expected range
        for i, (name, val) in enumerate(zip(self.config.joint_names, action_vals)):
            if name == "gripper":
                if val < -10 or val > 110:  # gripper should be [0, 100]
                    logger.warning(f"Action '{name}' out of expected range [0,100]: {val:.2f}")
            else:
                if val < -120 or val > 120:  # joints should be [-100, 100]
                    logger.warning(f"Action '{name}' out of expected range [-100,100]: {val:.2f}")

        # Denormalize action
        rad_action = self._denormalize_action(action)

        # Debug: Log denormalized action
        rad_vals = [rad_action.get(n, 0.0) for n in self.config.joint_names]
        if self._debug_frame_count <= 5:
            logger.debug(f"[send_action #{self._debug_frame_count}] Denormalized to radians: {[f'{v:.3f}' for v in rad_vals]}")

        # Apply to actuators
        for name, rad in rad_action.items():
            if name in self._joint_to_act:
                ai = self._joint_to_act[name]
                # Clip to actuator range
                if int(self.model.actuator_ctrllimited[ai]) != 0:
                    lo = float(self.model.actuator_ctrlrange[ai, 0])
                    hi = float(self.model.actuator_ctrlrange[ai, 1])
                    rad = max(lo, min(hi, rad))
                self.data.ctrl[ai] = rad

        # Step simulation
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
            self._update_grasp_assist()

        # Sync viewer
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        # Return the action we applied (in normalized form)
        return action

    def disconnect(self) -> None:
        """Cleanup simulation resources."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        if self.renderer is not None:
            self.renderer = None

        self.model = None
        self.data = None
        self._connected = False
        logger.info(f"{self} disconnected.")

    def reset_cube_position(self, index: int | None = None, reset_arm: bool = False) -> None:
        """
        Reset cube to a preset position, and optionally move the place target.

        Args:
            index: Position index. If None, calculates from episode count and trials_per_position.
            reset_arm: If True, also reset arm to home position.
        """
        # Reset arm to home position first if requested
        if reset_arm:
            self.reset_arm_to_home()

        if not self._cube_positions or self._cube_qpos_adr is None:
            return

        if index is None:
            # Calculate position index from episode count and trials_per_position
            trials = max(1, self.config.trials_per_position)
            index = (self._episode_count // trials) % len(self._cube_positions)
            logger.info(f"Episode {self._episode_count}, trials_per_position={trials}, using position index {index}")

        if index >= len(self._cube_positions):
            return

        # Track current position index for get_current_preset()
        self._cube_pos_idx = index

        preset = self._cube_positions[index]
        if "pos" in preset:
            pos = np.array(preset["pos"], dtype=np.float64)
            self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = pos
            logger.info(f"Set cube position[{index}]: {pos.tolist()}")
        if "quat" in preset:
            quat = np.array(preset["quat"], dtype=np.float64)
            self.data.qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = quat

        if self._cube_dof_adr is not None:
            self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

        # Also set place target position if specified in preset
        if "target_pos" in preset:
            self.set_place_target_position(preset["target_pos"])

        self._attached = False
        self._set_cube_collision(True)
        mujoco.mj_forward(self.model, self.data)

    def reset_arm_to_home(self) -> None:
        """
        Reset robot arm to home position using keyframe.
        Preserves cube and place target positions.
        """
        if self.model.nkey > 0:
            # Save cube position before reset (keyframe resets everything)
            cube_qpos = None
            if self._cube_qpos_adr is not None:
                cube_qpos = self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7].copy()

            # Save place target position
            place_target_pos = None
            if self._place_target_body_id is not None:
                place_target_pos = self.model.body_pos[self._place_target_body_id].copy()

            # Use mj_resetDataKeyframe for complete reset (includes qpos, qvel, ctrl, etc.)
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            # Restore cube position
            if cube_qpos is not None and self._cube_qpos_adr is not None:
                self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 7] = cube_qpos
                if self._cube_dof_adr is not None:
                    self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

            # Restore place target position
            if place_target_pos is not None and self._place_target_body_id is not None:
                self.model.body_pos[self._place_target_body_id] = place_target_pos

            mujoco.mj_forward(self.model, self.data)

            # Log the actual normalized values for debugging
            state = self._normalize_state()
            state_vals = [state.get(f"{n}.pos", 0.0) for n in self.config.joint_names]
            logger.info(f"Reset arm to home position: {[f'{v:.2f}' for v in state_vals]}")

    def prepare_next_episode(self) -> None:
        """
        Prepare for the next episode by resetting arm and cube position.
        This should be called at the end of each episode before starting the next one.
        """
        self._episode_count += 1
        self.reset_cube_position(index=None, reset_arm=True)
        logger.info(f"Prepared for episode {self._episode_count}")

    def get_cube_position(self) -> np.ndarray | None:
        """
        Get current cube position.

        Returns:
            Cube position as numpy array [x, y, z], or None if cube not found.
        """
        if self._cube_qpos_adr is None:
            return None
        return self.data.qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3].copy()

    def get_place_target_position(self) -> np.ndarray | None:
        """
        Get current place target (dish) position.

        Returns:
            Place target position as numpy array [x, y, z], or None if not found.
        """
        if self._place_target_body_id is None:
            return None
        return self.data.xpos[self._place_target_body_id].copy()

    def set_place_target_position(self, pos: list | tuple | np.ndarray) -> None:
        """
        Set place target (dish) position.

        Args:
            pos: New position [x, y, z].
        """
        if self._place_target_body_id is None:
            logger.warning("Place target body not found in scene")
            return

        pos_arr = np.array(pos, dtype=np.float64)
        # Update body position via mocap or direct manipulation
        # Note: For static bodies, we modify the body's com position
        self.model.body_pos[self._place_target_body_id] = pos_arr
        logger.info(f"Set place target position: {pos_arr.tolist()}")

    def get_current_preset(self) -> dict | None:
        """
        Get the current position preset (cube pos, target pos, etc.).

        Returns:
            Current preset dict, or None if no presets loaded.
        """
        if not self._cube_positions:
            return None
        # Return the preset at current index
        idx = min(self._cube_pos_idx, len(self._cube_positions) - 1)
        return self._cube_positions[idx]

    def get_distance_to_target(self) -> float | None:
        """
        Calculate horizontal distance from cube to place target.

        Returns:
            Distance in meters (XY plane only), or None if positions unavailable.
        """
        cube_pos = self.get_cube_position()
        if cube_pos is None:
            return None

        # Prefer target_pos from current preset (this is the authoritative source)
        target_pos = None
        preset = self.get_current_preset()
        if preset and "target_pos" in preset:
            target_pos = np.array(preset["target_pos"], dtype=np.float64)

        # Fallback to place_target body position if no preset
        if target_pos is None:
            target_pos = self.get_place_target_position()

        if target_pos is None:
            return None

        # XY distance only (ignore Z)
        return float(np.linalg.norm(cube_pos[:2] - target_pos[:2]))

    def is_cube_on_target(self) -> bool | None:
        """
        Check if cube is in contact with place target (dish).

        Uses MuJoCo contact detection to determine if the cube geom
        is touching any of the place target geoms.

        Returns:
            True if cube is touching place target, False if not, None if detection unavailable.
        """
        if self._cube_geom_id is None or not self._place_target_geom_ids:
            return None

        # Check all active contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            # Check if this contact involves cube and any place target geom
            if geom1 == self._cube_geom_id and geom2 in self._place_target_geom_ids:
                return True
            if geom2 == self._cube_geom_id and geom1 in self._place_target_geom_ids:
                return True

        return False
