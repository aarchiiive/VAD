from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torchvision import transforms

from navsim.agents.vad.vad_config import VADConfig
from navsim.common.dataclasses import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


class VADFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for VAD."""

    def __init__(self, config: VADConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of VAD
        """
        self._config = config
        self.img2tensor = transforms.ToTensor()
        self.camera_names = [
            "cam_f0",
            "cam_l0",
            "cam_l1",
            "cam_l2",
            "cam_r0",
            "cam_r1",
            "cam_r2",
            "cam_b0",
        ]

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "vad_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, Any]:
        """Inherited, see superclass."""
        features: Dict[str, Any] = {}

        # Aggregate aligned ego features first (matches NuScenes style keys)
        features.update(self._align_with_nuscenes_format(agent_input))
        features.update(self._get_ego_trajectories(agent_input))
        features["ego_status"] = self._get_ego_status(agent_input)

        # Sensor data in multiple granularities
        features.update(self._get_sensor_data(agent_input))
        features.update(self._get_sensor_paths_aligned(agent_input))
        features.update(self._get_temporal_sensor_paths(agent_input))
        features["lidar_sweeps"] = self._get_lidar_sweeps(agent_input)

        # Calibration and temporal info
        features["camera_calibration"] = self._get_camera_calibration(agent_input)
        features.update(self._get_temporal_info(agent_input))

        return features

    def _get_ego_status(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Extract ego status from AgentInput.
        :param agent_input: input dataclass
        :return: dict of ego tensors
        """
        ego_status = agent_input.ego_statuses[-1]
        return {
            "ego_pose": torch.tensor(ego_status.ego_pose, dtype=torch.float32),
            "ego_velocity": torch.tensor(ego_status.ego_velocity, dtype=torch.float32),
            "ego_acceleration": torch.tensor(ego_status.ego_acceleration, dtype=torch.float32),
            "driving_command": torch.tensor(ego_status.driving_command, dtype=torch.int64),
        }

    def _get_ego_trajectories(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Extract ego historical trajectory offsets.
        :param agent_input: input dataclass
        :return: dict containing history offsets and absolute poses
        """
        num_history = len(agent_input.ego_statuses)
        ego_poses = np.array([status.ego_pose for status in agent_input.ego_statuses], dtype=np.float32)

        if num_history > 1:
            ego_his_trajs = ego_poses[1:, :2] - ego_poses[:-1, :2]
        else:
            ego_his_trajs = np.zeros((0, 2), dtype=np.float32)

        return {
            "ego_his_trajs": torch.tensor(ego_his_trajs, dtype=torch.float32),
            "ego_poses_history": torch.tensor(ego_poses[:, :2], dtype=torch.float32),
        }

    def _get_sensor_data(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Extract sensor data with temporal hierarchy.
        :param agent_input: input dataclass
        :return: hierarchical sensor data
        """
        num_frames = len(agent_input.cameras)
        image_paths_temporal: Dict[str, Dict[str, str]] = {}
        lidar_paths_temporal: List[str] = []

        for frame_idx in range(num_frames):
            cameras = agent_input.cameras[frame_idx]
            lidar = agent_input.lidars[frame_idx]

            frame_paths: Dict[str, str] = {}
            for cam_name in self.camera_names:
                cam_data = getattr(cameras, cam_name)
                frame_paths[cam_name] = str(cam_data.camera_path) if cam_data.camera_path is not None else ""
            image_paths_temporal[f"frame_{frame_idx}"] = frame_paths

            lidar_path = str(lidar.lidar_path) if lidar.lidar_path is not None else ""
            lidar_paths_temporal.append(lidar_path)

        current_image_paths = image_paths_temporal[f"frame_{num_frames - 1}"]
        current_lidar_path = lidar_paths_temporal[-1] if lidar_paths_temporal else ""

        return {
            "image_paths_temporal": image_paths_temporal,
            "lidar_paths_temporal": lidar_paths_temporal,
            "image_paths": current_image_paths,
            "lidar_path": current_lidar_path,
        }

    def _get_camera_calibration(self, agent_input: AgentInput) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract camera calibration information.
        :param agent_input: input dataclass
        :return: dict of calibration tensors per camera
        """
        cameras = agent_input.cameras[-1]

        intrinsics: Dict[str, torch.Tensor] = {}
        cam2lidar: Dict[str, torch.Tensor] = {}
        lidar2cam: Dict[str, torch.Tensor] = {}
        img2lidar: Dict[str, torch.Tensor] = {}
        lidar2img: Dict[str, torch.Tensor] = {}

        for cam_name in self.camera_names:
            cam_data = getattr(cameras, cam_name)

            if cam_data.intrinsics is not None:
                intrinsics[cam_name] = torch.tensor(cam_data.intrinsics, dtype=torch.float32)

            if (
                cam_data.sensor2lidar_rotation is not None
                and cam_data.sensor2lidar_translation is not None
                and cam_data.intrinsics is not None
            ):
                cam2lidar_tf = np.eye(4, dtype=np.float32)
                cam2lidar_tf[:3, :3] = cam_data.sensor2lidar_rotation
                cam2lidar_tf[:3, 3] = cam_data.sensor2lidar_translation
                cam2lidar[cam_name] = torch.tensor(cam2lidar_tf, dtype=torch.float32)

                lidar2cam_tf = np.linalg.inv(cam2lidar_tf)
                lidar2cam[cam_name] = torch.tensor(lidar2cam_tf, dtype=torch.float32)

                k_inv = np.linalg.inv(cam_data.intrinsics)
                k_inv_tf = np.eye(4, dtype=np.float32)
                k_inv_tf[:3, :3] = k_inv

                img2lidar_tf = cam2lidar_tf @ k_inv_tf
                img2lidar[cam_name] = torch.tensor(img2lidar_tf, dtype=torch.float32)

                k_tf = np.eye(4, dtype=np.float32)
                k_tf[:3, :3] = cam_data.intrinsics
                lidar2img_tf = k_tf @ lidar2cam_tf
                lidar2img[cam_name] = torch.tensor(lidar2img_tf, dtype=torch.float32)

        return {
            "intrinsics": intrinsics,
            "cam2lidar": cam2lidar,
            "lidar2cam": lidar2cam,
            "img2lidar": img2lidar,
            "lidar2img": lidar2img,
        }

    def _get_temporal_info(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Extract temporal information from AgentInput.
        :param agent_input: input dataclass
        :return: temporal metadata tensors
        """
        num_history = len(agent_input.ego_statuses)
        return {
            "num_history_frames": torch.tensor(num_history, dtype=torch.int64),
            "frame_sequence_length": torch.tensor(num_history, dtype=torch.int64),
        }

    def _get_lidar_sweeps(self, agent_input: AgentInput) -> List[Dict[str, Any]]:
        """
        Extract LiDAR sweep information for temporal context.
        :param agent_input: input dataclass
        :return: list of sweep metadata
        """
        sweeps: List[Dict[str, Any]] = []
        for idx, lidar in enumerate(agent_input.lidars[:-1]):
            if lidar.lidar_path is not None:
                sweeps.append(
                    {
                        "data_path": str(lidar.lidar_path),
                        "frame_idx": idx,
                        "is_current": False,
                    }
                )

        current_lidar = agent_input.lidars[-1]
        if current_lidar.lidar_path is not None:
            sweeps.append(
                {
                    "data_path": str(current_lidar.lidar_path),
                    "frame_idx": len(agent_input.lidars) - 1,
                    "is_current": True,
                }
            )

        return sweeps

    def _get_sensor_paths_aligned(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Get sensor paths in NuScenes-like format for the current frame.
        :param agent_input: input dataclass
        :return: dictionary containing camera and lidar metadata
        """
        cameras = agent_input.cameras[-1]
        lidar = agent_input.lidars[-1]

        cams: Dict[str, Dict[str, Any]] = {}
        for cam_name in self.camera_names:
            cam_data = getattr(cameras, cam_name)
            cams[cam_name] = {
                "data_path": str(cam_data.camera_path) if cam_data.camera_path is not None else "",
                "cam_intrinsic": cam_data.intrinsics.tolist() if cam_data.intrinsics is not None else None,
                "sensor2lidar_rotation": cam_data.sensor2lidar_rotation.tolist()
                if cam_data.sensor2lidar_rotation is not None
                else None,
                "sensor2lidar_translation": cam_data.sensor2lidar_translation.tolist()
                if cam_data.sensor2lidar_translation is not None
                else None,
            }

        return {
            "cams": cams,
            "lidar_path": str(lidar.lidar_path) if lidar.lidar_path is not None else "",
        }

    def _get_temporal_sensor_paths(self, agent_input: AgentInput) -> Dict[str, Any]:
        """
        Get temporal sensor paths for all frames.
        :param agent_input: input dataclass
        :return: temporal sensor metadata
        """
        num_frames = len(agent_input.cameras)
        cams_temporal: Dict[str, Dict[str, Dict[str, Any]]] = {}
        lidar_paths_temporal: List[str] = []

        for frame_idx in range(num_frames):
            cameras = agent_input.cameras[frame_idx]
            lidar = agent_input.lidars[frame_idx]

            frame_cams: Dict[str, Dict[str, Any]] = {}
            for cam_name in self.camera_names:
                cam_data = getattr(cameras, cam_name)
                frame_cams[cam_name] = {
                    "data_path": str(cam_data.camera_path) if cam_data.camera_path is not None else "",
                    "cam_intrinsic": cam_data.intrinsics.tolist() if cam_data.intrinsics is not None else None,
                    "sensor2lidar_rotation": cam_data.sensor2lidar_rotation.tolist()
                    if cam_data.sensor2lidar_rotation is not None
                    else None,
                    "sensor2lidar_translation": cam_data.sensor2lidar_translation.tolist()
                    if cam_data.sensor2lidar_translation is not None
                    else None,
                }

            cams_temporal[f"frame_{frame_idx}"] = frame_cams
            lidar_paths_temporal.append(str(lidar.lidar_path) if lidar.lidar_path is not None else "")

        return {
            "cams_temporal": cams_temporal,
            "lidar_paths_temporal": lidar_paths_temporal,
        }

    def _align_with_nuscenes_format(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Align with original NuScenes data format keys and structure.
        :param agent_input: input dataclass
        :return: aligned tensors mimicking NuScenes keys
        """
        current_ego = agent_input.ego_statuses[-1]
        num_history = len(agent_input.ego_statuses)

        aligned_features: Dict[str, torch.Tensor] = {
            "frame_idx": torch.tensor(num_history - 1, dtype=torch.int64),
            "frame_sequence_length": torch.tensor(num_history, dtype=torch.int64),
            "timestamp": torch.tensor(0, dtype=torch.int64),
            "ego_pose": torch.tensor(current_ego.ego_pose, dtype=torch.float32),
            "ego_velocity": torch.tensor(current_ego.ego_velocity, dtype=torch.float32),
            "ego_acceleration": torch.tensor(current_ego.ego_acceleration, dtype=torch.float32),
            "ego_speed": torch.tensor(np.linalg.norm(current_ego.ego_velocity), dtype=torch.float32),
            "ego_heading": torch.tensor(current_ego.ego_pose[2], dtype=torch.float32),
            "ego_length": torch.tensor(5.176, dtype=torch.float32),
            "ego_width": torch.tensor(2.297, dtype=torch.float32),
            "driving_command": torch.tensor(current_ego.driving_command, dtype=torch.int64),
        }

        driving_cmd_onehot = np.zeros(3, dtype=np.float32)
        if hasattr(current_ego.driving_command, "__len__") and len(current_ego.driving_command) > 0:
            cmd = current_ego.driving_command[0]
        else:
            cmd = int(current_ego.driving_command)

        if cmd == 0:
            driving_cmd_onehot[0] = 1.0
        elif cmd == 1:
            driving_cmd_onehot[1] = 1.0
        else:
            driving_cmd_onehot[2] = 1.0

        aligned_features["ego_fut_cmd"] = torch.tensor(driving_cmd_onehot, dtype=torch.float32)

        ego_poses = np.array([status.ego_pose for status in agent_input.ego_statuses], dtype=np.float32)
        if num_history > 1:
            ego_his_trajs = ego_poses[1:, :2] - ego_poses[:-1, :2]
        else:
            ego_his_trajs = np.zeros((0, 2), dtype=np.float32)
        aligned_features["ego_his_trajs"] = torch.tensor(ego_his_trajs, dtype=torch.float32)

        ego_lcf_feat = np.zeros(9, dtype=np.float32)
        ego_lcf_feat[0:2] = current_ego.ego_velocity
        ego_lcf_feat[2:4] = current_ego.ego_acceleration
        ego_lcf_feat[4] = 0.0  # yaw rate placeholder
        ego_lcf_feat[5] = 5.176  # ego length
        ego_lcf_feat[6] = 2.297  # ego width
        ego_lcf_feat[7] = np.linalg.norm(current_ego.ego_velocity)
        ego_lcf_feat[8] = 0.0  # steering / curvature placeholder
        aligned_features["ego_lcf_feat"] = torch.tensor(ego_lcf_feat, dtype=torch.float32)

        return aligned_features
