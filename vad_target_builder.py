from __future__ import annotations
from typing import Any, Dict, List, Tuple

import os
import cv2
import numpy as np
import numpy.typing as npt

import torch

from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely import affinity
from shapely.geometry import LineString, Polygon

from navsim.agents.vad.vad_config import VADConfig
from navsim.common.dataclasses import Annotations, Scene, NAVSIM_INTERVAL_LENGTH
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder


NUPLAN_MAPS_ROOT = os.environ["NUPLAN_MAPS_ROOT"]


class VADTargetBuilder(AbstractTargetBuilder):
    """
    Output target builder for VAD with proper ego-relative coordinate system.

    All trajectories are computed in ego-relative coordinates to handle ego motion correctly.
    """

    def __init__(self, trajectory_sampling: TrajectorySampling, config: VADConfig):
        """
        Initialize target builder with coordinate system specification.

        Args:
            trajectory_sampling: Trajectory sampling configuration.
            config: VAD configuration object.
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

        # NavSim uses a reduced label set; keep a single authoritative order here
        self.category_order: List[TrackedObjectType] = [
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.BARRIER,
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.GENERIC_OBJECT,
        ]
        self.agent_type_mapping = {category.fullname: idx for idx, category in enumerate(self.category_order)}
        self.category_names = [category.fullname for category in self.category_order]

        # Pacifica vehicle dimensions (accurate parameters)
        self.ego_width = 1.1485 * 2.0   # 2.297m
        self.ego_front_length = 4.049   # 4.049m
        self.ego_rear_length = 1.127    # 1.127m
        self.ego_length = self.ego_front_length + self.ego_rear_length  # 5.176m
        self.ego_height = 1.777         # 1.777m
        self.ego_wheel_base = 3.089     # 3.089m
        self.ego_cog_from_rear = 1.67   # 1.67m

    def get_unique_name(self) -> str:
        """Return unique identifier for this target builder."""
        return "vad_target"

    def compute_targets(self, scene: Scene) -> Dict[str, Any]:
        """
        Compute target tensors from the full scene dataclass.

        Args:
            scene: Scene dataclass containing privileged information.

        Returns:
            Dictionary with all VAD supervision targets.
        """
        targets: Dict[str, Any] = {}

        current_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_idx]
        annotations = current_frame.annotations
        ego_pose = StateSE2(*current_frame.ego_status.ego_pose)

        # Core VAD targets (trajectory, detections, semantic map)
        targets["trajectory"] = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses).poses,
            dtype=torch.float32,
        )

        agent_states, agent_labels = self._compute_agent_targets(annotations)
        targets["agent_states"] = agent_states
        targets["agent_labels"] = agent_labels
        targets["bev_semantic_map"] = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        # Scene-level ground truth
        targets.update(self._get_scene_gt_info(scene))

        # Agent futures (with proper ego-relative coordinate handling)
        agent_future = self._get_agent_future_trajectories(scene)
        targets.update(agent_future)

        # Additional agent feature targets (uses future info)
        targets.update(self._get_agent_lcf_features(scene, agent_future))

        # Compatibility and bookkeeping information
        targets.update(self._get_nuscenes_compatibility_info(scene))

        return targets

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 2D agent bounding boxes in ego coordinates.

        All coordinate data kept in NavSim coordinate system for consistency.

        Args:
            annotations: Current frame annotations.

        Returns:
            Tuple of (agent_states, agent_labels) tensors.
        """
        max_agents = getattr(self._config, "num_bounding_boxes", 200)
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float) -> bool:
            """Check if position is within LiDAR detection range."""
            return (
                getattr(self._config, "lidar_min_x", -50.0) <= x <= getattr(self._config, "lidar_max_x", 50.0)
                and getattr(self._config, "lidar_min_y", -50.0) <= y <= getattr(self._config, "lidar_max_y", 50.0)
            )

        if annotations is not None:
            for box, name in zip(annotations.boxes, annotations.names):
                box_x, box_y = box[0], box[1]
                if name == "vehicle" and _xy_in_lidar(box_x, box_y):
                    # Keep all data in NavSim coordinate system
                    box_heading = box[6]  # NavSim heading - no conversion
                    box_length = box[3]
                    box_width = box[4]
                    agent_states_list.append(
                        np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                    )

        # Initialize output arrays
        agent_states = np.zeros((max_agents, 5), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agent_states_list) > 0:
            agents_states_arr = np.array(agent_states_list)
            # Sort by distance to ego vehicle
            distances = np.linalg.norm(agents_states_arr[:, :2], axis=-1)
            argsort = np.argsort(distances)[:max_agents]
            agents_states_arr = agents_states_arr[argsort]
            agent_states[:len(agents_states_arr)] = agents_states_arr
            agent_labels[:len(agents_states_arr)] = True

        return torch.tensor(agent_states, dtype=torch.float32), torch.tensor(agent_labels, dtype=torch.bool)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Create semantic map in BEV following configured layers.

        Args:
            annotations: Current frame annotations.
            map_api: Map API for retrieving map elements.
            ego_pose: Current ego pose.

        Returns:
            BEV semantic map tensor.
        """
        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)

        if map_api is not None:
            for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
                if entity_type == "polygon":
                    entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
                elif entity_type == "linestring":
                    entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
                else:
                    entity_mask = self._compute_box_mask(annotations, layers)
                bev_semantic_map[entity_mask] = label

        return torch.tensor(bev_semantic_map, dtype=torch.int64)

    def _get_scene_gt_info(self, scene: Scene) -> Dict[str, Any]:
        """
        Extract ground-truth metadata and ego trajectories.

        ALL COORDINATE DATA KEPT IN NAVSIM SYSTEM FOR CONSISTENCY.

        Args:
            scene: Scene containing ground truth information.

        Returns:
            Dictionary of ground truth information.
        """
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_idx]
        annotations = current_frame.annotations
        ego_status = current_frame.ego_status

        gt_info: Dict[str, Any] = {
            "scene_token": scene.scene_metadata.scene_token,
            "log_name": scene.scene_metadata.log_name,
            "map_location": scene.scene_metadata.map_name,
            "timestamp": torch.tensor(current_frame.timestamp, dtype=torch.int64),
            "roadblock_ids": current_frame.roadblock_ids,
            "traffic_lights": current_frame.traffic_lights,
        }

        if annotations is not None and len(annotations.boxes) > 0:
            # Keep all box data in NavSim coordinate system
            agent_boxes = np.zeros((len(annotations.boxes), 7), dtype=np.float32)
            agent_names: List[str] = []

            for i, (box, name) in enumerate(zip(annotations.boxes, annotations.names)):
                # Store box data in NavSim format - NO COORDINATE CONVERSION
                agent_boxes[i, 0:3] = box[0:3]  # x, y, z (NavSim coordinates)
                agent_boxes[i, 3] = box[3]      # length (keep original order)
                agent_boxes[i, 4] = box[4]      # width
                agent_boxes[i, 5] = box[5]      # height
                agent_boxes[i, 6] = box[6]      # heading (NavSim coordinate system)

                # Map agent names to standard categories
                mapped_idx = self.agent_type_mapping.get(name)
                if mapped_idx is not None and mapped_idx < len(self.category_names):
                    agent_names.append(self.category_names[mapped_idx])
                else:
                    agent_names.append(name)

            gt_info.update({
                "gt_boxes": torch.tensor(agent_boxes, dtype=torch.float32),
                "gt_names": agent_names,
                "gt_velocity_3d": torch.tensor(annotations.velocity_3d, dtype=torch.float32),
                "gt_velocity": torch.tensor(annotations.velocity_3d[:, :2], dtype=torch.float32),
                "instance_tokens": annotations.instance_tokens,
                "track_tokens": annotations.track_tokens,
                "num_agents": torch.tensor(len(annotations.boxes), dtype=torch.int64),
                "valid_flag": torch.ones(len(annotations.boxes), dtype=torch.bool),
                "num_lidar_pts": torch.zeros(len(annotations.boxes), dtype=torch.int64),
                "num_radar_pts": torch.zeros(len(annotations.boxes), dtype=torch.int64),
            })
        else:
            # Empty scene handling
            gt_info.update({
                "gt_boxes": torch.zeros((0, 7), dtype=torch.float32),
                "gt_names": [],
                "gt_velocity_3d": torch.zeros((0, 3), dtype=torch.float32),
                "gt_velocity": torch.zeros((0, 2), dtype=torch.float32),
                "instance_tokens": [],
                "track_tokens": [],
                "num_agents": torch.tensor(0, dtype=torch.int64),
                "valid_flag": torch.zeros((0,), dtype=torch.bool),
                "num_lidar_pts": torch.zeros((0,), dtype=torch.int64),
                "num_radar_pts": torch.zeros((0,), dtype=torch.int64),
            })

        # Ego history trajectory (as position offsets)
        # history_traj = scene.get_history_trajectory()
        # history_offsets = history_traj.poses.astype(np.float32)
        # if history_offsets.shape[0] > 1:
        #     history_deltas = history_offsets[1:, :2] - history_offsets[:-1, :2]
        # else:
        #     history_deltas = np.zeros((0, 2), dtype=np.float32)
        # gt_info["gt_ego_his_trajs"] = torch.tensor(history_deltas, dtype=torch.float32)

        history_traj = scene.get_history_trajectory()
        history_offsets = history_traj.poses.astype(np.float32)
        if history_offsets.shape[0] > 1:
            # History should be reversed: older -> newer
            history_deltas = history_offsets[:-1, :2] - history_offsets[1:, :2]
            history_deltas = history_deltas[::-1].copy()  # Add .copy() to fix negative stride
        else:
            history_deltas = np.zeros((0, 2), dtype=np.float32)
        gt_info["gt_ego_his_trajs"] = torch.tensor(history_deltas, dtype=torch.float32)

        # Ego future trajectory (as position offsets)
        future_traj = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        future_offsets = future_traj.poses.astype(np.float32)
        if future_offsets.shape[0] > 1:
            ego_fut_trajs = future_offsets[1:, :2] - future_offsets[:-1, :2]
        else:
            ego_fut_trajs = np.zeros((0, 2), dtype=np.float32)
        gt_info["gt_ego_fut_trajs"] = torch.tensor(ego_fut_trajs, dtype=torch.float32)
        gt_info["gt_ego_fut_masks"] = torch.ones(future_offsets.shape[0] - 1, dtype=torch.float32)

        # Driving command (one-hot encoded from final trajectory offset)
        if future_offsets.shape[0] > 0:
            final_offset = future_offsets[-1]
            if final_offset[0] >= 2.0:
                ego_cmd = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # forward
            elif final_offset[0] <= -2.0:
                ego_cmd = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # backward
            else:
                ego_cmd = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # maintain/stop
        else:
            ego_cmd = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        gt_info["gt_ego_fut_cmd"] = torch.tensor(ego_cmd, dtype=torch.float32)
        gt_info["fut_valid_flag"] = torch.tensor(True, dtype=torch.bool)

        # Ego LCF feature vector (vx, vy, ax, ay, yaw_rate, length, width, speed, curvature)
        ego_lcf_feat = np.zeros(9, dtype=np.float32)
        ego_lcf_feat[0:2] = ego_status.ego_velocity     # vx, vy
        ego_lcf_feat[2:4] = ego_status.ego_acceleration # ax, ay

        # Calculate yaw rate from future trajectory
        if future_offsets.shape[0] > 1:
            yaw_diff = future_offsets[1, 2] - future_offsets[0, 2]
            ego_lcf_feat[4] = yaw_diff / NAVSIM_INTERVAL_LENGTH
        else:
            ego_lcf_feat[4] = 0.0

        ego_lcf_feat[5] = self.ego_length                              # length (5.176m)
        ego_lcf_feat[6] = self.ego_width                               # width (2.297m)
        ego_lcf_feat[7] = np.linalg.norm(ego_status.ego_velocity)      # speed
        ego_lcf_feat[8] = 0.0                                          # curvature placeholder

        gt_info["gt_ego_lcf_feat"] = torch.tensor(ego_lcf_feat, dtype=torch.float32)

        return gt_info

    def _get_agent_future_trajectories(self, scene: Scene) -> Dict[str, Any]:
        """
        Extract agent future trajectories in proper ego-relative coordinates.

        FIXED: Now handles ego motion correctly by transforming to ego-relative coordinates.

        Args:
            scene: Scene containing agent trajectory information.

        Returns:
            Dictionary of agent future trajectory data.
        """
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_annotations = scene.frames[current_idx].annotations
        fut_ts = self._trajectory_sampling.num_poses

        if current_annotations is None or len(current_annotations.track_tokens) == 0:
            return {
                "gt_agent_fut_trajs": torch.zeros((0, fut_ts * 2), dtype=torch.float32),
                "gt_agent_fut_masks": torch.zeros((0, fut_ts), dtype=torch.float32),
                "gt_agent_fut_yaw": torch.zeros((0, fut_ts), dtype=torch.float32),
            }

        num_agents = len(current_annotations.track_tokens)
        agent_future_trajs = np.zeros((num_agents, fut_ts, 2), dtype=np.float32)
        agent_future_masks = np.zeros((num_agents, fut_ts), dtype=np.float32)
        agent_future_yaw = np.zeros((num_agents, fut_ts), dtype=np.float32)

        current_ego_pose = StateSE2(*scene.frames[current_idx].ego_status.ego_pose)

        for agent_idx, track_token in enumerate(current_annotations.track_tokens):
            current_box = current_annotations.boxes[agent_idx]
            current_local_pos = current_box[:2]
            prev_global_pos = self._transform_from_ego_frame(current_local_pos, current_ego_pose)
            prev_pos_current = self._transform_to_ego_frame(prev_global_pos, current_ego_pose)
            prev_global_yaw = self._wrap_angle(current_box[6] + current_ego_pose.heading)

            for fut_step in range(fut_ts):
                future_idx = current_idx + 1 + fut_step
                if future_idx >= len(scene.frames):
                    break

                future_frame = scene.frames[future_idx]
                future_annotations = future_frame.annotations
                if future_annotations is None or track_token not in future_annotations.track_tokens:
                    break

                future_agent_idx = future_annotations.track_tokens.index(track_token)
                future_box = future_annotations.boxes[future_agent_idx]
                future_local_pos = future_box[:2]
                future_ego_pose = StateSE2(*future_frame.ego_status.ego_pose)

                future_global_pos = self._transform_from_ego_frame(future_local_pos, future_ego_pose)
                future_pos_current = self._transform_to_ego_frame(future_global_pos, current_ego_pose)

                agent_future_trajs[agent_idx, fut_step] = future_pos_current - prev_pos_current
                agent_future_masks[agent_idx, fut_step] = 1.0

                future_global_yaw = self._wrap_angle(future_box[6] + future_ego_pose.heading)
                agent_future_yaw[agent_idx, fut_step] = self._wrap_angle(future_global_yaw - prev_global_yaw)

                prev_global_pos = future_global_pos
                prev_pos_current = future_pos_current
                prev_global_yaw = future_global_yaw

        return {
            "gt_agent_fut_trajs": torch.tensor(agent_future_trajs.reshape(num_agents, -1), dtype=torch.float32),
            "gt_agent_fut_masks": torch.tensor(agent_future_masks, dtype=torch.float32),
            "gt_agent_fut_yaw": torch.tensor(agent_future_yaw, dtype=torch.float32),
        }

    @staticmethod
    def _transform_from_ego_frame(local_pos: np.ndarray, ego_pose: StateSE2) -> np.ndarray:
        """
        Transform position from ego-relative frame to global coordinates.

        Args:
            local_pos: Position expressed in the ego frame.
            ego_pose: Ego pose in global coordinates.

        Returns:
            Position in global coordinates.
        """
        cos_h, sin_h = np.cos(ego_pose.heading), np.sin(ego_pose.heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        return rotation_matrix @ local_pos + np.array([ego_pose.x, ego_pose.y])

    def _transform_to_ego_frame(self, global_pos: np.ndarray, ego_pose: StateSE2) -> np.ndarray:
        """
        Transform global position to ego-relative frame.

        Args:
            global_pos: Position in global coordinates [x, y].
            ego_pose: Ego vehicle pose.

        Returns:
            Position in ego-relative coordinates.
        """
        # Translate to ego origin
        translated = global_pos - np.array([ego_pose.x, ego_pose.y])

        # Rotate to ego heading
        cos_h, sin_h = np.cos(-ego_pose.heading), np.sin(-ego_pose.heading)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        return rotation_matrix @ translated

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _get_agent_lcf_features(self, scene: Scene, agent_future: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute agent latent curve features leveraging future predictions.

        ALL FEATURE DATA KEPT IN NAVSIM COORDINATE SYSTEM.

        Args:
            scene: Scene containing agent information.
            agent_future: Dictionary containing agent future trajectory data.

        Returns:
            Dictionary of agent LCF features and goal classifications.
        """
        current_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[current_idx].annotations

        if annotations is None or len(annotations.boxes) == 0:
            return {
                "gt_agent_lcf_feat": torch.zeros((0, 9), dtype=torch.float32),
                "gt_agent_fut_goal": torch.zeros((0,), dtype=torch.float32),
            }

        num_agents = len(annotations.boxes)
        agent_lcf_feat = np.zeros((num_agents, 9), dtype=np.float32)

        for i, (box, name) in enumerate(zip(annotations.boxes, annotations.names)):
            # All features in NavSim coordinate system
            agent_lcf_feat[i, 0:2] = box[:2]                          # x, y position
            agent_lcf_feat[i, 2] = box[6]                             # heading (NavSim)
            agent_lcf_feat[i, 3:5] = annotations.velocity_3d[i, :2]   # vx, vy
            agent_lcf_feat[i, 5] = box[4]                             # width
            agent_lcf_feat[i, 6] = box[3]                             # length
            agent_lcf_feat[i, 7] = box[5]                             # height
            agent_lcf_feat[i, 8] = float(self.agent_type_mapping.get(name, -1))  # type

        # Compute goal classification from future trajectories (ego-relative coordinates)
        agent_future_goal = np.zeros(num_agents, dtype=np.float32)
        fut_trajs_tensor = agent_future["gt_agent_fut_trajs"]
        fut_trajs_np = fut_trajs_tensor.reshape(num_agents, -1, 2).cpu().numpy() if num_agents > 0 else np.zeros(
            (0, self._trajectory_sampling.num_poses, 2), dtype=np.float32
        )

        for i in range(num_agents):
            traj = fut_trajs_np[i]
            if traj.size == 0:
                agent_future_goal[i] = 9  # static class
                continue

            # Get final position from cumulative trajectory (ego-relative coordinates)
            cumulative = np.cumsum(traj, axis=0)
            final_offset = cumulative[-1] if cumulative.size > 0 else np.zeros(2, dtype=np.float32)

            if np.linalg.norm(final_offset) < 1.0:
                agent_future_goal[i] = 9  # static class
            else:
                # Discretize direction into 8 bins (ego-relative coordinate system)
                goal_yaw = np.arctan2(final_offset[1], final_offset[0]) + np.pi
                agent_future_goal[i] = np.floor(goal_yaw / (np.pi / 4.0)) % 8

        return {
            "gt_agent_lcf_feat": torch.tensor(agent_lcf_feat, dtype=torch.float32),
            "gt_agent_fut_goal": torch.tensor(agent_future_goal, dtype=torch.float32),
        }

    def _get_nuscenes_compatibility_info(self, scene: Scene) -> Dict[str, Any]:
        """
        Collect additional metadata for NuScenes-style consumption.

        Args:
            scene: Scene containing compatibility information.

        Returns:
            Dictionary of NuScenes compatibility data.
        """
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_idx]
        ego_status = current_frame.ego_status

        # Navigation tokens for temporal relationships
        # prev_token = scene.frames[current_idx - 1].token if current_idx > 0 else ""
        # next_token = scene.frames[current_idx + 1].token if current_idx < len(scene.frames) - 1 else ""

        # Navigation tokens for temporal relationships - with proper bounds checking
        prev_token = ""
        next_token = ""

        if current_idx > 0 and current_idx - 1 < len(scene.frames):
            prev_token = scene.frames[current_idx - 1].token

        if current_idx + 1 < len(scene.frames):
            next_token = scene.frames[current_idx + 1].token

        # CAN bus data simulation (18-element array for NuScenes compatibility)
        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0:2] = ego_status.ego_pose[:2]                               # x, y position
        can_bus[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)     # quaternion placeholder
        can_bus[7:9] = ego_status.ego_acceleration                           # ax, ay
        can_bus[13:15] = ego_status.ego_velocity                             # vx, vy

        return {
            "token": current_frame.token,
            "prev": prev_token,
            "next": next_token,
            "can_bus": torch.tensor(can_bus, dtype=torch.float32),
            "frame_idx": torch.tensor(current_idx, dtype=torch.int64),

            # Coordinate transformation placeholders (handled in converter)
            "lidar2ego_translation": torch.zeros(3, dtype=torch.float32),
            "lidar2ego_rotation": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "ego2global_translation": torch.tensor(ego_status.ego_pose[:3], dtype=torch.float32),
            "ego2global_rotation": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
            "sweeps": [],  # Empty for NavSim (no LiDAR sweeps)
        }

    # Map helper methods for semantic BEV generation
    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute mask for map polygon elements.

        Args:
            map_api: Map API for retrieving elements.
            ego_pose: Current ego pose.
            layers: Semantic map layers to process.

        Returns:
            Boolean mask for polygon elements.
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)

        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(mask, [exterior], color=255)

        mask = np.rot90(mask)[::-1]
        return mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute mask for map linestring elements.

        Args:
            map_api: Map API for retrieving elements.
            ego_pose: Current ego pose.
            layers: Semantic map layers to process.

        Returns:
            Boolean mask for linestring elements.
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)

        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(mask, [points], isClosed=False, color=255, thickness=2)

        mask = np.rot90(mask)[::-1]
        return mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute mask for agent bounding boxes.

        Args:
            annotations: Current frame annotations.
            layers: Object types to include.

        Returns:
            Boolean mask for agent boxes.
        """
        mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)

        if annotations is not None:
            for name, box_value in zip(annotations.names, annotations.boxes):
                agent_type = tracked_object_types[name]
                if agent_type in layers:
                    x, y, heading = box_value[0], box_value[1], box_value[-1]
                    box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]

                    agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                    exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                    exterior = self._coords_to_pixel(exterior)
                    cv2.fillPoly(mask, [exterior], color=255)

        mask = np.rot90(mask)[::-1]
        return mask > 0

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform geometry to local coordinate system.

        Args:
            geometry: Shapely geometry object.
            origin: Origin pose for transformation.

        Returns:
            Transformed geometry in local coordinates.
        """
        a, b = np.cos(origin.heading), np.sin(origin.heading)
        d, e = -np.sin(origin.heading), np.cos(origin.heading)
        translated = affinity.affine_transform(geometry, [1, 0, 0, 1, -origin.x, -origin.y])
        rotated = affinity.affine_transform(translated, [a, b, d, e, 0, 0])
        return rotated

    def _coords_to_pixel(self, coords: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
        """
        Convert world coordinates to pixel coordinates.

        Args:
            coords: World coordinates array.

        Returns:
            Pixel coordinates array.
        """
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)

    # def get_local_vector_map(
    #     self,
    #     scene: Scene,
    #     frame_token: str,
    #     radius: float = 50.0,
    #     layers: List[SemanticMapLayer] = [
    #         SemanticMapLayer.LANE,
    #         SemanticMapLayer.LANE_CONNECTOR,
    #         SemanticMapLayer.ROADBLOCK,
    #         SemanticMapLayer.CROSSWALK,
    #     ],
    # ) -> Dict[SemanticMapLayer, List[Any]]:
    #     """Retrieve local vector map objects around ego pose for a specific frame."""
    #     from nuplan.common.actor_state.state_representation import StateSE2
    #     from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    #     # Find frame by token
    #     frame = next((f for f in scene.frames if f.token == frame_token), None)
    #     if frame is None:
    #         raise KeyError(f"Frame token {frame_token} not found in scene")

    #     # Ego pose and map API
    #     ego_pose = StateSE2(*frame.ego_status.ego_pose)

    #     # ✅ FIX: use global NUPLAN_MAPS_ROOT and static version string
    #     map_api = get_maps_api(
    #         map_root=NUPLAN_MAPS_ROOT,
    #         map_version="nuplan-maps-v1.0",
    #         map_name=scene.scene_metadata.map_name,
    #     )

    #     # Query nearby vector map objects
    #     local_map_objects = map_api.get_proximal_map_objects(
    #         point=ego_pose.point,
    #         radius=radius,
    #         layers=layers,
    #     )
    #     return local_map_objects


    def get_local_vector_map(
        self,
        scene: Scene,
        frame_token: str,
        radius: float = 50.0,
        layers: List[SemanticMapLayer] = list(SemanticMapLayer),
    ) -> Dict[str, Any]:
        """Retrieve all vector map layers (GeoDataFrames) around ego pose for a specific frame."""
        from shapely.geometry import box
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

        # Find target frame
        frame = next((f for f in scene.frames if f.token == frame_token), None)
        if frame is None:
            raise KeyError(f"Frame token {frame_token} not found in scene")

        # Ego pose and map API
        ego_pose = StateSE2(*frame.ego_status.ego_pose)

        # Use global map root + fixed version
        map_api = get_maps_api(
            map_root=NUPLAN_MAPS_ROOT,
            map_version="nuplan-maps-v1.0",
            map_name=scene.scene_metadata.map_name,
        )

        # Define bounding box around ego position
        x, y = ego_pose.x, ego_pose.y
        patch = box(x - radius, y - radius, x + radius, y + radius)

        # Collect vector map layers safely
        local_map_objects: Dict[str, Any] = {}

        for layer in layers:
            try:
                df = map_api._get_vector_map_layer(layer)
                if "geometry" not in df.columns:
                    continue
                # Select features inside patch
                subset = df[df["geometry"].intersects(patch)]
                if len(subset) > 0:
                    local_map_objects[layer.name] = subset
            except Exception:
                # Skip unsupported or missing layers
                continue

        return local_map_objects
