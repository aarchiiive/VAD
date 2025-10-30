from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mmcv
import numpy as np
import torch
from pyquaternion import Quaternion
from tqdm import tqdm

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.vad.vad_config import VADConfig
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractTargetBuilder
from vad_target_builder import VADTargetBuilder


def _build_sensor_config(num_history_frames: int) -> SensorConfig:
    """Load camera metadata for each history frame while skipping LiDAR blobs."""
    history_indices = list(range(num_history_frames))
    return SensorConfig(
        cam_f0=history_indices,
        cam_l0=history_indices,
        cam_l1=history_indices,
        cam_l2=history_indices,
        cam_r0=history_indices,
        cam_r1=history_indices,
        cam_r2=history_indices,
        cam_b0=history_indices,
        lidar_pc=False,
    )


def _to_rotation_matrix(rotation: Any) -> np.ndarray:
    """Convert quaternion or rotation matrix input to rotation matrix."""
    rot = np.asarray(rotation)
    if rot.shape == (3, 3):
        return rot
    if rot.shape == (4,):
        quat = Quaternion(rotation)
        return quat.rotation_matrix
    raise ValueError(f"Unsupported rotation format: shape={rot.shape}")


def _to_quaternion(rotation_matrix: np.ndarray) -> List[float]:
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    quat = Quaternion(matrix=rotation_matrix)
    return [float(quat.w), float(quat.x), float(quat.y), float(quat.z)]


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous matrix from rotation and translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _compute_sensor2ego(
    sensor_rot: Any,
    sensor_trans: Any,
    lidar_rot: Any,
    lidar_trans: Any,
) -> Tuple[List[float], List[float]]:
    """Compute sensor-to-ego transform given sensor-to-lidar and lidar-to-ego."""
    if sensor_rot is None or sensor_trans is None or lidar_rot is None or lidar_trans is None:
        return None, None

    sensor2lidar_rot = _to_rotation_matrix(sensor_rot)
    sensor2lidar_trans = np.asarray(sensor_trans, dtype=np.float64)
    lidar2ego_rot = _to_rotation_matrix(lidar_rot)
    lidar2ego_trans = np.asarray(lidar_trans, dtype=np.float64)

    T_sensor2lidar = _make_transform(sensor2lidar_rot, sensor2lidar_trans)
    T_lidar2ego = _make_transform(lidar2ego_rot, lidar2ego_trans)
    T_sensor2ego = T_lidar2ego @ T_sensor2lidar

    rotation = _to_quaternion(T_sensor2ego[:3, :3])
    translation = T_sensor2ego[:3, 3].astype(float).tolist()
    return rotation, translation


def _build_camera_entries(
    raw_frame: Dict[str, Any],
    lidar_rot: Any,
    lidar_trans: Any,
) -> Dict[str, Dict[str, Any]]:
    """Construct camera dictionary aligned with NuScenes format."""
    cams_info: Dict[str, Dict[str, Any]] = {}
    cams = raw_frame.get("cams", {})

    for cam_name, cam_data in cams.items():
        # Normalize key to lowercase like agent input
        key = cam_name.lower()
        data_path = cam_data.get("data_path", "")
        cam_intrinsic = cam_data.get("cam_intrinsic")
        sensor2lidar_rot = cam_data.get("sensor2lidar_rotation")
        sensor2lidar_trans = cam_data.get("sensor2lidar_translation")
        sensor2ego_rot = cam_data.get("sensor2ego_rotation")
        sensor2ego_trans = cam_data.get("sensor2ego_translation")

        if sensor2ego_rot is None or sensor2ego_trans is None:
            sensor2ego_rot, sensor2ego_trans = _compute_sensor2ego(
                sensor2lidar_rot,
                sensor2lidar_trans,
                lidar_rot,
                lidar_trans,
            )

        if sensor2lidar_rot is not None:
            sensor2lidar_rot = _to_rotation_matrix(sensor2lidar_rot).astype(np.float32)
        if sensor2lidar_trans is not None:
            sensor2lidar_trans = np.asarray(sensor2lidar_trans, dtype=np.float32)
        if cam_intrinsic is not None:
            cam_intrinsic = np.asarray(cam_intrinsic, dtype=np.float32)

        cams_info[key] = {
            "data_path": str(data_path),
            "cam_intrinsic": cam_intrinsic,
            "sensor2lidar_rotation": sensor2lidar_rot,
            "sensor2lidar_translation": sensor2lidar_trans,
            "sensor2ego_rotation": sensor2ego_rot,
            "sensor2ego_translation": sensor2ego_trans,
        }

    return cams_info


def _build_sweeps(
    raw_frames: List[Dict[str, Any]],
    current_idx: int,
    max_sweeps: int,
) -> List[Dict[str, Any]]:
    """Build LiDAR sweeps list similar to NuScenes info files."""
    sweeps: List[Dict[str, Any]] = []
    if current_idx >= len(raw_frames):
        return sweeps

    current_frame = raw_frames[current_idx]
    curr_lidar_rot = _to_rotation_matrix(current_frame["lidar2ego_rotation"])
    curr_lidar_trans = np.asarray(current_frame["lidar2ego_translation"], dtype=np.float64)
    curr_ego_rot = _to_rotation_matrix(current_frame["ego2global_rotation"])
    curr_ego_trans = np.asarray(current_frame["ego2global_translation"], dtype=np.float64)

    T_curr_lidar2ego = _make_transform(curr_lidar_rot, curr_lidar_trans)
    T_curr_ego2global = _make_transform(curr_ego_rot, curr_ego_trans)
    T_global2curr_ego = np.linalg.inv(T_curr_ego2global)
    T_curr_ego2curr_lidar = np.linalg.inv(T_curr_lidar2ego)

    prev_idx = current_idx - 1
    while prev_idx >= 0 and len(sweeps) < max_sweeps:
        prev_frame = raw_frames[prev_idx]
        prev_lidar_path = prev_frame.get("lidar_path")
        if prev_lidar_path is None:
            prev_idx -= 1
            continue

        prev_lidar_rot = _to_rotation_matrix(prev_frame["lidar2ego_rotation"])
        prev_lidar_trans = np.asarray(prev_frame["lidar2ego_translation"], dtype=np.float64)
        prev_ego_rot = _to_rotation_matrix(prev_frame["ego2global_rotation"])
        prev_ego_trans = np.asarray(prev_frame["ego2global_translation"], dtype=np.float64)

        T_prev_lidar2ego = _make_transform(prev_lidar_rot, prev_lidar_trans)
        T_prev_ego2global = _make_transform(prev_ego_rot, prev_ego_trans)
        T_prev_lidar2global = T_prev_ego2global @ T_prev_lidar2ego

        T_prev_lidar2curr = T_curr_ego2curr_lidar @ T_global2curr_ego @ T_prev_lidar2global
        sweeps.append(
            {
                "data_path": str(prev_lidar_path),
                "type": "lidar",
                "sensor2ego_translation": prev_frame["lidar2ego_translation"],
                "sensor2ego_rotation": prev_frame["lidar2ego_rotation"],
                "ego2global_translation": prev_frame["ego2global_translation"],
                "ego2global_rotation": prev_frame["ego2global_rotation"],
                "timestamp": prev_frame["timestamp"],
                "sensor2lidar_rotation": T_prev_lidar2curr[:3, :3].tolist(),
                "sensor2lidar_translation": T_prev_lidar2curr[:3, 3].astype(float).tolist(),
            }
        )
        prev_idx -= 1

    return sweeps


def _tensor_to_numpy(value: Any) -> Any:
    """Convert torch tensors to numpy arrays recursively."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _tensor_to_numpy(v) for k, v in value.items()}
    return value


def _scene_to_info(
    scene_loader: SceneLoader,
    token: str,
    target_builder: AbstractTargetBuilder,
    max_sweeps: int,
) -> Dict[str, Any]:
    """Convert a NavSim scene token into an info dictionary."""
    scene = scene_loader.get_scene_from_token(token)
    raw_frames = scene_loader.scene_frames_dicts[token]
    current_idx = scene.scene_metadata.num_history_frames - 1
    raw_frame = raw_frames[current_idx]

    targets = target_builder.compute_targets(scene)
    targets_np = _tensor_to_numpy(targets)

    info: Dict[str, Any] = {}
    info["token"] = raw_frame["token"]
    info["scene_token"] = scene.scene_metadata.scene_token

    lidar_path = raw_frame.get("lidar_path")
    info["lidar_path"] = str(lidar_path) if lidar_path is not None else ""

    info["prev"] = raw_frame.get("sample_prev", "") or raw_frame.get("prev", "")
    info["next"] = raw_frame.get("sample_next", "") or raw_frame.get("next", "")
    info["frame_idx"] = raw_frame.get("frame_idx", current_idx)
    info["timestamp"] = raw_frame["timestamp"]
    info["map_location"] = raw_frame.get("map_location", scene.scene_metadata.map_name)

    # Sensor calibration
    info["lidar2ego_translation"] = raw_frame["lidar2ego_translation"]
    info["lidar2ego_rotation"] = raw_frame["lidar2ego_rotation"]
    info["ego2global_translation"] = raw_frame["ego2global_translation"]
    info["ego2global_rotation"] = raw_frame["ego2global_rotation"]

    # Camera metadata
    info["cams"] = _build_camera_entries(
        raw_frame,
        raw_frame["lidar2ego_rotation"],
        raw_frame["lidar2ego_translation"],
    )

    # LiDAR sweeps
    info["sweeps"] = _build_sweeps(raw_frames, current_idx, max_sweeps)

    # CAN bus data (prefer raw if available)
    if "can_bus" in raw_frame and raw_frame["can_bus"] is not None:
        info["can_bus"] = np.asarray(raw_frame["can_bus"], dtype=np.float32)
    else:
        info["can_bus"] = targets_np["can_bus"]

    # Ground-truth fields
    info["fut_valid_flag"] = bool(targets_np["fut_valid_flag"])
    info["gt_boxes"] = targets_np["gt_boxes"]
    info["gt_names"] = targets["gt_names"]
    info["gt_velocity"] = targets_np["gt_velocity"]
    info["num_lidar_pts"] = targets_np["num_lidar_pts"]
    info["num_radar_pts"] = targets_np["num_radar_pts"]
    info["valid_flag"] = targets_np["valid_flag"]
    info["gt_agent_fut_trajs"] = targets_np["gt_agent_fut_trajs"]
    info["gt_agent_fut_masks"] = targets_np["gt_agent_fut_masks"]
    info["gt_agent_lcf_feat"] = targets_np["gt_agent_lcf_feat"]
    info["gt_agent_fut_yaw"] = targets_np["gt_agent_fut_yaw"]
    info["gt_agent_fut_goal"] = targets_np["gt_agent_fut_goal"]
    info["gt_ego_his_trajs"] = targets_np["gt_ego_his_trajs"]
    info["gt_ego_fut_trajs"] = targets_np["gt_ego_fut_trajs"]
    info["gt_ego_fut_masks"] = targets_np["gt_ego_fut_masks"]
    info["gt_ego_fut_cmd"] = targets_np["gt_ego_fut_cmd"]
    info["gt_ego_lcf_feat"] = targets_np["gt_ego_lcf_feat"]

    return info


def _generate_infos(
    scene_loader: SceneLoader,
    tokens: List[str],
    target_builder: AbstractTargetBuilder,
    max_sweeps: int,
) -> List[Dict[str, Any]]:
    """Generate info dictionaries for a provided token list."""
    infos: List[Dict[str, Any]] = []
    for token in tqdm(tokens, desc="Generating NavSim infos"):
        infos.append(_scene_to_info(scene_loader, token, target_builder, max_sweeps))
    return infos


def _prepare_scene_filter(base_filter: SceneFilter, desired_logs: List[str]) -> SceneFilter:
    """Clone and specialise a scene filter for specific log names."""
    scene_filter = deepcopy(base_filter)
    if scene_filter.log_names is not None:
        scene_filter.log_names = [log for log in scene_filter.log_names if log in desired_logs]
    else:
        scene_filter.log_names = desired_logs
    return scene_filter


def create_navsim_infos(
    data_root: Path,
    sensor_root: Path,
    output_dir: Path,
    split: str,
    max_sweeps: int,
    fut_horizon: float,
    interval: float,
    mode: str,
    frame_interval: Optional[int],
    train_start: int = 0,
    train_count: Optional[int] = None,
    val_start: int = 0,
    val_count: Optional[int] = None,
) -> None:
    """Main routine producing train/val info files for NavSim VAD."""
    GlobalHydra.instance().clear()
    with initialize_config_module(
        config_module="navsim.planning.script.config.training",
        version_base=None,
    ):
        cfg = compose(
            config_name="default_training",
            overrides=[f"train_test_split={split}"],
        )

    train_filter = _prepare_scene_filter(instantiate(cfg.train_test_split.scene_filter), cfg.train_logs)
    val_filter = _prepare_scene_filter(instantiate(cfg.train_test_split.scene_filter), cfg.val_logs)

    if frame_interval is not None:
        if frame_interval <= 0:
            raise ValueError("frame_interval must be positive.")
        train_filter.frame_interval = frame_interval
        val_filter.frame_interval = frame_interval
    elif mode == "openscene":
        train_filter.frame_interval = 1
        val_filter.frame_interval = 1

    sensor_config = _build_sensor_config(train_filter.num_history_frames)

    def build_loader(scene_filter: SceneFilter) -> SceneLoader:
        return SceneLoader(
            data_path=data_root,
            original_sensor_path=sensor_root,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

    train_loader = build_loader(train_filter)
    val_loader = build_loader(val_filter)

    config = VADConfig()
    trajectory_sampling = TrajectorySampling(time_horizon=fut_horizon, interval_length=interval)
    target_builder = VADTargetBuilder(trajectory_sampling=trajectory_sampling, config=config)

    def _select_tokens(
        loader: SceneLoader,
        start: int,
        count: Optional[int],
    ) -> Tuple[List[str], List[str]]:
        if start < 0:
            raise ValueError("start index must be non-negative.")

        if mode == "openscene":
            tokens_per_log = loader.get_tokens_list_per_log()
            log_names = sorted(tokens_per_log.keys())
            if start >= len(log_names):
                return [], []
            selected_logs = log_names[start:]
            if count is not None:
                selected_logs = selected_logs[: max(count, 0)]
            selected_tokens: List[str] = []
            for log_name in selected_logs:
                selected_tokens.extend(tokens_per_log[log_name])
            return selected_tokens, selected_logs

        tokens = loader.tokens
        if start >= len(tokens):
            return [], []
        subset = tokens[start:]
        if count is not None:
            subset = subset[: max(count, 0)]
        return subset, []

    train_tokens, train_logs = _select_tokens(train_loader, train_start, train_count)
    val_tokens, val_logs = _select_tokens(val_loader, val_start, val_count)

    train_infos = _generate_infos(train_loader, train_tokens, target_builder, max_sweeps)
    val_infos = _generate_infos(val_loader, val_tokens, target_builder, max_sweeps)

    effective_interval = train_filter.frame_interval

    metadata = {
        "version": split,
        "mode": mode,
        "frame_interval": effective_interval,
        "subset": {
            "train": {
                "start": train_start,
                "requested_count": train_count,
                "actual_token_count": len(train_infos),
                "actual_log_count": len(train_logs) if train_logs else None,
                "log_names": train_logs if train_logs else None,
            },
            "val": {
                "start": val_start,
                "requested_count": val_count,
                "actual_token_count": len(val_infos),
                "actual_log_count": len(val_logs) if val_logs else None,
                "log_names": val_logs if val_logs else None,
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = {"infos": train_infos, "metadata": metadata}
    val_data = {"infos": val_infos, "metadata": metadata}

    def _build_filename(
        prefix: str,
        default_name: str,
        subset_start: int,
        subset_count: Optional[int],
        actual_tokens: int,
        actual_logs: Optional[int],
    ) -> Path:
        subset_applied = subset_start > 0 or subset_count is not None
        if subset_applied:
            count_value = actual_logs if (actual_logs is not None) else actual_tokens
            return output_dir / f"{prefix}_start{subset_start}_count{count_value}.pkl"
        return output_dir / default_name

    train_path = _build_filename(
        prefix="vad_navsim_infos_temporal_train",
        default_name="vad_navsim_infos_temporal_train.pkl",
        subset_start=train_start,
        subset_count=train_count,
        actual_tokens=len(train_infos),
        actual_logs=len(train_logs) if train_logs else None,
    )
    val_path = _build_filename(
        prefix="vad_navsim_infos_temporal_val",
        default_name="vad_navsim_infos_temporal_val.pkl",
        subset_start=val_start,
        subset_count=val_count,
        actual_tokens=len(val_infos),
        actual_logs=len(val_logs) if val_logs else None,
    )

    mmcv.dump(train_data, str(train_path))
    mmcv.dump(val_data, str(val_path))

    train_log_msg = f" ({len(train_logs)} logs)" if train_logs else ""
    val_log_msg = f" ({len(val_logs)} logs)" if val_logs else ""
    print(f"Saved {len(train_infos)} training samples{train_log_msg} to {train_path}")
    print(f"Saved {len(val_infos)} validation samples{val_log_msg} to {val_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NavSim logs into VAD info files.")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to NavSim log directory.")
    parser.add_argument("--sensor-root", type=Path, required=True, help="Path to NavSim sensor directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store generated info files.")
    parser.add_argument("--split", type=str, default="navtrain", help="Training split defined in Hydra config.")
    parser.add_argument("--max-sweeps", type=int, default=10, help="Maximum number of historical LiDAR sweeps.")
    parser.add_argument(
        "--fut-horizon",
        type=float,
        default=3.0,
        help="Future trajectory horizon in seconds (must align with model training).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval between consecutive frames in seconds.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="navsim",
        choices=["navsim", "openscene"],
        help="Generation mode. 'openscene' enables sliding-window sampling over entire logs.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        help="Custom frame interval for sliding window (overrides default in selected mode).",
    )
    parser.add_argument("--train-start", type=int, default=0, help="Start index within the training tokens.")
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Maximum number of training tokens to convert (subset).",
    )
    parser.add_argument("--val-start", type=int, default=0, help="Start index within the validation tokens.")
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Maximum number of validation tokens to convert (subset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Environment defaults (can be overridden by user before invocation)
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(args.data_root / "maps"))
    os.environ.setdefault("NAVSIM_EXP_ROOT", str(args.data_root))
    os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(Path(__file__).resolve().parents[2]))
    os.environ.setdefault("OPENSCENE_DATA_ROOT", str(args.data_root))

    create_navsim_infos(
        data_root=args.data_root,
        sensor_root=args.sensor_root,
        output_dir=args.output_dir,
        split=args.split,
        max_sweeps=args.max_sweeps,
        fut_horizon=args.fut_horizon,
        interval=args.interval,
        mode=args.mode,
        frame_interval=args.frame_interval,
        train_start=args.train_start,
        train_count=args.train_count,
        val_start=args.val_start,
        val_count=args.val_count,
    )


if __name__ == "__main__":
    main()
