from typing import Any, Dict

import os
import pickle
from pathlib import Path

data_dir = "/home/work/datasets/navsim_full/navsim/dataset"
os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = f"{data_dir}/maps"
os.environ["NAVSIM_EXP_ROOT"] = "/home/work/song99/VAD/navsim/exp"
os.environ["NAVSIM_DEVKIT_ROOT"] = "/home/work/song99/VAD/navsim/"
os.environ["OPENSCENE_DATA_ROOT"] = f"{data_dir}"



from hydra import initialize_config_module, compose
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from navsim.agents.vad.vad_config import VADConfig
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from vad_feature_builder import VADFeatureBuilder
from vad_target_builder import VADTargetBuilder


def safe_collate(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that safely handles strings in dicts.

    Args:
        batch (list[Dict[str, Any]]): List of samples (usually dicts) from the dataset.

    Returns:
        Dict[str, Any]: Collated batch where string values are left unchanged.
    """
    # Handle dicts with string values safely
    if isinstance(batch[0], dict):
        result = {}
        for key in batch[0]:
            values = [d[key] for d in batch]
            # If all values are strings, skip default_collate
            if all(isinstance(v, str) for v in values):
                result[key] = values
            else:
                result[key] = default_collate(values)
        return result

    # Fallback to default collate for non-dict batches
    return default_collate(batch)



if __name__ == "__main__":
    split = "navtrain"

    # Compose the training config via Hydra
    GlobalHydra.instance().clear()
    with initialize_config_module(
        config_module="navsim.planning.script.config.training",
        version_base=None,
    ):
        cfg = compose(
            config_name="default_training",
            overrides=[
                "experiment_name=jupyter_debug",
                f"train_test_split={split}",
                "cache_path=/home/work/song99/VAD/cache",
                "agent=transfuser_agent",
                # "train_test_split.scene_filter.max_scenes=4",
            ],
        )

    # Build agent, datasets, and dataloaders
    agent = instantiate(cfg.agent)

    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [
            log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs
        ]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    history_steps = [0, 1, 2, 3]
    sensor_config = SensorConfig(
        cam_f0=history_steps,
        cam_l0=history_steps,
        cam_l1=history_steps,
        cam_l2=history_steps,
        cam_r0=history_steps,
        cam_r1=history_steps,
        cam_r2=history_steps,
        cam_b0=history_steps,
        lidar_pc=history_steps,
    )

    # train_scene_loader = SceneLoader(
    #     original_sensor_path=original_sensor_path,
    #     data_path=data_path,
    #     scene_filter=train_scene_filter,
    #     sensor_config=agent.get_sensor_config(),
    # )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=sensor_config,
    )

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    config = VADConfig()
    feature_builder = VADFeatureBuilder(config)
    target_builder = VADTargetBuilder(trajectory_sampling, config)

    # train_data = Dataset(
    #     scene_loader=train_scene_loader,
    #     feature_builders=agent.get_feature_builders(),
    #     target_builders=agent.get_target_builders(),
    #     cache_path=cfg.cache_path,
    #     force_cache_computation=cfg.force_cache_computation,
    # )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=[feature_builder],
        target_builders=[target_builder],
        # cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    # train_loader = DataLoader(train_data, shuffle=True, **cfg.dataloader.params)
    val_loader = DataLoader(val_data, shuffle=False, collate_fn=safe_collate, **cfg.dataloader.params)