# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
from data_converter.create_gt_database import create_groundtruth_database
from data_converter import nuscenes_converter as nuscenes_converter
from data_converter import lyft_converter as lyft_converter
from data_converter import kitti_converter as kitti
from data_converter import indoor_converter as indoor
import argparse
import os
from os import path as osp
import sys

import mmcv
sys.path.append('.')


def kitti_data_prep(root_path, info_prefix, version, out_dir):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
    """
    kitti.create_kitti_info_file(root_path, info_prefix)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(root_path,
                                  f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)

    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version)
        # create_groundtruth_database(dataset_name, root_path, info_prefix,
        #                             f'{out_dir}/{info_prefix}_infos_train.pkl')


def _default_map_ann_filename(version):
    return 'nuscenes_mini_map_anns_val.json' if 'mini' in version else 'nuscenes_map_anns_val.json'


def _append_scene_suffix(path, scene_limit):
    base, ext = osp.splitext(path)
    return f'{base}_scenes{scene_limit}{ext}'


def _subsample_infos_by_scene(info_data, scene_limit):
    allowed_scenes = []
    allowed_set = set()
    filtered_infos = []
    for info in info_data.get('infos', []):
        scene_token = info.get('scene_token')
        if scene_token is None:
            filtered_infos.append(info)
            continue
        if scene_token in allowed_set:
            filtered_infos.append(info)
            continue
        if len(allowed_scenes) < scene_limit:
            allowed_scenes.append(scene_token)
            allowed_set.add(scene_token)
            filtered_infos.append(info)
        else:
            continue
    new_data = dict(info_data)
    new_data['infos'] = filtered_infos
    metadata = dict(new_data.get('metadata', {}))
    metadata['scene_limit'] = scene_limit
    metadata['scene_tokens'] = allowed_scenes
    new_data['metadata'] = metadata
    return new_data, len(allowed_scenes)


def maybe_apply_scene_limit(info_path, scene_limit):
    if scene_limit <= 0:
        return None
    if not osp.exists(info_path):
        print(f'[SceneLimit] {info_path} not found, skip scene sampling.')
        return None
    data = mmcv.load(info_path)
    limited_data, kept = _subsample_infos_by_scene(data, scene_limit)
    if kept == 0:
        print(f'[SceneLimit] No scenes retained from {info_path}.')
        return None
    limited_path = _append_scene_suffix(info_path, kept)
    mmcv.dump(limited_data, limited_path)
    print(f'[SceneLimit] Saved {kept} scenes to {limited_path}')
    return limited_path


def maybe_generate_vad_map_annotations(root_path,
                                       out_dir,
                                       info_prefix,
                                       dataset_version,
                                       map_ann_filename,
                                       generate=False,
                                       overwrite=False):
    if not generate:
        return

    val_info_path = osp.join(out_dir, f'{info_prefix}_infos_temporal_val.pkl')
    if not osp.exists(val_info_path):
        print(f'[MapAnn] Skip generation because {val_info_path} does not exist.')
        return

    map_ann_path = osp.join(root_path, map_ann_filename)
    if osp.exists(map_ann_path) and not overwrite:
        print(f'[MapAnn] {map_ann_path} exists. Use --overwrite-map-anns to regenerate.')
        return

    os.makedirs(osp.dirname(map_ann_path), exist_ok=True)
    print(f'[MapAnn] Generating {map_ann_path} from {val_info_path}')
    from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import VADCustomNuScenesDataset

    vad_dataset = VADCustomNuScenesDataset(
        ann_file=val_info_path,
        data_root=root_path,
        pipeline=None,
        classes=None,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        box_type_3d='LiDAR',
        test_mode=True,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        map_ann_file=map_ann_path,
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        version=dataset_version)
    vad_dataset._format_gt()
    print(f'[MapAnn] Saved to {map_ann_path}')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    from tools.data_converter import waymo_converter as waymo

    splits = ['training', 'validation', 'testing']

    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'test'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)

    create_groundtruth_database(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required='False',
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--generate-map-anns',
    action='store_true',
    help='Generate VAD vector map annotations for nuScenes val split.')
parser.add_argument(
    '--map-ann-filename',
    type=str,
    default='',
    help='Relative filename (under root-path) for generated nuScenes map annotations.')
parser.add_argument(
    '--overwrite-map-anns',
    action='store_true',
    help='Overwrite existing map annotations when generating.')
parser.add_argument(
    '--train-scene-limit',
    type=int,
    default=0,
    help='If >0, limit nuScenes training infos to the first N scenes. '
         'Validation/test remain untouched. The limited file will be saved '
         'with _scenes{N} suffix.')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        train_info_path = osp.join(args.out_dir, f'{args.extra_tag}_infos_temporal_train.pkl')
        limited_path = maybe_apply_scene_limit(train_info_path, args.train_scene_limit)
        if limited_path:
            print(f'[SceneLimit] Training infos limited file: {limited_path}')
        map_ann_filename = args.map_ann_filename or _default_map_ann_filename(
            train_version)
        maybe_generate_vad_map_annotations(
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.extra_tag,
            dataset_version=train_version,
            map_ann_filename=map_ann_filename,
            generate=args.generate_map_anns,
            overwrite=args.overwrite_map_anns)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        train_info_path = osp.join(args.out_dir, f'{args.extra_tag}_infos_temporal_train.pkl')
        limited_path = maybe_apply_scene_limit(train_info_path, args.train_scene_limit)
        if limited_path:
            print(f'[SceneLimit] Training infos limited file: {limited_path}')
        map_ann_filename = args.map_ann_filename or _default_map_ann_filename(
            train_version)
        maybe_generate_vad_map_annotations(
            root_path=args.root_path,
            out_dir=args.out_dir,
            info_prefix=args.extra_tag,
            dataset_version=train_version,
            map_ann_filename=map_ann_filename,
            generate=args.generate_map_anns,
            overwrite=args.overwrite_map_anns)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
