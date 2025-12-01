#!/usr/bin/env python

import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction

from mmdet3d.datasets import build_dataset
from mmdet.apis import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run evaluation from a saved format_state.pkl')
    parser.add_argument('format_state', help='Path to format_state.pkl')
    parser.add_argument(
        '--config',
        default=None,
        help='Optional config path (overrides path stored in format_state)')
    parser.add_argument(
        '--out',
        default=None,
        help='Where to save the evaluation bundle (defaults to '
             '<jsonfile_prefix>/bbox_results_eval.pkl)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings, same format as test.py')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.format_state):
        raise FileNotFoundError(args.format_state)
    state = mmcv.load(args.format_state)
    cfg_path = args.config or state.get('config')
    if cfg_path is None:
        raise RuntimeError('format_state missing config path; please specify --config')
    cfg = Config.fromfile(cfg_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)
        else:
            _module_dir = os.path.dirname(cfg_path)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            importlib.import_module(_module_path)
    # ensure deterministic behavior similar to test.py
    if args.cfg_options is None and cfg.get('seed') is not None:
        set_random_seed(cfg.seed, deterministic=False)

    test_cfg = cfg.data.test
    for key in ['samples_per_gpu', 'workers_per_gpu', 'prefetch_factor',
                'shuffler_sampler', 'nonshuffler_sampler']:
        test_cfg.pop(key, None)
    dataset = build_dataset(test_cfg)
    bbox_results = state['bbox_results']
    eval_kwargs = state['eval_kwargs']
    evaluation_results = dataset.evaluate(bbox_results, **eval_kwargs)
    summary_paths = getattr(dataset, 'latest_summary_paths', [])
    formatted_results = state.get('formatted_results', {})
    jsonfile_prefix = state.get('jsonfile_prefix', 'test/eval_resume')

    bundle = dict(
        bbox_results=bbox_results,
        formatted_results=formatted_results,
        evaluation_metrics=evaluation_results,
        summary_paths=summary_paths,
        jsonfile_prefix=jsonfile_prefix,
    )
    bundle_path = args.out or os.path.join(jsonfile_prefix, 'bbox_results_eval.pkl')
    mmcv.mkdir_or_exist(os.path.dirname(bundle_path))
    print(f'\nSaving resumed-eval bundle to {bundle_path}')
    mmcv.dump(bundle, bundle_path)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('fork')
    main()
