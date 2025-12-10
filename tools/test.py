#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Testing script for MapTR with MMEngine.

Example:
    Single GPU:
        python tools/test.py configs/maptr_av2_example.py checkpoints/latest.pth
    
    Multiple GPUs:
        bash tools/dist_test.sh configs/maptr_av2_example.py checkpoints/latest.pth 8
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Test a 3D detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Register all modules
    register_all_modules(init_default_scope=False)
    
    # Register MapTR plugin
    import projects.mmdet3d_plugin  # noqa: F401

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # Show prediction results
    if args.show or args.show_dir:
        cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.show = args.show
        if args.show_dir:
            cfg.default_hooks.visualization.vis_backends = [
                dict(type='LocalVisBackend', save_dir=args.show_dir)
            ]

    # Build runner
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    # Start testing
    runner.test()


if __name__ == '__main__':
    main()
