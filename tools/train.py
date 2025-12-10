#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Training script for MapTR with MMEngine.

This script demonstrates how to train MapTR using the new MMEngine runner.

Example:
    Single GPU:
        python tools/train.py configs/maptr_av2_example.py
    
    Multiple GPUs:
        bash tools/dist_train.sh configs/maptr_av2_example.py 8
"""

import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
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

    # Register all modules in mmdet3d into the registries
    register_all_modules(init_default_scope=False)
    
    # Register MapTR plugin modules
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
        # Use config filename as default work_dir if not specified
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported for semi-supervised detection'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'

    # Resume training
    cfg.resume = args.resume

    # Build the runner from config
    if 'runner_type' not in cfg:
        # Build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # Build customized runner from the registry
        # If 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    main()
