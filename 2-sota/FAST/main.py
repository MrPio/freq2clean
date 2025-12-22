# Copyright (C) [2025] [Yiqun Wang]
# SPDX-License-Identifier: GPL-3.0-or-later


from train import goTraining
from test import goTesting
from utils.config import json2args
import os
import torch
import argparse


def print_mode(mode):
    print(f"Running in {mode} mode")


def validate_path(path):
    if not path or not os.path.exists(path):
        raise ValueError(f"Invalid path: {path}")


def main(config_path="./params.json", test_path=None):
    try:
        print(f"Load configuration file path: {config_path}")
        args = json2args(config_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        assert torch.cuda.is_available(), "Currently, we only support CUDA version"
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        if args.mode == "train":
            print_mode("Training")
            goTraining(args)
        elif args.mode == "test":
            args.test_path = test_path
            validate_path(args.test_path)
            print_mode("Testing")
            goTesting(args)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', '--config_path',
                        dest='config_path',
                        type=str,
                        default='./params.json')
    parser.add_argument('--test-path', '--test_path',
                        dest='test_path',
                        type=str,
                        default=None)
    cmd_args = parser.parse_args()

    main(config_path=cmd_args.config_path, test_path=cmd_args.test_path)
