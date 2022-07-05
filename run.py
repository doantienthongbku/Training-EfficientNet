from load_config import *
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training EfficinetNet')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="dataset",
                        help='Directory of the DLModel dataset')
    parser.add_argument('--config', default='config/training_config.yaml',
                        help='Path to config yaml file')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    config_path = args.config