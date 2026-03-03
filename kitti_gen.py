import argparse
from tracker.config import cfg_from_yaml_file, cfg
from dataset.kitti_dataset import build_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="config/kitti_train.yaml", help='specify the config for tracking')
    args = parser.parse_args()
    config = cfg_from_yaml_file(args.cfg_file, cfg)

    build_dataset(cfg, mode='train')
    build_dataset(cfg, mode='val')