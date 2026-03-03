from pathlib import Path
import torch
import numpy as np
from PIL import Image
import argparse
from collections import defaultdict
import torchvision.transforms as transforms
from random import randint
from torch.utils.data import Dataset
from tqdm import tqdm
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from dataset.kitti_dataset_base import load_tracking_label
from tracker.utils import save_pickle, load_pickle
from tracker.config import cfg_from_yaml_file, cfg
from dataset.kitti_dataset_base import Calibration


def build_dataset(cfg, mode, num_points=512, crop_size=(80, 80)):

    assert mode in ('train', 'val'), "mode must be 'train' or 'val'"

    root = Path(cfg.dataset_path) / 'training'
    out_base = root / 'rebuild_dataset'
    out_base.mkdir(parents=True, exist_ok=True)
    cls_map_r = {v: k for k, v in cfg.class_map.items()}

    # Prepare image transform
    img_transform = transforms.Compose([
        transforms.Resize(crop_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    samples = defaultdict(list)
    label_dir = root / 'label_02'
    image_dir = root / 'image_02'
    velodyne_dir = root / 'velodyne'
    calib_dir = root / 'calib'

    for seq in cfg.tracking_seqs[mode]:
        seq = f"{int(seq):04d}"
        seq_out = out_base / seq
        seq_out.mkdir(exist_ok=True)

        labels = load_tracking_label(label_dir / f"{seq}.txt", cfg.class_map)
        frames = sorted(np.unique(labels[:, 0].astype(int)))
        calib = Calibration(calib_dir / f"{seq}.txt")

        for frame in tqdm(frames, desc=f"Processing sequence: {seq}"):
            # Paths
            img_path = image_dir / seq / f"{frame:06d}.png"
            pc_path = velodyne_dir / seq / f"{frame:06d}.bin"
            if not (img_path.exists() and pc_path.exists()):
                continue

            # Load data
            img = Image.open(img_path)
            pc_all = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

            # Filter labels by frame
            for obj in labels[labels[:, 0] == frame]:
                tid, cls = map(int, obj[1:3])
                # Compute 3D box in camera coords
                box3d_cam = obj[10:17][[3, 4, 5, 2, 0, 1, 6]][None]
                box2d_img = (obj[6: 10]).astype(int)

                # Skip invalid crops
                x1, y1, x2, y2 = box2d_img
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop and preprocess image
                img_crop = img_transform(img.crop((x1, y1, x2, y2)))

                # Convert box to LiDAR coords
                loc_lidar = calib.rect_to_lidar(box3d_cam[:, :3])
                l, h, w, rot = box3d_cam[:, 3:4], box3d_cam[:, 4:5], box3d_cam[:, 5:6], box3d_cam[:, 6:7]
                loc_lidar[:, 2] += (h[:, 0] / 2)
                box3d_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rot)], axis=1)

                # Crop point cloud
                pc_coords = torch.from_numpy(pc_all[:, :3]).float().cuda()[None]
                box_t = torch.from_numpy(box3d_lidar).float().cuda()[None]
                mask = roiaware_pool3d_utils.points_in_boxes_gpu(pc_coords, box_t).cpu().numpy()[0] == 0
                pc_crop = pc_all[mask]
                if pc_crop.size == 0:
                    continue

                # Sample fixed number of points
                idx = np.random.choice(len(pc_crop), num_points, replace=True)
                pc_crop = pc_crop[idx]

                # Generate random points inside the oriented 3D box
                center = box3d_lidar[0, :3]
                dx, dy, dz, yaw = box3d_lidar[0, 3], box3d_lidar[0, 4], box3d_lidar[0, 5], box3d_lidar[0, 6]
                # local offsets
                local = np.stack([
                    np.random.uniform(-dx/2, dx/2, num_points),
                    np.random.uniform(-dy/2, dy/2, num_points),
                    np.random.uniform(-dz/2, dz/2, num_points)
                ], axis=1)
                # rotation matrix around z-axis
                c, s = np.cos(yaw), np.sin(yaw)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                pc_geo_crop = local.dot(R.T) + center
                # add dummy reflectance
                pc_geo_crop = np.hstack([pc_geo_crop, np.zeros((num_points, 1))]).astype(np.float32)

                # with open('points.bin', 'wb') as f:
                #     pc_crop[:, :3].tofile(f)
                # print("done")

                # Crop image planer geometry
                img_geo_crop = np.column_stack((np.random.uniform(x1, x2, 512),
                                                np.random.uniform(y1, y2, 512))) / np.array(img.size)

                # Save sample
                filename = f"{frame:06d}_{tid}.pkl"
                save_path = seq_out / filename
                save_pickle({
                    'seq': seq,
                    'frame': frame,
                    'tid': tid,
                    'class_name': cls_map_r[cls],
                    'img': img_crop,
                    'pts': pc_crop,
                    'img_geo': img_geo_crop,
                    'pts_geo': pc_geo_crop,
                    'box2d_img': box2d_img,
                    'box3d_lidar': box3d_lidar},
                    save_path)
                samples[f"{seq}_{frame:06d}"].append(str(save_path))

    # Save sample index
    save_pickle(samples, root / f"{mode}.pkl")
    print(f"Dataset {mode} built with {len(samples)} samples.")


class KittiDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        assert mode in ('train', 'val'), "mode must be 'train' or 'val'"
        self.root = Path(cfg.dataset_path) / 'training'
        self.mode = mode
        self.SEQ = cfg.tracking_seqs
        self.class_map = cfg.class_map

        # Lists of (last_filename, cur_filename)
        self.pairs = []
        self.positive_pairs = defaultdict(list)
        self.negative_pairs = defaultdict(list)
        self._build_pairs()

        self.flag = False

    def _build_pairs(self):
        # Load index mapping from prebuilt pickle
        sample_info = load_pickle(self.root / f'{self.mode}.pkl')

        seq_list = self.SEQ[self.mode]
        # Iterate with progress bar
        for seq in seq_list:
            seq = f"{int(seq):04d}"
            labels = load_tracking_label(self.root / 'label_02' / f"{seq}.txt", self.class_map)
            frames = sorted(np.unique(labels[:, 0].astype(int)))

            for frame in tqdm(frames, desc=f"[{self.mode} set] Building pairs for Seq {seq}", leave=True):
                frame_interval = np.random.randint(1, 2)
                cur_frame  = f"{seq}_{frame:06d}"
                next_frame = f"{seq}_{frame+frame_interval:06d}"
                cur_files  = sample_info.get(cur_frame, [])
                next_files = sample_info.get(next_frame, [])
                if not cur_files or not next_files:
                    continue

                for cur_path in cur_files:
                    cur_data = load_pickle(cur_path)
                    for next_path in next_files:
                        next_data  = load_pickle(next_path)
                        if cur_data['tid'] == next_data['tid']:
                            self.positive_pairs[cur_data['class_name']].append((cur_data, next_data))
                        else:
                            self.negative_pairs[cur_data['class_name']].append((cur_data, next_data))

        for k, v in self.class_map.items():
            print(f"[{self.mode} set] [{k}] Total positive pairs: {len(self.positive_pairs[k])}")
            print(f"[{self.mode} set] [{k}] Total negative pairs: {len(self.negative_pairs[k])}")

        # Flatten the lists for easier access
        self.positive_pairs = [pair for pairs in self.positive_pairs.values() for pair in pairs]
        self.negative_pairs = [pair for pairs in self.negative_pairs.values() for pair in pairs]
        self.pairs = self.positive_pairs + self.negative_pairs

        # Shuffle negative to balance? optional
        np.random.shuffle(self.pairs)

    @staticmethod
    def _load_pair(pair):
        sample1 = pair[0]
        sample2 = pair[1]
        img1, img2 = sample1['img'], sample2['img']
        pts1, pts2 = sample1['pts'], sample2['pts']
        img_geo1, img_geo2 = sample1['img_geo'], sample2['img_geo']
        is_positive_pair = int(sample1['tid'] == sample2['tid'])
        return (img1, img2), (pts1, pts2), (img_geo1, img_geo2), is_positive_pair

    def __getitem__(self, item):
        return self._load_pair(self.pairs[item])

    def __len__(self):
        return len(self.pairs)

