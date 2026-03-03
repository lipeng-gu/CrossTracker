import os
import torch
import argparse
from datetime import datetime
import time
import copy
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from model.m3_model import Net
from tracker.config import cfg, cfg_from_yaml_file
from tracker.utils import set_random_seed, greedy_matching
from tracker.tracker import Box3D, Box2D, OtherInfo, Tracker3D, Tracker2D
from dataset.kitti_dataset_base import Calibration, Oxts
from tracker.box_op import compute_iou2d_matrix, compute_iou2d
from efficiency_analysis import analyze_system_cpu_memory, analyze_system_gpu_memory


# ---------------------------------------------------------------------------- #
# Detection processing and sorting
# ---------------------------------------------------------------------------- #
class DetectionProcessor:
    def __init__(self, cfg):
        mode = 'testing' if cfg.mode == 'test' else 'training'
        self.class_name = cfg.class_name
        self.path3d = f"{cfg.detections_path}/3D_{cfg.detector3d}_{cfg.class_name[:3]}_{mode[:-3]}"
        self.path2d = f"{cfg.detections_path}/2D_{cfg.detector2d}_{cfg.class_name[:3]}_{mode[:-3]}"
        self.input_score3d = cfg.input_score['3d']
        self.input_score2d = cfg.input_score['2d']

    def process_detections(self, detections, seq_name=None, is_3d=True):
        processed = []
        for det in detections:
            if is_3d:
                f, x, y, z = int(float(det[0])), det[10], det[11], det[12]
                h, w, l, ry = det[7], det[8], det[9], det[13]
                cls, alpha = int(float(det[1])), det[-1]
                xx1, yy1, xx2, yy2, score = det[2], det[3], det[4], det[5], det[6]
                if xx1 >= xx2 or yy1 >= yy2 or score < self.input_score3d: continue
                processed.append([seq_name, f, cls, alpha, xx1, yy1, xx2, yy2, h, w, l, x, y, z, ry, score])
            else:
                f, x1, y1, x2, y2, score = int(float(det[0])), det[1], det[2], det[3], det[4], det[5]
                if x1 >= x2 or y1 >= y2 or score < self.input_score2d: continue
                processed.append([seq_name, f, max(x1, 0), max(y1, 0), x2, y2, score])
        return processed

    def sort_and_structure(self, det_list, cls_map, is_3d=True):
        structured = {}
        for frame, dets in det_list.items():
            if not dets: continue
            arr = np.array(dets)
            idx = np.argsort(arr[:, -1].astype(np.float64))[::-1]
            sorted_dets = arr[idx]
            temp = []
            for det in sorted_dets:
                if is_3d:
                    box3d = Box3D(x=float(det[11]), y=float(det[12]), z=float(det[13]),
                                  h=float(det[8]), w=float(det[9]), l=float(det[10]), ry=float(det[14]))
                    box2d = Box2D(xx1=float(det[4]), yy1=float(det[5]), xx2=float(det[6]), yy2=float(det[7]))
                    info = OtherInfo(seq=det[0], frame=int(det[1]), cls=int(det[2]), alpha=float(det[3]), score=float(det[15]))
                    temp.append([box3d, box2d, info])
                else:
                    box2d = Box2D(xx1=float(det[2]), yy1=float(det[3]), xx2=float(det[4]), yy2=float(det[5]))
                    info = OtherInfo(seq=det[0], frame=int(det[1]), cls=cls_map[self.class_name], score=float(det[6]))
                    temp.append([box2d, info])
            structured[frame] = temp
        return structured

    def build_detections(self, seq_name, cls_map):
        path3d = os.path.join(self.path3d, f'{seq_name}.txt')
        path2d = os.path.join(self.path2d, f'{seq_name}.txt')
        raw3d = np.loadtxt(path3d, delimiter=',').reshape(-1, 15) #if os.path.exists(path3d) else np.empty((0,15))
        raw2d = np.loadtxt(path2d, delimiter=',').reshape(-1, 6) #if os.path.exists(path2d) else np.empty((0,6))

        st = min(int(raw3d[:,0].min()) if raw3d.size else 0, int(raw2d[:,0].min()) if raw2d.size else 0)
        lf = max(int(raw3d[:,0].max()) if raw3d.size else 0, int(raw2d[:,0].max()) if raw2d.size else 0)

        det3d_info = defaultdict(list)
        det2d_info = defaultdict(list)
        for f in range(st, lf+1):
            det3d_temp = raw3d[raw3d[:,0]==f] if raw3d.size else []
            det2d_temp = raw2d[raw2d[:,0]==f] if raw2d.size else []
            det3d_info[f] = self.process_detections(det3d_temp, seq_name, True)
            det2d_info[f] = self.process_detections(det2d_temp, seq_name, False)

        det3d_struct = self.sort_and_structure(det3d_info, cls_map, True)
        det2d_struct = self.sort_and_structure(det2d_info, cls_map, False)
        return det3d_struct, det2d_struct, st, lf

class CrossTracker:
    def __init__(self, cfg, seq_name, model, dataset):
        set_random_seed(cfg.seed)
        self.cfg = cfg
        self.seq_name = f"{int(seq_name):04d}"
        self.cls_map = cfg.class_map
        self.cls_map_r = {v: k for k, v in cfg.class_map.items()}
        self.model = model
        self.mode = 'testing' if self.cfg.mode == 'test' else 'training'
        self.dataset_path = f"{self.cfg.dataset_path}/{self.mode}"
        self.embed_dim = cfg.embed_dim
        self.thr_stage1 = self.cfg.thr_stage1
        self.thr_stage2 = self.cfg.thr_stage2

        self._init_paths()
        self._init_trackers()
        self.oxts = Oxts(self.oxts_path)
        self.dets_3d, self.dets_2d, self.sf, self.lf = dataset.build_detections(
            self.seq_name,
            self.cls_map
        )

    def _init_paths(self):
        save_path = f"{self.cfg.save_path}/fusion_{self.cfg.detector3d}_{self.cfg.class_name}_{self.mode[:-3]}"
        os.makedirs(save_path, exist_ok=True)
        self.save_path = os.path.join(save_path, f"{self.seq_name}.txt")
        self.oxts_path = f"{self.dataset_path}/oxts/{self.seq_name}.txt"

    def _init_trackers(self):
        self.trackers3d = {}
        self.trackers2d = {}
        self.tid3d = 0
        self.tid2d = 10000
        self.output_per_frame = defaultdict(list)

        self.img_w, self.img_h = self.cfg.img_size
        self.img_trans = T.Compose([
            T.Resize((80, 80), interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def load_calibration(self):
        calib_path = f"{self.dataset_path}/calib/{self.seq_name}.txt"
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file {calib_path} does not exist.")
        return Calibration(calib_path)

    def load_image(self, frame, dets):
        if not self.cfg.use_ifm or not dets: return 100 * torch.ones((len(dets), self.embed_dim)).cuda()
        img_path = f"{self.dataset_path}/image_02/{self.seq_name}/{frame:06d}.png"
        img = Image.open(img_path)
        self.img_w, self.img_h = img.size

        crops = []
        for b in dets:
            b = b[-2]  # box2d in [box3d, box2d, other] or [box2d, other]
            coords = (b.xx1, b.yy1, b.xx2, b.yy2)
            crop = self.img_trans(img.crop(coords)).unsqueeze(0).cuda()
            crops.append(crop)
        return self.model.encode_img(torch.cat(crops, 0))

    def load_planer_geometry(self, dets, sample='random'):
        if not self.cfg.use_gfm or not dets: return 100 * torch.ones((len(dets), self.embed_dim)).cuda()
        crops = []
        for b in dets:
            b = b[-2]  # box2d in [box3d, box2d, other] or [box2d, other]

            if sample == 'random':
                # Random sampling of points within the bounding box
                crop = np.column_stack((np.random.uniform(b.xx1, b.xx2, 512),
                                        np.random.uniform(b.yy1, b.yy2, 512))) / (self.img_w, self.img_h)
            elif sample == 'grid':
                # Grid sampling of points within the bounding box
                x = np.linspace(b.xx1, b.xx2, int(np.sqrt(512))) / self.img_w
                y = np.linspace(b.yy1, b.yy2, int(np.sqrt(512))) / self.img_h
                crop = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
            else:
                raise ValueError("Sample method must be 'random' or 'grid'.")

            # 可视化散点
            # import matplotlib.pyplot as plt
            # plt.scatter(crop[:, 0], crop[:, 1], s=10, c='blue', alpha=0.5)
            # plt.title("Scatter Plot of Random Points")
            # plt.xlabel("Normalized X")
            # plt.ylabel("Normalized Y")
            # plt.grid(True)
            # plt.show()

            crop = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).transpose(2, 1).cuda()
            crops.append(crop)
        return self.model.encode_geo(torch.cat(crops, 0))

    def load_points(self, frame, dets, calib):
        if not self.cfg.use_pfm or not dets: return 100 * torch.ones((len(dets), self.embed_dim)).cuda()

        pts_path = f"{self.dataset_path}/velodyne/{self.seq_name}/{frame:06d}.bin"
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4) if os.path.exists(pts_path) else np.zeros((0, 4), dtype=np.float32)
        if pts.shape[0] == 0: return 100 * torch.ones((len(dets), self.embed_dim)).cuda()

        boxes = np.array([
            [b.x, b.y, b.z, b.l, b.h, b.w, b.ry]
            for b, _, _ in dets
        ], dtype=np.float32).reshape(-1, 7)  # (N,7)
        boxes = calib.convert_boxes3d_cam_to_lidar(boxes)
        boxes = torch.tensor(boxes, dtype=torch.float32).unsqueeze(1).cuda()  # (N,1,7)

        pts = torch.tensor(pts, dtype=torch.float32).unsqueeze(0).repeat(len(dets), 1, 1).cuda()  # (N,P,4)
        pts_in_box = roiaware_pool3d_utils.points_in_boxes_gpu(pts[..., :3], boxes)

        crops, has_pts = [], []
        for i in range(len(dets)):
            mask = pts_in_box[i] == 0
            crop = pts[i][mask]
            if crop.shape[0] > 0:
                idx = np.random.choice(crop.shape[0], 512, replace=True)
                crop = crop[idx]
                has_pts.append(True)
            else:
                crop = torch.ones((512, 4), dtype=torch.float32).cuda()
                has_pts.append(False)
            crops.append(crop.transpose(0, 1).unsqueeze(0))  # (1,4,512)
        crops = torch.cat(crops, dim=0)  # (N,4,512)
        crops = torch.cat([crops, F.normalize(crops[:, :3, :], p=2, dim=-1)], dim=1)  # (N,7,512)
        crops = self.model.encode_pts(crops)  # (N, EMBED_DIM)

        mask = torch.tensor(has_pts, device=crops.device)  # (N,)
        crops[~mask] = 100
        return crops

    def split_feats(self, feats_img, feats_geo, feats_pts, dets_3d, dets_2d):
        M = len(dets_3d)
        K = len(dets_2d)
        invalid_feat = torch.full((1, self.embed_dim), 100.0, dtype=torch.float32, device=feats_img.device)

        feat_3d = [
            [feats_img[i:i+1], feats_geo[i:i+1], feats_pts[i:i+1]]
            for i in range(M)
        ]

        feat_2d = [
            [feats_img[M + j:M + j + 1], feats_geo[M + j:M + j + 1], invalid_feat]
            for j in range(K)
        ]

        return feat_3d, feat_2d

    def ego_motion_compensation(self, frame, trks, calib):
        # inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching

        assert len(self.trackers3d) == len(trks)
        ego_xyz_imu, ego_rot_imu, left, right = self.oxts.get_ego_traj(frame, 1, 1, only_fut=True, inverse=True)
        for trk_id in self.trackers3d.keys():
            trk_tmp3d = trks[trk_id][0]
            xyz = np.array([trk_tmp3d.x, trk_tmp3d.y, trk_tmp3d.z]).reshape((1, -1))
            compensated_xyz3d = calib.egomotion_compensation_ID(xyz, ego_rot_imu, ego_xyz_imu, left, right)
            trk_tmp3d.x, trk_tmp3d.y, trk_tmp3d.z = compensated_xyz3d[0]

            # update compensated state in the Kalman filter
            try:
                self.trackers3d[trk_id].kf_box3d.x[:3] = copy.copy(compensated_xyz3d).reshape((-1))
            except:
                self.trackers3d[trk_id].kf_box3d.x[:3] = copy.copy(compensated_xyz3d).reshape((-1, 1))

        return trks

    def predict(self, is_3d=True):
        trackers = self.trackers3d if is_3d else self.trackers2d
        return {tid: trk.predict() for tid, trk in trackers.items()}

    def update(self, dets, det_feats, matched_indices, is_3d=True):
        trks = self.trackers3d if is_3d else self.trackers2d
        unmatched_det = [True] * len(dets)
        unmatched_trk = [True] * len(trks)
        tid_past = list(trks.keys())
        tid_now = []
        for det_ind, trk_ind in matched_indices:
            unmatched_det[det_ind] = False
            unmatched_trk[trk_ind] = False
            tid = tid_past[trk_ind]

            trk = trks[tid]
            trk.update(*dets[det_ind])
            trk.cur_det = dets[det_ind] + [tid]
            trk.features = det_feats[det_ind]

            tid_now.append(tid)

        return tid_now, unmatched_det, [tid_past[i] for i, flag in enumerate(unmatched_trk) if flag]

    def birth(self, dets, dets_feat, unmatched_dets, obj_id_now, is_3d=True):
        # create and initialise new trackers for unmatched detections
        trackers = self.trackers3d if is_3d else self.trackers2d
        for i in range(len(unmatched_dets)):
            if unmatched_dets[i]:
                if is_3d:
                    self.tid3d += 1
                    obj_id = self.tid3d
                    trk = Tracker3D(Box3D.bbox2array(dets[i][0]), Box2D.bbox2array(dets[i][1]), dets[i][2], obj_id, self.thr_stage1.min_hits)
                else:
                    self.tid2d += 1
                    obj_id = self.tid2d
                    trk = Tracker2D(Box2D.bbox2array(dets[i][0]), dets[i][1], obj_id, self.thr_stage1.min_hits)

                trk.cur_det = dets[i] + [obj_id]
                trackers[obj_id] = trk
                trackers[obj_id].features = dets_feat[i]

                # save obj_id_now
                obj_id_now.append(obj_id)

        return obj_id_now

    def die(self, is_3d=True):
        trackers = self.trackers3d if is_3d else self.trackers2d
        for tid in list(trackers):
            if trackers[tid].time_since_update >= self.thr_stage1.max_age: del trackers[tid]

    def fusion(self, frame):

        def check_traj_status(trk, status=1, check_boundary=False):
            box2d = trk.get_state() if len(trk.cur_det) == 3 else trk.get_state()[1]
            if check_boundary:
                is_not_on_boundary = box2d.is_not_on_the_boundary(self.img_w, self.img_h)
            else:
                is_not_on_boundary = True

            # 1: matched trajectories: hits >= 2 and time_since_update == 0
            if status == 1:
                return trk.hits >= 2 and trk.time_since_update == 0 and is_not_on_boundary
            # 2: unmatched detections: hits == 1 and time_since_update == 0
            elif status == 2:
                return trk.hits == 1 and trk.time_since_update == 0 and is_not_on_boundary
            # 3: unmatched trajectories: hits >= 1 and time_since_update > 0
            elif status == 3:
                return trk.hits >= 1 and trk.time_since_update > 0 and is_not_on_boundary
            else:
                assert False, "Invalid trajectory status"

        def check_hits(trk1, trk2):
            return min(trk1.hits, trk2.hits) >= self.thr_stage2.min_hits

        step1_a, step1_b = self.cfg.cross_correction.step1_a, self.cfg.cross_correction.step1_b  # address unmatched detections that represent either false detections new objects
        step2_c, step2_d = self.cfg.cross_correction.step2_c, self.cfg.cross_correction.step2_d  # address unmatched trajectories that can arise from trajectory termination or missed detections
        step3_e = self.cfg.cross_correction.step3_e   # address unmatched trajectories that may arise from simultaneous termination and missed detections in both modalities
        iou_in, iou_step1, iou_step2, iou_step3, iou_out = self.thr_stage2.iou_in, self.thr_stage2.iou_s1, self.thr_stage2.iou_s2, self.thr_stage2.iou_s3, self.thr_stage2.iou_out
        ids_3d = list(self.trackers3d.keys())
        ids_2d = list(self.trackers2d.keys())
        used_ids_3d, used_ids_2d = [], []

        iou2d_matrix = np.zeros((len(self.trackers3d), len(self.trackers2d)))
        for m, trk_3d in enumerate(self.trackers3d.values()):
            box3d_2d = trk_3d.get_state()[1] if trk_3d.time_since_update > 0 else trk_3d.cur_det[1]
            for n, trk_2d in enumerate(self.trackers2d.values()):
                box2d = trk_2d.get_state() if trk_2d.time_since_update > 0 else trk_2d.cur_det[0]
                iou2d_matrix[m][n] = compute_iou2d(np.array([box3d_2d.xx1, box3d_2d.yy1, box3d_2d.xx2, box3d_2d.yy2]),
                                                   np.array([box2d.xx1, box2d.yy1, box2d.xx2, box2d.yy2]))

        # 3d matched trajectories <-> 2d matched trajectories
        for m, trk_3d in enumerate(self.trackers3d.values()):
            costs = [0 for _ in range(len(self.trackers2d))]
            if len(costs) == 0 or trk_3d.id in used_ids_3d or not check_traj_status(trk_3d, status=1):
                continue
            for n, trk_2d in enumerate(self.trackers2d.values()):
                if trk_2d.id in used_ids_2d or not check_traj_status(trk_2d, status=1):
                    continue
                costs[n] = iou2d_matrix[m][n]

            if np.max(costs) > iou_in:
                used_ids_3d.append(trk_3d.id)
                used_ids_2d.append(ids_2d[np.argmax(costs)])

        if step1_a:
            # (a) 3d unmatched detections <- 2d matched trajectories
            for m, trk_3d in enumerate(self.trackers3d.values()):
                costs = [0 for _ in range(len(self.trackers2d))]
                if len(costs) == 0 or trk_3d.id in used_ids_3d or not check_traj_status(trk_3d, status=2):
                    continue
                for n, trk_2d in enumerate(self.trackers2d.values()):
                    if trk_2d.id in used_ids_2d or not check_traj_status(trk_2d, status=1):
                        continue
                    costs[n] = iou2d_matrix[m][n]

                if np.max(costs) > iou_step1:
                    trk_3d.hits += 1
                    used_ids_3d.append(trk_3d.id)
                    used_ids_2d.append(ids_2d[np.argmax(costs)])

        if step1_b:
            # (b) 3d unmatched detections <-> 2d unmatched detections
            for m, trk_3d in enumerate(self.trackers3d.values()):
                costs = [0 for _ in range(len(self.trackers2d))]
                if len(costs) == 0 or trk_3d.id in used_ids_3d or not check_traj_status(trk_3d, status=2):
                    continue
                for n, trk_2d in enumerate(self.trackers2d.values()):
                    if trk_2d.id in used_ids_2d or not check_traj_status(trk_2d, status=2):
                        continue
                    costs[n] = iou2d_matrix[m][n]

                if np.max(costs) > iou_step1:
                    trk_2d = self.trackers2d[ids_2d[np.argmax(costs)]]
                    trk_3d.hits += 1
                    trk_2d.hits += 1
                    used_ids_3d.append(trk_3d.id)
                    used_ids_2d.append(trk_2d.id)

        if step2_c:
            # (c) 3d unmatched trajectories <- 2d matched trajectories
            for m, trk_3d in enumerate(self.trackers3d.values()):
                costs = [0 for _ in range(len(self.trackers2d))]
                if len(costs) == 0 or trk_3d.id in used_ids_3d or not check_traj_status(trk_3d, status=3):
                    continue
                for n, trk_2d in enumerate(self.trackers2d.values()):
                    if trk_2d.id in used_ids_2d or not check_traj_status(trk_2d, status=1):
                        continue
                    costs[n] = iou2d_matrix[m][n]

                trk_2d = self.trackers2d[ids_2d[np.argmax(costs)]]
                if np.max(costs) > iou_step2 and check_hits(trk_2d, trk_3d):
                    box3d, box3d_2d = trk_3d.get_state()
                    info = trk_3d.info
                    info.frame = frame
                    box2d = trk_2d.cur_det[0]

                    trk_3d.update(box3d, box2d, info)
                    trk_3d.cur_det = [box3d, box2d, info, trk_3d.id]
                    trk_3d.features = self.trackers2d[trk_2d.id].features
                    used_ids_3d.append(trk_3d.id)
                    used_ids_2d.append(trk_2d.id)

        if step2_d:
            # (d) 2d unmatched trajectories <- 3d matched trajectories
            for n, trk_2d in enumerate(self.trackers2d.values()):
                costs = [0 for _ in range(len(self.trackers3d))]
                if len(costs) == 0 or trk_2d.id in used_ids_2d or not check_traj_status(trk_2d, status=3):
                    continue
                for m, trk_3d in enumerate(self.trackers3d.values()):
                    if trk_3d.id in used_ids_3d or not check_traj_status(trk_3d, status=1):
                        continue
                    costs[m] = iou2d_matrix[m][n]

                trk_3d = self.trackers3d[ids_3d[np.argmax(costs)]]
                if np.max(costs) > iou_step2 and check_hits(trk_2d, trk_3d):
                    box3d_2d = trk_3d.cur_det[1]
                    info = trk_2d.info
                    info.frame = frame

                    trk_2d.update(box3d_2d, info)
                    trk_2d.cur_det = [box3d_2d, info, trk_2d.id]
                    trk_2d.features = self.trackers3d[trk_3d.id].features
                    used_ids_2d.append(trk_2d.id)
                    used_ids_3d.append(trk_3d.id)

        if step3_e:
            # (e) 3d unmatched trajectories <-> 2d unmatched trajectories
            for m, trk_3d in enumerate(self.trackers3d.values()):
                costs = [0 for _ in range(len(self.trackers2d))]
                if len(costs) == 0 or trk_3d.id in used_ids_3d or not check_traj_status(trk_3d, status=3, check_boundary=True):
                    continue
                for n, trk_2d in enumerate(self.trackers2d.values()):
                    if trk_2d.id in used_ids_2d or not check_traj_status(trk_2d, status=3, check_boundary=True):
                        continue
                    costs[n] = iou2d_matrix[m][n]

                trk_2d = self.trackers2d[ids_2d[np.argmax(costs)]]
                if np.max(costs) > iou_step3 and check_hits(trk_2d, trk_3d):
                    box3d, box3d_2d = trk_3d.get_state()
                    info_3d = trk_3d.info
                    info_3d.frame = frame

                    box2d = trk_2d.get_state()
                    info_2d = trk_2d.info
                    info_2d.frame = frame

                    trk_3d.update(box3d, box3d_2d, info_3d)
                    trk_3d.cur_det = [box3d, box3d_2d, info_3d, trk_3d.id]
                    trk_2d.update(box2d, info_2d)
                    trk_2d.cur_det = [box2d, info_2d, trk_2d.id]
                    used_ids_2d.append(trk_2d.id)
                    used_ids_3d.append(trk_3d.id)

        if step1_a or step1_b or step2_c or step2_d or step3_e:
            for m, trk_3d in enumerate(self.trackers3d.values()):
                if (len(iou2d_matrix[m]) > 0 and np.max(iou2d_matrix[m]) > iou_out): # or trk_3d.is_confirmed:
                    if check_traj_status(trk_3d, status=1):
                        self.output_per_frame[frame].append(trk_3d.cur_det)
        else:
            for m, trk_3d in enumerate(self.trackers3d.values()):
                if trk_3d.hits >= self.thr_stage1.min_hits and trk_3d.time_since_update == 0:
                    self.output_per_frame[frame].append(trk_3d.cur_det)

    # 连接损失预测
    def compute_sim(self, det_feats, trk_feats, embed_dim=512):

        mask1 = (det_feats[:, embed_dim * 0] != 100) & (trk_feats[:, embed_dim * 0] != 100)
        mask2 = (det_feats[:, embed_dim * 1] != 100) & (trk_feats[:, embed_dim * 1] != 100)
        mask3 = (det_feats[:, embed_dim * 2] != 100) & (trk_feats[:, embed_dim * 2] != 100)
        all_mask1 = mask1 & ~mask2 & ~mask3
        all_mask2 = mask2 & ~mask3
        all_mask3 = mask2 & mask3

        out1 = self.model.classifier1(det_feats[all_mask1, :embed_dim * 1], trk_feats[all_mask1, :embed_dim * 1])
        out2 = self.model.classifier2(det_feats[all_mask2, :embed_dim * 2], trk_feats[all_mask2, :embed_dim * 2])
        out3 = self.model.classifier3(det_feats[all_mask3, :embed_dim * 3], trk_feats[all_mask3, :embed_dim * 3])

        out = torch.zeros(len(det_feats), 2).cuda()
        out[all_mask1, :] = out1
        out[all_mask2, :] = out2
        out[all_mask3, :] = out3
        out = F.softmax(out, dim=1)
        return out.detach().cpu().numpy()[:, 1]

    def compute_cost_map(self, dets, dets_feat, trks, is_3d=True):
        cost_matrix = np.zeros((len(dets), len(trks)))
        if len(dets) == 0 or len(trks) == 0:
            return cost_matrix

        if is_3d:
            trackers = self.trackers3d
            trks_center = np.array([[box3d.x, box3d.y, box3d.z] for box3d, _ in [trk for _, trk in trks.items()]])
            dets_center = np.array([[box3d.x, box3d.y, box3d.z] for box3d, _, _ in dets])
            cost_matrix = np.linalg.norm(dets_center[:, np.newaxis, :] - trks_center[np.newaxis, :, :], axis=2)
        else:
            trackers = self.trackers2d
            trks_box2d = np.array([[box2d.xx1, box2d.yy1, box2d.xx2, box2d.yy2] for box2d in [trk for _, trk in trks.items()]])
            dets_box2d = np.array([[box2d.xx1, box2d.yy1, box2d.xx2, box2d.yy2] for box2d, _ in dets])
            cost_matrix = 1 - compute_iou2d_matrix(dets_box2d, trks_box2d)

        obj_id_past = list(trackers.keys())
        dets_feat = torch.cat([torch.cat(feat, dim=1) for feat in dets_feat], dim=0).repeat_interleave(len(trks), dim=0)
        trks_feat = torch.cat([torch.cat(trackers[obj_id].features, dim=1) for obj_id in obj_id_past], dim=0).repeat(len(dets), 1)
        score_matrix = 1 - self.compute_sim(dets_feat, trks_feat, self.cfg.embed_dim).reshape(-1, len(trks))

        sgc_score = self.thr_stage1.sgc_score3d if is_3d else self.thr_stage1.sgc_score2d
        sgc_score_strict = sgc_score.strict
        sgc_score_loose = sgc_score.loose

        mask1 = score_matrix < self.thr_stage1.m3_score
        mask2 = cost_matrix < sgc_score_strict
        mask3 = cost_matrix < sgc_score_loose
        mask = mask2 | (mask1 & mask3)
        # cost_matrix[mask] += score_matrix[mask]
        cost_matrix[~mask] = 1000

        return cost_matrix

    # 主函数
    def step(self):

        st = time.time()
        for frame in tqdm(range(self.sf, self.lf + 1), desc=f"Processing SEQ {self.seq_name}"):
            dets_3d = self.dets_3d.get(frame, [])
            dets_2d = self.dets_2d.get(frame, [])
            calib = self.load_calibration()

            # ************ Load image, point cloud data; extract multi-modal features ************
            dets_feat_img = self.load_image(frame, dets_3d + dets_2d)
            dets_feat_geo = self.load_planer_geometry(dets_3d + dets_2d)
            dets_feat_pc = self.load_points(frame, dets_3d, calib)
            dets_feat_3D, dets_feat_2D = self.split_feats(dets_feat_img, dets_feat_geo, dets_feat_pc, dets_3d, dets_2d)

            # ************ predict ************
            trks_3d = self.predict(is_3d=True)
            trks_2d = self.predict(is_3d=False)

            # ************ ego motion compensation for 3D trajectory ************
            if frame > self.sf:
                trks_3d = self.ego_motion_compensation(frame, trks_3d, calib)

            # ************ compute cost matrix ************
            matrix_3D = self.compute_cost_map(dets_3d, dets_feat_3D, trks_3d, is_3d=True)
            matrix_2D = self.compute_cost_map(dets_2d, dets_feat_2D, trks_2d, is_3d=False)

            # ************match************
            matched_indices_3D = greedy_matching(matrix_3D)
            matched_indices_2D = greedy_matching(matrix_2D)

            # ************update (process matched detections and tracks)************
            obj_id_now_3D, unmatched_dets_3D, unmatched_trks_id_3D = self.update(dets_3d, dets_feat_3D, matched_indices_3D, is_3d=True)
            obj_id_now_2D, unmatched_dets_2D, unmatched_trks_id_2D = self.update(dets_2d, dets_feat_2D, matched_indices_2D, is_3d=False)

            # ************birth (process unmatched detections)************
            self.birth(dets_3d, dets_feat_3D, unmatched_dets_3D, obj_id_now_3D, is_3d=True)
            self.birth(dets_2d, dets_feat_2D, unmatched_dets_2D, obj_id_now_2D, is_3d=False)

            # ************fuse detections************
            self.fusion(frame)

            # ************die (remove dead tracks)************
            self.die(is_3d=True)
            self.die(is_3d=False)

        self._save_results()

        return self.lf - self.sf + 1, time.time() - st

    def _save_results(self):
        with open(self.save_path, 'w') as f:
            for frame, trk in self.output_per_frame.items():
                for b3, b2, info, tid in trk:
                    f.write(
                        f"{frame} {tid} {self.cls_map_r[info.cls]} 0 0 {info.alpha} "
                        f"{b2.xx1:.2f} {b2.yy1:.2f} {b2.xx2:.2f} {b2.yy2:.2f} "
                        f"{b3.h:.2f} {b3.w:.2f} {b3.l:.2f} "
                        f"{b3.x:.2f} {b3.y:.2f} {b3.z:.2f} {b3.ry:.2f} {info.score:.2f}\n")


if __name__ == '__main__':

    torch.cuda.reset_peak_memory_stats()
    start_cpu_memory = analyze_system_cpu_memory()
    start_gpu_memory = analyze_system_gpu_memory()
    print(datetime.now())

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="configs/kitti_mot/pointgnn_rrc_car.yaml", help='specify the config for tracking')
    args = parser.parse_args()
    config = cfg_from_yaml_file(args.cfg_file, cfg)

    model = Net(embed_dim=config.embed_dim).cuda().eval()
    model.load_state_dict(torch.load(config.ckpt))
    dataset = DetectionProcessor(config)

    total_f, total_t = 0, 0
    for seq_name in config.tracking_seqs[config.mode]:
        tr = CrossTracker(config, seq_name, model, dataset)
        nf, dt = tr.step()
        total_f += nf
        total_t += dt
        print(f"Seq {seq_name} FPS:{nf / dt:.2f}")

    print(f"Overall FPS:{total_f / total_t:.2f}, Frames:{total_f}, Time:{total_t:.2f}")
    print(datetime.now())

    end_cpu_memory = analyze_system_cpu_memory()
    end_gpu_memory = analyze_system_gpu_memory()
    print(f"CPU Memory used: {end_cpu_memory - start_cpu_memory:.2f} MB, Start: {start_cpu_memory:.2f} MB, End: {end_cpu_memory:.2f} MB")
    print(f"GPU Memory used: {end_gpu_memory - start_gpu_memory:.2f} MB, Start: {start_gpu_memory:.2f} MB, End: {end_gpu_memory:.2f} MB")
