import torch
import random
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from sklearn.utils.linear_assignment_ import linear_assignment


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_planer_points(img_w, img_h, top_left, bottom_right, n_points, vis_points=False):
    # 创建一个网格
    grid = np.broadcast_to(np.arange(0, img_w), (img_h, img_w))

    # 裁剪区域
    cropped_area = grid[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]

    # 随机选择100个点的行和列索引
    random_indices = np.random.choice(cropped_area.size, size=n_points, replace=True)
    random_rows, random_cols = np.unravel_index(random_indices, cropped_area.shape)

    # 获取裁剪区域内的随机点和对应的二维坐标
    selected_coords = np.column_stack((random_cols, random_rows)) + np.array([top_left])

    if vis_points:
        # 可视化二维散点图
        plt.scatter(selected_coords[:, 0], selected_coords[:, 1], s=1)
        # 自适应坐标轴范围
        plt.axis('auto')
        plt.show()

    return selected_coords / (img_w, img_h)

def greedy_matching(cost_matrix, dist_thresh=1000):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1 \
                and cost_matrix[det_id, trk_id] < dist_thresh:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices).reshape(-1, 2)

def linear_matching(cost_matrix, dist_thresh=1000):
    indices = linear_assignment(cost_matrix)

    matched_indices = []
    for row, col in indices:
        if cost_matrix[row, col] < dist_thresh:
            matched_indices.append([row, col])

    return np.asarray(matched_indices).reshape(-1, 2)


def save_pickle(v, filename):
    with open(filename, 'wb') as f:
        pickle.dump(v, f)
    return filename


def load_pickle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)