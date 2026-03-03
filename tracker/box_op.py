import numpy as np


def compute_iou2d(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    return inter_area / float(box1_area + box2_area - inter_area)

def compute_iou2d_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    # Expand dimensions to calculate pairwise IoU
    boxes1 = np.expand_dims(boxes1, axis=1)  # Shape: (5, 1, 4)
    boxes2 = np.expand_dims(boxes2, axis=0)  # Shape: (1, 8, 4)

    # Calculate intersection
    inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    # Calculate areas of individual boxes
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Calculate union
    union_area = area1 + area2 - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou
