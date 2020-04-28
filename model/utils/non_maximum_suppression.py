import torch
import numpy as np


def nms(roi, nms_thresh):
    order = np.arange(roi.shape[0]).astype(np.int32)
    res = list()
    areas = np.prod(roi[:, 2:] - roi[:, :2], axis=1)
    while order.size > 0:
        res.append(order[0])
        """计算IoU"""
        tl = np.maximum(roi[order[0], :2], roi[order[1:], :2])
        br = np.minimum(roi[order[0], 2:], roi[order[1:], 2:])
        overlaps = np.prod(br - tl, axis=1) * (br > tl).all(axis=1)
        IoU = overlaps / (areas[order[1:]] + areas[order[0]] - overlaps)
        keep = np.where(IoU <= nms_thresh)[0] + 1
        order = order[keep]
    return np.array(res)


"""
boxes = np.array([[12, 190, 300, 399],[221, 250, 389, 500],[100, 100, 150, 168],
                  [166, 70, 312, 190],[28, 130, 134, 302]])

res = nms(boxes, 0.1)
print(res)
"""