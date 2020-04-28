import numpy as np


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    center_x = base_size / 2
    center_y = base_size / 2
    anchors = np.zeros((len(ratios) * len(scales), 4), dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = base_size * scales[j] * np.sqrt(ratios[i])
            w = base_size * scales[j] * np.sqrt(1./ratios[i])
            index = i * len(ratios) + j
            anchors[index, 0] = center_y - h * 0.5
            anchors[index, 1] = center_x - w * 0.5
            anchors[index, 2] = center_y + h * 0.5
            anchors[index, 3] = center_x + w * 0.5
    return anchors


def loc2bbox(src_box, loc):
    ctr_x = 0.5 * (src_box[:, 1] + src_box[:, 3])
    ctr_y = 0.5 * (src_box[:, 0] + src_box[:, 2])
    h = src_box[:, 2] - src_box[:, 0]
    w = src_box[:, 3] - src_box[:, 1]

    dy = loc[:, 0::4]   #从0开始每过4个取一个, (__,1)
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    #因为ctr_y为 (__, )故newaxis后与dy格式一致
    ctr_y = ctr_y[:, np.newaxis] + dy * h[:, np.newaxis]
    ctr_x = ctr_x[:, np.newaxis] + dx * w[:, np.newaxis]
    h = np.exp(dh) * h[:, np.newaxis]
    w = np.exp(dw) * w[:, np.newaxis]

    dst_box = np.zeros(loc.shape, dtype=loc.dtype)
    dst_box[:, 0::4] = ctr_y - 0.5 * h
    dst_box[:, 1::4] = ctr_x - 0.5 * w
    dst_box[:, 2::4] = ctr_y + 0.5 * h
    dst_box[:, 3::4] = ctr_x + 0.5 * w
    return dst_box


def bbox_iou(box1, box2):
    """
    return a (R1, R2) matrix
    """
    area1 = np.prod(box1[:, 2:] - box1[:, :2], axis=1)
    area2 = np.prod(box2[:, 2:] - box2[:, :2], axis=1)

    tl = np.maximum(box1[:, None, :2], box2[:, :2])
    br = np.minimum(box1[:, None, 2:], box2[:, 2:])

    overlaps = np.prod(br - tl, axis=2) * (tl < br).all(axis=2) #R1, R2
    return overlaps / (area1[:, None] + area2 - overlaps)


def bbox2loc(src_box, dst_box):
    x1 = (src_box[:, 1] + src_box[:, 3]) * 0.5
    y1 = (src_box[:, 0] + src_box[:, 2]) * 0.5
    w1 = (src_box[:, 3] - src_box[:, 1]) * 0.5
    h1 = (src_box[:, 2] + src_box[:, 0]) * 0.5

    x2 = (dst_box[:, 3] + dst_box[:, 1]) * 0.5
    y2 = (dst_box[:, 2] + dst_box[:, 0]) * 0.5
    w2 = (dst_box[:, 3] - dst_box[:, 1]) * 0.5
    h2 = (dst_box[:, 2] - dst_box[:, 0]) * 0.5

    eps = np.finfo(h1.dtype).eps
    h1 = np.maximum(eps, h1)
    w1 = np.maximum(eps, w1)

    dy = (y2 - y1) / h1
    dx = (x2 - x1) / w1
    dh = np.log(h2 / h1)
    dw = np.log(w2 / w1)
    loc = np.stack((dy, dx, dh, dw)).transpose()
    return loc

