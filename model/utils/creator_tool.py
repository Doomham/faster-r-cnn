import numpy as np
from model.utils.bbox_tool import loc2bbox, bbox_iou, bbox2loc
from model.utils.non_maximum_suppression import nms


class ProposalCreator(object):
    def __init__(self, model, nms_thresh=0.7,
                 n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms=2000, n_test_post_nms=300, min_size=16):
        self.model = model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)
        #clip
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        #除去太小的roi（任意边小与min_size)
        min_size = self.min_size * scale
        _h = roi[:, 2] - roi[:, 0]
        _w = roi[:, 3] - roi[:, 1]
        keep = np.where((_h >= min_size) & (_w >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        #nms进一步减少roi
        order = score.ravel().argsort()[::-1]   #argsort从小到大排
        order = order[:n_pre_nms]
        roi = roi[order, :] #已经是分数按高到低排的roi
        keep_ = nms(roi, self.nms_thresh)[:n_post_nms]
        return roi[keep_]


class AnchorTargetCreator(object):
    def __init__(self):
        self.n_sample = 256
        self.pos_iou_thresh = 0.7
        self.neg_iou_thresh = 0.3
        self.pos_ratio = 0.5

    def __call__(self, bbox, anchor, img_size):
        H, W = img_size
        n_anchor = anchor.shape[0]
        inside_index = _get_inside_idx(anchor, H, W)
        anchor = anchor[inside_index]
        """（在内部的anchor中）每个anchor与哪个bbox的IoU最大， 以及对应的label"""
        a2b_max_iou_idx, label = self._create_label(bbox, anchor, inside_index)
        a2b_loc = bbox2loc(anchor, bbox[a2b_max_iou_idx])

        loc = _unmap(a2b_loc, n_anchor, inside_index)
        label = _unmap(label, n_anchor, inside_index)
        return loc, label

    def _create_label(self, bbox, anchor, inside_index):
        label = np.empty(len(inside_index), dtype=np.int32)
        label.fill(-1)

        a2b_max_iou_idx, a2b_max_iou, b2a_max_iou_idx = self._cal_ious(bbox, anchor)
        label[a2b_max_iou < self.neg_iou_thresh] = 0
        label[a2b_max_iou >= self.pos_iou_thresh] = 1
        label[b2a_max_iou_idx] = 1

        n_pos = int(self.n_sample * self.pos_ratio)
        pos_idx = np.where(label == 1)[0]
        if len(pos_idx) > n_pos:
            disable_idx = np.random.choice(pos_idx, size=len(pos_idx)-n_pos, replace=False)
            label[disable_idx] = -1

        n_neg = self.n_sample - np.sum(label == 1)
        neg_idx = np.where(label == 0)[0]
        if len(neg_idx) > n_neg:
            disable_idx = np.random.choice(neg_idx, size=len(neg_idx)-n_neg, replace=False)
            label[disable_idx] = -1

        return a2b_max_iou_idx, label

    def _cal_ious(self, bbox, anchor):
        iou = bbox_iou(anchor, bbox)    #(n_anchor, n_bbox)
        a2b_max_iou_idx = np.argmax(iou, axis=1)
        b2a_max_iou_idx = np.argmax(iou, axis=0)
        a2b_max_iou = np.max(iou, axis=1)
        return a2b_max_iou_idx, a2b_max_iou, b2a_max_iou_idx


class ProposalTargetCreator(object):
    def __init__(self):
        self.n_sample = 128
        self.pos_ratio = 0.25

        self.pos_iou_thresh = 0.5

        self.neg_iou_thresh_h = 0.5
        self.neg_iou_thresh_l = 0.0 #notice: py-faster-rcnn 0.1

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """
        label: Notice that it's not binary one, 20

        Notice:
            Why we add gt_bbox to train?
            Answer is in the R-CNN paper, ONLY those boxes nearby gt_bbox is helpful
        """
        roi = np.concatenate((roi, bbox), axis=0)
        pos_roi_per_img = np.round(self.n_sample * self.pos_ratio)
        IoU = bbox_iou(roi, bbox)
        r2b = np.argmax(IoU, axis=1)
        r2b_max_iou = np.max(IoU, axis=1)

        roi_label = label[r2b] + 1  #[0, fg_class - 1] to [1, fg_class]

        pos_idx = np.where(r2b_max_iou >= self.pos_iou_thresh)[0]
        pos_roi_this_img = int(np.minimum(len(pos_idx), pos_roi_per_img))
        if pos_idx.size > 0:
            pos_idx = np.random.choice(pos_idx, size=pos_roi_this_img, replace=False)

        neg_idx = np.where((r2b_max_iou < self.neg_iou_thresh_h) & (r2b_max_iou >= self.neg_iou_thresh_l))[0]
        neg_roi_this_img = int(np.minimum(len(neg_idx), self.n_sample - pos_roi_this_img))
        if neg_idx.size > 0:
            neg_idx = np.random.choice(neg_idx, size=neg_roi_this_img, replace=False)

        keep_idx = np.append(pos_idx, neg_idx)
        gt_roi_label = roi_label[keep_idx]
        gt_roi_label[len(pos_idx):] = 0

        sample_roi = roi[keep_idx]

        gt_roi_loc = bbox2loc(sample_roi, bbox[r2b[keep_idx]])
        """Why do this normalize?"""
        gt_roi_loc = (gt_roi_loc - np.array(loc_normalize_mean, dtype=np.float32)
                      ) / np.array(loc_normalize_std, dtype=np.float32)
        return sample_roi, gt_roi_loc, gt_roi_label


def _get_inside_idx(anchor, h, w):
    index_inside = np.where((anchor[:, 0] >= 0) & (anchor[:, 2] <= h) &
                            (anchor[:, 1] >= 0) & (anchor[:, 3] <= w))[0]
    return index_inside


def _unmap(data, n_anchor, inside_index):
    """
    case1: data is label
    case2: data is loc
    """
    if len(data.shape) == 1:
        label = np.empty(n_anchor, dtype=data.dtype)
        label.fill(-1)
        label[inside_index] = data
        return label
    else:
        loc = np.empty((n_anchor, 4), dtype=data.dtype)
        loc.fill(0)
        loc[inside_index, :] = data
        return loc

