from __future__ import division
import torch
import numpy as np
from model.utils.bbox_tool import loc2bbox
from model.utils.non_maximum_suppression import nms
from torch import nn
from data.dataset import preprocess
import torch.nn.functional as F
from torch import optim


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.rpn = rpn
        self.extractor = extractor
        self.head = head
        """这段干啥用？"""
        self.loc_mean = (0., 0., 0., 0.)
        self.loc_std = (0.1, 0.1, 0.2, 0.2)
        self.use_preset('evaluate')

    def forward(self, img, scale=1.):
        img_size = img.shape[2:]   #H, W  because img.size=N, C, H, W
        feature_map = self.extractor(img)   #1, 512, H//16, W//16
        rpn_locs, rpn_scores, rois, anchors = self.rpn(feature_map, img_size, scale)
        roi_locs, roi_scores = self.head(feature_map, rois)
        return roi_locs, roi_scores, rois

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            for img in imgs:
                size = img.shape[1:]
                img = preprocess(img)
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bbox = list()
        score = list()
        label = list()

        for img, size in zip(prepared_imgs, sizes):
            img = torch.from_numpy(img).cuda().float()[None]  #(1, C, H, W)
            scale = torch.tensor(img.shape[3] / size[1])
            roi_locs, roi_scores, rois = self.forward(img, scale)

            roi = torch.from_numpy(rois).cuda() / scale

            mean = torch.tensor(self.loc_mean).cuda().repeat(21)[None]
            std = torch.tensor(self.loc_std).cuda().repeat(21)[None]

            roi_locs = (roi_locs * std) + mean
            roi_locs = roi_locs.view(-1, 21, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_locs)
            pred_bbox = loc2bbox(roi.cpu().numpy().reshape((-1, 4)),
                                 roi_locs.detach().cpu().numpy().reshape((-1, 4)))
            pred_bbox = torch.from_numpy(pred_bbox).cuda()
            pred_bbox = pred_bbox.view(-1, 21 * 4)
            pred_bbox[:, 0::2] = pred_bbox[:, 0::2].clamp(0, size[0])
            pred_bbox[:, 1::2] = pred_bbox[:, 1::2].clamp(0, size[1])

            prob = F.softmax(roi_scores, dim=1).detach().cpu().numpy()
            pred_bbox = pred_bbox.cpu().numpy()

            b, l, s = self._suppress(pred_bbox, prob)
            bbox.append(b)
            label.append(l)
            score.append(s)

        self.use_preset('evaluate')
        self.train()
        return bbox, label, score

    def _suppress(self, raw_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        raw_bbox = raw_bbox.reshape((-1, 21, 4))
        for l in range(1, 21):
            bbox_l = raw_bbox[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            bbox_l = bbox_l[mask]
            prob_l = prob_l[mask]
            order = np.argsort(prob_l)[::-1]
            sorted_bbox = bbox_l[order]
            keep = nms(sorted_bbox, self.nms_thresh)

            bbox.append(sorted_bbox[keep])
            label.append((l - 1) * np.ones(len(keep)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox).astype(np.float32)
        label = np.concatenate(label).astype(np.int32)
        score = np.concatenate(score).astype(np.float32)
        return bbox, label, score

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = 1e-3
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        self.optimizer = optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
