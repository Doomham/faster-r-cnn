from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from torch import nn
import torch
from torchnet.meter import ConfusionMeter, AverageValueMeter
from utils.config import opt
from utils.vis_tool import Visualizer

LossTuple = namedtuple('LossTuple',['rpn_loc_loss','rpn_cls_loss','roi_loc_loss','roi_cls_loss','total_loss'])


class FasterRCNNTrainer(nn.Module):
    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()
        self.loc_normalize_mean = faster_rcnn.loc_mean
        self.loc_normalize_std = faster_rcnn.loc_std
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        self.faster_rcnn = faster_rcnn
        self.optimizer = faster_rcnn.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss
        self.vis = Visualizer(opt.env)

    def forward(self, img, bbox, label, scale):
        N, C, H, W = img.shape

        feature_map = self.faster_rcnn.extractor(img)
        rpn_locs, rpn_scores, rois, anchors = self.faster_rcnn.rpn(feature_map, (H, W), scale)
        bbox = bbox[0]
        label = label[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        """RPN Loss"""
        """
        1.找出未越界的anchors
        2.对于这些未越界的anchors，先求出IoU矩阵 (n_anchor, n_bbox)
            (i)贴标签：(n_sample=256, pos_ratio=0.5)
                negative: max_iou < 0.3 的anchors
                positive: max_iou >= 0.7 或 某个bbox与其IoU最大
                don't care: otherwise
            (ii)求loc：
                求出这些anchors到使得IoU最大的bbox的loc
        3.将label, loc补全回n_anchor的shape
        """
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox.numpy(), anchors, (H, W))
        gt_rpn_label = torch.from_numpy(gt_rpn_label).long()
        gt_rpn_loc = torch.from_numpy(gt_rpn_loc)

        rpn_loc_loss = _loc_loss(rpn_loc.cuda(), gt_rpn_loc.cuda(), gt_rpn_label.cuda(), self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score.cuda(), gt_rpn_label.cuda(), ignore_index=-1)

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score[gt_rpn_label > -1]
        self.rpn_cm.add(_rpn_score.detach(), _gt_rpn_label.data.long())

        """ROI Loss"""
        """
        1.首先要找用来训练的RoI，即sample_roi，一部分来自roi一部分来自bbox。
          gt_roi_label 中 0=background
        2.用这样的sample_roi扔进RoIHead中获取对应的loc(n_sample, 21*4)和score。
          对每一个sample_roi找对应gt_label的loc
        """
        sample_roi, gt_roi_loc, gt_roi_label = \
            self.proposal_target_creator(roi, bbox.numpy(), label.numpy())
        roi_loc, roi_score = self.faster_rcnn.head(feature_map, sample_roi)

        n_sample = roi_loc.shape[0]
        roi_loc = roi_loc.view(n_sample, -1, 4)
        roi_loc = roi_loc[torch.arange(0, n_sample).long().cuda(), torch.from_numpy(gt_roi_label).cuda().long()]

        gt_roi_loc = torch.from_numpy(gt_roi_loc).cuda()
        gt_roi_label = torch.from_numpy(gt_roi_label).long().cuda()

        roi_loc_loss = _loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)
        self.roi_cm.add(roi_score.detach(), gt_roi_label.data.long())

        loss = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        loss = loss + [sum(loss)]
        return LossTuple(*loss)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        loss = self.forward(imgs, bboxes, labels, scale)
        loss.total_loss.backward()
        self.optimizer.step()
        self.update_meters(loss)
        return loss

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        torch.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, loss):
        loss_d = {k: v.view(1)[0] for k, v in loss._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff < (1/sigma2)).float()
    y = flag * sigma2 * 0.5 * diff ** 2 + (1-flag) * (abs_diff - 0.5 / sigma2)
    return y.sum()


def _loc_loss(pred_loc, gt_loc, gt_label, sigma):
    """ in_weight 用来帮助将特定的loc纳入loss的计算中"""
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    loc_loss /= (gt_label >= 0).sum()
    return loc_loss

