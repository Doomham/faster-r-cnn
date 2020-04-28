import os
import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import TrainDataset, TestDataset, inverse_normalize
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list
    for ii, (img, size, gt_bbox, gt_label, gt_difficult) in tqdm(enumerate(dataloader)):
        size = [size[0][0], size[1][0]] #why?
        pred_bbox, pred_label, pred_score = faster_rcnn.predict(img.numpy(), [size])
        gt_bboxes += list(gt_bbox.numpy())
        gt_labels += list(gt_label.numpy())
        gt_difficults += list(gt_difficult.numpy())
        pred_bboxes +=pred_bbox
        pred_labels += pred_label
        pred_scores += pred_score
        if ii == test_num:
            break
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    data_set = TrainDataset()
    print('load data.')
    data_loader = data_.DataLoader(data_set, batch_size=1, shuffle=True)
    testset = TestDataset()
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       shuffle=False,
                                       pin_memory=True
                                       )

    faster_rcnn = FasterRCNNVGG16()
    print('model construct.')

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    lr = opt.lr
    best_map = 0

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox, label, scale) in tqdm(enumerate(data_loader)):
            img = img.cuda()
            trainer.train_step(img, bbox, label, scale)
            if (ii + 1)%opt.plot_every==0:
                ipdb.set_trace()

                """plot loss"""
                trainer.vis.plot_many(trainer.get_meter_data())

                """plot gt_bbox"""
                ori_img = inverse_normalize(img[0].cpu().numpy())
                gt_img = visdom_bbox(ori_img, bbox[0].numpy(), label[0].numpy())
                trainer.vis.img('gt_img', gt_img)
                """plot predicted bbox"""
                pred_bbox, pred_label, pred_score = trainer.faster_rcnn.predict([ori_img], visualize=True)
                pred_img = visdom_bbox(ori_img, pred_bbox[0], pred_label[0], pred_score[0])
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', trainer.roi_cm.conf.float().cpu())

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr = lr * opt.lr_decay

        trainer.vis.plot('test_map', eval_result['map'])
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        if epoch == 13:
            print('finish!')
            break

train()