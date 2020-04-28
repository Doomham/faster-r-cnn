from data.dataset import TrainDataset
from torch.utils import data as data_
from tqdm import tqdm
from model.faster_rcnn_vgg16 import decom_vgg16, VGG16RoIHead
import torch
from model.region_proposal_network import RegionProposalNetwork
import numpy as np
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer

data_set = TrainDataset()
#data_loader = data_.DataLoader(data_set, batch_size=1, shuffle=False)

img, bbox, label, scale = data_set.__getitem__(0)
model = FasterRCNNVGG16().cuda()
trainer = FasterRCNNTrainer(model)
loss = trainer.forward(torch.from_numpy(img[None, :]).cuda(), bbox, label, scale)
print(loss)
"""
roi_locs, roi_scores, rpn_locs, rpn_scores = model.forward(torch.from_numpy(img[None, :]).cuda())
print(roi_locs.shape)
print(roi_scores.shape)
print(rpn_locs.shape)
print(rpn_scores.shape)
"""
"""
extractor, classifier = decom_vgg16()
feature_map = extractor.cuda()(torch.from_numpy(img[None, :]).cuda())
print(img.shape)#3, 600, 800
print(feature_map.shape)#1, 512, 37, 50


boxes = np.array([[12, 190, 300, 399],[221, 250, 389, 500],[100, 100, 150, 168],
                  [166, 70, 312, 190],[28, 130, 134, 302]])
head = VGG16RoIHead(classifier.cuda()).cuda()
locs, scores = head.forward(feature_map, boxes)
print(locs.shape)
print(scores.shape)
"""