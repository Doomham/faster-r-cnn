import torch
from torch import nn
from torchvision.models import vgg16
import numpy as np
from model.faster_rcnn import FasterRCNN
from model.region_proposal_network import RegionProposalNetwork


def decom_vgg16():
    model = vgg16(True)
    features = list(model.features)[:30]
    classifier = model.classifier
    del classifier[6]
    del classifier[5]
    del classifier[2]
    classifier = nn.Sequential(*classifier)

    #freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    return nn.Sequential(*features), classifier


class VGG16RoIHead(nn.Module):
    def __init__(self, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.loc = nn.Linear(4096, 21 * 4)
        self.score = nn.Linear(4096, 21)
        self.roi_pool = nn.AdaptiveMaxPool2d((7, 7))

        normal_init(self.loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

    def forward(self, feature_map, rois):
        rois = torch.from_numpy(rois).cuda().mul_(1./16).long()
        num_rois = rois.size(0)
        roi_pool_result = []
        for i in np.arange(num_rois):
            roi = rois[i]
            """notice that feature map is (1, 512, W, H)"""
            f = feature_map[0, :, roi[0]:(roi[2] + 1), roi[1]:(roi[3] + 1)]
            roi_pool_result.append(self.roi_pool(f))
        roi_pool_result = torch.cat(roi_pool_result)
        h = roi_pool_result.view(num_rois, -1)    #num_rois, 25088
        fc7 = self.classifier(h)
        locs = self.loc(fc7)
        scores = self.score(fc7)
        return locs, scores


class FasterRCNNVGG16(FasterRCNN):
    def __init__(self):
        rpn = RegionProposalNetwork()
        extractor, classifier = decom_vgg16()
        head = VGG16RoIHead(classifier)
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head)


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


"""
arr = np.arange(12).reshape(1, 1, 3, 4)
arr = torch.from_numpy(arr)
cut = torch.tensor([[0, 0, 2, 2], [0, 0, 2, 2]])
t = list()
for i in range(cut.size(0)):
    c = cut[i]
    t.append(arr[..., c[0]:(c[2] + 1), c[1]:(c[3] + 1)])
print(t)
print(torch.cat(t))
"""
print(torch.tensor([0] * 15))