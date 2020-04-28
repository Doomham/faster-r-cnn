import numpy as np
from torch.nn import functional as F
import torch
from torch import nn
from model.utils.bbox_tool import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base()
        self.conv = nn.Conv2d(512, 512, 3, 1, 1)

        k = self.anchor_base.shape[0]
        self.score = nn.Conv2d(512, 2 * k, 1)
        self.loc = nn.Conv2d(512, 4 * k, 1)

        self.proposal_layer = ProposalCreator(self)

        normal_init(self.conv, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale):
        N, C, H, W = x.shape
        anchors = _enumerate_shifted_anchors(self.anchor_base, (H, W))  #(__,4)

        h = F.relu(self.conv(x))    #(1, 512, H, W)

        scores = self.score(h)  #(1, 2 * k, H, W)
        locs = self.loc(h)  #(1, 4*k, H, W)

        """it does need a .contiguous() to use view method"""
        locs = locs.permute((0, 2, 3, 1)).contiguous().view(N, -1, 4)
        scores = scores.permute((0, 2, 3, 1)).contiguous()
        #why view as H, W rather than W, H?
        #answer: look the way we create anchors!
        fg_scores = scores.view(N, H, W, 9, 2)[:, :, :, :, 1].contiguous().view(N, -1)
        scores = scores.view(N, -1, 2)

        rois = self.proposal_layer(locs[0].cpu().data.numpy(),
                                   fg_scores[0].cpu().data.numpy(),
                                   anchors, img_size, scale.numpy())

        return locs, scores, rois, anchors


def _enumerate_shifted_anchors(anchor_base, size):
    """anchor_base.shape = k, 4"""
    H, W = size
    shifted_x = np.arange(0, W * 16, 16)#(W, )
    shifted_y = np.arange(0, H * 16, 16)#(H, )

    shifted_x, shifted_y = np.meshgrid(shifted_x, shifted_y)#(H, W) and（H, W)
    # .ravel() shape = (H*W, )
    anchors = np.stack((shifted_y.ravel(), shifted_x.ravel(), shifted_y.ravel(), shifted_x.ravel()), axis=1)#(H*W, 4)
    anchors = anchors[None, :].transpose((1, 0, 2)) + anchor_base
    return anchors.reshape(-1, 4).astype(np.float32)


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

a = torch.tensor(5).cuda()
b = a.detach().cpu()
print(b)
