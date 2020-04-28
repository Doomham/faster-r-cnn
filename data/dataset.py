from data.voc_dataset import VOCBboxDataset
import torch
from skimage import transform as sktsf
import numpy as np
from torchvision import transforms as tvtsf
from data.util import resize_bbox, random_flip, flip_bbox


def inverse_normalize(img):
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def normalize(img):
    n = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = n(torch.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    """
    input:
        an img(~numpy ndarray) with CHW RGB [0, 255]
    对输入的图像rescale，短边大于等于600长边小于等于1000并/255
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img /= 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    return normalize(img.astype(np.float32))


class Transform(object):
    def __init__(self):
        self.min_size = 600
        self.max_size = 1000

    def __call__(self, data):
        img, bbox, label = data
        _, H, W = img.shape
        img = preprocess(img)
        _, o_H, o_W = img.shape
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        #randomly horizontally flip
        img, x_flip = random_flip(img)
        if x_flip:
            bbox = flip_bbox(bbox, (o_H, o_W))

        scale = o_H / H
        return img, bbox, label, scale


class TrainDataset(object):
    def __init__(self):
        self.db = VOCBboxDataset('D:/pyworks/FasterRCNN/data/VOCdevkit/VOC2007', 'trainval')
        self.transform = Transform()

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.transform((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset(object):
    def __init__(self):
        self.db = VOCBboxDataset('D:/pyworks/FasterRCNN/data/testset/VOCdevkit/VOC2007', 'test', True)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)