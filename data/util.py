import numpy as np
import random
from PIL import Image


def read_image(img_file):
    f = Image.open(img_file)
    img = f.convert('RGB')
    f.close()
    img = np.asarray(img, dtype=np.float32)
    #transpose (H, W, C) to (C, H, W)
    return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] *= y_scale
    bbox[:, 2] *= y_scale
    bbox[:, 1] *= x_scale
    bbox[:, 3] *= x_scale
    return bbox


def random_flip(img):
    x_flip = random.choice([True, False])
    if x_flip:
        img = img[:, :, ::-1]
    return img, x_flip


def flip_bbox(bbox, size):
    h, w = size
    x_min = w - bbox[:, 3]
    x_max = w - bbox[:, 1]
    bbox[:, 1] = x_min
    bbox[:, 3] = x_max
    return bbox