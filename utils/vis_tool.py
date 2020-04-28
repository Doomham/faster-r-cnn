import time
import numpy as np
import matplotlib
import torch
import visdom

#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data.voc_dataset import VOCBboxDataset

VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)


def vis_img(img, ax=None):
    if ax is None:
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)
    """(C, H, W) to (H, W, C)"""
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):
    label_name = list(VOC_BBOX_LABEL_NAMES) + ['bg']

    ax = vis_img(img, ax)

    for i, box in enumerate(bbox):
        h = box[2] - box[0]
        w = box[3] - box[1]
        ax.add_patch(plt.Rectangle((box[1], box[0]), w, h, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
        caption.append(label_name[label[i]])

        caption.append('{:.2f}'.format(score[i]))

        ax.text(box[1], box[0], ':'.join(caption), style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    return ax


def fig2data(fig):
    """
    brief Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it

    @param fig： a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plt.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.


def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)
    return data


class Visualizer(object):
    """
    wrapper for visdom
    you can still access naive visdom function by
    self.line, self.scater,self._send,etc.
    due to the implementation of `__getattr__`
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self._vis_kw = kwargs

        # e.g.（’loss',23） the 23th value of loss
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        change the config of visdom
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(torch.tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self


"""
data = VOCBboxDataset('D:/pyworks/FasterRCNN/data/VOCdevkit/VOC2007', 'trainval')
img, bbox, label = data.get_example(0)
vis_bbox(img, bbox, label)
plt.show()"""
"""
arr = np.array([0, 1, 1, 0, 1], dtype=np.bool)
tp = np.cumsum(arr == 1)
fp = np.cumsum(arr == 0)
p = tp / (tp + fp)
mp0 = np.concatenate(([0], np.nan_to_num(p), [0]))
mp = np.maximum.accumulate(mp0[::-1])
print(p)
print(mp0)
print(mp)"""