import os
import torch
import shutil
import network.models as models
import numpy as np
import torchvision
import torchvision.transforms as transforms


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val*n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def cutout_fn(img, length):
    img = np.array(img)
    h, w = img.shape[0], img.shape[1]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    if len(img.shape) == 3:
        img *= mask.reshape([h, w, 1]).astype('uint8')
    else:
        img *= mask.astype('uint8')
    return img


def gaussian_noise(x, c):
    x = np.array(x) / 255.
    return (np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255).astype('uint8')


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6


def get_network(net_name, dset_name):
    if net_name == 'densenet':
        net_fn = models.densenet100_bc
    elif net_name == 'wideresnet':
        net_fn = models.wide_resnet40_4
    else:
        raise Exception("Argument --net_name is not specified correctly.")
    return net_fn
