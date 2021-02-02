from data.load_cifar import load_cifar10, load_cifar100
import numpy as np
import os
import pandas as pd
import pickle
import random
from scipy.io import loadmat
import skimage
from skimage import io
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms
from utils.utils import AverageMeter, cutout_fn, gaussian_noise
from utils.imagenet_c_small import corrupt as corrupt_small


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images, self.labels, self.transform = images, labels, transform
        self.labels = self.labels.astype(np.int_)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = F.to_pil_image(image)
        if self.transform:
            image = self.transform(image)
        return image, label


class PickleDataset(ImageDataset):
    def __init__(self, path, transform=None):
        self.images, self.labels = pickle.load(open(path, 'rb'))
        self.labels = self.labels.astype(np.int_)
        self.transform = transform


class DummyAug(object):
    def __init__(self, fn):
        self.function = fn

    def set_augment(self, args_dict):
        self.args_dict = args_dict

    def __call__(self, im):
        c = int(self.args_dict['c'])
        s = int(self.args_dict['s'])
        return self.function(im, severity=s, corruption_number=c)


class SoftTargetDatasetFolder(Dataset):
    def __init__(
            self, img_folder, transform=None, acc_list_path=None,
            augment_prob=1, augment=None, augment_args_dict=None, beta_neg=1, beta_pos=1,
        ):
        self.transform = transform
        if acc_list_path:
            self.acc_list = pd.read_csv(acc_list_path, delimiter=',')
            print(self.acc_list)
        self.beta = torch.distributions.beta.Beta(torch.Tensor([beta_neg]), torch.Tensor([beta_pos]))
        self.beta_aug = torch.distributions.beta.Beta(torch.Tensor([1.]), torch.Tensor([1.]))
        self.augment_prob, self.augment_args_dict = augment_prob, augment_args_dict
        class_dirs = sorted(os.listdir(img_folder))
        self.image_paths, self.labels = [], []
        for n, c in enumerate(class_dirs):
            class_dir = os.path.join(img_folder, c)
            for im in os.listdir(class_dir):
                self.image_paths.append(os.path.join(img_folder, c, im))
                self.labels.append(n)
        self.n_class = len(class_dirs)
        self.aug = DummyAug(corrupt_small)

    def __len__(self):
        return len(self.labels)

    def get_args_dict(self, alpha):
        idx = np.argmin(np.abs(self.acc_list['accuracy'].values - alpha.numpy()))
        acc = self.acc_list['accuracy'].values[idx]
        val = self.acc_list['value'].values[idx].split('-')
        return {'c': int(val[0]), 's': int(val[1])}, acc

    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx], self.labels[idx]
        image = io.imread(image_path)
        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)
        is_augment = self.beta_aug.sample().item() < self.augment_prob
        if is_augment:
            if not self.augment_args_dict:
                alpha = self.beta.sample() # Get the expected probability.
                alpha = 1/self.n_class + alpha * (1 - 1/self.n_class)
                args_dict, p = self.get_args_dict(alpha) # Used in student training.
            else:
                args_dict, p = self.augment_args_dict, 1 # Used in extracting corpus.
            self.aug.set_augment(args_dict)
            transform = transforms.Compose(self.transform.transforms[:-3] + [self.aug] + self.transform.transforms[-3:])
            rest_p = (1 - p) / (self.n_class - 1)
            y_onehot = torch.ones([self.n_class]) * rest_p
            y_onehot[label] = p
        else:
            y_onehot = torch.zeros([self.n_class])
            y_onehot[label] = 1
            transform = self.transform
        if transform:
            image = F.to_pil_image(image)
            image = transform(image)
        return image, label, y_onehot


class SoftTargetDatasetCifar(Dataset):
    def __init__(
            self, dset_name, split='train', transform=None, acc_list_path=None,
            augment_prob=1, augment=None, augment_args_dict=None, beta_neg=1, beta_pos=1,
        ):
        if dset_name == 'cifar10':
            x_tr, y_tr, x_te, y_te = load_cifar10()
        elif dset_name == 'cifar100':
            x_tr, y_tr, x_te, y_te = load_cifar100()
        self.images = x_tr if split == 'train' else x_te
        self.labels = y_tr if split == 'train' else y_te
        self.labels = self.labels.astype(np.int_)
        self.n_class = len(np.unique(self.labels))
        self.transform = transform
        if acc_list_path:
            self.acc_list = pd.read_csv(acc_list_path, delimiter=',')
            print(self.acc_list)
        self.beta = torch.distributions.beta.Beta(torch.Tensor([beta_neg]), torch.Tensor([beta_pos]))
        self.beta_aug = torch.distributions.beta.Beta(torch.Tensor([1.]), torch.Tensor([1.]))
        self.augment_prob, self.augment_args_dict = augment_prob, augment_args_dict
        self.aug = DummyAug(corrupt_small)


    def __len__(self):
        return len(self.labels)

    def get_args_dict(self, alpha):
        idx = np.argmin(np.abs(self.acc_list['accuracy'].values - alpha.numpy()))
        acc = self.acc_list['accuracy'].values[idx]
        val = self.acc_list['value'].values[idx].split('-')
        return {'c': int(val[0]), 's': int(val[1])}, acc

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        is_augment = self.beta_aug.sample().item() < self.augment_prob
        if is_augment:
            if not self.augment_args_dict:
                alpha = self.beta.sample() # Get the expected probability.
                alpha = 1/self.n_class + alpha * (1 - 1/self.n_class)
                args_dict, p = self.get_args_dict(alpha) # Used in student training.
            else:
                args_dict, p = self.augment_args_dict, 1 # Used in extracting corpus.
            self.aug.set_augment(args_dict)
            transform = transforms.Compose(self.transform.transforms[:-2] + [self.aug] + self.transform.transforms[-2:])
            rest_p = (1 - p) / (self.n_class - 1)
            y_onehot = torch.ones([self.n_class]) * rest_p
            y_onehot[label] = p
        else:
            y_onehot = torch.zeros([self.n_class])
            y_onehot[label] = 1
            transform = self.transform
        if transform:
            image = F.to_pil_image(image)
            image = transform(image)
        return image, label, y_onehot


class SoftTargetDatasetMat(Dataset):
    def __init__(
            self, mat_path, transform=None, acc_list_path=None,
            augment_prob=1, augment=None, augment_args_dict=None, beta_neg=1, beta_pos=1,
        ):
        data_ = loadmat(mat_path)
        self.images, self.labels = data_['X'].transpose([3, 0, 1, 2]), data_['y']
        self.labels = (self.labels % 10).astype('long').reshape(-1)
        self.n_class = len(np.unique(self.labels))
        self.transform = transform
        if acc_list_path:
            self.acc_list = pd.read_csv(acc_list_path, delimiter=',')
            print(self.acc_list)
        self.beta = torch.distributions.beta.Beta(torch.Tensor([beta_neg]), torch.Tensor([beta_pos]))
        self.beta_aug = torch.distributions.beta.Beta(torch.Tensor([1.]), torch.Tensor([1.]))
        self.augment_prob, self.augment_args_dict = augment_prob, augment_args_dict
        self.aug = DummyAug(corrupt_small)


    def __len__(self):
        return len(self.labels)

    def get_args_dict(self, alpha):
        idx = np.argmin(np.abs(self.acc_list['accuracy'].values - alpha.numpy()))
        acc = self.acc_list['accuracy'].values[idx]
        val = self.acc_list['value'].values[idx].split('-')
        return {'c': int(val[0]), 's': int(val[1])}, acc

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        is_augment = self.beta_aug.sample().item() < self.augment_prob
        if is_augment:
            if not self.augment_args_dict:
                alpha = self.beta.sample() # Get the expected probability.
                alpha = 1/self.n_class + alpha * (1 - 1/self.n_class)
                args_dict, p = self.get_args_dict(alpha) # Used in student training.
            else:
                args_dict, p = self.augment_args_dict, 1 # Used in extracting corpus.
            self.aug.set_augment(args_dict)
            transform = transforms.Compose(self.transform.transforms[:-3] + [self.aug] + self.transform.transforms[-3:])
            rest_p = (1 - p) / (self.n_class - 1)
            y_onehot = torch.ones([self.n_class]) * rest_p
            y_onehot[label] = p
        else:
            y_onehot = torch.zeros([self.n_class])
            y_onehot[label] = 1
            transform = self.transform
        if transform:
            image = F.to_pil_image(image)
            image = transform(image)
        return image, label, y_onehot
