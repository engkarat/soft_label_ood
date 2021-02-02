import os
import torch
import shutil
import network.models as models
import numpy as np
import torchvision
import torchvision.transforms as transforms
from data.datasets import *


def get_dataset_and_transform(
    dset_name, use_test_transform=False, augment=None, augment_args_dict=None,
    acc_list_path=None, apply_aug_to_test=False, aug_prob=1, beta_neg=1, beta_pos=1,
    common_augment=False, allowed_corruptions=None,
):
    if dset_name == 'cifar10':
        mean = np.array([0.49137255, 0.48235294, 0.44666667])
        std = np.array([0.24705882, 0.24352941, 0.26156863])
        n_class = 10
        tr_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        te_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        transform_ = tr_transform if not use_test_transform else te_transform
        if augment:
            tr_dset = SoftTargetDatasetCifar(
                dset_name, split='train', transform=transform_, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            tr_dset = torchvision.datasets.CIFAR10('data/datasets/', transform=tr_transform if not use_test_transform else te_transform)

        if apply_aug_to_test:
            te_dset = SoftTargetDatasetCifar(
                dset_name, split='test', transform=te_transform, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            te_dset = torchvision.datasets.CIFAR10('data/datasets/', transform=te_transform, train=False)

        va_dset = None
        
    elif dset_name == 'cifar100':
        mean = np.array([0.5071, 0.4865, 0.4409])
        std = np.array([0.2673, 0.2564, 0.2762])
        n_class = 100
        tr_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        te_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        transform_ = tr_transform if not use_test_transform else te_transform
        if augment:
            tr_dset = SoftTargetDatasetCifar(
                dset_name, split='train', transform=transform_, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            tr_dset = torchvision.datasets.CIFAR100('data/datasets/', transform=tr_transform if not use_test_transform else te_transform)

        if apply_aug_to_test:
            te_dset = SoftTargetDatasetCifar(
                dset_name, split='test', transform=te_transform, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            te_dset = torchvision.datasets.CIFAR100('data/datasets/', transform=te_transform, train=False)

        va_dset = None
    elif dset_name == 'svhn':
        mean = np.array([0.4376821, 0.4437697, 0.47280442])
        std = np.array([0.19803012, 0.20101562, 0.19703614])
        n_class = 10
        tr_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        te_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        mat_path = 'data/datasets/train_32x32.mat'
        transform_ = tr_transform if not use_test_transform else te_transform
        if augment:
            tr_dset = SoftTargetDatasetMat(
                mat_path, transform=transform_, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            tr_dset = torchvision.datasets.SVHN('data/datasets/', transform=tr_transform if not use_test_transform else te_transform)

        mat_path = 'data/datasets/test_32x32.mat'
        if apply_aug_to_test:
            te_dset = SoftTargetDatasetMat(
                dset_name, transform=te_transform, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            te_dset = torchvision.datasets.SVHN('data/datasets/', transform=te_transform, split='test')

        va_dset = None
    elif dset_name == 'food101':
        mean = np.array([0.54465765, 0.44358285, 0.34473051])
        std = np.array([0.2515994, 0.25035347, 0.2545649])
        n_class = 101
        tr_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        te_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        im_folder = 'data/datasets/realistic_images/food101_small/train/'
        transform_ = tr_transform if not use_test_transform else te_transform
        if augment:
            tr_dset = SoftTargetDatasetFolder(
                im_folder, transform=transform_, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            tr_dset = torchvision.datasets.ImageFolder(im_folder, transform=transform_)

        im_folder = 'data/datasets/realistic_images/food101_small/test/'
        if apply_aug_to_test:
            te_dset = SoftTargetDatasetFolder(
                im_folder, transform=te_transform, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            te_dset = torchvision.datasets.ImageFolder(im_folder, transform=te_transform)

        im_folder = 'data/datasets/realistic_images/food101_small/val/'
        if apply_aug_to_test:
            va_dset = SoftTargetDatasetFolder(
                im_folder, transform=te_transform, acc_list_path=acc_list_path,
                augment=augment, augment_args_dict=augment_args_dict, augment_prob=aug_prob,
                beta_neg=beta_neg, beta_pos=beta_pos,
            )
        else:
            va_dset = torchvision.datasets.ImageFolder(im_folder, transform=te_transform)
    else:
        raise NotImplementedError("The chosen dataset is not implemented.")
    return tr_dset, va_dset, te_dset, n_class, mean, std
