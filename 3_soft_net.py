import numpy as np
import os
import pandas as pd
import socket
import sys
import time
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import utils.utils as utils
import utils.dataset_utils as dataset_utils

from configs import get_train_soft_configs
from logger import Logger
from torch.optim import SGD

import warnings
warnings.filterwarnings('ignore')


args = get_train_soft_configs()
workers = args.workers
batch_size = args.batch_size
te_batch_size = args.te_batch_size
momentum = args.momentum
weight_decay = args.weight_decay
net_name = args.net_name
epochs = args.epochs
lr_epochs = args.lr_epochs
lr = args.lr
grad_clip = args.grad_clip
dset_name = args.dset_name
run_id = args.run_id
seed = args.seed
augment = 'imagenet_c'
acc_list_path = args.acc_list_path
aug_prob = args.aug_prob
beta_neg = args.beta_neg
beta_pos = args.beta_pos
exp_name = '3_soft_net'
project_path = os.path.dirname(os.path.abspath(__file__))
logger_path = os.path.join(project_path, 'log', exp_name, 'run-{}'.format(run_id))
logger = Logger(exp_name, log_dir=logger_path).get()
np.random.seed(seed+run_id)
torch.manual_seed(seed+run_id)
torch.cuda.manual_seed(seed+run_id)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()

    for step, (input, target, onehot_y) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        onehot_y = onehot_y.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(input)
        sm = F.softmax(logits, 1)
        loss = - torch.mean(torch.sum(onehot_y * torch.log(sm), 1)) # Soft-target loss
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(input)
            sm = F.softmax(logits, 1)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


if __name__ == '__main__':
    start_time = time.time()
    logger.info(args)
    com_name = socket.gethostname()
    logger.info("Running on: {}".format(com_name))

    tr_dset, va_dset, te_dset, n_class, mean, std = dataset_utils.get_dataset_and_transform(
        dset_name, augment=augment, acc_list_path=acc_list_path,
        aug_prob=aug_prob, beta_neg=beta_neg, beta_pos=beta_pos,
    )
    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=batch_size, drop_last=True,
        shuffle=True, pin_memory=True, num_workers=workers,
    )
    te_loader = torch.utils.data.DataLoader(
        te_dset, batch_size=batch_size,
        pin_memory=True, num_workers=workers,
    )

    #build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    net_fn = utils.get_network(net_name, dset_name)
    net = net_fn(n_class)
    if args.parallel:
        net = nn.DataParallel(net)
    net.cuda()
    logger.info('param size = %.1f MB', utils.count_parameters_in_MB(net))

    optim = SGD(
        net.parameters(), lr, momentum=momentum,
        weight_decay=weight_decay
    )

    for epoch in range(epochs):
        if epoch in lr_epochs:
            optim.param_groups[0]['lr'] *= 0.1
        lr_ = optim.param_groups[0]['lr']
        logger.info('epoch {} lr {:.4f}'.format(epoch+1, lr_))

        train_acc, _ = train(tr_loader, net, criterion, optim)
        logger.info('--- Train acc: {:.2f}% ---'.format(train_acc))
    valid_acc, _ = infer(te_loader, net, criterion)
    logger.info('--- Test acc: {:.2f}% ---'.format(valid_acc))
    ckpt_dir = os.path.join(logger_path, 'ckpt')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, 'last')
    if isinstance(net, torch.nn.DataParallel):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    torch.save(state_dict, ckpt_path)

    end_time = time.time()
    duration = end_time - start_time
    logger.info('Running time: %d(s)', duration)
