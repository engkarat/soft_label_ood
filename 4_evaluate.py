import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics as sk
import socket
import sys
import time
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision
import utils.utils as utils
import utils.dataset_utils as dataset_utils

from configs import get_eval_configs, config_oods
from data.datasets import PickleDataset, SoftTargetDatasetFolder, SoftTargetDatasetCifar
from logger import Logger
from torch.optim import SGD
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


args = get_eval_configs()
workers = args.workers
batch_size = args.batch_size
net_name = args.net_name
dset_name = args.dset_name
run_id = args.run_id
seed = args.seed
eval_exp = args.eval_exp
exp_name = '4_evaluate_metrics'
project_path = os.path.dirname(os.path.abspath(__file__))
offset = 1000 if eval_exp == 3 else 0
logger_path = os.path.join(project_path, 'log', exp_name, 'run-{}'.format(run_id+offset))
logger = Logger(exp_name, log_dir=logger_path).get()
np.random.seed(seed+run_id)
torch.manual_seed(seed+run_id)
torch.cuda.manual_seed(seed+run_id)
name_dict = {1: '1_std_net', 3: '3_soft_net'}
eval_name = name_dict[eval_exp]
eval_path = os.path.join(project_path, 'log', eval_name, 'run-{}'.format(run_id))


def infer(loader, model, criterion, pgd_steps=None):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()
    lgs, sms, corr, corr2 = [], [], [], []
    if pgd_steps:
        attacker = PGD(model, eps=8/255, alpha=2/255, steps=pgd_steps, random_start=True)

    for data in tqdm(loader):
        img, lbl = data[0], data[1]
        img = img.cuda()
        lbl = lbl.cuda(non_blocking=True)
        if pgd_steps:
            img = attacker(img, lbl, mean_ts, std_ts)

        with torch.no_grad():
            lg = model(img)
            lgs.append(lg)
            sm = F.softmax(lg, 1)
            sms.append(sm)
            lbl = torch.clamp(lbl, 0, sm.shape[1] - 1)
            loss = criterion(lg, lbl)
            corr.append(lbl == torch.argmax(sm, 1))
            corr2.append(sm[range(len(sm)), lbl])

        prec1, prec5 = utils.accuracy(lg, lbl, topk=(1, 5))

        n = img.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
    lgs = torch.cat(lgs)
    sms = torch.cat(sms)
    ent = - torch.sum(sms * torch.log(sms), 1, keepdim=True)
    ent[torch.isnan(ent)] = 0
    corr = torch.cat(corr)
    corr2 = torch.cat(corr2)
    return top1.avg, objs.avg, lgs, sms, ent, corr, corr2


def get_auc(id_mea, od_mea):
    y_score = np.concatenate([id_mea, od_mea])
    y_true = np.concatenate([np.ones(len(id_mea)), np.zeros(len(od_mea))])
    return sk.roc_auc_score(y_true, y_score) * 100


def get_tnr(id_mea, od_mea):
    tpr_95_mea = np.percentile(id_mea, 5)
    tnr = np.sum(od_mea < tpr_95_mea) / len(od_mea)
    return tnr * 100


def get_aupr(id_mea, od_mea):
    y_score = np.concatenate([id_mea, od_mea])
    y_true = np.concatenate([np.ones(len(id_mea)), np.zeros(len(od_mea))])
    return sk.average_precision_score(y_true, y_score) * 100


def cal_calib(sms, corr, n_bin=15):
    idx = np.argsort(sms)
    sms = sms[idx]
    corr = corr[idx]
    sample_per_bin = int(np.ceil(len(sms) / n_bin))
    ece = 0
    scatters = []
    for i in range(n_bin):
        sms_ = sms[i*sample_per_bin: (i+1)*sample_per_bin]
        corr_ = corr[i*sample_per_bin: (i+1)*sample_per_bin]
        conf = np.mean(sms_) * 100
        acc = np.mean(corr_) * 100
        scatters.append([conf, acc])
        # print("{:.4f} -> {:.4f}".format(conf, acc))
        ece_ = (len(sms_) / len(sms)) * np.abs(acc - conf)
        ece += ece_
    return ece, np.array(scatters)


def get_ood_dset(ood_dset_name):
    if ood_dset_name == 'cifar10':
        dset = torchvision.datasets.CIFAR10('data/datasets/', transform=te_transform, train=False)
    elif ood_dset_name == 'cifar100':
        dset = torchvision.datasets.CIFAR100('data/datasets/', transform=te_transform, train=False)
    elif ood_dset_name == 'svhn':
        dset = torchvision.datasets.SVHN('data/datasets/', transform=te_transform, download=True, split='test')
    else:
        im_dir = os.path.join(dataset_path, ood_dset_name)
        dset = torchvision.datasets.ImageFolder(os.path.join(im_dir), transform=te_transform)
    return dset


if __name__ == '__main__':
    start_time = time.time()
    logger.info(args)
    com_name = socket.gethostname()
    logger.info("Running on: {}".format(com_name))
    exp_type = 'experimental'
    collections = {}
    to_dict = lambda acc, lgs, sms, ent: {'acc': acc, 'lgs': lgs, 'sms': sms, 'ent': ent}

    # Evaluate test accuracy
    tr_dset, va_dset, te_dset, n_class, mean, std = dataset_utils.get_dataset_and_transform(dset_name)
    mean_ts, std_ts = torch.from_numpy(mean).cuda().float(), torch.from_numpy(std).cuda().float()
    te_loader = torch.utils.data.DataLoader(te_dset, batch_size=batch_size, num_workers=workers)

    # Build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    net_fn = utils.get_network(net_name, dset_name)
    net = net_fn(n_class)
    net.cuda()
    logger.info('param size = %.1f MB', utils.count_parameters_in_MB(net))
    ckpt_path = os.path.join(eval_path, 'ckpt', 'last')
    ckpt = torch.load(ckpt_path)
    for k in list(ckpt.keys()):
        if k.startswith('module.'):
            ckpt[k.replace('module.', '')] = ckpt[k]
            del ckpt[k]
    net.load_state_dict(ckpt)
    print("Ckpt is loaded from {}.".format(ckpt_path))
    net.eval()
    acc, _, lgs, sms, ent, corr, corr2 = infer(te_loader, net, criterion)
    logger.info('--- Test acc: {:.2f}% ---'.format(acc))
    collections['test'] = to_dict(acc, lgs, sms, ent)
    fig_path = os.path.join(eval_path, 'figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    with open(os.path.join(eval_path, 'eval_acc'), 'w') as f:
        f.write(str(acc))
    with open(os.path.join(eval_path, 'eval_calib'), 'w') as f:
        ece, scatters = cal_calib(np.max(sms.cpu().detach().numpy(), 1), corr.cpu().detach().numpy())
        plt.figure(figsize=[6, 6])
        plt.plot(range(0, 101), range(0, 101))
        plt.scatter(scatters[:, 0], scatters[:, 1])
        plt.savefig(os.path.join(fig_path, 'eval_calib'), format="jpeg", bbox_inches='tight')
        f.write(str(ece))

    te_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    ood_names = config_oods[exp_type]
    dataset_path = os.path.join(project_path, 'data', 'datasets', exp_type, 'oods')
    with open(os.path.join(eval_path, 'eval_ood'), 'w') as f:
        f.write('ood,auc_max_sms,auc_ent,tnr_max_sms,tnr_ent,aupr_max_sms,aupr_ent\n')
        for ood_dset_name in ood_names.keys():
            logger.info('Evaluating OOD: {}.'.format(ood_names[ood_dset_name]))
            dset = get_ood_dset(ood_dset_name)
            loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, pin_memory=True, num_workers=workers)
            acc, _, lgs, sms, ent, _, _ = infer(loader, net, criterion)
            auc_sms = get_auc(np.max(collections['test']['sms'].cpu().detach().numpy(), 1), np.max(sms.cpu().detach().numpy(), 1))
            auc_ent = get_auc(-np.max(collections['test']['ent'].cpu().detach().numpy(), 1), -np.max(ent.cpu().detach().numpy(), 1))
            tnr_sms = get_tnr(np.max(collections['test']['sms'].cpu().detach().numpy(), 1), np.max(sms.cpu().detach().numpy(), 1))
            tnr_ent = get_tnr(-np.max(collections['test']['ent'].cpu().detach().numpy(), 1), -np.max(ent.cpu().detach().numpy(), 1))
            aupr_sms = get_aupr(np.max(collections['test']['sms'].cpu().detach().numpy(), 1), np.max(sms.cpu().detach().numpy(), 1))
            aupr_ent = get_aupr(-np.max(collections['test']['ent'].cpu().detach().numpy(), 1), -np.max(ent.cpu().detach().numpy(), 1))
            msg = "{},{},{},".format(ood_names[ood_dset_name], auc_sms, auc_ent)
            msg += "{},{},".format(tnr_sms, tnr_ent)
            msg += "{},{}\n".format(aupr_sms, aupr_ent)
            sm_id = np.max(collections['test']['sms'].cpu().detach().numpy(), 1)
            sm_od = np.max(sms.cpu().detach().numpy(), 1)
            plt.figure(figsize=[6, 6])
            plt.hist(sm_id, bins=100, density=True, alpha=0.6)
            plt.hist(sm_od, bins=100, density=True, alpha=0.6)
            plt.xlim(0, 1)
            plt.savefig(os.path.join(fig_path, 'eval_ood_{}'.format(ood_dset_name)), format="jpeg", bbox_inches='tight')
            f.write(msg)

    end_time = time.time()
    duration = end_time - start_time
    logger.info('Running time: %d(s)', duration)
