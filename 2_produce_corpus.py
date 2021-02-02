import numpy as np
import os
import socket
import sys
import time
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import utils.utils as utils
import utils.dataset_utils as dataset_utils

from configs import get_train_configs
from logger import Logger
from torch.optim import SGD
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


args = get_train_configs()
workers = args.workers
batch_size = args.batch_size
net_name = args.net_name
dset_name = args.dset_name
run_id = args.run_id
seed = args.seed
exp_name = '2_produce_corpus'
augment = 'imagenet_c'
project_path = os.path.dirname(os.path.abspath(__file__))
logger_path = os.path.join(project_path, 'log', exp_name, 'run-{}'.format(run_id))
teacher_path = os.path.join(project_path, 'log', '1_std_net', 'run-{}'.format(run_id))
logger = Logger(exp_name, log_dir=logger_path).get()
np.random.seed(seed+run_id)
torch.manual_seed(seed+run_id)
torch.cuda.manual_seed(seed+run_id)

if __name__ == '__main__':
    logger.info(args)
    com_name = socket.gethostname()
    logger.info("Running on: {}".format(com_name))

    net_fn = utils.get_network(net_name, dset_name)
    if dset_name == 'cifar10': n_class = 10
    elif dset_name == 'cifar100': n_class = 100
    elif dset_name == 'svhn': n_class = 10
    elif dset_name == 'food101': n_class = 101
    net = net_fn(n_class)
    net.cuda()
    net.load_state_dict(torch.load(os.path.join(teacher_path, 'ckpt', 'last')))
    net.eval()

    values = []
    for c in range(15):
        for s in range(1, 6):
            values.append([c, s])

    f_name = 'corpus_{}'.format(augment)
    f_path = os.path.join(teacher_path, f_name)
    with open(f_path, 'w') as f:
        msg = 'accuracy,value\n'
        print(msg.strip())
        f.write(msg)
        for v in values:
            aug_args_dict = {'c': v[0], 's': v[1]}
            tr_dset, va_dset, te_dset, n_class, mean, std = dataset_utils.get_dataset_and_transform(
                dset_name, use_test_transform=True, augment=augment, augment_args_dict=aug_args_dict,
            )
            tr_loader = torch.utils.data.DataLoader(
                tr_dset, batch_size=batch_size, drop_last=False,
                shuffle=True, pin_memory=False, num_workers=12,
            )
            with torch.no_grad():
                corr, amt = 0, 0
                for x, y, _ in tqdm(tr_loader):
                    lg = net(x.cuda())
                    corr += torch.sum(torch.argmax(lg, 1) == y.cuda()).item()
                    amt += len(y)
            acc = corr/amt
            v = str(v[0]) + '-' + str(v[1])
            msg = '{},{}\n'.format(acc, v)
            logger.info(msg.strip())
            f.write(msg)
