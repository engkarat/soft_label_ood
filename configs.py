import argparse


def str2intlist(v):
    return [int(i) for i in v.split(',')]


def str2list(v):
    return [i for i in v.split(',')]


def aug_args_dict(v):
    pairs = v.split(',')
    args = {}
    for pair in pairs:
        k, v = pair.split('=')
        args[k] = v
    return args


def get_train_configs():
    parser = argparse.ArgumentParser("training")

    data = parser.add_argument_group("Data Loader hyper-parameters")
    data.add_argument('--workers', type=int, default=4, help='Workers amount.')
    data.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    data.add_argument('--te_batch_size', type=int, default=500, help='Non-train batch size.')

    optim = parser.add_argument_group("Optimizer hyper-parameters")
    optim.add_argument('--momentum', type=float, default=0.9, help='momentum')
    optim.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    model = parser.add_argument_group("Network configs")
    model.add_argument('--net_name', type=str, default='resnet', help='Network architecture.')

    train = parser.add_argument_group("Training hyper-parameters")
    train.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    train.add_argument('--lr_epochs', type=str2intlist, default='100,150', help='Epoch at which decay learning rate.')
    train.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    train.add_argument('--grad_clip', type=float, default=-1, help='Gradient clipping.')

    others = parser.add_argument_group("Other hyper-parameters")
    others.add_argument('--dset_name', type=str, required=True, help='Name of dataset used in experiment.')
    others.add_argument('--run_id', type=int, required=True, help='Identifier number of the experiment.')
    others.add_argument('--seed', type=int, default=12345, help='Random seed.')
    others.add_argument('--parallel', action='store_true', help='Specify if use parallel.')

    return parser.parse_args()


def get_train_soft_configs():
    parser = argparse.ArgumentParser("training")

    data = parser.add_argument_group("Data Loader hyper-parameters")
    data.add_argument('--workers', type=int, default=4, help='Workers amount.')
    data.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    data.add_argument('--te_batch_size', type=int, default=500, help='Non-train batch size.')

    optim = parser.add_argument_group("Optimizer hyper-parameters")
    optim.add_argument('--momentum', type=float, default=0.9, help='momentum')
    optim.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    model = parser.add_argument_group("Network configs")
    model.add_argument('--net_name', type=str, default='resnet', help='Network architecture.')

    train = parser.add_argument_group("Training hyper-parameters")
    train.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    train.add_argument('--lr_epochs', type=str2intlist, default='100,150', help='Epoch at which decay learning rate.')
    train.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    train.add_argument('--grad_clip', type=float, default=-1, help='Gradient clipping.')
    train.add_argument('--augment', type=str, default='imagenet_c', help='List of applied augmentations.')
    train.add_argument('--acc_list_path', type=str, required=True, help='List of applied augmentations.')
    train.add_argument('--aug_prob', type=float, default=0.2, help='Probability of applying the augmentation; default = 1.')
    train.add_argument('--beta_neg', type=float, default=1, help='Beta distribution neg param.')
    train.add_argument('--beta_pos', type=float, default=1, help='Beta distribution pos param.')

    others = parser.add_argument_group("Other hyper-parameters")
    others.add_argument('--dset_name', type=str, required=True, help='Name of dataset used in experiment.')
    others.add_argument('--run_id', type=int, required=True, help='Identifier number of the experiment.')
    others.add_argument('--seed', type=int, default=123456, help='Random seed.')
    others.add_argument('--parallel', action='store_true', help='Specify if use parallel.')

    return parser.parse_args()


def get_eval_configs():
    parser = argparse.ArgumentParser("training")

    data = parser.add_argument_group("Data Loader hyper-parameters")
    data.add_argument('--workers', type=int, default=4, help='Workers amount.')
    data.add_argument('--batch_size', type=int, default=500, help='Batch size.')

    model = parser.add_argument_group("Network configs")
    model.add_argument('--net_name', type=str, default='resnet', help='Network architecture.')
    model.add_argument('--eval_exp', type=int, help='Evaluated experiment id.')

    others = parser.add_argument_group("Other hyper-parameters")
    others.add_argument('--dset_name', type=str, required=True, help='Name of dataset used in experiment.')
    others.add_argument('--run_id', type=int, required=True, help='Identifier number of the experiment.')
    others.add_argument('--seed', type=int, default=1234567, help='Random seed.')
    others.add_argument('--load_ckpt_path', type=str, help="Path for ckpt loading, or set it as 'default'.")

    return parser.parse_args()

config_oods = {
    'experimental': {
        'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100',
        'imnet_cropped_mod': 'TINc', 'imnet_resized': 'TINr',
        'lsun_cropped_mod': 'LSUNc', 'lsun_resized': 'LSUNr',
        'isun': 'iSUN', 'svhn': 'SVHN', 'food101': 'Food-101',
        'gaus_noise': 'Gaus. noise', 'unif_noise': 'Unif. noise',
    },
}
