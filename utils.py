import os
import argparse
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.densenet import _DenseLayer
from netslim import get_pruning_layers, MaskedBatchNorm
from networks import deactivate


WEIGHT_POSTFIX = ".weight"
BIAS_POSTFIX = ".bias"


def parse_cifar_args(dataset="cifar"):
    default_data_path = {"cifar": "datasets/cifar-100", "ilsvrc12": "/opt/ILSVRC2012"}
    default_train_bs = {"cifar": 64, "ilsvrc12": 256}
    default_test_bs = {"cifar": 50, "ilsvrc12": 200}
    default_epochs = {"cifar": 160, "ilsvrc12": 90}
    default_init_lr = {"cifar": 0.1, "ilsvrc12": 0.1}
    default_lr_schedule = {"cifar": [0.5, 0.75], "ilsvrc12": [1/3, 2/3]}
    default_log_interval = {"cifar": 10, "ilsvrc12": 1}
    
    parser = argparse.ArgumentParser(description='PyTorch Cifar-100 Example for Network Slimming')
    parser.add_argument('--cifar10', action='store_true', default=False, 
                        help='Train CIFAR-10, by default the script train cifar-100')
    parser.add_argument('--batch-size', type=int, default=default_train_bs[dataset], metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=default_test_bs[dataset], metavar='N',
                        help='input batch size for testing (default: 50)')
    parser.add_argument('--resume-path', default='',
                        help='path to a trained model weight')
    parser.add_argument('--data-path', default=default_data_path[dataset],
                        help='path to dataset')
    parser.add_argument('--arch', default='resnet50',
                        help='network architecture')
    parser.add_argument('--epochs', type=int, default=default_epochs[dataset], metavar='EP',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=default_init_lr[dataset], metavar='LR',
                        help='base learning rate (default: 0.1)')
    parser.add_argument('--lr-decay-schedule', nargs='+', type=float, default=default_lr_schedule[dataset], metavar='LR-T',
                        help='the period ratio of epochs to decay LR')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1, metavar='LR-MUL',
                        help='decay factor of learning rate (default: 0.3162)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='L2',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--sparsity', type=float, default=0, metavar='L1',
                        help='coefficient for L1 regularization on BN (default: 0)')
    parser.add_argument('--sparsity-schedule', type=float, default=0, metavar='L1',
                        help='start to impose sparsity in training')
    parser.add_argument('--prune-ratio', type=float, default=-1, metavar='PR',
                        help='ratio of pruned channels to total channels, -1: do not prune, 1: optimal prune')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: the answer)')
    parser.add_argument('--postfix', default='', metavar='POSTNAME',
                        help='postfix of the output name')
    parser.add_argument('--log-interval', type=int, default=default_log_interval[dataset], metavar='LOG-T',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output', default='output', metavar='OUTPUT', 
                        help='folder to output images and model checkpoints')
    parser.add_argument('--tfs', action='store_true', default=False,
                        help='train from scratch')
    parser.add_argument('--test-at-0', action='store_true', default=False,
                        help='do a test before training')
    parser.add_argument('--prunable-layers-only', action='store_true', default=False,
                        help='sparsity and pruning on prunable layers only')
    parser.add_argument('--gamma', type=float, default=1, metavar='GAMMA',
                        help='gamma for Hard-Aware Back Propagation and Focal Loss')
    args = parser.parse_args()
    
    output_name = 'bs' # baseline
    if args.prune_ratio > 0.999:
        output_name = "ot" # optimal thresholding
    elif args.prune_ratio > 0:
        output_name = "ns_pr_{}".format(args.prune_ratio) # network slimming
    if args.sparsity > 0:
        output_name += "_sp_{}".format(args.sparsity)
    if args.weight_decay > 0:
        output_name += "_wd_{}".format(args.weight_decay)
    if args.tfs and args.prune_ratio > 0:
        output_name += "_tfs"
    if args.prunable_layers_only:
        output_name += "_po"
    if args.postfix:
        output_name += "_{}".format(args.postfix)
    
    if dataset == "cifar":
        args.num_classes = 10 if args.cifar10 else 100
        args.output += "-cifar{}-{}".format(args.num_classes, args.arch)
        args.output = os.path.join(args.output, output_name)
    elif dataset == "ilsvrc12":
        args.output += "-ilsvrc12-{}".format(args.arch)
        args.output = os.path.join(args.output, output_name)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    return args


class AverageMeter(object):
    """Computes and stores the average and current value, from PyTorch official examples repo"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def ns_post_process_resnet(model, prune_ratio, input_shape, device=None):
    prec_layers, succ_layers, bn_names = get_pruning_layers(model, input_shape, device)
    
    # find all scale weights in BN layers
    weights = model.state_dict()
    scale_weights = []
    norm_layer_names = list(set(succ_layers) & set(prec_layers))
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        weight = weights[norm_weight_name]
        scale_weights.extend([_.abs().item() for _ in list(weight)])

    # find threshold for pruning
    scale_weights.sort()
    prune_th_index = int(float(len(scale_weights)) * prune_ratio + 0.5)
    prune_th = scale_weights[prune_th_index]

    for m in model.modules():
        if isinstance(m, Bottleneck):
            if m.bn1.weight.abs().max().item() < prune_th or m.bn2.weight.abs().max().item() < prune_th or m.bn3.weight.abs().max().item() < prune_th:
                m.conv1.apply(deactivate)
                m.bn1.apply(deactivate)
                m.conv2.apply(deactivate)
                m.bn2.apply(deactivate)
                m.conv3.apply(deactivate)
                m.bn3.apply(deactivate)
                print("Skip the conv branch of a bottelneck block for resnet!")
        elif isinstance(m, BasicBlock):
            if m.bn1.weight.abs().max().item() < prune_th or m.bn2.weight.abs().max().item() < prune_th:
                m.conv1.apply(deactivate)
                m.bn1.apply(deactivate)
                m.conv2.apply(deactivate)
                m.bn2.apply(deactivate)
                print("Skip the conv branch of a basic block for resnet!")

                
def ns_post_process_densenet(model, prune_ratio, input_shape, device=None):
    prec_layers, succ_layers, bn_names = get_pruning_layers(model, input_shape, device)
    
    # find all scale weights in BN layers
    weights = model.state_dict()
    scale_weights = []
    norm_layer_names = list(set(succ_layers) & set(prec_layers))
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        weight = weights[norm_weight_name]
        scale_weights.extend([_.abs().item() for _ in list(weight)])

    # find threshold for pruning
    scale_weights.sort()
    prune_th_index = int(float(len(scale_weights)) * prune_ratio + 0.5)
    prune_th = scale_weights[prune_th_index]

    for m in model.modules():
        if isinstance(m, _DenseLayer):
            # in case that the layer was replaced by a masked BN
            if isinstance(m.norm1, MaskedBatchNorm):
                norm1 = m.norm1.bn
            else:
                norm1 = m.norm1
            if norm1.weight.abs().max().item() < prune_th or m.norm2.weight.abs().max().item() < prune_th:
                m.conv1.apply(deactivate)
                m.norm1.apply(deactivate)
                m.conv2.apply(deactivate)
                m.norm2.apply(deactivate)
                print("Skip the conv branch of a _DenseLayer block for densenet!")

