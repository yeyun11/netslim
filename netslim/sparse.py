import numpy
import torch
import torch.nn as nn
from .prune import WEIGHT_POSTFIX


def update_bn(model, s=1e-4):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) or \
            isinstance(m, nn.modules.instancenorm._InstanceNorm) or \
            isinstance(m, nn.GroupNorm):
            m.weight.grad.data.add_(s * torch.sign(m.weight.data))


def update_bn_by_names(model, sparsity_dict):
    for norm_layer_name, s in sparsity_dict.items():
        *container_names, module_name = norm_layer_name.split('.')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        m = container._modules[module_name]
        m.weight.grad.data.add_(s * torch.sign(m.weight.data))

        
"""
from .thresholding import css_thresholding
def update_sparse_dict(state_dict, sparse_dict, bn_names, comp, sparsity):
    keep_ratios = {}
    for bn_name in bn_names:
        bn_weight_name = bn_name + WEIGHT_POSTFIX
        scale_weight = state_dict[bn_weight_name]
        scale_weight_list = [_.abs().item() for _ in list(scale_weight)]

        prune_th, prune_info = css_thresholding(scale_weight_list)
        keep_mask = scale_weight > prune_th
        keep_ratio = keep_mask.sum().item() / len(scale_weight_list)
        keep_ratios[bn_name] = keep_ratio

    kr_mean = numpy.mean(list(keep_ratios.values()))
    kr_std = numpy.std(list(keep_ratios.values()))

    for bn_name in bn_names:
        sparse_dict[bn_name] = sparsity * numpy.exp(comp * (keep_ratios[bn_name] - kr_mean) / kr_std)

    return sparse_dict
"""