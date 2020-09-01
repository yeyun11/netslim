import copy
from functools import partial
import torch
import torch.nn as nn
from .graph_parser import get_pruning_layers
from .thresholding import css_thresholding, MIN_SCALING_FACTOR, OT_DISCARD_PERCENT

OUT_CHANNEL_DIM = 0
IN_CHANNEL_DIM = 1
WEIGHT_POSTFIX = ".weight"
BIAS_POSTFIX = ".bias"


class MaskedBatchNorm(nn.Module):
    """
    Select channels from the output of BatchNorm 1d/2d/3d layer. 
    This implementation is referred to 
        https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/network-slimming/models/channel_selection.py
    """

    def __init__(self, bn, mask):
        super(MaskedBatchNorm, self).__init__()
        self.bn = bn
        self.register_buffer('channel_indexes', mask.nonzero().flatten())

    def forward(self, x):
        x = self.bn(x)
        return x.index_select(1, self.channel_indexes)

    def extra_repr(self):
        return super(MaskedBatchNorm, self).extra_repr() + "num_selected_channels={}".format(len(self.channel_indexes))


def group_weight_names(weight_names):
    grouped_names = {}
    for weight_name in weight_names:
        group_name = '.'.join(weight_name.split('.')[:-1])
        if group_name not in grouped_names:
            grouped_names[group_name] = [weight_name, ]
        else:
            grouped_names[group_name].append(weight_name)
    return grouped_names


def prune_filters(norm_layer_name, keep_mask, weights, grouped_weight_names, prec_layers, succ_layers):
    keep_indices = torch.nonzero(keep_mask).flatten()

    # 1. prune source normalization layer
    for weight_name in grouped_weight_names[norm_layer_name]:
        weights[weight_name] = weights[weight_name].masked_select(keep_mask)

    # 2. prune target succeeding conv/linear/... layers
    for prune_layer_name in succ_layers[norm_layer_name]:
        # hard code for features.41 conv-bn-linear
        #if norm_layer_name == "features.41":
        #    new_keep_mask = torch.zeros(512*7*7)
        #    for prune_index in keep_indices:
        #        for k in range(prune_index.item()*49, prune_index.item()*49+49):
        #            new_keep_mask[k] = 1
        #    new_keep_indices = torch.nonzero(new_keep_mask).flatten()
        #    for weight_name in grouped_weight_names[prune_layer_name]:
        #        if weight_name.endswith(WEIGHT_POSTFIX):
        #            weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, new_keep_indices)
        for weight_name in grouped_weight_names[prune_layer_name]:
            if weight_name.endswith(WEIGHT_POSTFIX):
                weights[weight_name] = weights[weight_name].index_select(IN_CHANNEL_DIM, keep_indices)

    # 3. prune target preceding conv/linear/... layers
    for prune_layer_name in prec_layers[norm_layer_name]:
        for weight_name in grouped_weight_names[prune_layer_name]:
            if weight_name.endswith(WEIGHT_POSTFIX):
                weights[weight_name] = weights[weight_name].index_select(OUT_CHANNEL_DIM, keep_indices)
            elif weight_name.endswith(BIAS_POSTFIX):
                weights[weight_name] = weights[weight_name].index_select(0, keep_indices)


def uniform_scaling(weights, prune_ratio, prec_layers, succ_layers):
    """Better prune method

    Arguments:
        weights (OrderedDict): unpruned model weights
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
        thresholding: thresholding method, by default: optimal prune

    Returns:
        pruned_weights (OrderedDict): pruned model weights
    """

    # prune weights with calculated threshold of a uniform scaling
    norm_layer_names = list(set(succ_layers) & set(prec_layers))
    grouped_weight_names = group_weight_names(weights.keys())
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        scale_weight = weights[norm_weight_name].abs()
        scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
        keep_index = int(len(scale_weight_list) * prune_ratio + 0.5)
        scale_weight_sorted = sorted(scale_weight_list)
        prune_th = (scale_weight_sorted[keep_index]+scale_weight_sorted[keep_index+1]) / 2
        keep_mask = scale_weight > prune_th
        # prune_ratio = 1 - keep_mask.sum().item() / len(scale_weight_list)

        if keep_mask.sum().item() == scale_weight.size(0):
            continue

        prune_filters(norm_layer_name, keep_mask, weights, grouped_weight_names, prec_layers, succ_layers)

    return weights, None


def network_slimming(weights, prune_ratio, prec_layers, succ_layers, per_layer_normalization=False):
    """default pruning method as described in:
            Zhuang Liu et.al., "Learning Efficient Convolutional Networks through Network Slimming", in ICCV 2017"

    Arguments:
        weights (OrderedDict): unpruned model weights
        prune_ratio (float): ratio of be pruned channels to total channels
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
        per_layer_normalization (bool): normalized by layer??

    Returns:
        pruned_weights (OrderedDict): pruned model weights
    """

    # find all scale weights in BN layers
    scale_weights = []
    norm_layer_names = list(set(succ_layers) & set(prec_layers))
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        weight = weights[norm_weight_name]
        if per_layer_normalization:
            scale_weights.extend([(_.abs() / weight.sum()).item() for _ in list(weight)])
        else:
            scale_weights.extend([_.abs().item() for _ in list(weight)])

    # find threshold for pruning
    scale_weights.sort()
    prune_th_index = int(float(len(scale_weights)) * prune_ratio + 0.5)
    prune_th = scale_weights[prune_th_index]

    # unpruned_norm_layer_names = list(set(succ_layers) ^ set(prec_layers))
    grouped_weight_names = group_weight_names(weights.keys())
    eliminated_layers = list()
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        scale_weight = weights[norm_weight_name].abs()
        if per_layer_normalization:
            scale_weight = scale_weight / scale_weight.sum()
        keep_mask = scale_weight > prune_th
        if keep_mask.sum().item() == scale_weight.size(0):
            continue

        # in case not to prune the whole layer
        if keep_mask.sum() == 1:
            print("Warning: One channel left for {}!".format(norm_layer_name))
        elif keep_mask.sum() == 0:
            scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
            scale_weight_list.sort(reverse=True)
            keep_mask = scale_weight > (scale_weight_list[0]+scale_weight_list[1]) / 2
            print("Warning: Zero channels after pruning {}! The channel with the largest scaling factor was kept ...".format(norm_layer_name))
            eliminated_layers.append(norm_layer_name)

        prune_filters(norm_layer_name, keep_mask, weights, grouped_weight_names, prec_layers, succ_layers)

    return weights, prune_th, eliminated_layers


network_slimming_normalized_by_layer = partial(network_slimming, per_layer_normalization=True)


def network_slimming_keep_half(weights, prune_ratio, prec_layers, succ_layers, per_layer_normalization=False):
    """iterative network slimming described in :
            Zhuang Liu et.al., "Learning Efficient Convolutional Networks through Network Slimming", in ICCV 2017"

    Arguments:
        weights (OrderedDict): unpruned model weights
        prune_ratio (float): ratio of be pruned channels to total channels
        prec_layers (dict): mapping from BN names to preceding convs/linears
        succ_layers (dict): mapping from BN names to succeeding convs/linears
        per_layer_normalization (bool): normalized by layer??

    Returns:
        pruned_weights (OrderedDict): pruned model weights
    """

    # find all scale weights in BN layers
    scale_weights = []
    norm_layer_names = list(set(succ_layers) & set(prec_layers))
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        weight = weights[norm_weight_name]
        if per_layer_normalization:
            scale_weights.extend([(_.abs() / weight.sum()).item() for _ in list(weight)])
        else:
            scale_weights.extend([_.abs().item() for _ in list(weight)])

    # find threshold for pruning
    scale_weights.sort()
    prune_th_index = int(float(len(scale_weights)) * prune_ratio + 0.5)
    prune_th = scale_weights[prune_th_index]

    # unpruned_norm_layer_names = list(set(succ_layers) ^ set(prec_layers))
    grouped_weight_names = group_weight_names(weights.keys())
    eliminated_layers = list()
    for norm_layer_name in norm_layer_names:
        norm_weight_name = norm_layer_name + WEIGHT_POSTFIX
        scale_weight = weights[norm_weight_name].abs()
        if per_layer_normalization:
            scale_weight = scale_weight / scale_weight.sum()
        keep_mask = scale_weight > prune_th
        if keep_mask.sum().item() == scale_weight.size(0):
            continue

        # in case not to prune the whole layer
        scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
        if keep_mask.sum().item() < len(scale_weight_list) // 2:
            scale_weight_list.sort(reverse=True)
            mid_index = len(scale_weight_list) // 2
            keep_mask = scale_weight > (scale_weight_list[mid_index]+scale_weight_list[mid_index+1]) / 2
            print("Warning: More then half channels pruned for {}! Keep the largest half ...".format(norm_layer_name))
            eliminated_layers.append(norm_layer_name)

        prune_filters(norm_layer_name, keep_mask, weights, grouped_weight_names, prec_layers, succ_layers)

    return weights, prune_th, eliminated_layers


def _dirty_fix(module, param_name, pruned_shape):
    module_param = getattr(module, param_name)

    # identify the dimension to prune
    pruned_dim = 0
    for original_size, pruned_size in zip(module_param.shape, pruned_shape):
        if original_size != pruned_size:
            keep_indices = torch.LongTensor(range(pruned_size)).to(module_param.data.device)
            module_param.data = module_param.data.index_select(pruned_dim, keep_indices)

            # modify number of features/channels
            if param_name == "weight":
                if isinstance(module, nn.modules.batchnorm._BatchNorm) or \
                        isinstance(module, nn.modules.instancenorm._InstanceNorm) or \
                        isinstance(module, nn.GroupNorm):
                    module.num_features = pruned_size
                elif isinstance(module, nn.modules.conv._ConvNd):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_channels = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_channels = pruned_size
                elif isinstance(module, nn.Linear):
                    if pruned_dim == OUT_CHANNEL_DIM:
                        module.out_features = pruned_size
                    elif pruned_dim == IN_CHANNEL_DIM:
                        module.in_features = pruned_size
                else:
                    pass
        pruned_dim += 1


def load_pruned_model(model, pruned_weights, prefix='', load_pruned_weights=True, inplace=True):
    """load pruned weights to a unpruned model instance

    Arguments:
        model (pytorch model): the model instance
        pruned_weights (OrderedDict): pruned weights
        prefix (string optional): prefix (if has) of pruned weights
        load_pruned_weights (bool optional): load pruned weights to model according to the ICLR 2019 paper:
            "Rethinking the Value of Network Pruning", without finetuning, the model may achieve comparable or even
            better results
        inplace (bool, optional): if return a copy of the model

    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)
    """
    # inplace or return a new copy
    if not inplace:
        pruned_model = copy.deepcopy(model)
    else:
        pruned_model = model

    model_weight_names = pruned_model.state_dict().keys()
    pruned_weight_names = pruned_weights.keys()

    # patch for channel selection
    for model_weight_name in model_weight_names:
        model_weight_name = prefix + model_weight_name
        if model_weight_name not in pruned_weight_names:
            tokens = model_weight_name.split('.')
            tokens.insert(-1, "bn")
            bn_masked_model_weight_name = '.'.join(tokens)
            if bn_masked_model_weight_name in pruned_weight_names and bn_masked_model_weight_name.endswith(
                    WEIGHT_POSTFIX):
                nch = len(pruned_weights[bn_masked_model_weight_name])
                *container_names, module_name, param_name = model_weight_name.split('.')
                container = pruned_model
                for container_name in container_names:
                    container = container._modules[container_name]
                module = container._modules[module_name]
                container._modules[module_name] = MaskedBatchNorm(module, torch.ones(nch).bool())

    # check if module names match
    model_weight_names = pruned_model.state_dict().keys()
    assert set([prefix + _ for _ in model_weight_names]) == set(pruned_weight_names)

    # update modules with mis-matched weight
    model_weights = pruned_model.state_dict()
    for model_weight_name in model_weight_names:
        if model_weights[model_weight_name].shape != pruned_weights[prefix + model_weight_name].shape:
            *container_names, module_name, param_name = model_weight_name.split('.')
            container = pruned_model
            for container_name in container_names:
                container = container._modules[container_name]
            module = container._modules[module_name]
            _dirty_fix(module, param_name, pruned_weights[prefix + model_weight_name].shape)
    if load_pruned_weights:
        pruned_model.load_state_dict({k: v for k, v in pruned_weights.items()})
    return pruned_model


def replace_with_masked_norms(model, succ_layers, prune_th=None, inplace=True):
    """replace BN with masked BN and indices

    Arguments:
        model (pytorch model): the model instance
        succ_layers (Dict): BN names and corresponding convs/linears
        prune_th: threshold for pruning, 1: optimal prune, 0~1: network slimming
        inplace (bool, optional): if return a copy of the model

    Returns:
        a model instance with masked BNs
    """

    if not inplace:
        model = copy.deepcopy(model)

    weights = model.state_dict()
    grouped_weight_names = group_weight_names(weights.keys())

    for norm_layer_name in succ_layers:

        # 0. Calculate threshold for pruning
        scale_weight_name = norm_layer_name + WEIGHT_POSTFIX
        scale_weight = weights[scale_weight_name].abs()
        if not prune_th:
            prune_th = css_thresholding([_ for _ in scale_weight])
            keep_mask = scale_weight > prune_th
            prune_th = None
        else:
            keep_mask = scale_weight > prune_th
            # In case of zero channels, for skip connections, use post calculation for params and flops
            if keep_mask.sum() < 1:
                scale_weight_list = [_.abs().item() for _ in list(scale_weight)]
                scale_weight_list.sort(reverse=True)
                keep_mask = scale_weight > (scale_weight_list[0]+scale_weight_list[1]) / 2
                print("Warning: Zero channels after pruning {}! The channel with the largest scaling factor is kept ...".format(norm_layer_name))
            elif keep_mask.sum() == 1:
                print("Warning: One channel left for {}!".format(norm_layer_name))

        *container_names, module_name, param_name = scale_weight_name.split('.')
        container = model
        for container_name in container_names:
            container = container._modules[container_name]
        module = container._modules[module_name]
        container._modules[module_name] = MaskedBatchNorm(module, keep_mask)

        # 2. prune target succeeding conv/linear/... layers
        keep_indices = torch.nonzero(keep_mask).flatten()
        for prune_layer_name in succ_layers[norm_layer_name]:
            for weight_name in grouped_weight_names[prune_layer_name]:
                if weight_name.endswith(WEIGHT_POSTFIX):
                    # Prune weight & assign to module
                    *container_names, module_name, param_name = weight_name.split('.')
                    container = model
                    for container_name in container_names:
                        container = container._modules[container_name]
                    module = container._modules[module_name]
                    module_param = getattr(module, param_name)
                    module_param.data = weights[weight_name].data.index_select(IN_CHANNEL_DIM, keep_indices)
                    pruned_size = len(keep_indices)

                    # Update num of features for pruned module
                    if isinstance(module, nn.modules.conv._ConvNd):
                        module.in_channels = pruned_size
                    elif isinstance(module, nn.Linear):
                        module.in_features = pruned_size
                    else:
                        pass

    return model


def prune(model, input_shape, prune_method=network_slimming, prune_ratio=0.3, channel_select=True):
    """prune a model

    Arguments:
        model (pytorch model): the model instance
        input_shape (tuple): shape of the input tensor
        prune_ratio (float): ratio of be pruned channels to total channels
        prune_method (method): algorithm to prune weights

    Returns:
        a model instance with pruned structure (and weights if load_pruned_weights==True)

    Pipeline:
        1. generate mapping from tensors connected to BNs by parsing torch script traced graph
        2. identify corresponding BN and conv/linear like:
            conv/linear --> ... --> BN --> ... --> conv/linear
                                     |
                                    ...
                                     | --> relu --> ... --> conv/linear
                                    ...
                                     | --> ... --> maxpool --> ... --> conv/linear
            , where ... represents per channel operations. all the floating nodes must be conv/linear
        3. prune the weights of BN and connected conv/linear
        4. patch with masked BN for channel selection
        5. load weights to a unpruned model with pruned weights
    """
    # convert to CPU for simplicity
    src_device = next(model.parameters()).device
    model = model.cpu()

    # parse & generate mappings to BN layers
    prec_layers, succ_layers, bn_names = get_pruning_layers(model, input_shape)

    # prune weights
    pruned_weights, prune_th, eliminated_layers = prune_method(model.state_dict(), prune_ratio, prec_layers, succ_layers)

    # prune model according to pruned weights
    pruned_model = load_pruned_model(model, pruned_weights)

    # channel selection
    if channel_select:
        prunable_bn_layer_names = list(set(prec_layers) & set(succ_layers))
        for bn_name in prunable_bn_layer_names:
            succ_layers.pop(bn_name)

        pruned_model = replace_with_masked_norms(pruned_model, succ_layers, prune_th)

    return pruned_model.to(src_device)#, eliminated_layers
