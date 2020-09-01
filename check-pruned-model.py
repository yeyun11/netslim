import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import cifar

from networks import cifar_archs, ilsvrc12_archs
from networks import weights_init
from netslim import prune, load_pruned_model, update_bn, update_bn_by_names, \
    network_slimming, global_optimal_thresholding, get_norm_layer_names

num_classes = {
    "cifar10": 10, 
    "cifar100": 100, 
    "ilsvrc12": None
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check pruned model')
    parser.add_argument('resume', default='',
                        help='path to a trained model weight')
    parser.add_argument('--pr', type=float, default=-1, metavar='PR',
                        help='ratio of pruned channels to total channels, -1: do not prune, 1: optimal prune')
    args = parser.parse_args()

    _, dataset, arch = args.resume.split(os.sep)[-3].split('-')
    input_shape = (3, 32, 32) if dataset != "ilsvrc12" else (3, 224, 224)
    
    if "vgg" not in arch:
        from thop_res import profile
    else:
        from thop import profile
    
    archs = cifar_archs if num_classes[dataset] else ilsvrc12_archs
    model = archs[arch](num_classes=num_classes[dataset]) if num_classes[dataset] else archs[arch]()

    # prune related settings
    if args.resume:
        try:
            model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        except:
            print("Cannot load state_dict directly, trying to load pruned weight ...")
            model = load_pruned_model(model, torch.load(args.resume, map_location="cpu"))
        channel_select = True
        if args.pr > 0.999:
            model = prune(model, input_shape, prune_ratio=args.pr, channel_select=channel_select)
        elif args.pr > 0:
            model = prune(model, input_shape, prune_method=network_slimming, prune_ratio=args.pr, channel_select=channel_select)

    #print(model)
    #model.eval()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
    #print("FLOPS: {:,}\nParams: {:,}".format(int(flops), int(params)))
    
    from torchsummary import summary
    summary(model, (3, 32, 32), device="cpu")
    
    print("FLOPS: {:.2f}M\nParams: {:.2f}M".format(flops/1024/1024, params/1024/1024))