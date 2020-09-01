import os
import argparse
import time
import numpy
import torch

from networks import cifar_archs, ilsvrc12_archs
from netslim import load_pruned_model

num_classes = {
    "cifar10": 10, 
    "cifar100": 100, 
    "ilsvrc12": None
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='benchmark pruned model')
    parser.add_argument('resume', default='', help='path to a trained model weight')
    args = parser.parse_args()

    _, dataset, arch = args.resume.split(os.sep)[-3].split('-')
    
    if "vgg" not in arch:
        from thop_res import profile
    else:
        from thop import profile
    
    archs = cifar_archs if num_classes[dataset] else ilsvrc12_archs
    model = archs[arch](num_classes=num_classes[dataset]) if num_classes[dataset] else archs[arch]()

    try:
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    except:
        print("Cannot load state_dict directly, trying to load pruned weight ...")
        model = load_pruned_model(model, torch.load(args.resume, map_location="cpu"))

    model.eval()
    data_shape = (1, 3, 224, 224) if dataset == "ilsvrc12" else (1, 3, 32, 32)
    flops, params = profile(model, inputs=(torch.randn(*data_shape),), verbose=False)
    print("FLOPS: {:,}\nParams: {:,}".format(int(flops), int(params)))
    
    onnx_path = "{}-{}-{}.onnx".format(arch, dataset, int(flops))
    torch.onnx.export(model, torch.randn(*data_shape), onnx_path)
    
    """
    # Convert onnx to OpenVINO IR
    cmd_template = 'python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model {0} --input_shape "{1}" --data_type {3} --model_name {2}-{3}'
    cmd = cmd_template.format(
        onnx_path, 
        repr(input_shape), 
        "{}-{}".format(arch, dataset), 
        "FP16"
    )
    print("Converting to OpenVINO IR ...")
    os.system(cmd)
    #os.system("rm {}".format(onnx_path))
    """
