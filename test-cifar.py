import os
import time
import argparse
import torch
from torchvision import transforms
from torchvision.datasets import cifar

from networks import cifar_archs as archs
from netslim import load_pruned_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Cifar Example for Test Pruned Model')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 50)')
parser.add_argument('--resume-path', default='',
                    help='path to a trained model weight')
parser.add_argument('--arch', default='resnet18',
                    help='network architecture')
parser.add_argument('--cifar10', action='store_true', default=False, 
                    help='Train CIFAR-10, by default the script train cifar-100')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device('cuda' if args.cuda else 'cpu')

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
normalize = transforms.Normalize(mean=[0.4914, 0.482, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

dataset, datapath = (cifar.CIFAR10, "./datasets/cifar-10") if args.cifar10 else (cifar.CIFAR100, "./datasets/cifar-100")

test_loader = torch.utils.data.DataLoader(
    dataset(datapath, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = archs[args.arch](num_classes=10 if args.cifar10 else 100)
pruned_weights = torch.load(args.resume_path)
model = load_pruned_model(model, pruned_weights).to(device)

model.eval()
correct = 0
with torch.no_grad():
    t_start = time.perf_counter()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]   # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    t_all = time.perf_counter() - t_start

accuracy = 100. * float(correct) / float(len(test_loader.dataset))
print("Accuracy: {}/{} ({:.2f}%)".format(correct, len(test_loader.dataset), accuracy))
print("Total time: {:.2f} s".format(t_all))
#if args.test_batch_size == 1:
#    print("Estimated FPS: {:.2f}".format(1/(t_all/len(test_loader))))
