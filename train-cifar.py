import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import cifar

from networks import cifar_archs as archs
from networks import weights_init
from netslim import prune, load_pruned_model, update_bn, \
    update_bn_by_names, network_slimming, get_norm_layer_names
from utils import parse_cifar_args as parse_args
from utils import ns_post_process_resnet, ns_post_process_densenet

import torch.backends.cudnn as cudnn


if __name__ == "__main__":
    # Training settings
    args = parse_args()
    print(args)
    os.system('mkdir -p {}'.format(args.output))
    device = torch.device('cuda' if args.cuda else 'cpu')

    if args.seed > 0:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
    
    # Make data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    normalize = transforms.Normalize(mean=[0.4914, 0.482, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    make_dataset = cifar.CIFAR10 if args.cifar10 else cifar.CIFAR100
    data_path = args.data_path[:-1] if args.cifar10 else args.data_path
    
    train_loader = torch.utils.data.DataLoader(
        make_dataset(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           normalize
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        make_dataset(data_path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           normalize
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = archs[args.arch](num_classes=args.num_classes)
    if args.prunable_layers_only:
        prune_bn_names, mask_bn_names, bn_names = get_norm_layer_names(model, (3, 32, 32))
        sparse_dict = {bn_name: args.sparsity for bn_name in prune_bn_names}
    model = model.to(device)

    # prune related settings
    if args.resume_path:
        try:
            model.load_state_dict(torch.load(args.resume_path, map_location=(None if args.cuda else "cpu")))
        except:
            print("Cannot load state_dict directly, trying to load pruned weight ...")
            model = load_pruned_model(model, torch.load(args.resume_path, map_location=(None if args.cuda else "cpu")))
        channel_select = False if args.prunable_layers_only else True
        if args.prune_ratio > 0.999:
            model = prune(model, (3, 32, 32), channel_select=channel_select)
        elif args.prune_ratio > 0:
            model = prune(model, (3, 32, 32), prune_method=network_slimming, prune_ratio=args.prune_ratio, channel_select=channel_select)
        
        if "resnet" in args.arch:
            if args.prune_ratio > 0.999:
                pass # OT to be updated
            elif args.prune_ratio > 0:
                ns_post_process_resnet(model, args.prune_ratio, (3, 32, 32), device)
        
        if "densenet" in args.arch:
            if args.prune_ratio > 0.999:
                pass # OT to be updated
            elif args.prune_ratio > 0:
                ns_post_process_densenet(model, args.prune_ratio, (3, 32, 32), device)

        if args.tfs:
            model.apply(weights_init)
            print("Train pruned model from scratch ...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        nesterov=True, 
        weight_decay=args.weight_decay
    )
    
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if args.prunable_layers_only:
                update_bn_by_names(model, sparse_dict)
            else:
                update_bn(model, args.sparsity)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def test(epoch, test_name=''):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]   # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        accuracy = 100. * float(correct) / float(len(test_loader.dataset))
        print('\nTest Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset), accuracy))

        with open('{}/{}.log'.format(args.output, test_name if test_name else "training"), 'a') as f:
            f.write('{}\t{}\n'.format(epoch, accuracy))

        return accuracy
    
    lr = args.lr
    lr_decay_schedule = [int(_*args.epochs+0.5) for _ in args.lr_decay_schedule]
    max_accuracy = 0.
    if args.test_at_0:
        test(0, "before_training")
    for epoch in range(args.epochs):
        if epoch in lr_decay_schedule:
            lr *= args.lr_decay_factor
            print('Changing learning rate to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train(epoch)
        accuracy = test(epoch)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model.state_dict(), '{}/ckpt_best.pth'.format(args.output))
        torch.save(model.state_dict(), '{}/ckpt_last.pth'.format(args.output))
