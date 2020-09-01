import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets

from networks import ilsvrc12_archs as archs
from networks import weights_init
from netslim import prune, load_pruned_model, update_bn, update_bn_by_names, \
    network_slimming, get_norm_layer_names
from utils import parse_cifar_args as parse_args
from utils import AverageMeter

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

if __name__ == "__main__":
    # Training settings
    args = parse_args("ilsvrc12")
    print(args)
    os.system('mkdir -p {}'.format(args.output))
    device = torch.device('cuda' if args.cuda else 'cpu')

    if args.seed > 0:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    
    # Make data loader
    kwargs = {'num_workers': 10, 'pin_memory': False} if args.cuda else {}
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traindir = os.path.join(args.data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    valdir = os.path.join(args.data_path, 'val')
    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    model = archs[args.arch]()
    model = model.to(device)

    # prune related settings
    if args.resume_path:
        try:
            model.load_state_dict(torch.load(args.resume_path))
        except:
            print("Cannot load state_dict directly, trying to load pruned weight ...")
            model = load_pruned_model(model, torch.load(args.resume_path))
        if args.prune_ratio > 0.999:
            model = prune(model, (3, 224, 224), channel_select=channel_select)
        elif args.prune_ratio > 0:
            model = prune(model, (3, 224, 224), prune_method=network_slimming, prune_ratio=args.prune_ratio, channel_select=channel_select)
        
        if "resnet" in args.arch:
            if args.prune_ratio > 0.999:
                ot_post_process_resnet(model)
            elif args.prune_ratio > 0:
                ns_post_process_resnet(model, args.prune_ratio, (3, 224, 224), device)
        
        if "densenet" in args.arch:
            if args.prune_ratio > 0.999:
                ot_post_process_densenet(model)
            elif args.prune_ratio > 0:
                ns_post_process_densenet(model, args.prune_ratio, (3, 224, 224), device)
        
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
            update_bn(model, args.sparsity)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    def cal_accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
                
    def test(epoch, test_name=''):
        model.eval()
        top1 = AverageMeter("Acc@1", ":.4f")
        top5 = AverageMeter("Acc@5", ":.4f")
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.to(device), target.to(device)
                output = model(data)
                acc1, acc5 = cal_accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

        print('\nTest Accuracy | top1: {:.4f}%, top5: {:.4f}%\n'.format(top1.avg, top5.avg))
        with open('{}/{}.log'.format(args.output, test_name if test_name else "training"), 'a') as f:
            f.write('{}\t{}\t{}\n'.format(epoch, top1.avg, top5.avg))

        return top1.avg
    
    lr = args.lr
    lr_decay_schedule = [int(_*args.epochs+0.5) for _ in args.lr_decay_schedule]
    max_accuracy = 0.
    if args.test_at_0:
        test(0, "before_training")
    model = nn.DataParallel(model).to(device)
    for epoch in range(args.epochs):
        if epoch in lr_decay_schedule:
            lr *= args.lr_decay_factor
            print('Changing learning rate to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        train(epoch)
        if (epoch+1) % 10 == 0:
            accuracy = test(epoch)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                torch.save(model.module.state_dict(), '{}/ckpt_best.pth'.format(args.output))
        torch.save(model.module.state_dict(), '{}/ckpt_last.pth'.format(args.output))

