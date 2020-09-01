import math
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import _resnet, Bottleneck, BasicBlock, conv3x3, conv1x1
from netslim import MaskedBatchNorm


def weights_init(m):
    if isinstance(m, nn.Conv2d) and m.weight.data.abs().sum().item() > 0:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.weight.data.abs().sum().item() > 0:
            m.weight.data.fill_(0.5)
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear) and m.weight.data.abs().sum().item() > 0:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
    elif isinstance(m, MaskedBatchNorm) and m.bn.weight.data.abs().sum().item() > 0:
        m.bn.weight.data.fill_(0.5)
        m.bn.bias.data.zero_()


def deactivate(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.zero_()
        if m.bias is not None:
            m.bias = None
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        m.weight.data.zero_()
        m.bias = None
    elif isinstance(m, nn.Linear):
        m.weight.data.zero_()
        if m.bias is not None:
            m.bias = None
        

def vgg14c(num_classes=100):
    """Constructs a VGG-16 simplified model for CIFAR dataset"""
    model = models.vgg16_bn()
    model.avgpool = nn.Identity()
    model.classifier = nn.Linear(512, num_classes)
    model.apply(weights_init)
    return model


def resnet50c(num_classes=100):
    """Constructs a ResNet-50 model for CIFAR dataset"""
    model = models.resnet50(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.avgpool = nn.AvgPool2d(4, stride=1)
    model.maxpool = nn.Identity()
    model.apply(weights_init)
    return model


def densenet121c(num_classes=100):
    """Constructs a DenseNet-121 model for CIFAR dataset"""
    model = models.densenet121(num_classes=num_classes)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[3] = nn.Identity()
    #model.classifier = nn.Linear(1024, num_classes)
    model.apply(weights_init)
    return model


def vgg9c(num_classes=100):
    """Constructs a VGG-11 simplified model for CIFAR dataset"""
    model = models.vgg11_bn()
    model.avgpool = nn.Identity()
    model.classifier = nn.Linear(512, num_classes)
    model.apply(weights_init)
    return model


def resnet18c(num_classes=100):
    """Constructs a ResNet-18 model for CIFAR dataset"""
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.avgpool = nn.AvgPool2d(4, stride=1)
    model.maxpool = nn.Identity()
    model.apply(weights_init)
    return model


def densenet63c(num_classes=100):
    """Constructs a DenseNet-63 simplified model for CIFAR dataset"""
    num_init_features = 32
    model = models.densenet._densenet('densenet63', 32, (3, 6, 12, 8), num_init_features, pretrained=False, progress=False)
    model.features[0] = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[1] = nn.BatchNorm2d(num_init_features)
    model.features[3] = nn.Identity()
    model.classifier = nn.Linear(512, num_classes)
    model.apply(weights_init)
    return model


class CifarResNet20(nn.Module):

    def __init__(self, block, layers, num_classes=10, groups=1, width_per_group=64):
        super(CifarResNet20, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 16
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv3x3(3, self.inplanes)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet20c(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR dataset"""
    model = CifarResNet20(BasicBlock, [3, 3, 3], num_classes=num_classes)
    model.apply(weights_init)
    return model


cifar_archs = {
    "vgg14": vgg14c, 
    "vgg16": vgg14c, 
    "resnet50": resnet50c, 
    "densenet121": densenet121c, 
    "vgg9": vgg9c, 
    "vgg11": vgg9c, 
    "resnet18": resnet18c, 
    "densenet63": densenet63c, 
    "resnet20": resnet20c
}


def vgg16():
    """Constructs a VGG-16 model for ILSVRC12 dataset"""
    model = models.vgg16_bn()
    model.classifier[2] = nn.BatchNorm1d(4096)
    model.classifier[5] = nn.BatchNorm1d(4096)
    model.apply(weights_init)
    return model


def vgg19():
    """Constructs a VGG-19 model for ILSVRC12 dataset"""
    model = models.vgg19_bn()
    model.classifier[2] = nn.BatchNorm1d(4096)
    model.classifier[5] = nn.BatchNorm1d(4096)
    model.apply(weights_init)
    return model


def resnet50():
    """Constructs a ResNet-50 model for ILSVRC12 dataset"""
    model = models.resnet50()
    model.apply(weights_init)
    return model


def resnet101():
    """Constructs a ResNet-101 model for ILSVRC12 dataset"""
    model = models.resnet101()
    model.apply(weights_init)
    return model


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(PreActBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out
    

class PreActBottleneck(Bottleneck):
    """Bottleneck with pre-activation"""
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(PreActBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.bn1 = norm_layer(inplanes)
        self.bn3 = norm_layer(int(planes * (base_width / 64.)) * groups)
        
    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


def preresnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a PreResNet-50 model for ILSVRC12 dataset"""
    model = _resnet('preresnet18', PreActBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
    model.apply(weights_init)
    return model


def preresnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a PreResNet-50 model for ILSVRC12 dataset"""
    model = _resnet('preresnet34', PreActBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)
    model.apply(weights_init)
    return model

    
def preresnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a PreResNet-50 model for ILSVRC12 dataset"""
    model = _resnet('preresnet50', PreActBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    model.apply(weights_init)
    return model


def preresnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a PreResNet-101 model for ILSVRC12 dataset"""
    model = _resnet('preresnet101', PreActBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
    model.apply(weights_init)
    return model


ilsvrc12_archs = {
    "vgg16": vgg16,
    "vgg19": vgg19, 
    "resnet50": resnet50, 
    "resnet101": resnet101, 
    "preresnet18": preresnet18, 
    "preresnet34": preresnet34,
    "preresnet50": preresnet50, 
    "preresnet101": preresnet101
}


if __name__ == "__main__":
    from torchsummary import summary
    #model = vgg14c(num_classes=100)
    #model = resnet50c(num_classes=100)
    #model = vgg14c()
    #model = densenet121c()
    model = densenet121c(100)
    summary(model, (3, 32, 32), device="cpu")
    #from thop import profile
    from thop_res import profile
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), verbose=False)
    print("FLOPs: {:,}, Params: {:,}".format(int(flops), int(params)))
    K = 1024
    M = K * K
    G = K * M
    flops_unit = 'K'
    if flops > G:
        flops /= G
        flops_unit = 'G'
    elif flops > M:
        flops /= M
        flops_unit = 'M'
    else:
        flops /= K
    params_unit = 'K'
    if params > G:
        params /= G
        params_unit = 'G'
    elif params > M:
        params /= M
        params_unit = 'M'
    else:
        params /= K
    print("FLOPs: {:.2f}{}, Params: {:.2f}{}".format(flops, flops_unit, params, params_unit))
    