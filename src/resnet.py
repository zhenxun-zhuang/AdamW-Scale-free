'''
Adapted from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet110 |   218  |  3.5M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['resnet']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', no_batch_norm=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if not no_batch_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.no_batch_norm = no_batch_norm

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                if not no_batch_norm:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion * planes,
                                  kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(self.expansion * planes)
                    )
                else:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_planes, self.expansion * planes,
                                  kernel_size=1, stride=stride, bias=False)
                    )

    def forward(self, x):
        if not self.no_batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, no_batch_norm=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.no_batch_norm = no_batch_norm

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if not no_batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, no_batch_norm=no_batch_norm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, no_batch_norm=no_batch_norm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, no_batch_norm=no_batch_norm)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, no_batch_norm):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                no_batch_norm=no_batch_norm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.no_batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet(depth, num_classes, no_batch_norm):
    """
    Constructs a ResNet model.
    """
    if depth == 20:
        num_blocks = [3, 3, 3]
    elif depth == 32:
        num_blocks = [5, 5, 5]
    elif depth == 44:
        num_blocks = [7, 7, 7]
    elif depth == 56:
        num_blocks = [9, 9, 9]
    elif depth == 110:
        num_blocks = [18, 18, 18]
    elif depth == 218:
        num_blocks = [36, 36, 36]
    elif depth == 1202:
        num_blocks = [200, 200, 200]
    else:
        raise ValueError(f'Invalid depth {depth} for the ResNet model!')
    return ResNet(block=BasicBlock,
                  num_blocks=num_blocks,
                  num_classes=num_classes,
                  no_batch_norm=no_batch_norm)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for depth in [20, 32, 44, 56, 110, 218, 1202]:
        model = resnet(depth, num_classes=10, no_batch_norm=False)
        print(f"resnet{depth}")
        test(model)
        print()
