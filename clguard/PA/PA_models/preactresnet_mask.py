import torch
import torch.nn as nn
import torch.nn.functional as F
from clguard.PA.dcil import mnn


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return mnn.MaskConv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups, bias=bias, padding=padding)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return mnn.MaskConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.shortcut = None
        # 0 -> part use, 1-> full use
        self.type_value = 0
        self.feature_value = 0
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1_full = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_full = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1, padding=1, bias=False)
        self.ind = None

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
               conv1x1(in_planes, self.expansion * planes, stride=stride, bias=False)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(x))
        # switch the bn
        identity = x
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn1(x)
        else:
            out = self.bn1_full(x)
        out = F.relu(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = self.conv1(out)
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn2(out)
        else:
            out = self.bn2_full(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.ind is not None:
            out += identity[:, self.ind, :, :]
        else:
            out += identity
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        # 0 -> part use, 1-> full use
        self.type_value = 0

        self.feature_value = 0  # 是否掩盖特征

        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1_full = nn.BatchNorm2d(in_planes)
        self.conv1 = conv1x1(in_planes, planes, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_full = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn3_full = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion * planes, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride, bias=False)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(x))
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn1(x)
        else:
            out = self.bn1_full(x)
        out = F.relu(out)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        # out = self.conv2(F.relu(self.bn2(out)))
        if self.type_value == 0 or self.type_value == 2:
            out = self.bn2(out)
        else:
            out = self.bn2_full(out)
        out = F.relu(out)
        out = self.conv2(out)

        if self.type_value == 0 or self.type_value == 2:
            out = self.bn2(out)
        else:
            out = self.bn2_full(out)
        out = F.relu(out)
        # out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = mnn.MaskConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.classifier_part = nn.Linear(512 * block.expansion, num_classes)
        self.classifier_all = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, type_value, feature_value):

        for m in self.modules():
            if isinstance(m, PreActBlock):
                m.type_value = type_value
                m.feature_value = feature_value
            if isinstance(m, PreActBottleneck):
                m.type_value = type_value
                m.feature_value = feature_value
            if isinstance(m, mnn.MaskConv2d):
                m.type_value = type_value
                m.feature_value = feature_value

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # type 7 is sharing the fc
        if type_value == 0 or type_value == 2 or type_value == 7:
            out = self.classifier_part(out)
        else:
            out = self.classifier_all(out)
        return out

# Model configurations
cfgs = {
    '18':  (PreActBlock, [2, 2, 2, 2]),
    '34':  (PreActBlock, [3, 4, 6, 3]),
    '50':  (PreActBottleneck, [3, 4, 6, 3]),
    '101': (PreActBottleneck, [3, 4, 23, 3]),
    '152': (PreActBottleneck, [3, 8, 36, 3]),
}

def preactresnet(**kwargs):
    r"""ResNet models from "[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)"
    Args:
        data (str): the name of datasets
    """
    num_layers = str(kwargs.get('num_layers'))
    num_classes = int(kwargs.get('num_classes'))

    # set pruner
    global mnn
    mnn = kwargs.get('mnn')
    assert mnn is not None, "Please specify proper pruning method"
    if num_layers in cfgs.keys():
        block, layers = cfgs[num_layers]
        model = PreActResNet(block, layers, num_classes)
    return model

# def PreActResNet18(num_classes=10):
#     return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)
#
#
# def PreActResNet34():
#     return PreActResNet(PreActBlock, [3, 4, 6, 3])
#
#
# def PreActResNet50():
#     return PreActResNet(PreActBottleneck, [3, 4, 6, 3])
#
#
# def PreActResNet101():
#     return PreActResNet(PreActBottleneck, [3, 4, 23, 3])
#
#
# def PreActResNet152():
#     return PreActResNet(PreActBottleneck, [3, 8, 36, 3])