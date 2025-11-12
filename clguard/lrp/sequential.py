import torch

from . import Linear, Conv2d
from .maxpool import MaxPool2d
from .functional.utils import normalize

import torch.nn.functional as F
def grad_decorator_fn(module):
    """
        Currently not used but can be used for debugging purposes.
    """

    def fn(x):
        return normalize(x)

    return fn


avoid_normalization_on = ['relu', 'maxp']


def do_normalization(rule, module):
    if "pattern" not in rule.lower(): return False
    return not str(module)[:4].lower() in avoid_normalization_on


def is_kernel_layer(module):
    return isinstance(module, Conv2d) or isinstance(module, Linear) or isinstance(module, Bottleneck) or isinstance(
        module, BasicBlock) or isinstance(module,PreActBlock) or isinstance(module,PreActBottleneck)


def is_rule_specific_layer(module):
    return isinstance(module, MaxPool2d)


class Sequential(torch.nn.Sequential):
    def forward(self, input, explain=False, rule="epsilon", pattern=None):
        if not explain: return super(Sequential, self).forward(input)

        first = True

        # copy references for user to be able to reuse patterns
        if pattern is not None: pattern = list(pattern)

        for module in self:
            if do_normalization(rule, module):
                input.register_hook(grad_decorator_fn(module))

            if is_kernel_layer(module):
                P = None
                if pattern is not None:
                    P = pattern.pop(0)
                input = module.forward(input, explain=True, rule=rule, pattern=P)

            elif is_rule_specific_layer(module):
                input = module.forward(input, explain=True, rule=rule)

            else:  # Use gradient as default for remaining layer types
                input = module(input)
            first = False

        if do_normalization(rule, module):
            input.register_hook(grad_decorator_fn(module))

        return input


class Bottleneck(torch.nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        self.downsample = None

    def forward(self, x, explain=True, rule="epsilon", pattern=None):
        identity = x

        if pattern is not None:
            out = self.conv1(x, explain=explain, rule=rule, pattern=pattern[0])
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out, explain=explain, rule=rule, pattern=pattern[1])
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out, explain=explain, rule=rule, pattern=pattern[2])
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample[0](x, explain, rule, pattern=pattern[3])
                identity = self.downsample[1](identity)
        else:
            out = self.conv1(x, explain=explain, rule=rule)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out, explain=explain, rule=rule)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out, explain=explain, rule=rule)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample[0](x, explain, rule)
                identity = self.downsample[1](identity)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.downsample = None

    def forward(self, x, explain=True, rule="epsilon", pattern=None):
        identity = x

        if pattern is not None:
            out = self.conv1(x, explain=explain, rule=rule, pattern=pattern[0])
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out, explain=explain, rule=rule, pattern=pattern[1])
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample[0](x, explain, rule, pattern=pattern[2])
                identity = self.downsample[1](identity)
        else:
            out = self.conv1(x, explain=explain, rule=rule)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out, explain=explain, rule=rule)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample[0](x, explain, rule)
                identity = self.downsample[1](identity)

        out += identity
        out = self.relu(out)

        return out

class PreActBlock(torch.nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self):
        super(PreActBlock, self).__init__()
        self.shortcut = None

    def forward(self, x, explain=True, rule="epsilon", pattern=None):
        identity = x
        if pattern is not None:
            out = F.relu(self.bn1(x))
            out = self.conv1(out, explain=explain, rule=rule, pattern=pattern[0])
            out = F.relu(self.bn2(out))
            out = self.conv2(out, explain=explain, rule=rule, pattern=pattern[1])
            if self.shortcut is not None:
                identity = self.shortcut[0](identity, explain, rule, pattern=pattern[2])
        else:
            out = F.relu(self.bn1(x))
            out = self.conv1(out, explain=explain, rule=rule)
            out = F.relu(self.bn2(out))
            out = self.conv2(out, explain=explain, rule=rule)
            if self.shortcut is not None:
                identity = self.shortcut[0](identity, explain, rule)

        out += identity
        return out


class PreActBottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self):
        super(PreActBottleneck, self).__init__()
        self.shortcut = None

    def forward(self, x, explain=True, rule="epsilon", pattern=None):
        identity = x
        if pattern is not None:
            out = F.relu(self.bn1(x))
            out = self.conv1(out, explain=explain, rule=rule, pattern=pattern[0])
            out = F.relu(self.bn2(out))
            out = self.conv2(out, explain=explain, rule=rule, pattern=pattern[1])
            out = F.relu(self.bn3(out))
            out = self.conv3(out, explain=explain, rule=rule, pattern=pattern[2])
            if self.shortcut is not None:
                identity = self.shortcut[0](identity, explain, rule, pattern=pattern[3])
        else:
            out = F.relu(self.bn1(x))
            out = self.conv1(out, explain=explain, rule=rule)
            out = F.relu(self.bn2(out))
            out = self.conv2(out, explain=explain, rule=rule)
            out = F.relu(self.bn3(out))
            out = self.conv3(out, explain=explain, rule=rule)
            if self.shortcut is not None:
                identity = self.shortcut[0](identity, explain, rule)

        out += identity
        return out
