import torch
import torchvision

import models.preact_resnet
from .conv       import Conv2d
from .linear     import Linear
from .sequential import Sequential, Bottleneck, BasicBlock, PreActBlock, PreActBottleneck

conversion_table = {
        'Linear': Linear,
        'Conv2d': Conv2d
    }

# # # # # Convert torch.models.vggxx to lrp model
def convert_vgg(module, modules=None):
    # First time
    if modules is None:
        modules = []
        for m in module.children():
            convert_vgg(m, modules=modules)

            # Vgg model has a flatten, which is not represented as a module
            # so this loop doesn't pick it up.
            # This is a hack to make things work.
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, torch.nn.Sequential):
        for m in module.children():
            convert_vgg(m, modules=modules)

    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)
    # maxpool is handled with gradient for the moment

    elif isinstance(module, torch.nn.ReLU):
        # avoid inplace operations. They might ruin PatternNet pattern
        # computations
        modules.append(torch.nn.ReLU())
    else:
        modules.append(module)


# # # # # Convert torch.models.resnetxx to lrp model
def convert_resnet(module, modules=None):
    # First time
    if modules is None:
        modules = []
        for m in module.children():
            convert_resnet(m, modules=modules)

            # Vgg model has a flatten, which is not represented as a module
            # so this loop doesn't pick it up.
            # This is a hack to make things work.
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, torch.nn.Sequential):
        for m in module.children():
            convert_resnet(m, modules=modules)

    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)
    # maxpool is handled with gradient for the moment

    elif isinstance(module, torch.nn.ReLU):
        # avoid inplace operations. They might ruin PatternNet pattern
        # computations
        modules.append(torch.nn.ReLU())
    elif isinstance(module, torchvision.models.resnet.Bottleneck):
        # For torchvision Bottleneck
        bottleneck = Bottleneck()
        bottleneck.conv1 = Conv2d.from_torch(module.conv1)
        bottleneck.conv2 = Conv2d.from_torch(module.conv2)
        bottleneck.conv3 = Conv2d.from_torch(module.conv3)
        bottleneck.bn1 = module.bn1
        bottleneck.bn2 = module.bn2
        bottleneck.bn3 = module.bn3
        bottleneck.relu = torch.nn.ReLU()
        if module.downsample is not None:
            bottleneck.downsample = module.downsample
            bottleneck.downsample[0] = Conv2d.from_torch(module.downsample[0])
        modules.append(bottleneck)
    elif isinstance(module, torchvision.models.resnet.BasicBlock):
        # For torchvision BasicBlock
        basicblock = BasicBlock()
        basicblock.conv1 = Conv2d.from_torch(module.conv1)
        basicblock.conv2 = Conv2d.from_torch(module.conv2)
        basicblock.bn1 = module.bn1
        basicblock.bn2 = module.bn2
        basicblock.relu = torch.nn.ReLU()
        if module.downsample is not None:
            basicblock.downsample = module.downsample
            basicblock.downsample[0] = Conv2d.from_torch(module.downsample[0])
        modules.append(basicblock)
    else:
        modules.append(module)

# # # # # Convert preactresnet to lrp model
def convert_preactresnet(module, modules=None):
    # First time
    if modules is None:
        modules = []
        for m in module.children():
            convert_preactresnet(m, modules=modules)

            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                modules.append(torch.nn.Flatten())

        sequential = Sequential(*modules)
        return sequential

    # Recursion
    if isinstance(module, torch.nn.Sequential):
        for m in module.children():
            convert_preactresnet(m, modules=modules)

    elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
        class_name = module.__class__.__name__
        lrp_module = conversion_table[class_name].from_torch(module)
        modules.append(lrp_module)
    # maxpool is handled with gradient for the moment

    elif isinstance(module, torch.nn.ReLU):
        # avoid inplace operations. They might ruin PatternNet pattern
        # computations
        modules.append(torch.nn.ReLU())
    elif isinstance(module, models.preact_resnet.PreActBlock):
        # For torchvision BasicBlock
        preactBlock = PreActBlock()

        preactBlock.bn1 = module.bn1
        preactBlock.conv1 = Conv2d.from_torch(module.conv1)

        preactBlock.bn2 = module.bn2
        preactBlock.conv2 = Conv2d.from_torch(module.conv2)
        if module.shortcut is not None:
            preactBlock.shortcut = module.shortcut
            preactBlock.shortcut[0] = Conv2d.from_torch(module.shortcut[0])
        modules.append(preactBlock)
    elif isinstance(module, models.preact_resnet.PreActBottleneck):
        # For torchvision BasicBlock
        preactBottleneck = PreActBottleneck()

        preactBottleneck.bn1 = module.bn1
        if module.shortcut is not None:
            preactBottleneck.shortcut = module.shortcut
            preactBottleneck.shortcut[0] = Conv2d.from_torch(module.shortcut[0])
        preactBottleneck.conv1 = Conv2d.from_torch(module.conv1)

        preactBottleneck.bn2 = module.bn2
        preactBottleneck.conv2 = Conv2d.from_torch(module.conv2)
        preactBottleneck.bn3 = module.bn3
        preactBottleneck.conv3 = Conv2d.from_torch(module.conv3)

        modules.append(preactBottleneck)

    else:
        modules.append(module)

