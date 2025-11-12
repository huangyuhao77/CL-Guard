import torch
import torch.nn as nn
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


class VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        fc_part1 = nn.Linear(512 * 7 * 7, 4096)
        fc_all1 = nn.Linear(512 * 7 * 7, 4096)
        fc_part2 = nn.Linear(4096, 4096)
        fc_all2 = nn.Linear(4096, 4096)
        fc_part3 = nn.Linear(4096, num_classes)
        fc_all3 = nn.Linear(4096, num_classes)
        self.classifier_part = nn.Sequential(
            fc_part1,
            nn.ReLU(True),
            nn.Dropout(),
            fc_part2,
            nn.ReLU(True),
            nn.Dropout(),
            fc_part3,
        )
        self.classifier_all = nn.Sequential(
            fc_all1,
            nn.ReLU(True),
            nn.Dropout(),
            fc_all2,
            nn.ReLU(True),
            nn.Dropout(),
            fc_all3,
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor, type_value, feature_value) -> torch.Tensor:
        for layer in self.features:
            if "bn_part" in layer.__class__.__name__:
                if type_value == 0 or type_value == 2:
                    x = layer(x)
                    continue
            elif 'MaskConv2d' in layer.__class__.__name__:
                layer.type_value = type_value
                layer.feature_value = feature_value
                x = layer(x)
            else:
                x = layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if type_value == 0 or type_value == 2 or type_value == 7:
            x = self.classifier_part(x)
        else:
            x = self.classifier_all(x)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, norm_layer = None) -> nn.Sequential:
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = mnn.MaskConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                bn_part = norm_layer(v)
                bn_all = norm_layer(v)
                layers += [conv2d, bn_part, bn_all, nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg(**kwargs: Any) -> VGG:
    num_layers = str(kwargs.get('num_layers'))
    num_classes = int(kwargs.get('num_classes'))

    # set pruner
    global mnn
    mnn = kwargs.get('mnn')
    assert mnn is not None, "Please specify proper pruning method"

    if num_layers == '11':
        model = VGG(make_layers(cfgs['A'], batch_norm=False), num_classes=num_classes)
    elif num_layers == '11_bn':
        model = VGG(make_layers(cfgs['A'], batch_norm=True), num_classes=num_classes)
    elif num_layers == '13':
        model = VGG(make_layers(cfgs['B'], batch_norm=False), num_classes=num_classes)
    elif num_layers == '13_bn':
        model = VGG(make_layers(cfgs['B'], batch_norm=True), num_classes=num_classes)
    elif num_layers == '16':
        model = VGG(make_layers(cfgs['D'], batch_norm=False), num_classes=num_classes)
    elif num_layers == '16_bn':
        model = VGG(make_layers(cfgs['D'], batch_norm=True), num_classes=num_classes)
    elif num_layers == '19':
        model = VGG(make_layers(cfgs['E'], batch_norm=False), num_classes=num_classes)
    elif num_layers == '19_bn':
        model = VGG(make_layers(cfgs['E'], batch_norm=True), num_classes=num_classes)
    else:
        assert 'Please check the num_layer you input'
    return model

