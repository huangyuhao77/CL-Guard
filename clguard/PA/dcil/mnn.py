import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter



class Masker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class Masker_part(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad * mask, None



class Masker_full(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad*(1-mask), None




class Masker_dis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x*(1-mask)

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad*(1-mask), None



class Masker_full_use(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad, None


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias, padding_mode)
        self.mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)
        self.feature_m = nn.Parameter(requires_grad=False)
        # 0 -> part use, 1-> full use
        self.type_value = 0
        self.feature_value = 0

    def reset(self):
        init.ones_(self.feature_m)

    def forward(self, input):
        if self.feature_value == 0:
            # feature_value控制
            # 执行
            input_modified = input * self.feature_m
        else:
            input_modified = input

        if self.type_value == 0:
            masked_weight = Masker_part.apply(self.weight, self.mask)
        elif self.type_value == 2:
            masked_weight = Masker.apply(self.weight, self.mask)

        elif self.type_value == 3:
            masked_weight = Masker_dis.apply(self.weight, self.mask)
        elif self.type_value == 4:
            masked_weight = Masker_full_use.apply(self.weight, self.mask)
        else:
            masked_weight = Masker_full.apply(self.weight, self.mask)

        return super(MaskConv2d, self)._conv_forward(input_modified, masked_weight, self.bias)