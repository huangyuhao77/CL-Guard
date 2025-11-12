from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_weight_threshold(model, rate, args):
    importance_all = None
    for name, item in model.named_parameters():
        # if len(item.size())==4 and 'mask' not in name:
        if ('conv' in name or 'downsample_p.0' in name or 'features' in name) and 'weight' in name:
            weights = item.data.view(-1).cpu()
            grads = item.grad.data.view(-1).cpu()
        if args.prune_imp == 'L1':
            importance = weights.abs().numpy()
        elif args.prune_imp == 'L2':
            importance = weights.pow(2).numpy()
        elif args.prune_imp == 'grad':
            importance = grads.abs().numpy()
        elif args.prune_imp == 'syn':
            importance = (weights * grads).abs().numpy()

        if importance_all is None:
            importance_all = importance
        else:
            importance_all = np.append(importance_all, importance)

    importance_all = np.sort(importance_all)[::-1]
    threshold = importance_all[int((len(importance_all) - 1) * rate)]
    return threshold


def weight_prune(model, threshold, args):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'weight' in name:
            key = name.replace('weight', 'mask')
            if key in state.keys():
                if args.prune_imp == 'L1':
                    mat = item.data.abs()
                elif args.prune_imp == 'L2':
                    mat = item.data.pow(2)
                elif args.prune_imp == 'grad':
                    mat = item.grad.data.abs()
                elif args.prune_imp == 'syn':
                    mat = (item.data * item.grad.data).abs()
                # state[key].data.copy_(torch.gt(mat, threshold).float())
                state[key].data.copy_(torch.ge(mat, threshold).float())


def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():
        if 'mask' in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        if ('conv' in name or 'downsample_p.0' in name or 'features' in name) and 'weight' in name:
            # if 'weight' in name or 'bias' in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (mask_nonzeros / total_weights) * 100
    return total_weights, mask_nonzeros, sparsity


def load_state_dict(net, orig_state_dict):
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'classifier_part' in k:
            p = k.replace('classifier_part', 'classifier')
            new_state_dict[k] = orig_state_dict[p]
        # elif 'classifier_all' in k:
        #     a = k.replace('classifier_all', 'classifier')
        #     new_state_dict[k] = orig_state_dict[a]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)



