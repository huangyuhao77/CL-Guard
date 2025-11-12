import os
import pickle
import numpy as np
import torch

# # # rhos
identity_fn    = lambda w, b: (w, b)


def gamma_fn(gamma): 
    def _gamma_fn(w, b):
        w = w + w * torch.max(torch.tensor(0., device=w.device), w) * gamma
        if b is not None: b = b + b * torch.max(torch.tensor(0., device=b.device), b) * gamma
        return w, b
    return _gamma_fn


# # # incrs
add_epsilon_fn = lambda e: lambda x:   x + ((x > 0).float()*2-1) * e


# # # Other stuff
def safe_divide(a, b):
    return a / (b + (b == 0).float())

def normalize(x):
    n_dim = len(x.shape)

    # This is what they do in `innvestigate`. Have no idea why?
    # https://github.com/albermax/innvestigate/blob/1ed38a377262236981090bb0989d2e1a6892a0b1/innvestigate/layers.py#L321
    if n_dim == 2: return x
    
    abs = torch.abs(x.view(x.shape[0], -1))
    absmax = torch.max(abs, axis=1)[0].view(x.shape[0], 1)
    for i in range(2, n_dim): absmax = absmax.unsqueeze(-1)

    x = safe_divide(x, absmax)
    x = x.clamp(-1, 1)

    return x

# # # # # # # # # # # # # # # # # # # # #
# Patterns
# # # # # # # # # # # # # # # # # # # # #
def store_patterns(file_name, patterns):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name, 'wb') as f:
        serialized_patterns = []
        for p in patterns:
            if isinstance(p, torch.Tensor):
                # 如果是张量，将其转换为 NumPy 数组并存储
                serialized_patterns.append(p.detach().cpu().numpy())
            elif isinstance(p, list):
                # 如果是列表，递归处理列表中的每个元素
                serialized_sublist = [item.detach().cpu().numpy() for item in p]
                serialized_patterns.append(serialized_sublist)

        # 使用 pickle 存储处理后的列表
        pickle.dump(serialized_patterns, f)

def load_patterns(file_name, device):
    with open(file_name, 'rb') as f:
        # 从 pickle 文件中加载数据
        serialized_patterns = pickle.load(f)
        # 对每个元素进行处理
        loaded_patterns = []
        for p in serialized_patterns:
            if isinstance(p, np.ndarray):
                # 如果是 NumPy 数组，转换为张量
                loaded_patterns.append(torch.tensor(p).to(device))
            elif isinstance(p, list):
                # 如果是列表，递归处理列表中的每个元素
                loaded_sublist = [torch.tensor(item).to(device) for item in p]
                loaded_patterns.append(loaded_sublist)
            else:
                # 其他类型的元素，保持不变
                loaded_patterns.append(p)

        return loaded_patterns
