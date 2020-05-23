import os
import random
import numpy as np
import torch
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TotalAverage():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.mass = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, val, mass=1):
        self.val = val
        self.mass += mass
        self.sum += val * mass
        self.avg = self.sum / self.mass

class MovingAverage():
    def __init__(self, intertia=0.9):
        self.intertia = intertia
        self.reset()

    def reset(self):
        self.avg = 0.

    def update(self, val):
        self.avg = self.intertia * self.avg + (1 - self.intertia) * val

def setup_runtime(seed=0, cuda_dev_id=1):
    """Initialize CUDA, CuDNN and the random seeds. """
    # Setup CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if len(cuda_dev_id) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_dev_id[0])
        for i in cuda_dev_id[1:]:
            os.environ["CUDA_VISIBLE_DEVICES"] += "," + str(i)

    # global cuda_dev_id
    _cuda_device_id = cuda_dev_id
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False 
    # Fix random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
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


def absorb_bn(module, bn_module):
    w = module.weight.data
    if module.bias is None:
        if isinstance(module, nn.Linear):
            zeros = torch.Tensor(module.out_features).zero_().type(w.type())
        else:
            zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
        module.bias = nn.Parameter(zeros)
    b = module.bias.data
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    if isinstance(module, nn.Conv2d):
        w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
    else:
        w.mul_(invstd.unsqueeze(1).expand_as(w))
    b.add_(-bn_module.running_mean).mul_(invstd)

    if bn_module.affine:
        if isinstance(module, nn.Conv2d):
            w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
        else:
            w.mul_(bn_module.weight.data.unsqueeze(1).expand_as(w))
        b.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    bn_module.reset_parameters()
    bn_module.register_buffer('running_mean', None)
    bn_module.register_buffer('running_var', None)
    bn_module.affine = False
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)

def search_absorb_bn(model):
    prev = None
    for m in model.children():
        if is_bn(m) and is_absorbing(prev):
            print("absorbing",m)
            absorb_bn(prev, m)
        search_absorb_bn(m)
        prev = m

class View(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

def sequential_skipping_bn_cut(model):
    mods = []
    layers = list(model.features) + [View()]
    if 'sobel' in dict(model.named_children()).keys():
        layers = list(model.sobel) + layers
    for m in nn.Sequential(*(layers)).children():
        if not is_bn(m):
            mods.append(m)
    return nn.Sequential(*mods)
