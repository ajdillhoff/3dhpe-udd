import os
import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
import numpy as np
import torch


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

class AverageMeter(object):
    """Computes and stores the average and current value."""
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

def get_printer(msg):
    def printer(tensor):
        print(f"{msg} shape: {tensor.shape}")
        print(f"\n{tensor}\n****")
    return printer

def register_hook(tensor, msg):
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))


def convert_state_dict(d):
    """Converts weights saved from Multi GPU training."""
    new_dict = {}
    for key, value in d.items():
        new_key = key[7:]
        new_dict[new_key] = value
    return new_dict

'''
Util code from github.com/xinghaochen/awesome-hand-pose-estimation/
'''

def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, int(len(positions[0]) / 3), 3))


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu', 'msra'])


def get_dataset_file(dataset):
    return 'results/groundtruth/{}/{}_test_groundtruth_label.txt'.format(dataset, dataset)


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.03, -587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


def get_errors(dataset, in_file):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    labels = get_positions(get_dataset_file(dataset))
    outputs = get_positions(in_file)
    params = get_param(dataset)
    labels = pixel2world(labels, *params)
    outputs = pixel2world(outputs, *params)
    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    return errors


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def normalize_batch(imgs):
    """Normalize images to be in range [0, 1]."""
    orig_shape = imgs.shape
    bg_mask = (imgs == 0).view(orig_shape[0], -1)
    fg_mask = (imgs > 0).view(orig_shape[0], -1)
    bg_pad = torch.ones_like(imgs.view(orig_shape[0], -1)) * 255.0 * bg_mask
    fg_pad = torch.ones_like(imgs.view(orig_shape[0], -1)) * 255.0 * fg_mask
    min_vals, _ = (imgs.view(orig_shape[0], -1) + bg_pad).min(-1)
    max_vals, _ = imgs.view(orig_shape[0], -1).max(-1)
    min_vals = min_vals.unsqueeze(1).repeat(1, orig_shape[1] * orig_shape[2]).view(orig_shape)
    max_vals = max_vals.unsqueeze(1).repeat(1, orig_shape[1] * orig_shape[2]).view(orig_shape)
    imgs -= min_vals
    imgs /= (max_vals - min_vals)
    imgs *= fg_mask.view(orig_shape)
    return imgs
