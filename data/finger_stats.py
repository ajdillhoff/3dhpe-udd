import os
import sys
import math

import torch
import torchvision
import torch.nn.functional as F
from PIL import Image

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import data_loader.data_loaders as module_data


if __name__ == '__main__':
    data_path = '/home/alex/data/train'
    loader = module_data.FingerDataLoader(data_path, 1, False, 0, 1, False)
    means = torch.zeros(3)
    std_devs = torch.zeros(3)
    count = 0
    for batch_idx, (data, _) in enumerate(loader):
        means[0] += data[:, 0].mean()
        means[1] += data[:, 1].mean()
        means[2] += data[:, 2].mean()
        std_devs[0] += data[:, 0].std()
        std_devs[1] += data[:, 1].std()
        std_devs[2] += data[:, 2].std()
        count += 1

    means /= count
    std_devs /= count

    print('means: {}\nstd devs: {}'.format(means, std_devs))
