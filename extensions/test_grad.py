import sys
import math

import torch
from torch.autograd import gradcheck

if ".." not in sys.path:
    sys.path.append("..")

from model.Renderer import RendererFunction

torch.set_printoptions(threshold=10000)


def main():
    render_f = RendererFunction.apply

    input = (torch.ones(1, 1, 3, dtype=torch.float32, requires_grad=True, device=torch.device('cuda')) * 0.5, 2, 2, 1.0)
    test = gradcheck(render_f, input, eps=0.1)
    print(test)


if __name__ == "__main__":
    main()

