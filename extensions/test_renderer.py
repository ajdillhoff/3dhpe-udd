import sys
import math
from timeit import default_timer as timer

import torch

if ".." not in sys.path:
    sys.path.append("..")

from model import Renderer


def dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def render_point(point, img, r=1.0):
    x_range = [max(0, math.floor(point[0] - r)), min(img.shape[0], math.ceil(point[0] + r))]
    for i in range(x_range[0], x_range[1]):
        y_range = [max(0, math.floor(point[1] - r)), min(img.shape[0], math.ceil(point[1] + r))]
        for j in range(y_range[0], y_range[1]):
            d_point = dist(torch.tensor([i, j], dtype=torch.float32), point)
            if d_point**2 < r**2:
                phi = (1 - (d_point / r**2))**2
                depth = (1 - point[2]) * phi
                if depth > img[j, i]:
                    img[j, i] = depth


def main():
    batch_size = 32
    num_points = 20000
    height = 120
    width = 120
    radius = 2.0
    points_cpu = torch.rand(batch_size, num_points, 3)
    points_cpu[:, :2] *= 120
    points_cpu.requires_grad_(True)
    points_gpu = points_cpu.detach().clone().to(torch.device('cuda')).requires_grad_(True)
    target_cpu = torch.rand(batch_size, height, width)
    target_gpu = target_cpu.clone().to(torch.device('cuda'))

    l1_loss = torch.nn.functional.l1_loss

    renderer = Renderer.Renderer(height, width, radius)
    renderer.cuda()

    # GPU forward pass
    start = timer()
    depth_img_gpu = renderer(points_gpu)
    end = timer()
    print("GPU Rendering time: {} ms".format((end - start) * 1000.0))

    # CPU forward pass
    depth_img_cpu = torch.zeros(batch_size, height, width)
    start = timer()
    for b in range(batch_size):
        for p in range(num_points):
            render_point(points_cpu[b, p], depth_img_cpu[b])
        end = timer()
    print("CPU Rendering time: {} ms".format((end - start) * 1000.0))

    loss_cpu = l1_loss(depth_img_cpu, target_cpu)
    loss_gpu = l1_loss(depth_img_gpu, target_gpu)
    print("loss_cpu = {}".format(loss_cpu.item()))
    print("loss_gpu = {}".format(loss_gpu.item()))

    # GPU backward pass
    start = timer()
    loss_gpu.backward()
    end = timer()
    print("GPU Backward Pass: {} ms".format((end - start) * 1000.0))

    # CPU backward pass
    start = timer()
    loss_cpu.backward()
    end = timer()
    print("CPU Backward Pass: {} ms".format((end - start) * 1000.0))

    # Compare gradients of CPU and GPU
    same = torch.allclose(points_cpu.grad, points_gpu.grad.cpu())
    print(same)


if __name__ == "__main__":
    main()

