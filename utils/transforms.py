import torch
import numpy as np


def get_point_cloud(inputs, num_points, bg_val):
    """Gets point cloud representation from depth."""

    batch_size = inputs.shape[0]
    img_size = inputs.shape[2]

    output = torch.zeros(batch_size, num_points, 3,
                         dtype=inputs.dtype, device=inputs.device)

    for i in range(batch_size):
        depth = inputs[i].squeeze(0).transpose(0, 1) #.flip(1)
        depth_points = depth != bg_val
        z_vals = depth[depth != bg_val]
        depth_points = depth_points.nonzero().to(torch.float32) #/ img_size
        depth_points = torch.cat((depth_points, z_vals.view(-1, 1)), dim=1)
        if depth_points.shape[0] == 0:
            output[i] = torch.zeros(num_points,
                                    3,
                                    dtype=inputs.dtype,
                                    device=inputs.device,
                                    requires_grad=inputs.requires_grad)
            # Add small value for backward pass. Prevents div by zero error
            output[i] += np.finfo(float).eps
            output[i, :, 2] = 1 # set as background
            continue
        else:
            depth_points = depth_points.unsqueeze(0)
            depth_points = torch.nn.functional.interpolate(depth_points.transpose(1, 2),
                                                           size=num_points,
                                                           mode='nearest').transpose(2, 1)
        output[i] = depth_points

    return output
