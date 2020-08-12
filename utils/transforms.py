import torch
import numpy as np

from utils.general import create_barycentric_transform, get_barycentric_coordinates


def chp3d_keypoint_to_lh(keypoint_coords, libhand_joints):
    """Used for evaluation."""
    chp3d_keypoints_idx = [0, 12, 16, 20]
    lh_keypoints_idx = [0, 16, 11, 6]
    chp3d_map = [0, 4, 3, 2, 1, 21, 20, 19, 18, 17, 21, 16, 15, 14, 13, 21, 12, 11, 10, 9, 21, 8, 7, 6, 5]

    # Compute barycentric values for LH
    A_lh = libhand_joints[lh_keypoints_idx]
    r_lh = libhand_joints[5] # Bone
    T_lh = create_barycentric_transform(A_lh)
    coords = get_barycentric_coordinates(r_lh, T_lh, A_lh[3])

    # Compute barycentric values for CHP3D
    A_chp = keypoint_coords[:, chp3d_keypoints_idx]
    T_chp = create_barycentric_transform(A_chp)
    if A_chp.device != torch.device('cpu'):
        T_chp = T_chp.cuda()
        coords = coords.cuda()
    bone_estimates = T_chp @ coords + A_chp[:, 3]

    chp3d_est_pose = torch.cat((keypoint_coords,
                                bone_estimates.unsqueeze(1)), dim=1)
    # chp3d_est_pose = chp3d_est_pose[:, 1:]
    chp3d_est_pose = chp3d_est_pose[:, chp3d_map]
    # chp3d_est_pose -= chp3d_est_pose[:, 0]

    return chp3d_est_pose

def chp3d_to_lh(chp3d_coords, libhand_joints):
    """Converts CHP3D joint maps to LH."""

    chp3d_keypoints_idx = [0, 12, 16, 20]
    lh_keypoints_idx = [16, 11, 6]
    chp3d_map = [22, 4, 3, 2, 1, 21, 20, 19, 18, 17, 21, 16, 15, 14, 13, 21, 12, 11, 10, 9, 21, 8, 7, 6, 5]

    # Compute the palm (CHP3D does not estimate the wrist)
    palm_joint = (libhand_joints[0] + libhand_joints[16]) / 2

    # Compute barycentric values for LH
    A_lh = torch.cat((palm_joint.unsqueeze(0), libhand_joints[lh_keypoints_idx]))
    r_lh = libhand_joints[5] # Bone
    T_lh = create_barycentric_transform(A_lh)
    coords = get_barycentric_coordinates(r_lh, T_lh, A_lh[3])

    # Compute barycentric values for CHP3D
    A_chp = chp3d_coords[:, chp3d_keypoints_idx]
    T_chp = create_barycentric_transform(A_chp)
    if A_chp.device != torch.device('cpu'):
        T_chp = T_chp.cuda()
        coords = coords.cuda()
    bone_estimates = T_chp @ coords + A_chp[:, 3]

    # Compute the metacarpals
    palm_middle_diff = chp3d_coords[:, 12] - chp3d_coords[:, 0]
    mc = chp3d_coords[:, 0] - palm_middle_diff

    chp3d_est_pose = torch.cat((chp3d_coords,
                                bone_estimates.unsqueeze(1),
                                mc.unsqueeze(1)), dim=1)
    # chp3d_est_pose = chp3d_est_pose[:, 1:]
    chp3d_est_pose = chp3d_est_pose[:, chp3d_map]
    # chp3d_est_pose -= chp3d_est_pose[:, 0]

    return chp3d_est_pose

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
