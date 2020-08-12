import os
import sys
import time

import torch

import utils.quaternion as quat # TODO: This is copied from pytorch-skeleton-rig. Clean it up after testing.
import utils.camera_utils as camera_utils
from . import MeshTransform as mt
from . import HandModel as hm
from . import MeshMask as mm
from . import PointNet
from . import Skeleton
from base import BaseModel

from utils.general import *
from utils.transforms import *
from utils.util import register_hook


class DepthGenPointNet(BaseModel):
    """Depth generator network.
    HM variant output is hand model parameters:
      - object location,
      - object orientation,
      - relative joint orientations.
    """

    def __init__(self, mesh_path, skeleton_path, num_joints=17, image_height=256,
            image_width=256, sample_size=1000, output_joints=26):
        super(DepthGenPointNet, self).__init__()
        self.output_joints = output_joints
        self.estimator = PointNet.PointNetCaps(n_features=9, k=num_joints)

        self.image_height = image_height
        self.image_width = image_width
        self.sample_size = sample_size

        self.init_depth_net(mesh_path, skeleton_path, image_height,
                image_width)

    def init_depth_net(self, mesh_path, skeleton_path, image_height, image_width):
        rotations, positions, hand_parent_idxs = load_skeleton(skeleton_path)
        rotations = torch.from_numpy(rotations).to(torch.float32).cuda()
        positions = torch.from_numpy(positions).to(torch.float32).cuda()
        skeleton = Skeleton.Skeleton(hand_parent_idxs)

        # Camera and project matrices
        self.proj_matrix = torch.Tensor([[1.302294, 0.0, 0.0, 0.0],
                                    [0.0, 1.732051, 0.0, 0.0],
                                    [0.0, 0.0, -1.025316, -0.202532],
                                    [0.0, 0.0, -1.0, 1.0]])
        self.view_matrix = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]])

        self.clip_space_matrix = self.view_matrix @ self.proj_matrix
        self.clip_space_matrix = self.clip_space_matrix.to(torch.cuda.current_device())

        # Initialize HandModel
        trainable_idxs = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        self.hand_model = hm.HandModel(positions, rotations, skeleton, trainable_idxs)

        # Initialize MeshTransform
        T = skeleton.to_matrix(rotations.unsqueeze(0),
                               positions.unsqueeze(0),
                               torch.ones(1, rotations.shape[0], 3,
                                          device=torch.cuda.current_device()))
        T_inv = np.linalg.inv(T.cpu().numpy())
        mesh_vertices, normals, bone_weights, triangles = load_mesh_data(mesh_path)

        self.mesh_transform = mt.MeshTransform(np.concatenate(
            (mesh_vertices, np.ones((mesh_vertices.shape[0], 1))), axis=1),
            np.concatenate((normals, np.ones((normals.shape[0], 1))), axis=1),
                                          T_inv,
                                          bone_weights,
                                          self.clip_space_matrix)

        # Mesh Rendrer
        self.mesh_mask = mm.MeshMask([image_height, image_width],
                                     triangles.cuda())

    def generate_depth(self, x):
        """Generates a depth image given the current model parameters.

        Args:
            x - Tensor (B x (3 + num_joint) - Model parameters.

        Returns:
            z_buffer - Tensor (B x H x W) - Depth buffers.
        """
        local_transforms, coord_pred, rot_offset = self.hand_model(x)
        mesh, normals = self.mesh_transform(local_transforms)

        with torch.no_grad():
            points = mesh.clone().contiguous()
            # TODO: Temporarily necessary for synthetic data. This should be
            # adjusted by using a CoM calculation.
            # points[:, :, 1] -= 0.75
            points[:, :, 2] -= 0.2
            points = points @ self.clip_space_matrix.to(torch.cuda.current_device())
            points = points[:, :, :3] / points[:, :, 3, None]
            mask = self.mesh_mask(points)

        points = mesh[:, :, :3]
        out = []
        out_idxs = []

        for i in range(points.shape[0]):
            mask_idxs = (mask[i] == 1).nonzero().squeeze()
            if len(mask_idxs.size()) == 0 or mask_idxs.shape[0] < self.sample_size:
                sampled_pc = torch.zeros(self.sample_size, 3,
                        dtype=mesh.dtype, device=x.device, requires_grad=True)
                mask_idxs = torch.zeros(self.sample_size,
                        dtype=torch.long, device=x.device)
            else:
                # Random Sampling
                # perm = torch.randperm(mask_idxs.shape[0])
                # mask_idxs = mask_idxs[perm[:self.sample_size]]
                # Uniform Sampling
                sample_idxs = torch.linspace(0, mask_idxs.shape[0] - 1, self.sample_size, device=x.device)
                sample_idxs = sample_idxs.to(torch.long)
                mask_idxs = mask_idxs[sample_idxs]
                sampled_pc = points[i, mask_idxs]

            out_idxs.append(mask_idxs)
            out.append(sampled_pc)
        out = torch.stack(out)
        out_idxs = torch.stack(out_idxs)

        return out, out_idxs, local_transforms, coord_pred, rot_offset

    def forward(self, x):
        caps_out, seg_out, _, _ = self.estimator(x)
        points, out_idxs, transforms, coords, rot_offset = self.generate_depth(caps_out)

        # Normalization of output using first bone in index finger
        mean = coords[:, 1:].mean(1).unsqueeze(1).clone() # Don't include carpals
        norm_size = torch.norm(coords[:, 23] - coords[:, 22], dim=1)
        norm_size = norm_size.unsqueeze(1).unsqueeze(1)

        coords -= mean.repeat(1, self.output_joints, 1)
        coords /= norm_size.repeat(1, self.output_joints, 1)
        points -= mean.repeat(1, self.sample_size, 1)
        points /= norm_size.repeat(1, self.sample_size, 3)

        return [points,
                out_idxs,
                coords,
                transforms,
                norm_size,
                mean,
                rot_offset,
                self.mesh_transform.transforms_inv[:, 0],
                seg_out,
                self.mesh_transform.bone_weights]
