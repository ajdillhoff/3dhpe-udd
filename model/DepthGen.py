import numpy as np
import torch
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.structures import Meshes

from . import MeshTransform as mt
from . import HandModel as hm
from . import MeshMask as mm
from . import Skeleton
from utils.general import *
from utils.transforms import *


class DepthGen(torch.nn.Module):
    """3D generator module from depth input."""

    def __init__(self, mesh_path, skeleton_path, num_joints=17,
                 image_height=256, image_width=256, sample_size=1024,
                 output_joints=26):
        super(DepthGen, self).__init__()
        self.output_joints = output_joints
        self.image_height = image_height
        self.image_width = image_width
        self.sample_size = sample_size
        self.num_joints = num_joints

        self.init_depth_net(mesh_path, skeleton_path, image_height,
                            image_width)

    def init_depth_net(self, mesh_path, skeleton_path, image_height, image_width):
        rotations, positions, hand_parent_idxs = load_skeleton(skeleton_path)
        rotations = torch.from_numpy(rotations).to(torch.float32).cuda()
        positions = torch.from_numpy(positions).to(torch.float32).cuda()
        skeleton = Skeleton.Skeleton(hand_parent_idxs)

        # Camera and project matrices
        self.proj_matrix = torch.Tensor([[-1.732051, 0.0, 0.0, 0.0],
                                         [0.0, 1.732051, 0.0, 0.0],
                                         [0.0, 0.0, -1.025316, -0.202532],
                                         [0.0, 0.0, 0.0, 1.0]])
        self.view_matrix = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0, 1.0]])

        self.clip_space_matrix = self.view_matrix @ self.proj_matrix
        self.clip_space_matrix = self.clip_space_matrix.to(torch.cuda.current_device())

        # Initialize HandModel
        trainable_idxs = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        self.hand_model = hm.HandModel(positions, rotations, skeleton,
                                       trainable_idxs)

        # Initialize MeshTransform
        T = skeleton.to_matrix(rotations.unsqueeze(0),
                               positions.unsqueeze(0),
                               torch.ones(1, rotations.shape[0], 3,
                                          device=torch.cuda.current_device()))
        T_inv = np.linalg.inv(T.cpu().numpy())
        verts, normals, bone_weights, faces = load_mesh_data(mesh_path)
        self.faces = faces

        self.mesh_transform = mt.MeshTransform(
            np.concatenate((verts, np.ones((verts.shape[0], 1))), axis=1),
            np.concatenate((normals, np.ones((normals.shape[0], 1))), axis=1),
            T_inv,
            bone_weights,
            self.clip_space_matrix)

    def generate_depth(self, x):
        """Generates a depth image given the current model parameters.

        Args:
            x - Tensor (B x (3 + num_joint) - Model parameters.

        Returns:
            z_buffer - Tensor (B x H x W) - Depth buffers.
        """
        joint_idxs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 6]
        local_transforms, coord_pred, rot_offset = self.hand_model(x)
        mesh, _ = self.mesh_transform(local_transforms)

        coord_affine = torch.cat((coord_pred,
                                  torch.ones(
                                      coord_pred.shape[0],
                                      coord_pred.shape[1],
                                      1, device=torch.device('cuda'))), 2)
        mesh = mesh @ self.clip_space_matrix.to(torch.cuda.current_device())
        mesh = mesh[:, :, :3] / mesh[:, :, 3, None]
        coord_affine = coord_affine @ \
            self.clip_space_matrix.to(torch.cuda.current_device())
        coord_affine = coord_affine[:, :, :3] / coord_affine[:, :, 3, None]

        # Normalize output
        min_vals, _ = coord_affine[:, joint_idxs].min(1)
        max_vals, _ = coord_affine[:, joint_idxs].max(1)
        center = (max_vals + min_vals) / 2  # center of mass
        diff, _ = (max_vals - min_vals).max(1)

        mesh -= center.unsqueeze(1)
        mesh *= ((2 / diff.unsqueeze(1).unsqueeze(1).repeat(1, mesh.shape[1], 3)) * 0.65)
        mesh[:, :, 2] += 1

        coord_affine -= center.unsqueeze(1).repeat(1, coord_affine.shape[1], 1)
        coord_affine *= ((2 / diff.unsqueeze(1).unsqueeze(1).repeat(1, coord_affine.shape[1], 3)) * 0.65)

        # Initialize Meshes
        meshes = Meshes(mesh,
                        self.faces.unsqueeze(0).repeat(mesh.shape[0], 1, 1))

        # Rasterize
        _, zbuf, _, _ = rasterize_meshes(meshes.cuda(),
                                         image_size=120,
                                         cull_backfaces=True)

        # Crop
        out_img = zbuf[:, :, :, 0].unsqueeze(1)

        return out_img, coord_pred, rot_offset, local_transforms

    def forward(self, x):
        pose_params = x
        img, coords, rot_offset, local_transforms = self.generate_depth(pose_params)

        joint_idxs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 6]

        # Normalization of output using first bone in index finger
        mean = coords[:, joint_idxs].mean(1).unsqueeze(1).clone()  # Don't include carpals
        norm_size = torch.norm(coords[:, 23] - coords[:, 22], dim=1)
        norm_size = norm_size.unsqueeze(1).unsqueeze(1)

        coords -= mean.repeat(1, self.output_joints, 1)
        coords /= norm_size.repeat(1, self.output_joints, 1)
        # points -= mean.repeat(1, self.sample_size, 1)
        # points /= norm_size.repeat(1, self.sample_size, 3)

        # Normalize depth images
        for i in range(x.shape[0]):
            bg_mask = (img[i] == -1)
            fg_mask = (img[i] > -1)
            min_val = img[i, fg_mask].min()
            max_val = img[i, fg_mask].max()
            img[i, fg_mask] -= min_val
            img[i, fg_mask] /= (max_val - min_val)
            img[i, fg_mask] *= 2.0
            img[i, fg_mask] -= 1.0
            img[i, bg_mask] = 1.0

        return [img, coords, norm_size, mean, rot_offset, local_transforms,
                self.mesh_transform.transforms_inv[:, 0],
                self.mesh_transform.bone_weights]
