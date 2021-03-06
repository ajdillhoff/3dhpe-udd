import numpy as np
import torch

from . import MeshTransform as mt
from . import HandModel as hm
from . import MeshMask as mm
from . import Skeleton
from .Renderer import Renderer
from utils.general import *
from utils.transforms import *
from utils.util import normalize_batch


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
        self.proj_matrix = torch.Tensor([[1.732051, 0.0, 0.0, 0.0],
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

        self.renderer = Renderer(image_height, image_width, 2.0)

    @staticmethod
    def normalize_points(points, center, scale):
        points -= center.unsqueeze(1)
        points /= scale.unsqueeze(1).unsqueeze(1).repeat(1, points.shape[1], 3)
        return points

    def generate_depth(self, local_transforms, coord_pred):
        """Generates a depth image given the current model parameters.

        Args:
            local_transforms - Tensor (B x 4 x 4) - Local joint transforms.
            coord_pred - Tensor (B x J x 3) - Predicted hand keypoints.

        Returns:
            z_buffer - Tensor (B x H x W) - Depth buffers.
        """
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
        min_vals, _ = coord_affine.min(1)
        max_vals, _ = coord_affine.max(1)
        center = (max_vals + min_vals) / 2  # center of bounding box
        diff, _ = (max_vals[:, :2] - min_vals[:, :2]).max(1)
        scale = diff * (0.5 + 0.1)  # 0.1 is the padding

        mesh = self.normalize_points(mesh, center, scale)
        mesh[:, :, 2] += 0.5  # Centered between 0 and 1

        # Viewport Transform
        mesh[:, :, 0] = (mesh[:, :, 0] + 1.0) * 0.5 * (self.image_width - 1)
        mesh[:, :, 1] = (1 - (mesh[:, :, 1] + 1.0) * 0.5) * (self.image_height - 1)

        out_img = self.renderer(mesh)

        return out_img

    def forward(self, x, gen_depth=True):
        joint_idxs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 6]
        pose_params = x
        local_transforms, coords, rot_offset = self.hand_model(x)

        if gen_depth:
            img = self.generate_depth(local_transforms, coords[:, joint_idxs])

            # Normalize depth images
            img = normalize_batch(img)
        else:
            img = None

        # Normalization of output using first bone in index finger
        mean = coords[:, joint_idxs].mean(1).unsqueeze(1).clone()  # Don't include carpals
        norm_size = torch.norm(coords[:, 23] - coords[:, 22], dim=1)
        norm_size = norm_size.unsqueeze(1).unsqueeze(1)

        coords -= mean.repeat(1, self.output_joints, 1)
        coords /= norm_size.repeat(1, self.output_joints, 1)

        return [img, coords, norm_size, mean, rot_offset, local_transforms,
                self.mesh_transform.transforms_inv[:, 0],
                self.mesh_transform.bone_weights]
