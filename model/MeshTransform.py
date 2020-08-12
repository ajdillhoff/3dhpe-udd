import torch

from utils.util import register_hook


class MeshTransform(torch.nn.Module):
    """Mesh transform module.

    Transforms the default vertices into the pose defined by the input
    transformation matrices."""

    def __init__(self, w, normals, transforms_inv, bone_weights, clip_space_matrix):
        """Initialize weights and clip space matrix.

        Args:
            w: (n_v x 4) Vertices of the original mesh in homegeneous
               coordinates.
            transforms_inv: (n_j x 4 x 4) Tensor of transformations from object
                        space to bone space.
            bone_weights: (n_j x n_v) Matrix of bone weights per vertex.
            clip_space_matrix: (4 x 4) Matrix defining the transformation from
                               world space to camera space.
        """
        super(MeshTransform, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(w, dtype=torch.float32),
                                         requires_grad=False)
        self.transforms_inv = torch.nn.Parameter(
            torch.tensor(transforms_inv, dtype=torch.float32),
            requires_grad=False)
        self.bone_weights = torch.nn.Parameter(
            torch.tensor(bone_weights, dtype=torch.float32),
            requires_grad=False)
        self.clip_space_matrix = torch.nn.Parameter(
            clip_space_matrix.clone().detach(),
            requires_grad=False)
        self.normals = torch.tensor(normals, dtype=torch.float32).cuda()

    def forward(self, input):
        """Forward pass through layer.

        Args:
            input: (B x num_joints x 4 x 4) Tensor of transforms from local
                bone space to object space.

        Returns:
            (B x num_vertices x 3) Tensor of transformed mesh vertices in clip
            space.
        """
        x = input @ self.transforms_inv

        # TODO: Simple test for rendering normals. Ignoring translation.
        normal_transform = x.clone()
        normal_transform[:, :, :3, 3] = 0.0
        normal_transform = normal_transform.permute(0, 2, 3, 1) @ self.bone_weights
        normals = self.normals.to(torch.cuda.current_device())
        normals_t = torch.einsum('aijk,kj->aki', (normal_transform, normals))

        x = x.permute(0, 2, 3, 1) @ self.bone_weights
        x = torch.einsum('aijk,kj->aki', (x, self.weight))

        return x, normals_t
