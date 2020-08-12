import torch
import torch.nn.functional as F

from utils.quaternion import qmul


class HandModel(torch.nn.Module):
    """ Layer that converts model parameters into transformation matrices and
    3D joint locations."""
    def __init__(self, positions, rotations, skeleton, trainable_idxs):
        super(HandModel, self).__init__()
        self.positions = positions.to(torch.cuda.current_device())
        self.base_rotations = rotations.to(torch.cuda.current_device())
        self.skeleton = skeleton
        self.trainable_idxs = trainable_idxs

        self.skeleton.inherit_scale = False

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, 4)

        pos_pred = x[:, :, :4]
        # shape_pred = x[:, :, 4:]
        rot_offset = F.normalize(pos_pred, dim=-1)
        # scale_pred = torch.min(torch.max(torch.ones_like(shape_pred) * -0.5,
                                         # shape_pred),
                               # torch.ones_like(shape_pred) * 0.5)

        # Position
        positions = self.positions.to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Rotation
        rotations = self.base_rotations.to(torch.cuda.current_device()).repeat(batch_size, 1, 1)
        new_rotations = qmul(rotations[:, self.trainable_idxs].view(-1, 4), rot_offset.view(-1, 4))
        new_rotations = new_rotations.view(batch_size, -1, 4)
        rotations[:, self.trainable_idxs] = new_rotations

        # Scale parameter
        scale_params = torch.ones_like(positions)
        # if self.predict_shape is True:
            # scale_params[:, self.trainable_idxs] += scale_pred

        # Forward kinematics
        local_transforms = self.skeleton.to_matrix(rotations, positions, scale_params).cuda()
        coord_pred = self.skeleton.forward_kinematics2(rotations, positions, scale_params).cuda()
        coord_pred = coord_pred.permute((1, 0, 2))

        return local_transforms, coord_pred, rot_offset
