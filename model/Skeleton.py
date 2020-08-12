import torch
import numpy as np
from utils.quaternion import q_mult, qmul, q_rot, q_to_rotation_matrix_v


class Skeleton:
    def __init__(self, parents, inherit_scale=False):
        self.parents_ = np.array(parents)
        self.tip_length = 0.4 * 0.06
        self.inherit_scale = inherit_scale
        self.init_skeleton()

    def cuda(self):
        self.joints_ = self.joints_.cuda()
        return self

    def parents(self):
        return self.parents_

    def has_children(self):
        return self.has_children_

    def children(self):
        return self.children_

    def forward_kinematics(self, rotations, positions):
        """
        Forward kinematics using the given quaternion rotations and local
        positions.

        Args:
            rotations (J, 4): Unit quaternions describing local rotations
                for each joint.
            positions (J, 3): Joint positions relative to the parent.
        """
        rotations_world = []
        positions_world = []
        tips = []
        tip_idxs = []

        for i in range(rotations.shape[0]):
            if self.parents_[i] == -1:
                positions_world.append(positions[i])
                rotations_world.append(rotations[i])
            else:
                positions_world.append(
                    q_rot(rotations_world[self.parents_[i]], positions[i]) \
                    + positions_world[self.parents_[i]]
                )
                if self.has_children_[i]:
                    rotations_world.append(
                        q_mult(rotations_world[self.parents_[i]], rotations[i])
                    )
                else:
                    tip_idxs.append(i + len(tip_idxs) + 1)
                    tip_pos = torch.tensor([0., 1., 0.], device=rotations.device)
                    tip_pos = q_rot(q_mult(rotations_world[self.parents_[i]],
                                           rotations[i]), tip_pos)
                    tips.append(positions_world[i] + self.tip_length * tip_pos)
                    rotations_world.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotations.device))

        for i in range(len(tip_idxs)):
            positions_world.insert(tip_idxs[i], tips[i])

        return torch.stack(positions_world), torch.stack(rotations_world)

    def forward_kinematics2(self, rotations, positions, scales):
        """
        Forward kinematics using the given quaternion rotations and local
        positions.

        Args:
            rotations (J, 4): Unit quaternions describing local rotations
                for each joint.
            positions (J, 3): Joint positions relative to the parent.
            scales (J, 3): x, y, z scale for each joint.
        """
        rotations_world = []
        positions_world = []
        scales_world = []
        tips = []
        tip_idxs = []

        for i in range(rotations.shape[1]):
            if self.parents_[i] == -1:
                positions_world.append(positions[:, i])
                rotations_world.append(rotations[:, i])
                scales_world.append(scales[:, i])
            else:
                positions_world.append(
                    q_rot(rotations_world[self.parents_[i]], positions[:, i]) \
                    * scales_world[self.parents_[i]]
                    + positions_world[self.parents_[i]]
                )

                if self.inherit_scale:
                    scales_world.append(scales_world[self.parents_[i]] * scales[:, i])
                else:
                    scales_world.append(scales[:, i])

                if self.has_children_[i]:
                    rotations_world.append(
                        qmul(rotations_world[self.parents_[i]], rotations[:, i])
                    )
                else:
                    tip_idxs.append(i + len(tip_idxs) + 1)
                    tip_pos = torch.tensor([0., 1., 0.], device=rotations.device) * scales[:, i] * self.tip_length
                    # tip_pos = tip_pos.repeat(rotations_world[self.parents_[i]].shape[0], 1)
                    tip_pos = q_rot(qmul(rotations_world[self.parents_[i]],
                                           rotations[:, i]), tip_pos) \
                              + positions_world[i]
                    tips.append(tip_pos)
                    rotations_world.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=rotations.device))

        for i in range(len(tip_idxs)):
            positions_world.insert(tip_idxs[i], tips[i])

        return torch.stack(positions_world)

    def to_matrix(self, rotations, positions, scales):
        """
        Converts the local rotations and positions to matrix form.

        Args:
            rotations (N, J, 4): Unit quaternions describing local rotations
                for each joints.
            positions (N, J, 3): Joint positions relative to the parent.
            scales (N, J, 3): Scale parameters relative to the parent.
        """
        rotations_world = []
        positions_world = []
        scales_world = []
        transforms = []

        for i in range(rotations.shape[1]):
            if self.parents_[i] == -1:

                rotations_world.append(rotations[:, i])
                positions_world.append(positions[:, i])
                scales_world.append(scales[:, i])
                transform = q_to_rotation_matrix_v(rotations_world[i])
            else:
                positions_world.append(
                    q_rot(rotations_world[self.parents_[i]], positions[:, i]) \
                    * scales_world[self.parents_[i]]
                    + positions_world[self.parents_[i]]
                )
                q_w = qmul(rotations_world[self.parents_[i]], rotations[:, i])
                transform = q_to_rotation_matrix_v(q_w)

                tip_mod = 1.0

                if self.has_children_[i]:
                    rotations_world.append(q_w)
                else:
                    rotations_world.append(
                        torch.tensor([1., 0., 0., 0.],
                                     dtype=rotations.dtype,
                                     device=rotations.device))
                    tip_mod = self.tip_length

                if self.inherit_scale:
                    scales_world.append(scales_world[self.parents_[i]] * scales[:, i] * tip_mod)
                else:
                    scales_world.append(scales[:, i] * tip_mod)

            # Scale bone
            scale_matrix = torch.diag_embed(torch.cat((scales_world[i], torch.ones(scales.shape[0], 1, device=scales.device)), -1))
            transform = transform @ scale_matrix.to(transform.device)

            # Set position
            transform[:, :3, 3] = positions_world[i]

            transforms.append(transform)

        return torch.stack(transforms, dim=1)

    def init_skeleton(self):
        self.has_children_ = np.zeros(len(self.parents_)).astype(bool)
        for i, parent in enumerate(self.parents_):
            if parent != -1:
                self.has_children_[parent] = True

        self.children_ = []
        for i, parent in enumerate(self.parents_):
            self.children_.append([])
        for i, parent in enumerate(self.parents_):
            if parent != -1:
                self.children_[parent].append(i)
