import torch
import torch.nn.functional as F

import utils.quaternion as quat
import utils.general as general
from utils.util import register_hook
from base import BaseModel
from model import JointLayer as JointLayer


class SkeletonFinder(BaseModel):
    """Uses 3D joint parameters to calculate the transformation from a default
    pose."""

    def __init__(self, rotations, positions, skeleton, hand_parent_idxs,
                 pose_parent_idxs, joint_map):
        # TODO: Update the header
        """Initialize the layer.

        Args:
            skeleton: (n_j x 7) Tensor defining the skeleton. Each joint is
                      defined by 7 values. The first 3 are the locations
                      relative to the parent joint. The last 4 define the
                      orientation change relative to the parent joint as a
                      quaternion.
            hand_parent_idxs: (n_j) List which maps the joint index to the
                              corresponding parent index.
            pose_parent_idxs: (n_p) List which maps the pose index to the
                              corresponding parent index. Pose definitions
                              include finger tips.
            joint_map: (n_j) List which maps the skeleton joint index to the
                       pose index.
            clip_space_matrix (4 x 4): Camera and perspective matrix.
        """

        super(SkeletonFinder, self).__init__()
        self.positions = positions
        self.rotations = rotations
        self.skeleton = skeleton
        self.num_joints = len(hand_parent_idxs)
        self.hand_parent_idxs = hand_parent_idxs
        self.pose_parent_idxs = pose_parent_idxs
        self.joint_map = joint_map
        self.joint_parent_idxs = [-1, 0, 1]
        self.hand_map = [0, 1, -1]
        self.kp_hand_map = [-1, 0, 1]
        self.num_kp = len(self.kp_hand_map)

        # Orient the skeleton using rigid points on the hand ##
        p_world, p_rotations = self.skeleton.forward_kinematics(rotations, positions)

        # Compute model offsets
        # children are all joint_parent_idxs > 0
        # parents are all hand_map > 0
        parent_idxs = [i for i in self.joint_parent_idxs if i > 0]
        child_idxs = [i + 1 for i in parent_idxs]
        model_offsets = p_world[child_idxs] - p_world[parent_idxs]

        # Define JointLayers
        tip = torch.tensor([0.0, 0.15, 0.0])
        self.tip = JointLayer.JointLayer(tip, rotations[1])

        mid_base = p_world[1] - p_world[0]
        tip_base = p_world[2] - p_world[0]

        self.model_anchor = torch.stack((F.normalize(mid_base, dim=0),
                                         F.normalize(tip_base, dim=0)))

    def forward(self, x):
        """Forward pass through the layer.

        Converts the input to world space using known projection and view
        matrices. Once in world space, it is compared to the skeleton derived
        in world space. This comparison yields the model translation and
        rotation.

        Additionally, each local joint is compared to the predicted joints to
        find individual joint transformations.

        Args:
            input: (B x n_p x 3) Tensor of the 3D joint predictions. The x, y
            coordinates are in normalized render coordinates (0, 1) and the z
            value is in world space.

        Returns:
            transforms: (B x n_j x 4 x 4) Tensor defining the transforms for
                each joint.
        """
        # TODO: These matrices need to be passed into SF
        proj_matrix = torch.Tensor([[1.732051, 0.0, 0.0, 0.0],
                                    [0.0, 1.732051, 0.0, 0.0],
                                    [0.0, 0.0, -1.133333, -1.066667],
                                    [0.0, 0.0, -1.0, 0.0]]).cuda()
        view_matrix = torch.Tensor([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]).cuda()
        proj_matrix_inv = torch.inverse(proj_matrix)

        # Convert all input to world space
        x_h = torch.cat((x, torch.ones(x.shape[0], x.shape[1], 1,
                                       device=x.device)), 2)

        z = -x[:, :, 2]
        z_mult = torch.cat((z.repeat(2, 1, 1).view(x.shape[0], 2, -1),
                            torch.ones(x.shape[0],
                                       2, self.num_kp, device=x.device)), 1)
        x_clip = x_h.transpose(1, 2) * z_mult
        x_view = proj_matrix_inv @ x_clip

        x_world_unnorm = (torch.inverse(view_matrix) @ x_view).transpose(1, 2)
        x_world_unnorm[:, :, 2] = -z

        x_world = x_world_unnorm[:, :, :3]

        # Compute offsets for input
        parent_idxs = [i for i in self.joint_parent_idxs if i > -1]
        x_parents = torch.cat((torch.zeros(x.shape[0], 1, 3, device=x.device),
                               x_world[:, parent_idxs].clone().detach()), 1)

        x_world_offsets = x_world - x_parents

        # Local positions of the model are scaled based on the input
        bone_idxs = [i for i, val in enumerate(self.kp_hand_map) if val > -1]
        bone_lengths = torch.norm(x_world_offsets[:, bone_idxs], dim=2)

        # Model Rotation Calculation
        pose_joint_idxs = [0, 1, 2]
        pose_joints = x_world[:, pose_joint_idxs].clone()
        joints_mc = pose_joints[:, 1:] - pose_joints[:, 0].unsqueeze(1)

        # Find the normal of the triangle defined in the skeleton
        p2 = torch.stack((F.normalize(joints_mc[:, 0]),
                          F.normalize(joints_mc[:, 1])), dim=1)
        q = quat.find_q_v(self.model_anchor.repeat(x.shape[0], 1, 1), p2)
        model_rotation = q

        # Set new root rotation
        root_rotation = quat.qmul(model_rotation,
                                  self.rotations[0].repeat(
                                      model_rotation.shape[0], 1))

        # DEBUG: Joint 1
        # f2r_p = quat.q_rot(quat.q_inv(root_rotation), x_world_offsets[:, 1])
        tipr_p = quat.q_rot(quat.q_inv(root_rotation), x_world_offsets[:, 1])

        # Calculate joint rotations
        tip_q_parents, tip_q, tip_p = self.tip(x_world_offsets[:, 2],
                                               root_rotation)

        positions = []
        positions.append(x_world[:, 0])
        positions.append(tipr_p)
        positions = torch.stack(positions, dim=1)

        rotations = [root_rotation, tip_q]
        rotations = torch.stack(rotations, dim=1)

        local_transforms = self.skeleton.to_matrix(rotations, positions).cuda()

        return local_transforms, bone_lengths
