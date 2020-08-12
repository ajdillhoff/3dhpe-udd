# PoseNet.py
# Alex Dillhoff (ajdillhoff@gmail.com)
# Model definition for the 2D heatmap prediction network.

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from .TFConv2D import TFConv2D


class PosePrior(nn.Module):
    """Implements the PosePrior architecture.

    The purpose of this network is to estimate the 3D coordinates of the hand
    using 2D heatmaps.

    This architecture is defined in:
        Zimmermann, C., & Brox, T. (2017).
        Learning to Estimate 3D Hand Pose from Single RGB Images.
        Retrieved from http://arxiv.org/abs/1705.01389
    """

    def __init__(self, num_joints=9):
        """Defines and initializes the network."""

        super(PosePrior, self).__init__()
        self.num_joints = num_joints
        self.conv_pose_0_1 = nn.Conv2d(num_joints, 32, 3, padding=1)
        self.conv_pose_0_2 = TFConv2D(32, 32, 3, stride=2)
        self.conv_pose_1_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_pose_1_2 = TFConv2D(64, 64, 3, stride=2)
        self.conv_pose_2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_pose_2_2 = TFConv2D(128, 128, 3, stride=2)

        self.fc_rel0 = nn.Linear(2048, 512)
        self.fc_rel1 = nn.Linear(512, 512)
        self.fc_xyz = nn.Linear(512, num_joints * 3)

    def forward(self, x):
        """Forward pass through PosePrior.

        Args:
            x - (batch x num_joints x 256 x 256): 2D keypoint heatmaps.

        Returns:
            (batch x num_joints x 3): xyz coordinates of the hand in 3D space.
        """

        s = x.shape
        x = F.leaky_relu(self.conv_pose_0_1(x))
        x = F.leaky_relu(self.conv_pose_0_2(x))
        x = F.leaky_relu(self.conv_pose_1_1(x))
        x = F.leaky_relu(self.conv_pose_1_2(x))
        x = F.leaky_relu(self.conv_pose_2_1(x))
        x = F.leaky_relu(self.conv_pose_2_2(x))

        # Permute before reshaping since these weights are loaded from a TF
        # model.
        x = torch.reshape(x.permute(0, 2, 3, 1), (s[0], -1))

        x = F.leaky_relu(self.fc_rel0(x))
        x = F.leaky_relu(self.fc_rel1(x))
        x = self.fc_xyz(x)

        return torch.reshape(x, (s[0], self.num_joints, 3))


class PoseNet(BaseModel):
    """Implements the PoseNet architecture.

    This architecture is defined in:
        Zimmermann, C., & Brox, T. (2017).
        Learning to Estimate 3D Hand Pose from Single RGB Images.
        Retrieved from http://arxiv.org/abs/1705.01389
    """

    def __init__(self, num_joints):
        """Defines and initializes the network."""

        super(PoseNet, self).__init__()
        # Stage 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5_1 = nn.Conv2d(128, 512, 1)
        self.conv5_2 = nn.Conv2d(512, num_joints, 1)
        self.pool = nn.MaxPool2d(2, 2)

        # Stage 2
        self.conv6_1 = nn.Conv2d(128 + num_joints, 128, 7, padding=3)
        self.conv6_2 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_3 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_4 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_5 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_6 = nn.Conv2d(128, 128, 1)
        self.conv6_7 = nn.Conv2d(128, num_joints, 1)

        # Stage 3
        self.conv7_1 = nn.Conv2d(128 + num_joints, 128, 7, padding=3)
        self.conv7_2 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_3 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_4 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_5 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_6 = nn.Conv2d(128, 128, 1)
        self.conv7_7 = nn.Conv2d(128, num_joints, 1)

        self.pose_prior = PosePrior(num_joints)

    def forward(self, x):
        """Forward pass through PoseNet.

        Args:
            x - [batch x 3 x 256 x 256]: Color image containing a cropped
                image of the hand.

        Returns:
            [batch x num_joints x 32 x 32] hand keypoint heatmaps.
        """

        # Stage 1
        x = F.leaky_relu(self.conv1_1(x)) # 1
        x = F.leaky_relu(self.conv1_2(x)) # 2
        x = self.pool(x)          # 3
        x = F.leaky_relu(self.conv2_1(x)) # 4
        x = F.leaky_relu(self.conv2_2(x)) # 5
        x = self.pool(x)          # 6
        x = F.leaky_relu(self.conv3_1(x)) # 7
        x = F.leaky_relu(self.conv3_2(x)) # 8
        x = F.leaky_relu(self.conv3_3(x)) # 9
        x = F.leaky_relu(self.conv3_4(x)) # 10
        x = self.pool(x)          # 11
        x = F.leaky_relu(self.conv4_1(x)) # 12
        x = F.leaky_relu(self.conv4_2(x)) # 13
        x = F.leaky_relu(self.conv4_3(x)) # 14
        x = F.leaky_relu(self.conv4_4(x)) # 15
        x = F.leaky_relu(self.conv4_5(x)) # 16
        x = F.leaky_relu(self.conv4_6(x)) # 17
        encoding = F.leaky_relu(self.conv4_7(x)) # 18
        x = F.leaky_relu(self.conv5_1(encoding))
        scoremap = self.conv5_2(x)

        # Stage 2
        x = torch.cat([scoremap, encoding], dim=1)
        x = F.leaky_relu(self.conv6_1(x))
        x = F.leaky_relu(self.conv6_2(x))
        x = F.leaky_relu(self.conv6_3(x))
        x = F.leaky_relu(self.conv6_4(x))
        x = F.leaky_relu(self.conv6_5(x))
        x = F.leaky_relu(self.conv6_6(x))
        scoremap = self.conv6_7(x)

        # Stage 3
        x = torch.cat([scoremap, encoding], dim=1)
        x = F.leaky_relu(self.conv7_1(x))
        x = F.leaky_relu(self.conv7_2(x))
        x = F.leaky_relu(self.conv7_3(x))
        x = F.leaky_relu(self.conv7_4(x))
        x = F.leaky_relu(self.conv7_5(x))
        x = F.leaky_relu(self.conv7_6(x))
        x = self.conv7_7(x)

        return self.pose_prior(x)
