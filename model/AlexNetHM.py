import torch.nn as nn
import torch.nn.init as init


class AlexNetHM(nn.Module):
    """Modified AlexNet for pose estimation.
    Output are the hand model parameters:
      - object location,
      - object quaternion orientation,
      - and joint quaternion orientations.
    """

    def __init__(self, num_joints=16, joint_features=4):
        super(AlexNetHM, self).__init__()
        self.num_joints = num_joints
        self.num_joint_features = joint_features
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        final_out = nn.Linear(4096, num_joints * joint_features)

        self.estimator = nn.Sequential(
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            final_out
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.kaiming_uniform_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)
        #     if m is final_out:
        #         init.normal_(m.weight, mean=0.0, std=0.01)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.estimator(x)
        return x
