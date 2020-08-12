import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Capsule(nn.Module):
    def __init__(self, conv_block, num_pose_features, num_point_features):
        super(Capsule, self).__init__()

        self.num_pose_features = num_pose_features
        self.num_point_features = num_point_features

        # Pose encoder
        self.conv_block = conv_block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_fc = nn.Linear(512, self.num_pose_features)

        # Point map decoder
        up = nn.Upsample(mode='bilinear', scale_factor=2)
        u1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        u2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.map_decoder = nn.Sequential(*[
            up,
            u1,
            nn.ReLU(inplace=True),
            up,
            u2,
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(28*28, num_point_features),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x):
        conv_out = self.conv_block(x)
        pose_out = self.avgpool(conv_out)
        pose_out = pose_out.squeeze(-1).squeeze(-1)
        pose_out = self.pose_fc(pose_out)

        map_out = self.map_decoder(conv_out)

        return torch.cat((pose_out, map_out), -1)


class PartCapsuleNet(nn.Module):
    def __init__(self, layers, num_parts=1, num_features=4,
                 point_features=1024):
        super(PartCapsuleNet, self).__init__()
        self.num_parts = num_parts
        self.num_features = num_features
        self.point_features = point_features
        block = BasicBlock

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)

        self.capsules = nn.ModuleList([self.create_capsule(block)
                                       for i in range(num_parts)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def create_capsule(self, block):
        self.inplanes = 256
        b1 = self.make_layer(block, 512, 1, stride=2)

        capsule = Capsule(b1, self.num_features, self.point_features)

        return capsule

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        output = [capsule(x).unsqueeze(1) for capsule in self.capsules]
        output = torch.cat(output, 1)
        return output

