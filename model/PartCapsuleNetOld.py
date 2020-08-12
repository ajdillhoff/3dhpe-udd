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
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if norm_layer is nn.GroupNorm:
            self.bn1 = norm_layer(32, planes)
        else:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if norm_layer is nn.GroupNorm:
            self.bn2 = norm_layer(32, planes)
        else:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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
        u1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        u2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        u3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        u4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.map_decoder = nn.Sequential(*[
            up,
            u1,
            nn.ReLU(inplace=True),
            up,
            u2,
            nn.ReLU(inplace=True),
            up,
            u3,
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(56*56, num_point_features),
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
                 point_features=1024, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=nn.GroupNorm):
        super(PartCapsuleNet, self).__init__()
        self.num_parts = num_parts
        self.num_features = num_features
        self.point_features = point_features
        self.norm_layer = norm_layer
        block = BasicBlock

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if norm_layer is nn.GroupNorm:
            self.bn1 = norm_layer(32, self.inplanes)
        else:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2,
                                      dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2,
                                      dilate=replace_stride_with_dilation[1])

        self.capsules = nn.ModuleList([self.create_capsule(block) for i in range(num_parts)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_capsule(self, block):
        self.inplanes = 256
        b1 = self.make_layer(block, 512, 1, stride=2)

        capsule = Capsule(b1, self.num_features, self.point_features)

        return capsule

    def make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm_layer is nn.GroupNorm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(32, planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        output = [capsule(x).unsqueeze(1) for capsule in self.capsules]
        output = torch.cat(output, 1)
        return output

