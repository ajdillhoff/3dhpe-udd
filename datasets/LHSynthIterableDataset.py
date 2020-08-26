import os
import sys

import torch
import torchvision
import scipy.io as sio
import numpy as np
from PIL import Image

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

gen_path = os.path.abspath('/home/alex/dev/projects/libhand-mesh-generator/build/src/')
if gen_path not in sys.path:
    sys.path.append(gen_path)

import PoseGenerator
from utils.transforms import get_point_cloud


class LHSynthIterableDataset(torch.utils.data.IterableDataset):
    """Synthetic Dataset using Libhand with wrist. This is the model used in
    the paper by Tompson et al."""

    def __init__(self, scene_path, pose_config_path, shape_config_path,
                 sample_transform=None, target_transform=None,
                 num_points=1000, train=True, noise_coeff=0.0):
        """
        Args:
            root_dir (string): Path to the data.
            sample_transform (callable, optional): Optional transform to be
                applied to the sample.
            target_transform (callable, optional): Optional transform to be
                applied to the target.
            num_points (int, optional): Number of points to sample in the
                point cloud.
        """
        # self.pose_gen = PoseGenerator.PoseGenerator(scene_path, config_path)
        self.scene_path = scene_path
        self.pose_config_path = pose_config_path
        self.shape_config_path = shape_config_path
        self.sample_transform = sample_transform
        self.target_transform = target_transform
        self.num_points = num_points
        self.train = train
        self.num_kp = 26
        self.noise_coeff = noise_coeff
        self.proj = np.array([[1.302294, 0.0, 0.0, 0.0],
                              [0.0, 1.732051, 0.0, 0.0],
                              [0.0, 0.0, -1.025316, -0.202532],
                              [0.0, 0.0, -1.0, 1.0]])
        self.proj_inv = np.linalg.inv(self.proj)
        self.keypoint_names = ['carpals',
                               'metacarpals',
                               'finger5joint1',    # 2
                               'finger5joint2',
                               'finger5joint3',
                               'finger5joint3tip',
                               'Bone',
                               'finger1joint1',    # 7
                               'finger1joint2',
                               'finger1joint3',
                               'finger1joint3tip',
                               'Bone.001',
                               'finger2joint1',    # 12
                               'finger2joint2',
                               'finger2joint3',
                               'finger2joint3tip',
                               'Bone.002',
                               'finger3joint1',    # 17
                               'finger3joint2',
                               'finger3joint3',
                               'finger3joint3tip',
                               'Bone.003',
                               'finger4joint1',    # 22
                               'finger4joint2',
                               'finger4joint3',
                               'finger4joint3tip']

    def setup(self, idx):
        self.pose_gen = PoseGenerator.PoseGenerator(self.scene_path,
                                                    self.pose_config_path,
                                                    self.shape_config_path)

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_sample()

    def get_sample(self):
        pose_sample = self.pose_gen.GetSample()

        sample, joint_map = pose_sample.depth, pose_sample.joints
        h, w = sample.shape

        keypoint_gt = np.zeros((self.num_kp, 3))

        for idx, val in enumerate(self.keypoint_names):
            keypoint_gt[idx] = joint_map[val]
            keypoint_gt[idx, 2] *= -1000.0

        kps2d = keypoint_gt.copy()
        kps3d = self.uvd_to_xyz(kps2d.copy(), h, w)

        sample = np.asarray(sample, np.float32)
        sample = self.process_depth(sample)

        kps3d = torch.tensor(kps3d, dtype=torch.float32)

        bbox = self.get_bbox(kps2d[1:])
        norm_size = torch.norm(kps3d[23] - kps3d[22])
        center = kps3d[1:].mean(0)

        sample, padding = self.crop_depth(sample, bbox)
        # target = self.depth_to_pc(sample.copy(), bbox, padding)
        # target = torch.tensor(target, dtype=torch.float32)

        # target = self.normalize(target, center, norm_size)
        kps3d = self.normalize(kps3d, center, norm_size)

        if self.sample_transform:
            sample = self.sample_transform(sample)
        sample = self.normalize_depth(sample)

        if self.noise_coeff > 0:
            mask_idxs = sample != 1
            noise = torch.rand_like(sample) * self.noise_coeff
            sample[mask_idxs] += noise[mask_idxs]
            sample[sample > 1] = 1

        kps3d[:, 2] *= -1.0
        # target[:, 2] *= -1.0

        # return sample, target, kps3d
        return sample, kps3d

    def crop_depth(self, img, bbox):
        """Crop the depth image to the bounding box.

        If the cropped image is not square, 0-value padding will be added.

        Args:
            img (float, H x W x D): Depth array.
            bbox (float, 6): Bounding box of the hand in image space.

        Returns:
            Cropped image (float, H_c x W_c x D) and the row and column
            padding size added to the image (int, 2 x 2).
        """
        xstart = bbox[0]
        xend = bbox[1]
        ystart = bbox[2]
        yend = bbox[3]
        zstart = bbox[4]
        zend = bbox[5]

        cropped = img[max(ystart, 0):min(yend, img.shape[0]), max(xstart, 0):min(xend, img.shape[1])].copy()
        #
        # Crop z bound
        mask1 = np.logical_and(cropped < zstart, cropped != 0)
        mask2 = np.logical_and(cropped > zend, cropped != 0)
        cropped[mask1] = zstart
        cropped[mask2] = 0.0

        if cropped.shape[0] > cropped.shape[1]:
            diff = cropped.shape[0] - cropped.shape[1]
            row_pad = [0, 0]
            if diff % 2 == 1:
                col_pad = [int(diff / 2), int(diff / 2) + 1]
            else:
                col_pad = [int(diff / 2), int(diff / 2)]
        else:
            diff = cropped.shape[1] - cropped.shape[0]
            col_pad = [0, 0]
            if diff % 2 == 1:
                row_pad = [int(diff / 2), int(diff / 2) + 1]
            else:
                row_pad = [int(diff / 2), int(diff / 2)]

        return np.pad(cropped, (row_pad, col_pad), mode='constant', constant_values=0), (row_pad, col_pad)


    def get_bbox(self, keypoints, pad=20):
        """Calculates a 3d bounding box.

        Args:
            keypoints (array): 3d keypoints of the hand in either image or 3d
              space.
            pad (int): Amount of padding to add to the bounding box for all
              sides.
        Returns:
            6 values defining the bounding cube.
        """

        joints_min = keypoints.min(0) - pad
        joints_max = keypoints.max(0) + pad
        return np.array([joints_min[0], joints_max[0],
                         joints_min[1], joints_max[1],
                         joints_min[2], joints_max[2]]).astype(np.int)

    def process_depth(self, depth, depth_min=0.1, depth_max=8.0):
        depth = depth.copy()
        # The closest value is -1
        bg_idxs = depth == 0
        depth[bg_idxs] = 0.0
        return depth

    def uvd_to_xyz(self, uvd_points, height, width):
        proj = np.array([[1.302294, 0.0, 0.0, 0.0],
                         [0.0, 1.732051, 0.0, 0.0],
                         [0.0, 0.0, -1.025316, -0.202532],
                         [0.0, 0.0, -1.0, 0.0]])

        num_points = uvd_points.shape[0]

        z_vals = uvd_points[:, 2]
        half_height = height / 2
        half_width = width / 2
        #
        # Convert to HCS
        uvd_points[:, 0] = (uvd_points[:, 0] - half_width) / half_width
        uvd_points[:, 1] = ((half_height - uvd_points[:, 1]) / half_height)
        uvd_points[:, 0] *= uvd_points[:, 2]
        uvd_points[:, 1] *= uvd_points[:, 2]
        uvd_points = np.concatenate((uvd_points, np.ones((num_points, 1))), axis=1)

        # HCS -> World
        points_xyz = uvd_points @ self.proj_inv
        points_xyz = points_xyz[:, :3]
        points_xyz[:, 2] = z_vals

        return points_xyz

    def normalize_depth(self, depth_img):
        """Normalize depth image to be in range [0, 1].
        Returns a clone of the original image.
        """
        norm_img = depth_img.clone()
        bg_mask = (norm_img == 0)
        fg_mask = (norm_img > 0)
        min_val = norm_img[fg_mask].min()
        max_val = norm_img[fg_mask].max()
        norm_img[fg_mask] -= min_val
        norm_img[fg_mask] /= (max_val - min_val)
        norm_img[fg_mask] = 1 - norm_img[fg_mask]
        # norm_img[fg_mask] *= 2.0
        # norm_img[fg_mask] -= 1.0
        norm_img[bg_mask] = 0

        return norm_img

    def normalize(self, points, center, norm_size):
        """Normalize a set of points centered on `center` and scaled by
        `norm_size`.

        Args:
            center (float, array): Location to center object.
            norm_size (float): Scale factor.
        Returns:
            Normalized array of points.
        """
        #
        # Normalize
        norm_points = points.clone()
        norm_points -= center
        norm_points /= norm_size

        return norm_points

    def depth_to_pc(self, depth_img, bbox, padding):
        """Transforms the depth image into a point cloud representation.

        Args:
            depth_img (array): depth image.
            bbox (float, array): bounding box of the hand in (u, v, d).
            padding (int, array): row and column padding added to the cropped
              image from earlier pre-processing.
        Returns:
            Point cloud representation of hand.
        """

        xstart = bbox[0] - padding[1][0]
        ystart = bbox[2] - padding[0][0]

        # Convert to point cloud
        depth_img = torch.from_numpy(depth_img).unsqueeze(0)
        p_ndc = get_point_cloud(depth_img, self.num_points, 0)
        p_ndc = p_ndc.squeeze(0).numpy()
        p_ndc[:, 0] += xstart
        p_ndc[:, 1] += ystart
        pc = self.uvd_to_xyz(p_ndc, 480, 640)

        return pc
