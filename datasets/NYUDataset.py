import os
import sys

import torch
import scipy.io as sio
import numpy as np
from PIL import Image

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.transforms import get_point_cloud


class NYUDataset(torch.utils.data.Dataset):
    """NYU Dataset."""

    def __init__(self, root_dir, sample_transform, target_transform,
                 idxs=[], num_points=8192, train=False, synth=False):
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
        self.root_dir = root_dir
        self.sample_transform = sample_transform
        self.target_transform = target_transform
        self.num_points = num_points
        self.num_kp = 21
        self.train = train
        if len(idxs) == 0:
            if not train:
                idxs = list([i for i in range(8252)])
            else:
                idxs = list([i for i in range(72757)])
        self.idxs = idxs
        self.synth = synth

        # Load annotation file
        anno_mat = sio.loadmat(os.path.join(self.root_dir, "joint_data.mat"))
        self.annotations2d = anno_mat['joint_uvd'][0]
        self.annotations3d = anno_mat['joint_xyz'][0]
        self.eval_joints = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        self.joint_idxs = [0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19, 21, 23, 24, 25, 26, 28, 35, 32]
        self.nyu_to_model_idxs = [20, 19, 18, 17, 16, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 21]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx] + 1
        if self.synth:
            depth_name = os.path.join(self.root_dir,
                                      'synthdepth_1_{0:07d}.png'.format(idx))
        else:
            depth_name = os.path.join(self.root_dir,
                                      'depth_1_{0:07d}.png'.format(idx))

        depth = Image.open(depth_name)
        w, h = depth.size

        # Process depth
        r, g, b = depth.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        depth = np.bitwise_or(np.left_shift(g, 8), b)
        depth = np.asarray(depth, np.float32)

        # TODO: Manually selecting input for now
        sample = depth
        target = depth

        kps2d = self.annotations2d[idx-1, self.joint_idxs]
        kps3d = self.annotations3d[idx-1, self.joint_idxs]

        # Convert kps3d order to match hand model
        kps3d = kps3d[self.nyu_to_model_idxs]
        kps3d = torch.tensor(kps3d, dtype=torch.float32)

        # Process and normalize joints
        bbox = self.get_bbox(kps2d)
        norm_size = torch.norm(kps3d[18] - kps3d[17])
        center = kps3d.mean(0)

        # kps2d = self.annotations2d[idx-1, self.eval_joints]
        sample, padding = self.crop_depth(sample, bbox)
        target = self.depth_to_pc(sample)

        target = torch.tensor(target, dtype=torch.float32)
        kps3d = self.normalize(kps3d, center, norm_size)

        # NYU 14 joint eval
        kps14 = self.annotations3d[idx-1, self.eval_joints].copy()
        kps14 = torch.tensor(kps14, dtype=torch.float32)
        kps14 = self.normalize(kps14, center, norm_size)
        kps14[:, 2] *= -1.0

        if self.sample_transform:
            sample = self.sample_transform(sample)

        sample = self.normalize_depth(sample)

        target[:, 2] *= 1.0
        kps3d[:, 2] *= -1.0

        if idx >= 72757:
            has_anno = torch.BoolTensor([False])
        else:
            has_anno = torch.BoolTensor([True])

        return sample, target, kps3d, kps14, center, norm_size, bbox, padding, has_anno

    def convert_uvd_to_xyz(self, points_uvd):
        """Converts points in image space to 3D space."""
        x_res = 640
        y_res = 480
        xz_factor = 1.08836710
        yz_factor = 0.817612648

        norm_x = points_uvd[:, 0] / x_res - 0.5
        norm_y = 0.5 - points_uvd[:, 1] / y_res

        points_xyz = np.zeros_like(points_uvd)
        points_xyz[:, 2] = points_uvd[:, 2]
        points_xyz[:, 0] = norm_x * points_xyz[:, 2] * xz_factor
        points_xyz[:, 1] = norm_y * points_xyz[:, 2] * yz_factor

        return points_xyz

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

    def normalize_depth(self, depth_img):
        """Normalize depth image to be in range [-1, 1].
        Returns a clone of the original image.
        """
        norm_img = depth_img.clone()
        bg_mask = (norm_img == 0)
        fg_mask = (norm_img > 0)
        min_val = norm_img[fg_mask].min()
        max_val = norm_img[fg_mask].max()
        norm_img[fg_mask] -= min_val
        norm_img[fg_mask] /= (max_val - min_val)
        # norm_img[fg_mask] *= 2.0
        # norm_img[fg_mask] -= 1.0
        # norm_img[bg_mask] = 1.0

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

        # Normalize
        norm_points = points.clone()
        norm_points -= center
        norm_points /= norm_size

        return norm_points

    def depth_to_pc(self, depth_img, pad=0.05):
        """Transforms the depth image into a point cloud representation.

        Args:
            depth_img (array): depth image.
        Returns:
            Point cloud representation of hand.
        """

        # Convert to point cloud
        depth_img = torch.from_numpy(depth_img).unsqueeze(0)
        p_pixel = get_point_cloud(depth_img, self.num_points, 0)
        p_pixel = p_pixel.squeeze(0).numpy()

        # Convert to [-1, 1] cube
        min_vals = p_pixel.min(0)
        max_vals = p_pixel.max(0)
        diff = np.abs(max_vals - min_vals)
        center = (max_vals + min_vals) / 2
        scale = diff[:2].max() * (0.5 + pad)
        p_norm = p_pixel - center
        p_norm /= scale
        p_norm[:, 2] += 0.5

        return p_norm
