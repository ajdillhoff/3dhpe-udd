import os
import sys
import pickle

import cv2
import torch
import torchvision
import scipy.io as sio
import numpy as np
from PIL import Image

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.transforms import get_point_cloud


class ITSEDataset(torch.utils.data.Dataset):
    """Dataset class for ITSE experimental data."""

    def __init__(self, root_dir, sample_transform=None,
                 idx_range=[150, 6559], num_points=1024):
        """Class initialization.

        Args:
            root_dir (string): Path to the dataset.
            num_points (int): Number of points in point cloud.
        """
        self.root_dir = root_dir
        self.sample_transform = sample_transform
        self.idx_range = idx_range
        self.num_points = num_points
        self.pad = 20

    def __len__(self):
        return self.idx_range[1] - self.idx_range[0]

    def __getitem__(self, idx):
        idx += self.idx_range[0]
        color_path = os.path.join(self.root_dir, '{}_color.png'.format(idx))
        depth_path = os.path.join(self.root_dir, '{}_16bit.png'.format(idx))

        depth_img = Image.open(depth_path)
        color_img = Image.open(color_path).resize((640, 480))
        gray_img = color_img.convert('L')
        gray_arr = np.asarray(gray_img, np.int32)
        color_arr = np.asarray(color_img, np.int32)
        depth_arr = np.asarray(depth_img, np.float32)

        black_mask = (gray_arr < 30)
        # Tower is present in image, can do this manually
        black_mask[:, :100] = False
        black_mask[:, 500:] = False
        color_thresh = color_arr.copy()
        color_thresh[black_mask, :] = 0
        depth_thresh = depth_arr.copy()
        d = np.mean(depth_thresh[black_mask])
        depth_arr[depth_arr > d] = 0
        depth_arr[black_mask] = 0
        depth_arr[:, 450:] = 0
        depth_arr[:50, :] = 0
        sample = depth_arr.copy()

        # Morpholocal opening
        kernel = np.ones((7, 7), np.uint8)
        depth_arr = cv2.morphologyEx(depth_arr, cv2.MORPH_OPEN, kernel)

        # Get hand boundaries
        locs = np.argwhere(depth_arr > 0)
        min_y, min_x = np.min(locs, 0) - self.pad
        max_y, max_x = np.max(locs, 0) + self.pad
        bbox = [min_x, max_x, min_y, max_y]
        cropped_depth, padding = self.crop_depth(depth_arr, bbox)
        sample, _ = self.crop_depth(sample, bbox)
        target = self.depth_to_pc(cropped_depth.copy(), bbox, padding)
        target = torch.tensor(target, dtype=torch.float32)
        center = target.mean(0)
        norm_size = 1

        if self.sample_transform:
            sample = self.sample_transform(sample)
        else:
            sample = torch.tensor(sample, dtype=torch.float32)

        sample = self.normalize_depth(sample)
        target = self.normalize(target, center, 50)

        return sample, target

    def convert_uvd_to_xyz(self, points_uvd):
        """Converts points in image space to 3D space."""
        cx_d = 333.55
        cy_d = 227.51
        fx_d = 591.04
        fy_d = 592.63

        points_xyz = np.zeros_like(points_uvd)
        points_xyz[:, 0] = (points_uvd[:, 0] - cx_d) * points_uvd[:, 2] / fx_d
        points_xyz[:, 1] = (cy_d - points_uvd[:, 1]) * points_uvd[:, 2] / fy_d
        points_xyz[:, 2] = -points_uvd[:, 2]

        return points_xyz

    def crop_depth(self, img, bbox):
        """Crop the depth image to the bounding box.

        If the cropped image is not square, 0-value padding will be added.

        Args:
            img (float, H x W x D): Depth array.
            bbox (float, 4): Bounding box of the hand in image space.

        Returns:
            Cropped image (float, H_c x W_c x D) and the row and column
            padding size added to the image (int, 2 x 2).
        """
        xstart = bbox[0]
        xend = bbox[1]
        ystart = bbox[2]
        yend = bbox[3]

        cropped = img[max(ystart, 0):min(yend, img.shape[0]), max(xstart, 0):min(xend, img.shape[1])].copy()

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
        norm_img[fg_mask] *= 2.0
        norm_img[fg_mask] -= 1.0
        norm_img[bg_mask] = 1.0

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
        pc = self.convert_uvd_to_xyz(p_ndc)

        return pc
