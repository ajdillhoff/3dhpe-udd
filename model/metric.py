import math

import torch
import numpy as np


class EvalUtil:
    """Calculates PCK and EPE."""
    def __init__(self, device, num_kp=21):
        self.data = torch.empty(num_kp, 0, device=device)
        self.num_kp = num_kp

    def clear(self):
        """Reset data."""
        self.data = torch.empty(self.num_kp, 0, device=self.data.device)

    def feed(self, keypoint_gt, keypoint_prediction):
        """Feeds ground truths and predictions to the class.

        Works with batches."""

        if len(keypoint_gt.shape) == 2:
            keypoint_gt = keypoint_gt.unsqueeze(0)
        if len(keypoint_prediction.shape) == 2:
            keypoint_prediction = keypoint_prediction.unsqueeze(0)

        # Get keypoint-wise euclidean distance
        diffs = keypoint_gt - keypoint_prediction
        euclidean_dist = torch.norm(diffs, 2, 2)
        self.data = torch.cat((self.data, euclidean_dist.t()), dim=1)

    def _get_pck(self, kp_idx, threshold):
        """Returns PCKfor one keypoint within the given threshold."""
        if self.data.shape[1] == 0:
            return None

        pck = self.data[kp_idx, :]
        pck = pck[pck <= threshold].mean()

        return pck

    def _get_epe(self, kp_idx):
        """Returns the end point error for a single keypoint."""
        if self.data.shape[1] == 0:
            return None, None

        epe_mean = self.data[kp_idx, :].mean()
        epe_median = self.data[kp_idx, :].median()

        return epe_mean, epe_median

    def get_measures(self, val_min=0.0, val_max=0.05, steps=20):
        """Outputs the average mean and median error as well as the PCK."""
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)

        # Init mean measures.
        epe_mean_all = torch.zeros(self.num_kp)
        epe_median_all = torch.zeros(self.num_kp)
        pck_curve_all = torch.zeros(self.num_kp, steps)

        # Create one plot for each keypoint.
        for kp_idx in range(self.num_kp):
            mean, median = self._get_epe(kp_idx)

            if mean is None:
                continue

            epe_mean_all[kp_idx] = mean
            epe_median_all[kp_idx] = median

            # PCK
            pck_curve = list()
            for t_idx, t in enumerate(thresholds):
                pck = self._get_pck(kp_idx, t)
                pck_curve_all[kp_idx, t_idx] = pck

        epe_mean_all = epe_mean_all.mean()
        epe_median_all = epe_median_all.median()
        pck_curve_all = pck_curve_all.mean(0)

        return epe_mean_all, epe_median_all, pck_curve_all, thresholds
