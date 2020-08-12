import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance
from mpl_toolkits.mplot3d import Axes3D

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import model.DepthGen as DepthGen
from datasets.data_loaders import (
    LHSynthGenDataLoader,
    LHSynthDataLoader,
    NYUDataLoader,
    ITSEDataLoader,
    ITSEFDataLoader,
    NYUMultiDataLoader
)
from model.loss import physical_loss
from model import AlexNetHM, PartCapsuleNet
from model.ResNet import resnet18
from model.VGG import vgg16_bn, vgg16_gn
from utils.util import batched_index_select
from utils.visualization import plot_hands2d


class DepthGenLM(pl.LightningModule):

    def __init__(self, hparams):
        super(DepthGenLM, self).__init__()

        if hparams.estimator_arch == 'AlexNetHM':
            estimator = AlexNetHM.AlexNetHM(hparams.num_joints,
                                            hparams.num_features)
        elif hparams.estimator_arch == 'PartCapsuleNet':
            estimator = PartCapsuleNet.PartCapsuleNet([1, 1, 1],
                                                      hparams.num_joints,
                                                      hparams.num_features)
        elif hparams.estimator_arch == 'VGGbn':
            estimator = vgg16_bn(num_joints=hparams.num_joints,
                                 num_features=hparams.num_features)
        elif hparams.estimator_arch == 'VGGgn':
            estimator = vgg16_gn(num_joints=hparams.num_joints,
                                 num_features=hparams.num_features)
        elif hparams.estimator_arch == 'ResNet':
            estimator = resnet18(num_joints=hparams.num_joints,
                                 num_features=hparams.num_features)

        self.estimator = estimator
        self.depth_gen = DepthGen.DepthGen(hparams.mesh_path,
                                           hparams.skeleton_path,
                                           image_height=hparams.height,
                                           image_width=hparams.width,
                                           sample_size=hparams.num_points)
        self.learning_rate = hparams.learning_rate
        self.hparams = hparams
        self.step_idx = 0

    def forward(self, x):
        pose_params = self.estimator(x)
        preds = self.depth_gen(pose_params)

        return preds

    def training_step(self, batch, batch_idx):
        tensorboard_logs = {}
        if self.hparams.train_dataset == 'nyu_synth':
            loss_dict = self.lh_synth_step(batch)
            loss = loss_dict['loss']
            epe = loss_dict['epe']
            tensorboard_logs['train_epe'] = epe.mean()
            # if self.hparams.use_chamfer_loss:
                # tensorboard_logs['chamfer_loss'] = loss_dict['chamfer_loss']
        elif self.hparams.train_dataset == 'nyu':
            loss, epe, _ = self.nyu_step(batch)
            tensorboard_logs['train_epe'] = epe.mean()
        elif self.hparams.train_dataset == 'nyu_multi' or\
            self.hparams.train_dataset == 'nyu_multi_synth':
            loss_dict = self.nyu_multi_step(batch)
            loss = loss_dict['loss']
            epe_21 = loss_dict['epe_21']
            tensorboard_logs['multiview_loss'] = loss_dict['multiview_loss']
            tensorboard_logs['train_epe21'] = epe_21.mean()
            # if self.hparams.use_chamfer_loss:
                # tensorboard_logs['chamfer_loss'] = loss_dict['chamfer_loss']
        elif self.hparams.train_dataset == 'lh_synth_gen':
            loss_dict = self.lh_synth_step(batch)
            loss = loss_dict['loss']
            epe = loss_dict['epe']
            tensorboard_logs['train_epe'] = epe.mean()
            # if self.hparams.use_chamfer_loss:
                # tensorboard_logs['chamfer_loss'] = loss_dict['chamfer_loss']
        elif self.hparams.train_dataset == 'itse':
            loss_dict = self.itse_step(batch, batch_idx)
            loss = loss_dict['loss']
            # tensorboard_logs['chamfer_loss'] = loss_dict['chamfer_loss']

        self.step_idx += 1
        tensorboard_logs['train_loss'] = loss
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        if self.hparams.val_dataset == 'lh_synth':
            loss_dict = self.lh_synth_step(batch)
            loss = loss_dict['loss']
            epe = loss_dict['epe']
            epe_21 = torch.zeros_like(epe)
        elif self.hparams.val_dataset == 'nyu':
            loss, epe, epe_21 = self.nyu_step(batch)

        return {'val_loss': loss, 'val_epe': epe, 'val_epe_21': epe_21}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        epe_mean = torch.cat([x['val_epe'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_epe_mean': epe_mean}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        if self.hparams.test_dataset == 'lh_synth':
            loss_dict = self.lh_synth_step(batch)
            # TODO: Refactor this
            loss = loss_dict['loss']
            test_epe = loss_dict['epe']
            test_epe_21 = torch.zeros_like(test_epe)
        elif self.hparams.test_dataset == 'nyu':
            loss, test_epe, test_epe_21 = self.nyu_step(batch)
        elif self.hparams.test_dataset == 'itsef':
            loss_dict = self.itsef_step(batch)
            loss = loss_dict['loss']
            test_epe = loss_dict['epe']
            test_epe_21 = torch.zeros_like(test_epe)

        return {'test_loss': loss, 'test_epe': test_epe, 'test_epe_21': test_epe_21}

    def test_end(self, outputs):
        epe = torch.cat([x['test_epe'] for x in outputs])
        epe_21 = torch.cat([x['test_epe_21'] for x in outputs])

        np.save('./results/temp.npy', epe.cpu().numpy())
        np.save('./results/temp21.npy', epe_21.cpu().numpy())

        self.report_joint_error(epe)
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        epe_mean = epe.mean()
        epe_21_mean = epe_21.mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_epe_mean': epe_mean, 'test_epe_21_mean': epe_21_mean}

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                lr=self.learning_rate)

    def train_dataloader(self):
        data_path = self.hparams.train_dataset_path
        dataset_name = self.hparams.train_dataset
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        num_points = self.hparams.num_points

        if dataset_name == 'lh_synth':
            return LHSynthDataLoader(data_path, batch_size,
                                     self.hparams.train_noise_coeff, 0,
                                     num_workers, num_points)
        elif dataset_name == 'nyu':
            return NYUDataLoader(data_path, batch_size, True,
                                 num_workers, num_points, True)
        elif dataset_name == 'nyu_multi':
            return NYUMultiDataLoader(data_path, batch_size, True,
                                      num_workers, num_points, False, True)
        elif dataset_name == 'nyu_multi_synth':
            return NYUMultiDataLoader(data_path, batch_size, True,
                                      num_workers, num_points, True, True)
        elif dataset_name == 'lh_synth_gen':
            return LHSynthGenDataLoader(self.hparams.train_scene_path,
                                        self.hparams.train_pose_config_path,
                                        self.hparams.train_shape_config_path,
                                        10,
                                        batch_size,
                                        num_workers,
                                        num_points,
                                        self.hparams.train_noise_coeff)
        elif dataset_name == 'itse':
            return ITSEDataLoader(data_path, batch_size, True,
                                  num_workers, self.hparams.idx_range,
                                  num_points)
        elif dataset_name == 'itsef':
            return ITSEFDataLoader(data_path, batch_size, True,
                                   num_workers, num_points)

    def val_dataloader(self):
        data_path = self.hparams.val_dataset_path
        dataset_name = self.hparams.val_dataset
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        num_points = self.hparams.num_points

        if dataset_name == 'lh_synth':
            return LHSynthDataLoader(data_path, batch_size,
                                     self.hparams.val_noise_coeff, 0,
                                     num_workers, num_points)
        elif dataset_name == 'nyu':
            return NYUDataLoader(data_path, batch_size, False,
                                 num_workers, num_points, False)
        elif dataset_name == 'itsef':
            return ITSEFDataLoader(data_path, batch_size, False,
                                   num_workers, num_points)

    def test_dataloader(self):
        data_path = self.hparams.test_dataset_path
        dataset_name = self.hparams.test_dataset
        batch_size = self.hparams.batch_size
        num_workers = self.hparams.num_workers
        num_points = self.hparams.num_points
        if dataset_name == 'lh_synth':
            return LHSynthDataLoader(data_path, batch_size,
                                     self.hparams.train_noise_coeff, 0,
                                     num_workers, num_points)
        elif dataset_name == 'nyu':
            return NYUDataLoader(data_path, batch_size, False,
                                 num_workers, num_points, False)
        elif dataset_name == 'itsef':
            return ITSEFDataLoader(data_path, batch_size, False,
                                   num_workers, num_points)

    ############################
    ## Dataset-specific steps ##
    ############################

    def lh_synth_step(self, batch):
        x, kps_gt = batch
        preds = self.forward(x)
        imgs = preds[0]
        coords_pred = preds[1]
        rot_offset = preds[4]
        bone_weights = preds[7]
        model_idxs = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 5]

        # Calculate losses
        loss_dict = {}
        loss = 0.0
        if self.hparams.use_coord_loss is True:
            coord_loss = F.mse_loss(coords_pred, kps_gt)
            loss_dict['coord_loss'] = coord_loss
            loss += coord_loss
        if self.hparams.use_physical_loss is True:
            p_loss = self.hparams.physical_lambda * physical_loss(rot_offset)
            loss_dict['p_loss'] = p_loss
            loss += p_loss
        # if self.hparams.use_chamfer_loss is True:
            # chamfer_loss, _ = chamfer_distance(sampled_points, y)
            # loss_dict['chamfer_loss'] = chamfer_loss
            # chamfer_loss *= self.hparams.chamfer_lambda
            # loss += chamfer_loss
        if self.hparams.use_ss_loss is True:
            ss_loss = F.l1_loss(imgs, x)
            loss += ss_loss

        # EPE Metric (mm)
        coords_pred_mm = coords_pred.contiguous().view(coords_pred.shape[0], -1) * 50.0
        coords_pred_mm = coords_pred_mm.view(coords_pred.shape)
        keypoint_gt_mm = kps_gt.view(kps_gt.shape[0], -1) * 50.0
        keypoint_gt_mm = keypoint_gt_mm.view(kps_gt.shape)

        # Get keypoint-wise euclidean distance
        epe = torch.norm(coords_pred_mm - keypoint_gt_mm, 2, 2)

        loss_dict['loss'] = loss
        loss_dict['epe'] = epe

        return loss_dict

    def nyu_step(self, batch):
        x, y, kps_gt, kps14_gt, _, norm_size, _, _, has_anno = batch
        preds = self.forward(x)
        imgs = preds[0]
        coords_pred = preds[1]
        n_size = preds[2]
        mean = preds[3]
        rot_offset = preds[4]
        transforms = preds[5]
        bone_weights = preds[7]
        coords_pred_14 = coords_pred.clone()

        has_anno = has_anno.squeeze(-1)

        # `Bone` joints are not considered for keypoint loss.
        joint_idxs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 6]
        eval_21_idxs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]
        model_idxs = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 5]

        # Calculate losses
        loss = 0.0
        if self.hparams.use_coord_loss is True:
            coord_loss = F.mse_loss(coords_pred[:, joint_idxs], kps_gt)
            loss += coord_loss
        if self.hparams.use_physical_loss is True:
            p_loss = self.hparams.physical_lambda * physical_loss(rot_offset)
            loss += p_loss
        # if self.hparams.use_chamfer_loss is True:
            # chamfer_loss, _ = chamfer_distance(sampled_points, y)
            # chamfer_loss *= self.hparams.chamfer_lambda
            # loss += chamfer_loss
        if self.hparams.use_ss_loss is True:
            ss_loss = F.l1_loss(imgs, x)
            loss += ss_loss

        with torch.no_grad():
            # NYU 14 joint eval
            joint_to_nyu_idx = [10, 8, 15, 13, 20, 18, 25, 23, 5, 4, 3, 11]
            coords_pred_mm = coords_pred_14[:, joint_to_nyu_idx]

            # Apply transform to palm and wrist points
            M = transforms[:, 1] @ self.depth_gen.mesh_transform.transforms_inv[:, 1]
            extra_points = torch.tensor([
                [-0.025, -0.1, 0.0, 1.0],
                [0.015, -0.1, 0.0, 1.0],
            ], device=y.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
            extra_points_t = torch.einsum('bij,bjk->bik', M, extra_points.permute(0, 2, 1))
            extra_points_t = extra_points_t.permute(0, 2, 1)
            extra_points_t = extra_points_t[:, :, :3]
            extra_points_t -= mean.repeat(1, 2, 1)
            extra_points_t /= n_size.repeat(1, 2, 1)
            coords_pred_mm = torch.cat((coords_pred_mm[:, :-1], extra_points_t, coords_pred_mm[:, -1].unsqueeze(1)), 1)

            coords_pred_mm = coords_pred_mm.contiguous().view(coords_pred.shape[0], -1) * norm_size.unsqueeze(1)
            coords_pred_mm = coords_pred_mm.view(kps14_gt.shape)
            keypoint_gt_mm = kps14_gt.view(kps14_gt.shape[0], -1) * norm_size.unsqueeze(1)
            keypoint_gt_mm = keypoint_gt_mm.view(kps14_gt.shape)

            # Get keypoint-wise euclidean distance
            epe = torch.norm(coords_pred_mm - keypoint_gt_mm, 2, 2)

            # 21 joint eval
            coords_pred_21_mm = coords_pred[:, eval_21_idxs]
            pred_21_shape = coords_pred_21_mm.shape
            coords_pred_21_mm = coords_pred_21_mm.contiguous().view(coords_pred_21_mm.shape[0], -1) * norm_size.unsqueeze(1)
            coords_pred_21_mm = coords_pred_21_mm.view(pred_21_shape)

            kps_gt_21_mm = kps_gt[:, :-1]
            kps_gt_21_mm_shape = kps_gt_21_mm.shape
            kps_gt_21_mm = kps_gt_21_mm.view(kps_gt_21_mm.shape[0], -1) * norm_size.unsqueeze(1)
            kps_gt_21_mm = kps_gt_21_mm.view(kps_gt_21_mm_shape)

            epe_21 = torch.norm(coords_pred_21_mm - kps_gt_21_mm, 2, 2)

        return loss, epe, epe_21

    def nyu_multi_step(self, batch):
        view1, view2 = batch
        x1, y1, kps1, _, _, norm_size1, _, _ = view1
        x2, y2, kps2, _, _, _, _, _ = view2
        preds1, _ = self.forward(x1)
        points1 = preds1[0]
        coords1 = preds1[2]
        params1 = preds1[6]
        preds2, _ = self.forward(x2)
        points2 = preds2[0]
        coords2 = preds2[2]
        params2 = preds2[6]

        joint_idxs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25, 6]
        eval_21_idxs = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25]

        # Calculate losses
        loss_dict = {}
        loss = 0.0
        if self.hparams.use_coord_loss is True:
            coord_loss1 = F.mse_loss(coords1[:, joint_idxs], kps1)
            coord_loss2 = F.mse_loss(coords2[:, joint_idxs], kps2)
            coord_loss = coord_loss1 + coord_loss2
            loss_dict['coord_loss'] = coord_loss
            loss += coord_loss
        if self.hparams.use_physical_loss is True:
            p_loss1 = self.hparams.physical_lambda * physical_loss(params1)
            p_loss2 = self.hparams.physical_lambda * physical_loss(params2)
            p_loss = p_loss1 + p_loss2
            loss_dict['p_loss'] = p_loss
            loss += p_loss
        if self.hparams.use_chamfer_loss is True:
            chamfer_loss1, _ = chamfer_distance(points1, y1)
            chamfer_loss1 *= self.hparams.chamfer_lambda
            chamfer_loss2, _ = chamfer_distance(points2, y2)
            chamfer_loss2 *= self.hparams.chamfer_lambda
            chamfer_loss = chamfer_loss1 + chamfer_loss2
            loss_dict['chamfer_loss'] = chamfer_loss
            loss += chamfer_loss

        # For multiview, estimated parameters should be identical except
        # for estimated global rotation.
        multiview_loss = F.mse_loss(params1[:, 1:], params2[:, 1:])

        loss_dict['multiview_loss'] = multiview_loss
        loss_dict['loss'] = loss

        with torch.no_grad():
            # 21 joint eval
            coords_pred_21_mm = coords1[:, eval_21_idxs]
            pred_21_shape = coords_pred_21_mm.shape
            coords_pred_21_mm = coords_pred_21_mm.contiguous().view(coords_pred_21_mm.shape[0], -1) * norm_size1.unsqueeze(1)
            coords_pred_21_mm = coords_pred_21_mm.view(pred_21_shape)

            kps_gt_21_mm = kps1[:, :-1]
            kps_gt_21_mm_shape = kps_gt_21_mm.shape
            kps_gt_21_mm = kps_gt_21_mm.view(kps_gt_21_mm.shape[0], -1) * norm_size1.unsqueeze(1)
            kps_gt_21_mm = kps_gt_21_mm.view(kps_gt_21_mm_shape)

            epe_21 = torch.norm(coords_pred_21_mm - kps_gt_21_mm, 2, 2)
            loss_dict['epe_21'] = epe_21

        return loss_dict

    def itse_step(self, batch, batch_idx):
        x, y = batch
        preds, default_preds = self.forward(x, True)
        sampled_points = preds[0]
        out_idxs = preds[1]
        bone_weights = preds[8]

        if batch_idx == 0:
            # log points
            x_vis = sampled_points[0].detach().cpu()
            y_vis = y[0].cpu()
            c = x_vis.mean(0)
            b = 1.5
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(90, -90)
            ax.set_xlim([c[0] - b, c[0] + b])
            ax.set_ylim([c[1] - b, c[1] + b])
            ax.set_zlim([c[2] - b, c[2] + b])
            ax.scatter(x_vis[:, 0], x_vis[:, 1], x_vis[:, 2])
            ax.scatter(y_vis[:, 0], y_vis[:, 1], y_vis[:, 2])
            fig.tight_layout()
            self.logger.experiment.add_figure('pc', fig, self.step_idx)

        default_points = default_preds[0]

        loss = 0.0

        loss_dict = {}
        if self.hparams.use_physical_loss is True:
            rot_offset = preds[6]
            p_loss = self.hparams.physical_lambda * physical_loss(rot_offset)
            loss_dict['p_loss'] = p_loss
            loss += p_loss
        if self.hparams.use_chamfer_loss is True:
            chamfer_loss, _ = chamfer_distance(sampled_points, y,
                                               batch_reduction=None)
            chamfer_loss *= self.hparams.chamfer_lambda
            loss += chamfer_loss
            loss_dict['chamfer_loss'] = chamfer_loss
        if self.hparams.use_hand_prior is True:
            chamfer_loss, _ = chamfer_distance(sampled_points, y,
                                               batch_reduction=None)

            # Check default loss
            default_chamfer, _ = chamfer_distance(default_points, y,
                                                  batch_reduction=None)

            default_index = default_chamfer < chamfer_loss

            if default_index.any().item():
                params_est = preds[6]
                default_params = default_preds[6]
                param_loss = F.l1_loss(params_est[default_index],
                                       default_params[default_index])
                loss += param_loss
            if ~default_index.any().item():
                chamfer_loss = chamfer_loss[~default_index].mean()
                chamfer_loss *= self.hparams.chamfer_lambda
                loss += chamfer_loss
                loss_dict['chamfer_loss'] = chamfer_loss

        loss_dict['loss'] = loss

        return loss_dict

    def itsef_step(self, batch):
        x, y, x_3d = batch
        preds, default_preds = self.forward(x, True)
        sampled_points = preds[0]
        out_idxs = preds[1]
        y_pred = preds[2]
        bone_weights = preds[8]

        default_points = default_preds[0]

        loss = 0.0

        loss_dict = {}
        if self.hparams.use_physical_loss is True:
            rot_offset = preds[6]
            p_loss = self.hparams.physical_lambda * physical_loss(rot_offset)
            loss_dict['p_loss'] = p_loss
            loss += p_loss
        if self.hparams.use_chamfer_loss is True:
            chamfer_loss, _ = chamfer_distance(sampled_points, x_3d)
            chamfer_loss *= self.hparams.chamfer_lambda
            loss += chamfer_loss
            loss_dict['chamfer_loss'] = chamfer_loss
        if self.hparams.use_hand_prior is True:
            chamfer_loss, _ = chamfer_distance(sampled_points, x_3d,
                                               batch_reduction=None)

            # Check default loss
            default_chamfer, _ = chamfer_distance(default_points, x_3d,
                                                  batch_reduction=None)

            default_index = default_chamfer < chamfer_loss

            if default_index.any().item():
                params_est = preds[6]
                default_params = default_preds[6]
                param_loss = F.l1_loss(params_est[default_index],
                                       default_params[default_index])
                loss += param_loss
            if ~default_index.any().item():
                chamfer_loss = chamfer_loss[~default_index].mean()
                chamfer_loss *= self.hparams.chamfer_lambda
                loss += chamfer_loss
                loss_dict['chamfer_loss'] = chamfer_loss

        # Evaluation
        joint_idxs = [5, 25, 20, 15, 10, 1, 6]

        epe = torch.norm(y_pred[:, joint_idxs, :2] - y[:, :, :2], 2, 2)

        for i in range(x.shape[0]):
            kp_vis = y[i].detach().cpu()
            coords_vis = y_pred[i, joint_idxs].detach().cpu()
            kp_fig = plt.figure()
            kp_ax = kp_fig.add_subplot(111)
            kp_ax.set_title("MSE: {}".format(epe[i].mean()))
            kp_ax.axis('equal')
            plot_hands2d(kp_ax, kp_vis[:, :2], color='r')
            plot_hands2d(kp_ax, coords_vis[:, :2], color='b')
            annotations = [str(i) for i in range(7)]
            for j, anno in enumerate(annotations):
                kp_ax.text(kp_vis[j, 0], kp_vis[j, 1], anno)
                kp_ax.text(coords_vis[j, 0], coords_vis[j, 1], anno)

            kp_fig.savefig("figs/{}.png".format(i))
            plt.close()

        loss_dict['epe'] = epe
        loss_dict['loss'] = loss

        return loss_dict

    def report_joint_error(self, joint_errors):
        joint_idx = list(range(13, -1, -1))+[14]
        names = ['Palm', 'Wrist1', 'Wrist2', 'Thumb.R1', 'Thumb.R2', 'Thumb.T', 'Index.R', 'Index.T', 'Mid.R', 'Mid.T', 'Ring.R', 'Ring.T', 'Pinky.R', 'Pinky.T', 'Mean']
        mean = joint_errors.mean(0)

        print("Individual Mean Joint Errors (mm)")
        print("*********************************")
        for i, v in enumerate(joint_idx):
            if i == 14:
                err = mean.mean()
            else:
                err = mean[v]
            print("{0:10}: {1}".format(names[i], err))

    def report_joint_error2(self, joint_errors):
        joint_idx = list(range(6, -1, -1))+[7]
        names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little', 'Wrist', 'Palm', 'Mean']
        mean = joint_errors.mean(0)

        print("Individual Mean Joint Errors (mm)")
        print("*********************************")
        for i, v in enumerate(joint_idx):
            if i == 7:
                err = mean.mean()
            else:
                err = mean[v]
            print("{0:10}: {1}".format(names[i], err))
