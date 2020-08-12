import torch
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from utils.general import chamfer_dist, chamfer_dist_mask
from utils.quaternion import q_to_euler


class HLoss(Function):
    @staticmethod
    def forward(ctx, p1_vec, p2_vec, p1_loss, p1_act):
        output = (1 - p1_act) * p2_vec + p1_act * p1_vec
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output, None, None, None

def p_loss(diff, idx1, idx2, child_idxs, parent_loss, parent_vec, i, alpha):
    hloss = HLoss.apply

    # Second part
    input_diff = diff[i, child_idxs, idx1[i, child_idxs]]
    input_vec = input_diff.mean(0)

    # Target -> Input loss
    l = [[idx, val.item()] for idx, val in enumerate(idx2[i]) if val in child_idxs]
    if l:
        l = np.array(l)
        target_diff = diff[i, l[:, 0], l[:, 1]]
        target_vec = target_diff.mean(0)
        part_vec = (input_vec + target_vec)
    else:
        part_vec = input_vec

    parent_act = 1 / (1 + torch.exp(alpha - parent_loss.detach()) ** 16)
    part_loss = hloss(parent_vec, part_vec, parent_loss, parent_act)
    part_loss = part_loss.norm()

    return part_loss, part_vec

def h_loss(output, target, part_weights, alpha=1.0):
    """Hierarchical part loss.

    Arguments:
        output (B,N) - Estimated point clouds.
        target (B,N) - Target point clouds.
        part_weights (B,P,N) - Vertex weights of object model.
        alpha (float) - Activation parameter.
    """
    batch_size = output.shape[0]
    dist1, idx1, dist2, idx2, diff = chamfer_dist(output, target)
    loss = torch.empty(batch_size, part_weights.shape[1], requires_grad=False)

    for i in range(batch_size):
        p1_idxs = part_weights[i, 0].nonzero().squeeze()
        # Get the difference vectors associated with the first part
        p11_diff = diff[i, p1_idxs, idx1[i, p1_idxs]]
        p11_vec = p11_diff.mean(0)

        # Target -> Input loss
        l = [[idx, val.item()] for idx, val in enumerate(idx2[i]) if val in p1_idxs]
        l = np.array(l)
        p12_diff = diff[i, l[:, 0], l[:, 1]]
        p12_vec = p12_diff.mean(0)
        p1_vec = (p11_vec + p12_vec)
        p1_loss = p1_vec.norm()

        f1j1_loss, f1j1_vec = p_loss(diff, idx1, idx2, part_weights[i, 1].nonzero().squeeze(), p1_loss, p1_vec, i, alpha)
        f1j2_loss, f1j2_vec = p_loss(diff, idx1, idx2, part_weights[i, 2].nonzero().squeeze(), f1j1_loss, f1j1_vec, i, alpha)
        f1j3_loss, f1j3_vec = p_loss(diff, idx1, idx2, part_weights[i, 3].nonzero().squeeze(), f1j2_loss, f1j2_vec, i, alpha)
        f2j1_loss, f2j1_vec = p_loss(diff, idx1, idx2, part_weights[i, 4].nonzero().squeeze(), p1_loss, p1_vec, i, alpha)
        f2j2_loss, f2j2_vec = p_loss(diff, idx1, idx2, part_weights[i, 5].nonzero().squeeze(), f2j1_loss, f2j1_vec, i, alpha)
        f2j3_loss, f2j3_vec = p_loss(diff, idx1, idx2, part_weights[i, 6].nonzero().squeeze(), f2j2_loss, f2j2_vec, i, alpha)
        f3j1_loss, f3j1_vec = p_loss(diff, idx1, idx2, part_weights[i, 7].nonzero().squeeze(), p1_loss, p1_vec, i, alpha)
        f3j2_loss, f3j2_vec = p_loss(diff, idx1, idx2, part_weights[i, 8].nonzero().squeeze(), f3j1_loss, f3j1_vec, i, alpha)
        f3j3_loss, f3j3_vec = p_loss(diff, idx1, idx2, part_weights[i, 9].nonzero().squeeze(), f3j2_loss, f3j2_vec, i, alpha)
        f4j1_loss, f4j1_vec = p_loss(diff, idx1, idx2, part_weights[i, 10].nonzero().squeeze(), p1_loss, p1_vec, i, alpha)
        f4j2_loss, f4j2_vec = p_loss(diff, idx1, idx2, part_weights[i, 11].nonzero().squeeze(), f4j1_loss, f4j1_vec, i, alpha)
        f4j3_loss, f4j3_vec = p_loss(diff, idx1, idx2, part_weights[i, 12].nonzero().squeeze(), f4j2_loss, f4j2_vec, i, alpha)
        f5j1_loss, f5j1_vec = p_loss(diff, idx1, idx2, part_weights[i, 13].nonzero().squeeze(), p1_loss, p1_vec, i, alpha)
        f5j2_loss, f5j2_vec = p_loss(diff, idx1, idx2, part_weights[i, 14].nonzero().squeeze(), f5j1_loss, f5j1_vec, i, alpha)
        f5j3_loss, f5j3_vec = p_loss(diff, idx1, idx2, part_weights[i, 15].nonzero().squeeze(), f5j2_loss, f5j2_vec, i, alpha)

        loss[i, 0] = p1_loss # metacarpals
        loss[i, 1] = f1j1_loss # finger1joint1
        loss[i, 2] = f1j2_loss # finger1joint2
        loss[i, 3] = f1j3_loss # finger1joint3
        loss[i, 4] = f2j1_loss # finger2joint1
        loss[i, 5] = f2j2_loss # finger2joint2
        loss[i, 6] = f2j3_loss # finger2joint3
        loss[i, 7] = f3j1_loss # finger3joint1
        loss[i, 8] = f3j2_loss # finger3joint2
        loss[i, 9] = f3j3_loss # finger3joint3
        loss[i, 10] = f4j1_loss # finger4joint1
        loss[i, 11] = f4j2_loss # finger4joint2
        loss[i, 12] = f4j3_loss # finger4joint3
        loss[i, 13] = f5j1_loss # finger5joint1
        loss[i, 14] = f5j2_loss # finger5joint2
        loss[i, 15] = f5j3_loss # finger5joint3

    return loss


def mse_loss(output, target):
    return F.mse_loss(output, target)


def huber_loss(output, target):
    return F.smooth_l1_loss(output, target)


def physical_loss(output):
    """Penalize invalid predictions."""
    targets = torch.tensor([[[-1.57, 1.57],     # carpals
                             [-1.57, 1.57],
                             [-1.57, 1.57]],
                            [[-1.05, 1.05],     # metacarpals
                             [0, 0],
                             [-0.79, 0.26]],
                            [[-0.52, 0.52],     # finger5joint1
                             [0, 0],
                             [-0.35, 0.79]],
                            [[-0.26, 0.26],     # finger5joint2
                             [0, 0],
                             [0, 0]],
                            [[-1.05, 0],        # finger5joint3
                             [0, 0],
                             [0, 0]],
                            [[0, 0],            # Bone
                             [0, 0],
                             [0, 0]],
                            [[-1.05, 0.26],     # finger1joint1
                             [0, 0],
                             [-0.52, 0.26]],
                            [[-0.52, 0],        # finger1joint2
                             [0, 0],
                             [0, 0]],
                            [[-0.52, 0],        # finger1joint3
                             [0, 0],
                             [0, 0]],
                            [[0, 0],            # Bone.001
                             [0, 0],
                             [0, 0]],
                            [[-1.05, 0.26],     # finger2joint1
                             [0, 0],
                             [-0.52, 0.26]],
                            [[-0.52, 0],        # finger2joint2
                             [0, 0],
                             [0, 0]],
                            [[-0.52, 0],        # finger2joint3
                             [0, 0],
                             [0, 0]],
                            [[0, 0],            # Bone.002
                             [0, 0],
                             [0, 0]],
                            [[-1.05, 0.26],     # finger3joint1
                             [0, 0],
                             [-0.52, 0.26]],
                            [[-0.52, 0],        # finger3joint2
                             [0, 0],
                             [0, 0]],
                            [[-0.52, 0],        # finger3joint3
                             [0, 0],
                             [0, 0]],
                            [[0, 0],            # Bone.003
                             [0, 0],
                             [0, 0]],
                            [[-1.05, 0.26],     # finger4joint1
                             [0, 0],
                             [-0.52, 0.26]],
                            [[-0.52, 0],        # finger4joint2
                             [0, 0],
                             [0, 0]],
                            [[-0.52, 0],        # finger4joint3
                             [0, 0],
                             [0, 0]]], dtype=torch.float32, device=output.device)
    # targets = torch.tensor([[[-1.57, 1.57],     # carpals
                             # [-1.57, 1.57],
                             # [-1.57, 1.57]],
                            # [[-1.05, 1.05],     # metacarpals
                             # [0, 0],
                             # [-0.79, 0.26]],
                            # [[-0.52, 0.52],     # finger5joint1
                             # [0, 0],
                             # [-0.35, 0.79]],
                            # [[-0.26, 0.26],     # finger5joint2
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],        # finger5joint3
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],            # Bone
                             # [0, 0],
                             # [0, 0]],
                            # [[-1.05, 0.26],     # finger1joint1
                             # [0, 0],
                             # [-0.52, 0.26]],
                            # [[-0.52, 0],        # finger1joint2
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],        # finger1joint3
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],            # Bone.001
                             # [0, 0],
                             # [0, 0]],
                            # [[-1.05, 0.26],     # finger2joint1
                             # [0, 0],
                             # [-0.52, 0.26]],
                            # [[-0.52, 0],        # finger2joint2
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],        # finger2joint3
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],            # Bone.002
                             # [0, 0],
                             # [0, 0]],
                            # [[-1.05, 0.26],     # finger3joint1
                             # [0, 0],
                             # [-0.52, 0.26]],
                            # [[-0.52, 0],        # finger3joint2
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],        # finger3joint3
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],            # Bone.003
                             # [0, 0],
                             # [0, 0]],
                            # [[-1.05, 0.26],     # finger4joint1
                             # [0, 0],
                             # [-0.52, 0.26]],
                            # [[-0.52, 0],        # finger4joint2
                             # [0, 0],
                             # [0, 0]],
                            # [[0, 0],        # finger4joint3
                             # [0, 0],
                             # [0, 0]]], dtype=torch.float32, device=output.device)

    loss = 0.0
    for i in range(output.shape[1]):
        joint_angles = q_to_euler(output[:, i])
        loss += joint_loss(joint_angles, targets[i])

    return (loss / output.shape[1]).mean()


def joint_loss(output, target):
    """Compare Euler angles to given ranges.

    Args:
        output (B, 3) - Euler angles for a single joint.
        target (3, 2) - Acceptable Euler angle range for each dimension.

    Returns:
        Mean error across all items in the batch.
    """
    x_loss_min = target[0, 0] - output[:, 0]
    x_loss_min = torch.max(x_loss_min, torch.zeros_like(x_loss_min)).mean()
    x_loss_max = output[:, 0] - target[0, 1]
    x_loss_max = torch.max(x_loss_max, torch.zeros_like(x_loss_max)).mean()
    x_loss = x_loss_min + x_loss_max

    y_loss_min = target[1, 0] - output[:, 1]
    y_loss_min = torch.max(y_loss_min, torch.zeros_like(y_loss_min)).mean()
    y_loss_max = output[:, 1] - target[1, 1]
    y_loss_max = torch.max(y_loss_max, torch.zeros_like(y_loss_max)).mean()
    y_loss = y_loss_min + y_loss_max

    z_loss_min = target[2, 0] - output[:, 2]
    z_loss_min = torch.max(z_loss_min, torch.zeros_like(z_loss_min)).mean()
    z_loss_max = output[:, 2] - target[2, 1]
    z_loss_max = torch.max(z_loss_max, torch.zeros_like(z_loss_max)).mean()
    z_loss = z_loss_min + z_loss_max

    return (x_loss + y_loss + z_loss).mean()


def full_loss(preds, targets, chamfer_lambda, physical_lambda, has_anno=None, joint_idxs=None):
    """Blanket function for all losses used."""

    coords_pred = preds[2]
    if has_anno is not None:
        coords_pred = coords_pred[has_anno]
    if joint_idxs is not None:
        coords_pred = coords_pred[:, joint_idxs]

    # Coord Loss Only
    coord_loss = F.mse_loss(coords_pred, targets[0])
    loss = coord_loss

    # Chamfer Loss
    # with torch.no_grad():
        # coord_loss = F.mse_loss(coords_pred, targets[0])
    dist1, idx1, dist2, idx2, _ = chamfer_dist(preds[0], targets[1])
    chamfer_loss = (dist1 + dist2).mean()
    # loss = chamfer_lambda * chamfer_loss + coord_loss
    # loss = chamfer_loss

    # Physical loss
    p_loss = physical_loss(preds[6])
    # loss += (physical_lambda * p_loss)

    return loss, coord_loss, chamfer_loss, p_loss


class FullLoss(torch.nn.Module):
    def __init__(self, chamfer_lambda=0.1, physical_lambda=0.1):
        super(FullLoss, self).__init__()

        self.chamfer_lambda = chamfer_lambda
        self.physical_lambda = physical_lambda

    def forward(self, input, target, has_anno=None, joint_idxs=None):
        return full_loss(input, target, self.chamfer_lambda,
                         self.physical_lambda, has_anno=has_anno,
                         joint_idxs=joint_idxs)


def pointnet_loss(preds, targets, chamfer_lambda, physical_lambda, joint_lambda, has_anno=None, joint_idxs=None):
    """Implements loss used with PointNet."""

    gen_pc = preds[0]
    out_idxs = preds[1]
    coords_pred = preds[2]
    seg_out = preds[8]
    seg_out = seg_out.argmin(-1)
    bone_weights = preds[9]
    target_pc = targets[1]

    if joint_idxs is not None:
        bone_weights = bone_weights[joint_idxs]
    if has_anno is not None:
        coords_pred = coords_pred[has_anno]
    if joint_idxs is not None:
        coords_pred = coords_pred[:, joint_idxs]

    # Chamfer loss
    N = gen_pc.shape[1]
    M = target_pc.shape[1]

    pc1_expand = gen_pc.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand = target_pc.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand - pc2_expand
    pc_dist = (pc_diff ** 2).sum(-1)
    pc_dist = torch.sqrt(pc_dist)
    dist1, idx1 = pc_dist.min(2)
    dist2, idx2 = pc_dist.min(1)
    chamfer_loss = (dist1 + dist2).mean()
    loss = chamfer_lambda * chamfer_loss

    # Joint loss
    joint_loss = []
    for i in range(bone_weights.shape[0]):
        # Gen mask
        joint_weight = bone_weights[i]
        joint_weight = joint_weight.unsqueeze(0).expand(coords_pred.shape[0], -1)
        joint_weight = torch.gather(joint_weight, -1, out_idxs)
        joint_weight[joint_weight > 0.15] = 1 # threshold
        joint_weight[joint_weight < 1] = 0
        joint_weight_exp = joint_weight.unsqueeze(-1).expand(joint_weight.shape[0], joint_weight.shape[1], joint_weight.shape[1]).byte()

        # Target mask TODO: clean up masking
        part_mask = seg_out.clone()
        part_mask[part_mask != i] = -1
        part_mask[part_mask == i] = 1
        part_mask[part_mask == -1] = 0
        part_mask_exp = part_mask.byte().unsqueeze(1).expand(part_mask.shape[0], part_mask.shape[1], part_mask.shape[1])
        mask = joint_weight_exp & part_mask_exp
        part_dist = pc_dist.clone()
        part_dist[mask == 0] = 10

        d1, _ = part_dist.min(2)
        d2, _ = part_dist.min(1)

        # This is very hacky and I suspect it will cause all points to acquire
        # a gradient value.
        d1[d1 == 10] = 0
        d2[d2 == 10] = 0
        joint_loss.append((d1 + d2).mean())

    joint_loss = torch.tensor(joint_loss, device=target_pc.device)
    loss += (joint_lambda * joint_loss.mean())

    # Coord Loss Only
    coord_loss = F.mse_loss(preds[2], targets[0])
    loss += coord_loss

    # with torch.no_grad():
        # coord_loss = F.mse_loss(coords_pred, targets[0])

    # Physical loss
    p_loss = physical_loss(preds[6])
    loss += (physical_lambda * p_loss)

    return loss, coord_loss, joint_loss, chamfer_loss, p_loss


class PointNetLoss(torch.nn.Module):
    def __init__(self, chamfer_lambda=0.1, physical_lambda=0.1, joint_lambda=0.1):
        super(PointNetLoss, self).__init__()

        self.chamfer_lambda = chamfer_lambda
        self.physical_lambda = physical_lambda
        self.joint_lambda = joint_lambda

    def forward(self, input, target, has_anno=None, joint_idxs=None):
        return pointnet_loss(input, target,
                             self.chamfer_lambda, self.physical_lambda,
                             self.joint_lambda,
                             has_anno=has_anno, joint_idxs=joint_idxs)
