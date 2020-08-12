import math

import numpy as np
import torch
import torch.nn.functional as f

from utils.util import register_hook


def rot_to_euler(M):
    """Convert a rotation matrix to Euler representation."""
    x = torch.atan2(M[:, 2, 1], M[:, 2, 2])
    M21s = M[:, 2, 1] * M[:, 2, 1]
    M22s = M[:, 2, 2] * M[:, 2, 2]
    y = torch.atan2(-M[:, 2, 0], torch.sqrt(M21s + M22s))
    z = torch.atan2(M[:, 1, 0], M[:, 0, 0])
    return torch.stack((x, y, z)).transpose(0, 1)


def axisangle_to_q(v, theta):
    """Converts an axis-angle representation to a Quaternion."""
    if not type(v) == torch.Tensor:
        v = torch.Tensor(v)

    norm = torch.norm(v)
    if norm == 0:
        norm_v = v
    else:
        norm_v = v / norm
    q = torch.cat((torch.zeros(1, device=v.device), norm_v))
    theta = theta / 2
    q[0] = np.cos(theta)
    q[1:] = q[1:] * np.sin(theta)

    return q


def q_to_axisangle(q):
    """Converts a Quaternion to axis-angle representation."""
    w, v = q[0], q[1:]
    theta = torch.arccos(w) * 2.0

    return v / torch.norm(v), theta


def q_to_euler(q):
    """Converts a Quaternion vector to an Euler angle vector.

    Arguments:
        q (B,4) - Quaternion vectors

    Returns:
        euler (B,3) - Euler angle vectors.
    """
    sinr_cosp = 2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    cosr_cosp = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
    x = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])
    idx = (sinp.abs() >= 1).nonzero()
    y = torch.asin(sinp)
    y[idx] = math.pi / 2.0

    siny_cosp = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
    cosy_cosp = 1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    z = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((x, y, z)).transpose(0, 1)


def euler_to_q(euler):
    """Converts an Euler angle vector to Quaternion.
    The Euler angle vector is expected to be (X, Y, Z).

    Arguments:
        euler (B,3) - Euler angle vectors.

    Returns:
        q (B,4) - Quaternions.
    """
    euler *= 0.5
    cx = euler[:, 0].cos()
    sx = euler[:, 0].sin()
    cy = euler[:, 1].cos()
    sy = euler[:, 1].sin()
    cz = euler[:, 2].cos()
    sz = euler[:, 2].sin()

    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = sz * cy * sx + cz * sy * cx
    z = sz * cy * cx - cz * sy * sx

    return torch.stack((w, x, y, z)).transpose(0, 1)


def q_mult(q1, q2):
    """Quaternion multiplication."""
    w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    x = q1[1] * q2[0] + q1[0] * q2[1] + q1[2] * q2[3] - q1[3] * q2[2]
    y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return torch.stack((w, x, y, z))


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    IMPLEMENATION BY https://github.com/facebookresearch/QuaterNet
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def q_conjugate(q):
    """Conjuage of Quaternion q."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)


def q_rot(q, v):
    """Rotate vector by the quaternion.
    Implementation by facebookresearch/QuaterNet.
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:,:1] * uv + uuv)).view(original_shape)


def qv_mult(q, v):
    """Quaternion-Vector multiplication."""
    q2 = torch.cat((torch.zeros(1, dtype=q.dtype, device=q.device), v))
    return q_mult(q_mult(q, q2), q_conjugate(q))


def q_inv(q1):
    """Inverse of quaternion q1."""
    qinv = q_conjugate(q1)
    d = torch.bmm(q1.view(q1.shape[0], 1, 4), q1.view(q1.shape[0], 4, 1)).squeeze(1)
    return qinv / d


def get_rotation_to(v1, v2):
    """Calculates the quaternion rotation from vector v1 to v2."""
    v1 = torch.squeeze(v1)
    v2 = torch.squeeze(v2)

    v1_norm = torch.norm(v1)
    v2_norm = torch.norm(v2)

    if v1_norm != 0:
        v1 = v1 / v1_norm
    if v2_norm != 0:
        v2 = v2 / v2_norm

    d = torch.dot(v1, v2)
    # TODO: Do these cases lead to erroneous gradient calculations?
    if d >= 1.0:
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=v1.device)

    if d <= -1.0:
        return torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=v1.device)

    normal = torch.cross(v1, v2)
    # Add small epsilon to avoid d/dx sqrt(0)
    a = torch.sqrt((1 + torch.dot(v1, v2)) * 2)
    inv = 1 / a
    b = normal * inv
    a = a * 0.5
    test = torch.cat((a.reshape(1), b))
    q = test / torch.norm(test)
    q = q.unsqueeze(0)

    return q


def get_rotation_to_v(input, other):
    """Calculates the quaternion rotation which transforms input to other."""
    input = f.normalize(input)
    other = f.normalize(other)
    a = torch.bmm(input.view(input.shape[0], 1, input.shape[1]),
                  other.view(other.shape[0], other.shape[1], 1)).squeeze(1)
    parallel_same = (a >= (1.0 - 1e-6))
    parallel_opp = (a < (1e-6 - 1.0))
    rest = (parallel_same + parallel_opp == 0).nonzero()[:, 0]
    s = torch.sqrt((1.0 + a[rest]) * 2)
    inv = 1.0 / s
    c = torch.cross(input, other)[rest]
    b = c * inv
    w = s * 0.5
    q = torch.cat((w, b), 1)

    r = torch.zeros(input.shape[0], 4, device=q.device, dtype=q.dtype)

    r[rest] = q
    r[parallel_same.nonzero()[:, 0], 0] = 1.0
    r[parallel_opp.nonzero()[:, 0], 1] = 1.0
    # q[parallel_same] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
    # q[parallel_opp] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
    return f.normalize(r)


def calc_normal(p):
    """Calculates the vector that is perpendicular to the plane defined by
    p."""
    if p.shape[0] == 2:
        normal = torch.cross(p[0], p[1])
        norm = torch.norm(normal)
        if norm != 0:
            normal = normal / torch.norm(normal)
        return normal

    a = p[1] - p[0]
    b = p[2] - p[0]

    normal = torch.cross(a, b)
    norm = torch.norm(normal)
    if norm == 0:
        return normal
    return normal / norm


def calc_normal_v(input, other):
    """Calculate the normalized cross product of 2 vectors."""
    n_dim = len(input.shape) - 1
    normal = torch.cross(input, other)
    return f.normalize(normal, dim=n_dim)


def find_q(p1, p2):
    """Calculates the rotation required to align the points in p1 with those
    in p2."""

    if p1.shape[0] == 1:
        return get_rotation_to(p1, p2)[0]

    n_p1 = calc_normal(p1)
    n_p2 = calc_normal(p2)

    q = get_rotation_to(n_p1, n_p2)

    p1_t = q_rot(q[0], p1[0])
    p1_t = p1_t / torch.norm(p1_t)
    q_r = get_rotation_to(p1_t, p2[0])

    qf = q_mult(q_r[0], q[0])

    return qf


def find_q_v(input, other):
    """Higher level function for calculating the quaternion rotation between
    two vectors or sets of vectors.

    If `input` and `other` contain more than one vector, this function first
    aligns the plane defined by the vectors in `input` to the plane defined by
    the vectors in `other`.
    """
    n_dims = len(input.shape)
    if n_dims == 2:
        return get_rotation_to_v(input, other)

    input_normal = calc_normal_v(input[:, 0], input[:, 1])
    other_normal = calc_normal_v(other[:, 0], other[:, 1])

    q = get_rotation_to_v(input_normal, other_normal)

    input_t = q_rot(q, input[:, 0])
    input_t = f.normalize(input_t)
    q_r = get_rotation_to_v(input_t, other[:, 0])

    return qmul(q_r, q)


def q_to_rotation_matrix(q):
    """Converts a quaternion (w, x, y, z) vector to a rotation matrix."""
    R = torch.empty(3, 3, dtype=q.dtype, device=q.device)
    R[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
    R[0, 1] = 2 * (q[1] * q[2] - q[3] * q[0])
    R[0, 2] = 2 * (q[1] * q[3] + q[2] * q[0])
    R[1, 0] = 2 * (q[1] * q[2] + q[3] * q[0])
    R[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
    R[1, 2] = 2 * (q[2] * q[3] - q[1] * q[0])
    R[2, 0] = 2 * (q[1] * q[3] - q[2] * q[0])
    R[2, 1] = 2 * (q[2] * q[3] + q[1] * q[0])
    R[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2

    return R

def q_to_rotation_matrix_v(q):
    """Batch implementation for converting quaternions into rotation matrices.
    """
    q = f.normalize(q)
    b = q.shape[0]
    r1_0 = 1 - 2 * q[:, 2] ** 2 - 2 * q[:, 3] ** 2
    r1_1 = 2 * (q[:, 1] * q[:, 2] - q[:, 3] * q[:, 0])
    r1_2 = 2 * (q[:, 1] * q[:, 3] + q[:, 2] * q[:, 0])
    r1_3 = torch.zeros(b, device=q.device, dtype=q.dtype)
    r1 = torch.stack((r1_0, r1_1, r1_2, r1_3)).transpose(0, 1).view(b, 1, -1)
    r2_0 = 2 * (q[:, 1] * q[:, 2] + q[:, 3] * q[:, 0])
    r2_1 = 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 3] ** 2
    r2_2 = 2 * (q[:, 2] * q[:, 3] - q[:, 1] * q[:, 0])
    r2_3 = torch.zeros(b, device=q.device, dtype=q.dtype)
    r2 = torch.stack((r2_0, r2_1, r2_2, r2_3)).transpose(0, 1).view(b, 1, -1)
    r3_0 = 2 * (q[:, 1] * q[:, 3] - q[:, 2] * q[:, 0])
    r3_1 = 2 * (q[:, 2] * q[:, 3] + q[:, 1] * q[:, 0])
    r3_2 = 1 - 2 * q[:, 1] ** 2 - 2 * q[:, 2] ** 2
    r3_3 = torch.zeros(b, device=q.device, dtype=q.dtype)
    r3 = torch.stack((r3_0, r3_1, r3_2, r3_3)).transpose(0, 1).view(b, 1, -1)
    r4 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=q.device, dtype=q.dtype)
    r4 = r4.repeat(b, 1).view(b, 1, -1)
    result = torch.cat((r1, r2, r3, r4), dim=1)
    return result

