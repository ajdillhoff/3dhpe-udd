import os
import sys
import math

import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import utils.quaternion as quat


def chamfer_dist(pc1, pc2):
    """Chamfer distance between two point clouds."""
    N = pc1.shape[1]
    M = pc2.shape[1]

    pc1_expand = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand - pc2_expand
    pc_dist = (pc_diff ** 2).sum(-1)
    # pc_dist = torch.sqrt(pc_dist)
    # pc_dist = F.smooth_l1_loss(pc1_expand, pc2_expand, reduction='none')
    dist1, idx1 = pc_dist.min(2)
    dist2, idx2 = pc_dist.min(1)
    return dist1, idx1, dist2, idx2, pc_diff


def chamfer_dist_mask(pc1, pc2, mask, val=10.0):
    """Chamfer distance between two point clouds.

    The mask indicates the selected points corresponding between the two
    point clouds. The 0 values of the mask are set to a high value as to be
    ruled out of the minimum."""
    N = pc1.shape[1]
    M = pc2.shape[1]

    pc1_expand = pc1.unsqueeze(2).repeat(1, 1, M, 1)
    pc2_expand = pc2.unsqueeze(1).repeat(1, N, 1, 1)
    pc_diff = pc1_expand - pc2_expand
    pc_dist = (pc_diff ** 2).sum(-1)
    pc_dist = torch.sqrt(pc_dist)
    pc_dist[mask == 0] = val
    dist1, idx1 = pc_dist.min(2)
    dist2, idx2 = pc_dist.min(1)
    return dist1, idx1, dist2, idx2, pc_diff


def create_barycentric_transform(A):
    """Creates a transformation matrix used to calculate the barycentric
    coordinates of a point."""

    if len(A.shape) == 2:
        T = torch.tensor([[A[0, 0] - A[3, 0], A[1, 0] - A[3, 0], A[2, 0] - A[3, 0]],
                         [A[0, 1] - A[3, 1], A[1, 1] - A[3, 1], A[2, 1] - A[3, 1]],
                         [A[0, 2] - A[3, 2], A[1, 2] - A[3, 2], A[2, 2] - A[3, 2]]],
                         dtype=A.dtype, device=A.device)

    if len(A.shape) == 3:
        T = torch.zeros(A.shape[0], 3, 3, dtype=A.dtype, device=A.device)
        T[:, 0, 0] = A[:, 0, 0] - A[:, 3, 0]
        T[:, 0, 1] = A[:, 1, 0] - A[:, 3, 0]
        T[:, 0, 2] = A[:, 2, 0] - A[:, 3, 0]
        T[:, 1, 0] = A[:, 0, 1] - A[:, 3, 1]
        T[:, 1, 1] = A[:, 1, 1] - A[:, 3, 1]
        T[:, 1, 2] = A[:, 2, 1] - A[:, 3, 1]
        T[:, 2, 0] = A[:, 0, 2] - A[:, 3, 2]
        T[:, 2, 1] = A[:, 1, 2] - A[:, 3, 2]
        T[:, 2, 2] = A[:, 2, 2] - A[:, 3, 2]

    return T

def get_barycentric_coordinates(r, T, r4):
    """Returns the barycentric coordinates of r using transformation T and vertex r4."""

    if T.shape[0] == 1:
        T_inv = torch.inverse(T)
    else:
        T_inv = b_inv(T)

    coords = T_inv @ (r - r4)
    return coords

def b_inv(b_mat):
    """PyTorch batch matrix inverse.

    https://stackoverflow.com/questions/46595157/how-to-apply-the-torch-inverse-function-of-pytorch-to-every-sample-in-the-batc
    """

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def load_skeleton(skeleton_path):
    """Loads an Ogre skeletal model from XML.

    Args:
        skeleton_path (string): Path to skeleton XML file.

    Returns:
        skeleton (num_bones x 7): Skeleton tensor. The first 3 values are the
            position of the bone relative to the parent. The last 4 values are
            the rotation relative to the parent bone, represented as a
            quaternion.
        parent_map (list): A mapping from bone index to parent bone index.
        TODO: UPDATE HEADER
    """

    tree = ET.parse(skeleton_path)
    root = tree.getroot()

    # Process bones
    bones = root[0]
    num_bones = len(root[0])
    bone_names = []
    rotations = np.zeros((num_bones, 4))
    positions = np.zeros((num_bones, 3))

    for i in range(num_bones):
        bone_names.append(bones[i].attrib['name'])
        position = bones[i][0]
        rotation = bones[i][1]
        axis = rotation[0]
        positions[i] = np.array([float(position.attrib['x']),
                                 float(position.attrib['y']),
                                 float(position.attrib['z'])])
        rotations[i] = quat.axisangle_to_q([
                                    float(axis.attrib['x']),
                                    float(axis.attrib['y']),
                                    float(axis.attrib['z'])
                                ], float(rotation.attrib['angle']))

    # Process hierarchy
    bone_hierarchy = root[1]
    parent_map = [-1] # The root does not have a parent
    for i in range(len(bone_hierarchy)):
        parent_map.append(bone_names.index(bone_hierarchy[i].attrib['parent']))

    return rotations, positions, parent_map

def load_mesh_data(mesh_path):
    """Loads mesh vertices, bone assignments, and triangle IDs.

    Args:
        mesh_path - string: Path to the OGRE XML mesh data.

    Returns:
        mesh_vertices - array (N_v x 3): Mesh vertices, where N_v is the
            number of vertices.
        bone_weights - array (N_b x N_v): Bone weights, where N_b is the bone
            count and N_v is the number of vertices.
        triangles - array (N_f x 3): Triangle IDs, where N_f is the number of
            triangle faces in the mesh.
    """

    tree = ET.parse(mesh_path)
    root = tree.getroot()

    # Store all bone assignments
    bone_assignment_dict = {}
    bone_weight_dict = {}
    num_bones = 0
    for child in root[4]:
        key = 'vertex_' + str(child.attrib['vertexindex'])
        bone_index = int(child.attrib['boneindex'])
        if bone_index > num_bones:
            num_bones = bone_index

        if key in bone_assignment_dict:
            bone_weight_dict[key] = np.append(bone_weight_dict[key], np.array([float(child.attrib['weight'])]))
            bone_assignment_dict[key] = np.append(bone_assignment_dict[key], np.array([bone_index]))
        else:
            bone_weight_dict[key] = np.array([float(child.attrib['weight'])])
            bone_assignment_dict[key] = np.array([bone_index])

    num_bones += 1 # because num_bones is only as large as the biggest index.

    # Store the vertices
    mesh_vertices = np.zeros((int(root[0].attrib['vertexcount']), 3))
    normals = np.zeros((int(root[0].attrib['vertexcount']), 3))
    i = 0
    for child in root[0][0]:
        mesh_vertices[i, 0] = child[0].attrib['x']
        mesh_vertices[i, 1] = child[0].attrib['y']
        mesh_vertices[i, 2] = child[0].attrib['z']
        normals[i, 0] = child[1].attrib['x']
        normals[i, 1] = child[1].attrib['y']
        normals[i, 2] = child[1].attrib['z']
        i += 1

    # Build the bone_weights matrix
    # TODO: Testing needed
    bone_weights = np.zeros((num_bones, len(mesh_vertices)))
    i = 0
    for key, value in bone_assignment_dict.items():
        bone_assignments = value
        bone_weight = bone_weight_dict[key]
        bone_weights[bone_assignments, i] = bone_weight
        i += 1

    triangles_idxs = None
    vertex_map = [1, 2, 0]
    i = 0

    for submesh in root[1]:
        for faces in submesh:
            num_faces = int(faces.attrib['count'])
            if triangles_idxs is None:
                triangles_idxs = np.zeros((num_faces, 3), dtype=int)
            else:
                triangles_idxs = np.append(triangles_idxs, np.zeros((num_faces, 3), dtype=int), axis=0)

            for face in faces:
                j = 0
                for _, value in face.attrib.items():
                    triangles_idxs[i, vertex_map[j]] = int(value)
                    j += 1
                i += 1

    triangles = torch.from_numpy(triangles_idxs.astype(np.int32))

    return mesh_vertices, normals, bone_weights, triangles

def crop_and_resize(image, centers, crop_size, scale, mode='nearest'):
    """Crops and resizes the image using `torch.nn.functional.interpolate`.

    Args:
        image - Tensor (B x C x H x W): The input image.
        centers - Tensor (B x 2): Centers of the bounding boxes corresponding
            to each image.
        crop_size - int: The desired size in which to resize the result.
        scale - Tensor (B x 1): Scale factor for each image.

    Returns:
        cropped_images - Tensor (B x C x crop_size x crop_size): The resulting
            cropped and resized images.

    TODO: Only works on single images for now.
    """

    s = image.shape
    assert len(s) == 4, "Image needs to be of shape (B x C x H x W)"
    crop_location = centers.to(torch.float32)

    crop_size_scaled = math.ceil(float(crop_size) / scale)
    y1 = int(crop_location[:, 0] - crop_size_scaled // 2)
    y2 = int(y1 + crop_size_scaled)
    boxes = torch.tensor([0, 0, crop_size_scaled, crop_size_scaled], dtype=torch.int32)

    offset_y = 0
    if y1 < 0:
        offset_y = -y1
        boxes[0] = int(offset_y)
        y1 += offset_y
    if y2 > s[2]:
        offset_y = s[2] - y2
        boxes[2] = int(offset_y)
        y2 += offset_y

    x1 = int(crop_location[:, 1] - crop_size_scaled // 2)
    x2 = int(x1 + crop_size_scaled)
    offset_x = 0
    if x1 < 0:
        offset_x = -x1
        boxes[1] = int(offset_x)
        x1 += offset_x
    if x2 > s[3]:
        offset_x = s[3] - x2
        boxes[3] = int(offset_x)
        x2 += offset_x

    cropped_images = torch.zeros(s[0], s[1], crop_size_scaled, crop_size_scaled)
    cropped_images[:, :, boxes[0]:boxes[2], boxes[1]:boxes[3]] = image[:, :, y1:y2, x1:x2]
    cropped_images = F.interpolate(cropped_images, size=crop_size)

    return cropped_images


def calculate_padding(input_size, kernel_size, stride):
    """Calculates the amount of padding to add according to Tensorflow's
    padding strategy."""

    cond = input_size % stride

    if cond == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - cond, 0)

    if pad % 2 == 0:
        pad_val = pad // 2
        padding = (pad_val, pad_val)
    else:
        pad_val_start = pad // 2
        pad_val_end = pad - pad_val_start
        padding = (pad_val_start, pad_val_end)

    return padding


def plot_acc_curve(joint_errors):
    """Plot number of samples within moving accuracy threshold."""
    num_samples = joint_errors.shape[0]

    # Reported Accuracy (measured from paper)
    x_rep = np.array([0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 75.0])
    y_rep = np.array([0.0, 0.0, 0.04, 0.216, 0.43, 0.612, 0.75, 0.836, 0.866])
    x_vals = np.linspace(0, 75.0, num=1000)
    # y_int = np.interp(x_vals, x_rep, y_rep)
    f = interp1d(x_rep, y_rep, kind='cubic')
    y_int = f(x_vals)
    x = np.linspace(0.0, 75.0, num=1000)
    y = np.zeros((len(x)))
    for i in range(len(x)):
        y[i] = float((joint_errors < x[i]).sum()) / num_samples
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, c='r', label='This model')
    ax.plot(x_vals, y_int, c='b', label='Result from paper')
    ax.set_xlabel('Maximum allowed distance to GT (mm)')
    ax.set_ylabel('Fraction of frames within distance')
    ax.grid(True)
    ax.legend()
    plt.show()
