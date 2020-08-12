"""Utility functions related to 3D camera manipulation."""

import math
import numpy as np


def build_perspective_matrix(aspect_ratio, fov, near_clip, far_clip):
    """Builds a perspective transformation matrix.

    Implementation following (https://github.com/google/tf_mesh_renderer/) and
    gluPerspective (third_party/GL/glu/include/GLU/glu.h).

    Arguments:
        aspect_ratio: [float] The image aspect ratio.
        fov: [float] Field of view in degrees.
        near_clip: [float] Near clipping plane distance.
        far_clip: [float] Far clipping plane distance.

    Returns:
        [float numpy array shape=(4, 4)] Matrix transformation from eye space
        to clip space.
    """

    focal_lengths_y = 1. / math.tan(fov * (math.pi / 360.))
    depth_range = far_clip - near_clip

    perspective_matrix = np.zeros((4, 4))

    # Set zoom values
    perspective_matrix[0, 0] = focal_lengths_y / aspect_ratio
    perspective_matrix[1, 1] = focal_lengths_y

    # Set projection values
    perspective_matrix[2, 2] = -(far_clip + near_clip) / depth_range
    perspective_matrix[2, 3] = -2. * far_clip * near_clip / depth_range
    perspective_matrix[3, 2] = -1.

    return perspective_matrix


def camera_look_at(eye, center, up):
    """Builds a camera transformation matrix.

    Implemented following (https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml).

    Arguments:
        eye: [float vector] Position of the eye point.
        center: [float vector] Position of the reference point.
        up: [float vector] Direction of the up vector.

    Returns:
        camera_matrix: [float numpy array shape=(4, 4)] The resulting
                       transformation matrix.
    """

    Z = eye - center
    Z_norm = np.linalg.norm(Z)
    Z /= Z_norm

    Y = up / np.linalg.norm(up)

    X = np.cross(Y, Z)

    Y = np.cross(Z, X / np.linalg.norm(X))

    camera_matrix = np.zeros((4, 4))
    camera_matrix[0, :3] = X
    camera_matrix[1, :3] = Y
    camera_matrix[2, :3] = Z
    camera_matrix[3, 3] = 1.

    # translate by -eye
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = eye.T

    camera_matrix = camera_matrix @ translation_matrix

    # Implementation from https://stackoverflow.com/questions/21830340/understanding-glmlookat
    # Z = eye - center
    # Z /= np.linalg.norm(Z)
    # Y = up
    # X = np.cross(Y, Z)

    # Y = np.cross(Z, X)
    # X /= np.linalg.norm(X)
    # Y /= np.linalg.norm(Y)
    # camera_matrix[:3, 0] = X
    # camera_matrix[:3, 1] = Y
    # camera_matrix[:3, 2] = Z
    # camera_matrix[3, 3] = 1.0

    # camera_matrix[3, 0] = -np.dot(X, eye)
    # camera_matrix[3, 1] = -np.dot(Y, eye)
    # camera_matrix[3, 2] = -np.dot(Z, eye)

    return camera_matrix
