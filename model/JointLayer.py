import torch
import torch.nn.functional as F

import utils.quaternion as quat
from base import BaseModel
from utils.util import register_hook


class JointLayer(BaseModel):
    """Calculates rotation between two vectors."""
    def __init__(self,  offset, orientation):
        super(JointLayer, self).__init__()
        self.orientation = torch.nn.Parameter(orientation, requires_grad=False)
        self.offset = torch.nn.Parameter(offset, requires_grad=False)

    def forward(self, input_offsets, q_parents):
        """ Calculates the quaternion rotations which transform the model
        offsets to the input offsets.

        Inputs are transformed to the space shared by the model using the
        inverse of `q_parents`. Being in the same space allows a simple
        calculation of the difference in rotation.

        Args:
            input_offsets (N, 3): Input vectors of the keypoint w.r.t. their
                parent.
            q_parents (N, 4): Quaternion transformations which provide the
                total derived transformation up to the parent.
        """
        model_offsets = self.offset.repeat(input_offsets.shape[0], 1)
        derived_orientation = quat.qmul(q_parents,
                                        self.orientation.repeat(q_parents.shape[0], 1))
        inputs_object = quat.q_rot(quat.q_inv(derived_orientation), input_offsets)
        q_diffs = quat.find_q_v(F.normalize(model_offsets),
                                F.normalize(inputs_object))
        new_qs = quat.qmul(self.orientation.repeat(q_diffs.shape[0], 1),
                           q_diffs)
        new_q_parents = quat.qmul(q_parents, new_qs)
        new_ps = quat.q_rot(quat.q_inv(new_q_parents), input_offsets)

        return new_q_parents, new_qs, new_ps
