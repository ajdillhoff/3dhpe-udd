import torch
import depth_renderer_cuda, depth_renderer


class RendererFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, height, width, radius):
        if input.device == torch.device('cpu'):
            output = depth_renderer.forward(input, height, width, radius)
        else:
            output = depth_renderer_cuda.forward(input, height, width, radius)

        ctx.save_for_backward(input, output)
        ctx.height = height
        ctx.width = width
        ctx.radius = radius

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.device == torch.device('cpu'):
            raise NotImplementedError
        else:
            points, z_buffer = ctx.saved_tensors
            grad = depth_renderer_cuda.backward(grad_output,
                    points,
                    z_buffer,
                    ctx.height,
                    ctx.width,
                    ctx.radius)

        return grad, None, None, None


class Renderer(torch.nn.Module):
    """Implementation of the differential renderer defined in
    "How to Refine 3D Hand Pose Estimation from Unlabelled Depth Data ?" by
    Endri Dibra et al.

    Properties:
        height (int): output image height.
        width (int): output image width.
        radius (float): radius of circle used to threshold distance between
            pixels and vertices.
    """

    def __init__(self, height, width, radius):
        super(Renderer, self).__init__()
        self.height = height
        self.width = width
        self.radius = radius

    def forward(self, input):
        """Forward pass of the renderer.

        Renders the `input` points and produces a depth image.

        Args:
            input (Tensor, B x N x 3): 3D points to be rendered.

        Returns:
            depth_img (Tensor, B x H x W): Rendered depth image.
        """

        depth_img = RendererFunction.apply(input,
                                           self.height,
                                           self.width,
                                           self.radius)
        return depth_img
