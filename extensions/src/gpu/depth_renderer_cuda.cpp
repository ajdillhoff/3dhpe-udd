#include <torch/extension.h>


torch::Tensor depth_renderer_forward(
        torch::Tensor input,
        int height,
        int width,
        float radius);

torch::Tensor depth_renderer_backward(
        torch::Tensor grad_output,
        torch::Tensor points,
        torch::Tensor z_buffer,
        int height,
        int width,
        float radius);

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor depth_renderer_forward_cuda(
        torch::Tensor input,
        int height,
        int width,
        float radius) {
    CHECK_INPUT(input);

    return depth_renderer_forward(input,
            height,
            width,
            radius);
}

torch::Tensor depth_renderer_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor points,
        torch::Tensor z_buffer,
        int height,
        int width,
        float radius) {
    CHECK_INPUT(z_buffer);
    CHECK_INPUT(points);

    return depth_renderer_backward(grad_output,
            points,
            z_buffer,
            height,
            width,
            radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depth_renderer_forward_cuda, "Depth Renderer (Forward, CUDA)");
    m.def("backward", &depth_renderer_backward_cuda, "Depth Renderer (Backward, CUDA)");
}
