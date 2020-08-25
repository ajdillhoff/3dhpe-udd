#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <vector>
#include <algorithm>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *) addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *) addr, __float_as_uint(value)));

    return old;
}

// Renderer kernel in which each thread represents a 3D point.
__global__ void kernel_depth_renderer_point_forward(const float* points,
        const int batch_size, const int num_points,
        float* z_buffer, const float radius,
        const int height, const int width) {

    const int point_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (point_id >= num_points) {
        return;
    }

    for (int b = 0; b < batch_size; b++) {
        float x = points[b * num_points * 3 + point_id * 3 + 0];
        float y = points[b * num_points * 3 + point_id * 3 + 1];
        float z = points[b * num_points * 3 + point_id * 3 + 2];

        int x_min = max(0, __float2int_rd(x - radius));
        int x_max = min(width, __float2int_ru(x + radius));
        int y_min = max(0, __float2int_rd(y - radius));
        int y_max = min(height, __float2int_ru(y + radius));

        for (int i = x_min; i < x_max; i++) {
            for (int j = y_min; j < y_max; j++) {
                float d_point = (i - x) * (i - x) + (j - y) * (j - y);
                if (d_point < radius * radius) {
                    float phi = 1 - (d_point / (radius * radius));
                    phi *= phi;
                    float depth = (1 - z) * phi;
                    atomicMaxFloat(&z_buffer[height * width * b + j * width + i], depth);
                }
            }
        }
    }
}

__global__ void kernel_depth_renderer_point_backward(const float* grad_output,
        const float* points,
        const float* z_buffer,
        float* grad,
        const int batch_size,
        const int num_points,
        const float radius,
        const int height,
        const int width) {

    const int point_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_id >= num_points) {
        return;
    }

    for (int b = 0; b < batch_size; b++) {
        float x = points[b * num_points * 3 + point_id * 3 + 0];
        float y = points[b * num_points * 3 + point_id * 3 + 1];
        float z = points[b * num_points * 3 + point_id * 3 + 2];

        int x_min = max(0, __float2int_rd(x - radius));
        int x_max = min(width, __float2int_ru(x + radius));
        int y_min = max(0, __float2int_rd(y - radius));
        int y_max = min(height, __float2int_ru(y + radius));

        for (int i = x_min; i < x_max; i++) {
            for (int j = y_min; j < y_max; j++) {
                float d_point = (i - x) * (i - x) + (j - y) * (j - y);
                if (d_point < radius * radius) {
                    float phi = 1 - (d_point / (radius * radius));
                    phi *= phi;
                    float depth = (1 - z) * phi;

                    if (depth == z_buffer[b * height * width + j * width + i]) {
                        float h = (4.0 * (1.0 - z) / (radius * radius)) * (1 - (d_point / (radius * radius)));
                        float g_out = grad_output[b * height * width + j * width + i];
                        grad[b * num_points * 3 + point_id * 3 + 2] += g_out * -phi;
                        if (phi > 0) {
                            grad[b * num_points * 3 + point_id * 3 + 0] += g_out * h * (i - x);
                            grad[b * num_points * 3 + point_id * 3 + 1] += g_out * h * (j - y);
                        }
                    }
                }
            }
        }
    }
}

// Renderer kernel in which each thread represents a specific pixel.
__global__ void kernel_depth_renderer_pixel_forward(const float* points,
        const int batch_size, const int num_points,
        float* img, const float radius,
        const int height, const int width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) {
        return;
    }

    float radius_inv = 1.0f / radius;

    for (int b = 0; b < batch_size; b++) {
        for (int p = 0; p < num_points; p++) {
            float x = points[b * num_points * 3 + p * 3 + 0];
            float y = points[b * num_points * 3 + p * 3 + 1];
            float z = points[b * num_points * 3 + p * 3 + 2];

            int x_min = max(0, __float2int_rd(x - radius));
            int x_max = min(width, __float2int_ru(x + radius));
            int y_min = max(0, __float2int_rd(y - radius));
            int y_max = min(height, __float2int_ru(y + radius));

            if (i < x_min || i >= x_max || j < y_min || j >= y_max) {
                continue;
            }

            float d_point = (x - i) * (x - i) + (y - j) * (y - j);
            if (d_point * d_point < radius * radius) {
                float phi = 1 - (d_point * radius_inv) * (d_point * radius_inv);
                phi *= phi;
                float depth = (1 - z) * phi;
                if (depth > img[height * width * b + j * width + i]) {
                    img[height * width * b + j * width + i] = depth;
                }
            }
        }
    }
}

torch::Tensor depth_renderer_forward(torch::Tensor input, int height, int width, float radius) {
    const int batch_size = input.size(0);
    const int num_points = input.size(1);
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(input.device())
        .requires_grad(true);
    auto img = torch::zeros({batch_size, height, width}, options);

    /*dim3 block(32, 32);*/
    /*dim3 grid(ceil(height / 32.0f), ceil(width / 32.0f));*/

    /*kernel_depth_renderer_forward<<<grid, block>>>(*/
            /*input.data<float>(),*/
            /*batch_size,*/
            /*num_points,*/
            /*img.data<float>(),*/
            /*radius,*/
            /*height,*/
            /*width);*/

    dim3 block(1024);
    dim3 grid(ceil(num_points / 1024.0f));

    kernel_depth_renderer_point_forward<<<grid, block>>>(
            input.data<float>(),
            batch_size,
            num_points,
            img.data<float>(),
            radius,
            height,
            width);

    cudaDeviceSynchronize();

    gpuErrorcheck(cudaPeekAtLastError());

    return img;
}

torch::Tensor depth_renderer_backward(torch::Tensor grad_output,
        torch::Tensor points,
        torch::Tensor z_buffer,
        int height,
        int width,
        float radius) {

    const int batch_size = grad_output.size(0);
    const int num_points = points.size(1);
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(grad_output.device())
        .requires_grad(true);
    auto grad = torch::zeros({batch_size, num_points, 3}, options);

    dim3 block(1024);
    dim3 grid(ceil(num_points / 1024.0f));

    kernel_depth_renderer_point_backward<<<grid, block>>>(
            grad_output.data<float>(),
            points.data<float>(),
            z_buffer.data<float>(),
            grad.data<float>(),
            batch_size,
            num_points,
            radius,
            height,
            width);

    cudaDeviceSynchronize();

    gpuErrorcheck(cudaPeekAtLastError());

    return grad;
}
