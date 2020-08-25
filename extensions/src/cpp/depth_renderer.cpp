// CPP implementation of depth renderer described in
// "How to Refine 3D Hand Pose Estimation from Unlabelled Depth Data ?"
// by Dibra et al.
//
// Implemented by Alex Dillhoff (alex.dillhoff@mavs.uta.edu)

#include <cmath>
#include <algorithm>

#include <torch/extension.h>


torch::Tensor depth_renderer_forward(torch::Tensor points, int height, int width, float radius) {
    int batch_size = points.size(0);
    int num_points = points.size(1);

    torch::Tensor imgs = torch::zeros({batch_size, height, width}, torch::kFloat);
    auto points_ = points.contiguous();
    auto points_data = points_.data<float>();
    auto imgs_ = imgs.contiguous();
    auto imgs_data = imgs_.data<float>();

    for (int b = 0; b < batch_size; b++) {
        for (int p = 0; p < num_points; p++) {
            float x = points_data[b * num_points * 3 + p * 3 + 0];
            float y = points_data[b * num_points * 3 + p * 3 + 1];
            float z = points_data[b * num_points * 3 + p * 3 + 2];

            int x_min = std::max(0, (int)ceil(x - radius));
            int x_max = std::min(width, (int)floor(x + radius));
            int y_min = std::max(0, (int)ceil(y - radius));
            int y_max = std::min(height, (int)floor(y + radius));

            for (int i = x_min; i < x_max; i++) {
                for (int j = y_min; j < y_max; j++) {
                    float d_point = (i - x) * (i - x) + (j - y) * (j - y);
                    if (d_point < radius * radius) {
                        float phi = 1 - (d_point / (radius * radius));
                        phi *= phi;
                        float depth = (1 - z) * phi;
                        if (depth > imgs_data[b * height * width + j * width + i]) {
                            imgs_data[b * height * width + j * width + i] = depth;
                        }
                    }
                }
            }
        }
    }

    return imgs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &depth_renderer_forward, "Depth Renderer (Forward)");
}
