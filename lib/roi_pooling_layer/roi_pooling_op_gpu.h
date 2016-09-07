#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_ROIPOOLING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_ROIPOOLING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool ROIPoolForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data, const Eigen::GpuDevice& d);

bool ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const int* argmax_data, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
