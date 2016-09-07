#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_FEATUREEXTRAPOLATING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_FEATUREEXTRAPOLATING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.

bool FeatureExtrapolatingForwardLaucher(
    const float* bottom_data, const int num_scale_base, 
    const int num_scale, const int num_top, const int channels_trace, const int height, const int width, const int channels,
    const int* is_real_scales, const int* which_base_scales, const float* rescaling_factors,
    float* top_data, float* trace_data, const Eigen::GpuDevice& d);

bool FeatureExtrapolatingBackwardLaucher(const float* top_diff, const int num_scale_base, const int num_scale, const int channels_trace, const int batch_size,
    const int height, const int width, const int channels, const int* which_base_scales, const float* rescaling_factors,
    float* bottom_diff, const float* trace_data, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MAXPOOLING_OP_GPU_H_
