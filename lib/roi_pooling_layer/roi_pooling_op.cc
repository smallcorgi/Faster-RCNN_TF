/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include <stdio.h>
#include <cfloat>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "work_sharder.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("RoiPool")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .Output("argmax: int32");

REGISTER_OP("RoiPoolGrad")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("argmax: int32")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class RoiPoolOp : public OpKernel {
 public:
  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    auto bottom_data_flat = bottom_data.flat<T>();
    auto bottom_rois_flat = bottom_rois.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // Create output tensors
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->template flat<T>();

    Tensor* argmax_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
    auto argmax = argmax_tensor->template flat<int>();

    int pooled_height = pooled_height_;
    int pooled_width = pooled_width_;
    float spatial_scale = spatial_scale_;

    auto shard = [pooled_height, pooled_width, spatial_scale,
                  num_rois, batch_size, data_height, data_width, num_channels,
                  &bottom_data_flat, &bottom_rois_flat, &output, &argmax]
                  (int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b)
      {
        // (n, ph, pw, c) is an element in the pooled output
        int n = b;
        int c = n % num_channels;
        n /= num_channels;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;

        const float* bottom_rois = bottom_rois_flat.data() + n * 5;
        int roi_batch_ind = bottom_rois[0];
        int roi_start_w = round(bottom_rois[1] * spatial_scale);
        int roi_start_h = round(bottom_rois[2] * spatial_scale);
        int roi_end_w = round(bottom_rois[3] * spatial_scale);
        int roi_end_h = round(bottom_rois[4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        const T bin_size_h = static_cast<T>(roi_height)
                           / static_cast<T>(pooled_height);
        const T bin_size_w = static_cast<T>(roi_width)
                           / static_cast<T>(pooled_width);

        int hstart = static_cast<int>(floor(ph * bin_size_h));
        int wstart = static_cast<int>(floor(pw * bin_size_w));
        int hend = static_cast<int>(ceil((ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil((pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = std::min(std::max(hstart + roi_start_h, 0), data_height);
        hend = std::min(std::max(hend + roi_start_h, 0), data_height);
        wstart = std::min(std::max(wstart + roi_start_w, 0), data_width);
        wend = std::min(std::max(wend + roi_start_w, 0), data_width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        const float* bottom_data = bottom_data_flat.data() + roi_batch_ind * num_channels * data_height * data_width;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = (h * data_width + w) * num_channels + c;
            if (bottom_data[bottom_index] > maxval) {
              maxval = bottom_data[bottom_index];
              maxidx = bottom_index;
            }
          }
        }
        output(b) = maxval;
        argmax(b) = maxidx;
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        num_rois * num_channels * pooled_height * pooled_width * spatial_scale;
    Shard(worker_threads.num_threads, worker_threads.workers,
          output.size(), shard_cost, shard);
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

bool ROIPoolForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, int* argmax_data, const Eigen::GpuDevice& d);

static void RoiPoolingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois,
    const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape)
{
  Tensor* output = nullptr;
  Tensor* argmax = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &argmax));

  if (!context->status().ok()) {
    return;
  }

  ROIPoolForwardLaucher(
    bottom_data->flat<float>().data(), spatial_scale, num_rois, height,
    width, channels, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class RoiPoolOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    RoiPoolingKernel(context, &bottom_data, &bottom_rois, spatial_scale_, num_rois, data_height,
      data_width, num_channels, pooled_height_, pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

// compute gradient
template <class Device, class T>
class RoiPoolGradOp : public OpKernel {
 public:
  explicit RoiPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& argmax_data = context->input(2);
    const Tensor& out_backprop = context->input(3);

    auto bottom_data_flat = bottom_data.flat<T>();
    auto bottom_rois_flat = bottom_rois.flat<T>();
    auto argmax_data_flat = argmax_data.flat<int32>();
    auto out_backprop_flat = out_backprop.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data.dims() == 4,
                errors::InvalidArgument("argmax_data must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    // Create output tensors
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->template flat<T>();

    int pooled_height = pooled_height_;
    int pooled_width = pooled_width_;
    float spatial_scale = spatial_scale_;

    auto shard = [pooled_height, pooled_width, spatial_scale,
                  num_rois, batch_size, data_height, data_width, num_channels,
                  &bottom_data_flat, &bottom_rois_flat, &argmax_data_flat,
                  &out_backprop_flat, &output](int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b)
      {
        // (n, h, w, c) coords in bottom data
        int n = b;
        int c = n % num_channels;
        n /= num_channels;
        int w = n % data_width;
        n /= data_width;
        int h = n % data_height;
        n /= data_height;

        float gradient = 0.0;
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n)
        {
          const float* offset_bottom_rois = bottom_rois_flat.data() + roi_n * 5;
          int roi_batch_ind = offset_bottom_rois[0];
          // Skip if ROI's batch index doesn't match n
          if (n != roi_batch_ind) {
            continue;
          }

          int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
          int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
          int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
          int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

          // Skip if ROI doesn't include (h, w)
          const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h);
          if (!in_roi) {
            continue;
          }

          int offset = roi_n * pooled_height * pooled_width * num_channels;
          const float* offset_top_diff = out_backprop_flat.data() + offset;
          const int* offset_argmax_data = argmax_data_flat.data() + offset;

          // Compute feasible set of pooled units that could have pooled
          // this bottom unit

          // Force malformed ROIs to be 1x1
          int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
          int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);

          const T bin_size_h = static_cast<T>(roi_height)
                             / static_cast<T>(pooled_height);
          const T bin_size_w = static_cast<T>(roi_width)
                             / static_cast<T>(pooled_width);

          int phstart = floor(static_cast<int>(h - roi_start_h) / bin_size_h);
          int phend = ceil(static_cast<int>(h - roi_start_h + 1) / bin_size_h);
          int pwstart = floor(static_cast<int>(w - roi_start_w) / bin_size_w);
          int pwend = ceil(static_cast<int>(w - roi_start_w + 1) / bin_size_w);

          phstart = std::min(std::max(phstart, 0), pooled_height);
          phend = std::min(std::max(phend, 0), pooled_height);
          pwstart = std::min(std::max(pwstart, 0), pooled_width);
          pwend = std::min(std::max(pwend, 0), pooled_width);

          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (offset_argmax_data[(ph * pooled_width + pw) * num_channels + c] == (h * data_width + w) * num_channels + c)
              {
                gradient += offset_top_diff[(ph * pooled_width + pw) * num_channels + c];
              }
            }
          }
        }
        output(b) = gradient;
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        num_rois * num_channels * pooled_height * pooled_width * spatial_scale;
    Shard(worker_threads.num_threads, worker_threads.workers,
          output.size(), shard_cost, shard);
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

bool ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const int* argmax_data, const Eigen::GpuDevice& d);

static void RoiPoolingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois, const Tensor* argmax_data, const Tensor* out_backprop,
    const float spatial_scale, const int batch_size, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape)
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  ROIPoolBackwardLaucher(
    out_backprop->flat<float>().data(), spatial_scale, batch_size, num_rois, height,
    width, channels, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax_data->flat<int>().data(), context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class RoiPoolGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  explicit RoiPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& argmax_data = context->input(2);
    const Tensor& out_backprop = context->input(3);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data.dims() == 4,
                errors::InvalidArgument("argmax_data must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int height = bottom_data.dim_size(1);
    // data width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int channels = bottom_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    RoiPoolingGradKernel(
      context, &bottom_data, &bottom_rois, &argmax_data, &out_backprop,
      spatial_scale_, batch_size, num_rois, height, width, channels, pooled_height_,
      pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("RoiPoolGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolGradOp<CPUDevice, float>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiPoolOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("RoiPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiPoolGradOp<Eigen::GpuDevice, float>);
#endif
