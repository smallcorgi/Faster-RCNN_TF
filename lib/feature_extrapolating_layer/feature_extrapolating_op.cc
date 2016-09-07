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

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("FeatureExtrapolating")
    .Attr("T: {float, double}")
    .Attr("scales_base: list(float)")
    .Attr("num_scale_base: int")
    .Attr("num_per_octave: int")
    .Input("bottom_data: T")
    .Output("top_data: T")
    .Output("trace_data: T");

REGISTER_OP("FeatureExtrapolatingGrad")
    .Attr("T: {float, double}")
    .Attr("scales_base: list(float)")
    .Attr("num_scale_base: int")
    .Attr("num_per_octave: int")
    .Input("bottom_data: T")
    .Input("trace_data: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class FeatureExtrapolatingOp : public OpKernel {
 public:
  explicit FeatureExtrapolatingOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the scales
    scales_base_.clear();
    OP_REQUIRES_OK(context,
                   context->GetAttr("scales_base", &scales_base_));
    // Get the num_scale_base
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_scale_base", &num_scale_base_));
    // Check that num_scale_base is positive
    OP_REQUIRES(context, num_scale_base_ >= 0,
                errors::InvalidArgument("Need num_scale_base >= 0, got ",
                                        num_scale_base_));
    // Get the num_per_octave
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_per_octave", &num_per_octave_));
    // Check that num_per_octave is positive
    OP_REQUIRES(context, num_per_octave_ >= 0,
                errors::InvalidArgument("Need num_per_octave >= 0, got ",
                                        num_per_octave_));

    num_scale_ = (num_scale_base_ - 1) * num_per_octave_ + 1;

    // compute scales
    scales_.clear();
    for(int i = 0; i < num_scale_; i++)
    {
      int index_scale_base = i / num_per_octave_;
      float sbase = scales_base_[index_scale_base];
      int j = i % num_per_octave_;
      float step = 0;
      if(j == 0)
        scales_.push_back(sbase);
      else
      {
        float sbase_next = scales_base_[index_scale_base+1];
        step = (sbase_next - sbase) / num_per_octave_;
        scales_.push_back(sbase + j * step);
      }
      printf("%f\n", scales_[i]);
    }

    // flags of real scales or approximated scales
    is_real_scales_.clear();
    for(int i = 0; i < num_scale_; i++)
      is_real_scales_.push_back(0);
    for(int i = 0; i < num_scale_base_; i++)
      is_real_scales_[i * num_per_octave_] = 1;

    // scale mapping
    which_base_scales_.clear();
    for(int i = 0; i < num_scale_; i++)
      which_base_scales_.push_back(int(roundf(float(i) / float(num_per_octave_))));

    // rescaling factors
    rescaling_factors_.clear();
    for(int i = 0; i < num_scale_; i++)
    {
      int scale_base_index = which_base_scales_[i];
      float scale_base = scales_base_[scale_base_index];
      float scale = scales_[i];
      rescaling_factors_.push_back(scale / scale_base);
    }

  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    auto bottom_data_flat = bottom_data.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // counting
    int num_image = batch_size / num_scale_base_;
    int num_top = num_image * num_scale_;

    // construct the output shape
    int dims[4];
    dims[0] = num_top;
    dims[1] = data_height;
    dims[2] = data_width;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // Create output tensors
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->template flat<T>();

    // trace data 8 channels
    int channels_trace = 8;
    dims[3] = channels_trace;
    TensorShape trace_shape;
    TensorShapeUtils::MakeShape(dims, 4, &trace_shape);

    Tensor* trace_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, trace_shape, &trace_tensor));
    auto trace = trace_tensor->template flat<T>();

    // compute extrapolated features
    for(int n = 0; n < num_top; n++)
    {
      int index_image = n / num_scale_;
      int index_scale = n % num_scale_;
      // flag for approximation or not
      int flag = is_real_scales_[index_scale];
      // which base scale to use
      int index_scale_base = which_base_scales_[index_scale];
      // rescaling factor
      double factor = rescaling_factors_[index_scale];

      // bottom batch image
      int index_batch = index_image * num_scale_base_ + index_scale_base;

      for(int h = 0; h < data_height; h++)
      {
        for(int w = 0; w < data_width; w++)
        {
          for (int c = 0; c < num_channels; ++c)
          {
            const int index = (h * data_width + w) * num_channels + c;
            output(n * data_height * data_width * num_channels + index) = 0;
            if(flag == 1) // no approximation
            {
              output(n * data_height * data_width * num_channels + index) = bottom_data_flat(index_batch * data_height * data_width * num_channels + index);
              // set tracing info
              if(c == 0)
              {
                for(int i = 0; i < channels_trace / 2; i++)
                {
                  trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i) = index_batch * data_height * data_width * num_channels + index;
                  trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i + 1) = 0.25;
                }
              }
            }
            else
            {
              // bilinear interpolation
              double xp = w / factor;
              double yp = h / factor;
              double cx[2], cy[2], ux, uy;
              int xi, yi, dx, dy, i;
              T val;
              if(xp >= 0 && xp < data_width && yp >= 0 && yp < data_height)
              {
                xi = (int)floor(xp); 
                yi = (int)floor(yp);
                ux = xp - (double)xi;
                uy = yp - (double)yi;
                cx[0] = ux;
                cx[1] = 1 - ux;
                cy[0] = uy;
                cy[1] = 1 - uy;

                val = 0;
                i = 0;
                for(dx = 0; dx <= 1; dx++)
                {
                  for(dy = 0; dy <= 1; dy++)
                  {
                    if(xi+dx >= 0 && xi+dx < data_width && yi+dy >= 0 && yi+dy < data_height)
                    {
                      val += cx[1-dx] * cy[1-dy] * bottom_data_flat(index_batch * data_height * data_width * num_channels + ((yi+dy) * data_width + (xi+dx)) * num_channels + c);
                      if(c == 0)
                      {
                        trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i) = 
                          index_batch * data_height * data_width * num_channels + ((yi+dy) * data_width + (xi+dx)) * num_channels + c;
                        trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i + 1) = cx[1-dx] * cy[1-dy];
                      }
                    }
                    else
                    {
                      if(c == 0)
                      {
                        trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i) = -1;
                        trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i + 1) = 0;
                      }
                    }
                    i++;
                  }
                }
                output(n * data_height * data_width * num_channels + index) = val;
              }
              else
              {
                // set tracing info
                if(c == 0)
                {
                  for(int i = 0; i < channels_trace / 2; i++)
                  {
                    trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i) = -1;
                    trace(n * channels_trace * data_height * data_width + (h * data_width + w) * channels_trace + 2 * i + 1) = 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
 private:
  int num_scale_base_;
  int num_scale_;
  int num_per_octave_;
  std::vector<float> scales_base_;
  std::vector<float> scales_;
  std::vector<int> is_real_scales_;
  std::vector<int> which_base_scales_;
  std::vector<float> rescaling_factors_;
};

REGISTER_KERNEL_BUILDER(Name("FeatureExtrapolating").Device(DEVICE_CPU).TypeConstraint<float>("T"), FeatureExtrapolatingOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("FeatureExtrapolating").Device(DEVICE_CPU).TypeConstraint<double>("T"), FeatureExtrapolatingOp<CPUDevice, double>);

void copy_gpu_data(void* data_gpu, void* data_cpu, int size);

bool FeatureExtrapolatingForwardLaucher(
    const float* bottom_data, const int num_scale_base, 
    const int num_scale, const int num_top, const int channels_trace, const int height, const int width, const int channels,
    const int* is_real_scales, const int* which_base_scales, const float* rescaling_factors,
    float* top_data, float* trace_data, const Eigen::GpuDevice& d);

static void FeatureExtrapolatingingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const int num_scale_base, 
    const int num_scale, const int num_top, const int channels_trace, const int height, const int width, const int channels, 
    const int* is_real_scales, const int* which_base_scales, const float* rescaling_factors,
    const TensorShape& tensor_output_shape, const TensorShape& tensor_trace_shape) 
{
  Tensor* output = nullptr;
  Tensor* trace = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_trace_shape, &trace));

  if (!context->status().ok()) {
    return;
  }

  FeatureExtrapolatingForwardLaucher(
    bottom_data->flat<float>().data(), num_scale_base, num_scale, num_top, channels_trace, height,
    width, channels, is_real_scales, which_base_scales, rescaling_factors,
    output->flat<float>().data(), trace->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class FeatureExtrapolatingOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit FeatureExtrapolatingOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the scales
    scales_base_.clear();
    OP_REQUIRES_OK(context,
                   context->GetAttr("scales_base", &scales_base_));
    // Get the num_scale_base
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_scale_base", &num_scale_base_));
    // Check that num_scale_base is positive
    OP_REQUIRES(context, num_scale_base_ >= 0,
                errors::InvalidArgument("Need num_scale_base >= 0, got ",
                                        num_scale_base_));
    // Get the num_per_octave
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_per_octave", &num_per_octave_));
    // Check that num_per_octave is positive
    OP_REQUIRES(context, num_per_octave_ >= 0,
                errors::InvalidArgument("Need num_per_octave >= 0, got ",
                                        num_per_octave_));

    num_scale_ = (num_scale_base_ - 1) * num_per_octave_ + 1;

    // compute scales
    scales_.clear();
    for(int i = 0; i < num_scale_; i++)
    {
      int index_scale_base = i / num_per_octave_;
      float sbase = scales_base_[index_scale_base];
      int j = i % num_per_octave_;
      float step = 0;
      if(j == 0)
        scales_.push_back(sbase);
      else
      {
        float sbase_next = scales_base_[index_scale_base+1];
        step = (sbase_next - sbase) / num_per_octave_;
        scales_.push_back(sbase + j * step);
      }
    }

    // flags of real scales or approximated scales
    is_real_scales_.clear();
    for(int i = 0; i < num_scale_; i++)
      is_real_scales_.push_back(0);
    for(int i = 0; i < num_scale_base_; i++)
      is_real_scales_[i * num_per_octave_] = 1;

    // scale mapping
    which_base_scales_.clear();
    for(int i = 0; i < num_scale_; i++)
      which_base_scales_.push_back(int(roundf(float(i) / float(num_per_octave_))));

    // rescaling factors
    rescaling_factors_.clear();
    for(int i = 0; i < num_scale_; i++)
    {
      int scale_base_index = which_base_scales_[i];
      float scale_base = scales_base_[scale_base_index];
      float scale = scales_[i];
      rescaling_factors_.push_back(scale / scale_base);
    }

    // construct the output shape
    int dims[4];
    dims[0] = 1;
    dims[1] = 1;
    dims[2] = 1;
    dims[3] = num_scale_;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    is_initialized_ = 0;
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT32, output_shape, &flags_, nullptr));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT32, output_shape, &mapping_, nullptr));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, output_shape, &factors_, nullptr));

    if (!context->status().ok()) {
      return;
    }

  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // counting
    int num_image = batch_size / num_scale_base_;
    int num_top = num_image * num_scale_;

    // construct the output shape
    int dims[4];
    dims[0] = num_top;
    dims[1] = data_height;
    dims[2] = data_width;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // trace data 8 channels
    int dims_trace[4];
    int channels_trace = 8;
    dims_trace[0] = num_top;
    dims_trace[1] = data_height;
    dims_trace[2] = data_width;
    dims_trace[3] = channels_trace;
    TensorShape trace_shape;
    TensorShapeUtils::MakeShape(dims_trace, 4, &trace_shape);

    Tensor* tensor_flags = flags_.AccessTensor(context);
    Tensor* tensor_mapping = mapping_.AccessTensor(context);
    Tensor* tensor_factors = factors_.AccessTensor(context);

    if(is_initialized_ == 0)
    {
      copy_gpu_data((void*)tensor_flags->flat<int>().data(), (void*)is_real_scales_.data(), sizeof(int)*num_scale_);
      copy_gpu_data((void*)tensor_mapping->flat<int>().data(), (void*)which_base_scales_.data(), sizeof(int)*num_scale_);  
      copy_gpu_data((void*)tensor_factors->flat<float>().data(), (void*)rescaling_factors_.data(), sizeof(float)*num_scale_);
      is_initialized_ = 1;
    }

    FeatureExtrapolatingingKernel(context, &bottom_data, num_scale_base_, num_scale_, num_top, channels_trace, data_height,
      data_width, num_channels, tensor_flags->flat<int>().data(), tensor_mapping->flat<int>().data(), tensor_factors->flat<float>().data(), 
      output_shape, trace_shape);

  }
 private:
  int num_scale_base_;
  int num_scale_;
  int num_per_octave_;
  std::vector<float> scales_base_;
  std::vector<float> scales_;
  std::vector<int> is_real_scales_;
  std::vector<int> which_base_scales_;
  std::vector<float> rescaling_factors_;
  PersistentTensor flags_;
  PersistentTensor mapping_;
  PersistentTensor factors_;
  int is_initialized_;
};

REGISTER_KERNEL_BUILDER(Name("FeatureExtrapolating").Device(DEVICE_GPU).TypeConstraint<float>("T"), FeatureExtrapolatingOp<Eigen::GpuDevice, float>);


bool FeatureExtrapolatingBackwardLaucher(const float* top_diff, const int num_scale_base, const int num_scale, const int channels_trace, const int batch_size,
    const int height, const int width, const int channels, const int* which_base_scales,
    const float* rescaling_factors, float* bottom_diff, const float* trace_data, const Eigen::GpuDevice& d);

static void FeatureExtrapolatingingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* trace_data, const Tensor* out_backprop,
    const int num_scale_base, const int num_scale, const int channels_trace, const int batch_size, const int height, const int width, const int channels, 
    const int* which_base_scales, const float* rescaling_factors,
    const TensorShape& tensor_output_shape) 
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  FeatureExtrapolatingBackwardLaucher(
    out_backprop->flat<float>().data(), num_scale_base, num_scale, channels_trace, batch_size, height,
    width, channels, which_base_scales, rescaling_factors,
    output->flat<float>().data(), trace_data->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}


// compute gradient
template <class Device, class T>
class FeatureExtrapolatingGradOp : public OpKernel {
 public:
  explicit FeatureExtrapolatingGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the scales
    scales_base_.clear();
    OP_REQUIRES_OK(context,
                   context->GetAttr("scales_base", &scales_base_));
    // Get the num_scale_base
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_scale_base", &num_scale_base_));
    // Check that num_scale_base is positive
    OP_REQUIRES(context, num_scale_base_ >= 0,
                errors::InvalidArgument("Need num_scale_base >= 0, got ",
                                        num_scale_base_));
    // Get the num_per_octave
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_per_octave", &num_per_octave_));
    // Check that num_per_octave is positive
    OP_REQUIRES(context, num_per_octave_ >= 0,
                errors::InvalidArgument("Need num_per_octave >= 0, got ",
                                        num_per_octave_));

    num_scale_ = (num_scale_base_ - 1) * num_per_octave_ + 1;

    // compute scales
    scales_.clear();
    for(int i = 0; i < num_scale_; i++)
    {
      int index_scale_base = i / num_per_octave_;
      float sbase = scales_base_[index_scale_base];
      int j = i % num_per_octave_;
      float step = 0;
      if(j == 0)
        scales_.push_back(sbase);
      else
      {
        float sbase_next = scales_base_[index_scale_base+1];
        step = (sbase_next - sbase) / num_per_octave_;
        scales_.push_back(sbase + j * step);
      }
    }

    // flags of real scales or approximated scales
    is_real_scales_.clear();
    for(int i = 0; i < num_scale_; i++)
      is_real_scales_.push_back(0);
    for(int i = 0; i < num_scale_base_; i++)
      is_real_scales_[i * num_per_octave_] = 1;

    // scale mapping
    which_base_scales_.clear();
    for(int i = 0; i < num_scale_; i++)
      which_base_scales_.push_back(int(roundf(float(i) / float(num_per_octave_))));

    // rescaling factors
    rescaling_factors_.clear();
    for(int i = 0; i < num_scale_; i++)
    {
      int scale_base_index = which_base_scales_[i];
      float scale_base = scales_base_[scale_base_index];
      float scale = scales_[i];
      rescaling_factors_.push_back(scale / scale_base);
    }

    // construct the output shape
    int dims[4];
    dims[0] = 1;
    dims[1] = 1;
    dims[2] = 1;
    dims[3] = num_scale_;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    is_initialized_ = 0;
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_INT32, output_shape, &mapping_, nullptr));
    OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, output_shape, &factors_, nullptr));

    if (!context->status().ok()) {
      return;
    }

  }

  void Compute(OpKernelContext* context) override 
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& trace_data = context->input(1);
    const Tensor& out_backprop = context->input(2);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    OP_REQUIRES(context, trace_data.dims() == 4,
                errors::InvalidArgument("trace_data must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int height = bottom_data.dim_size(1);
    // data width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int channels = bottom_data.dim_size(3);
    int channels_trace = 8;

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    Tensor* tensor_mapping = mapping_.AccessTensor(context);
    Tensor* tensor_factors = factors_.AccessTensor(context);

    if(is_initialized_ == 0)
    {
      copy_gpu_data((void*)tensor_mapping->flat<int>().data(), (void*)which_base_scales_.data(), sizeof(int)*num_scale_);  
      copy_gpu_data((void*)tensor_factors->flat<float>().data(), (void*)rescaling_factors_.data(), sizeof(float)*num_scale_);
      is_initialized_ = 1;
    }

    FeatureExtrapolatingingGradKernel(
      context, &bottom_data, &trace_data, &out_backprop, num_scale_base_, num_scale_, channels_trace, batch_size, height,
      width, channels, tensor_mapping->flat<int>().data(), tensor_factors->flat<float>().data(), output_shape);

  }
 private:
  int num_scale_base_;
  int num_scale_;
  int num_per_octave_;
  std::vector<float> scales_base_;
  std::vector<float> scales_;
  std::vector<int> is_real_scales_;
  std::vector<int> which_base_scales_;
  std::vector<float> rescaling_factors_;
  PersistentTensor mapping_;
  PersistentTensor factors_;
  int is_initialized_;
};

REGISTER_KERNEL_BUILDER(Name("FeatureExtrapolatingGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), FeatureExtrapolatingGradOp<Eigen::GpuDevice, float>);
