#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <cfloat>
#include "feature_extrapolating_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// namespace tensorflow {
using namespace tensorflow;

void copy_gpu_data(void* data_gpu, void* data_cpu, int size)
{
  cudaError_t err;

  err = cudaMemcpy(data_gpu, data_cpu, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    std::cerr << "Unable to copy memory" << std::endl;
}

template <typename Dtype>
__global__ void FeatureExtrapolatingForward(const int nthreads, const Dtype* bottom_data,
    const int num_scale_base, const int num_scale, const int channels_trace, 
    const int height, const int width, const int channels,
    const int* is_real_scales, const int* which_base_scales,
    const float* rescaling_factors,
    Dtype* top_data, Dtype* trace_data)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, h, w, c) is an element in the output
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    int index_image = n / num_scale;
    int index_scale = n % num_scale;

    // flag for approximation or not
    int flag = is_real_scales[index_scale];
    // which base scale to use
    int index_scale_base = which_base_scales[index_scale];
    // rescaling factor
    float factor = rescaling_factors[index_scale];
    // bottom batch image
    int index_batch = index_image * num_scale_base + index_scale_base;
    const Dtype* batch_data = bottom_data + index_batch * height * width * channels;

    top_data[index] = 0;
    if(flag == 1) // no approximation
    {
      top_data[index] = batch_data[(h * width + w) * channels + c];
      // set tracing info
      if(c == 0)
      {
        for(int i = 0; i < channels_trace / 2; i++)
        {
          trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i] = index_batch * height * width * channels + (h * width + w) * channels + c;
          trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i + 1] = 0.25;
        }
      }
    }
    else
    {
      // bilinear interpolation
      float xp = w / factor;
      float yp = h / factor;
      float cx[2], cy[2], ux, uy;
      int xi, yi, dx, dy, i;
      Dtype val;
      if(xp >= 0 && xp < width && yp >= 0 && yp < height)
      {
        xi = (int)floor(xp); 
        yi = (int)floor(yp);
        ux = xp - (float)xi;
        uy = yp - (float)yi;
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
            if(xi+dx >= 0 && xi+dx < width && yi+dy >= 0 && yi+dy < height)
            {
              val += cx[1-dx] * cy[1-dy] * batch_data[((yi+dy) * width + (xi+dx)) * channels + c];
              if(c == 0)
              {
                trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i] = index_batch * channels * height * width + ((yi+dy) * width + (xi+dx)) * channels + c;
                trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i + 1] = cx[1-dx] * cy[1-dy];
              }
            }
            else
            {
              if(c == 0)
              {
                trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i] = -1;
                trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i + 1] = 0;
              }
            }
            i++;
          }
        }
        top_data[index] = val;
      }
      else
      {
        // set tracing info
        if(c == 0)
        {
          for(int i = 0; i < channels_trace / 2; i++)
          {
            trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i] = -1;
            trace_data[n * channels_trace * height * width + (h * width + w) * channels_trace + 2 * i + 1] = 0;
          }
        }
      }
    }
  }
}

bool FeatureExtrapolatingForwardLaucher(
    const float* bottom_data, const int num_scale_base, 
    const int num_scale, const int num_top, const int channels_trace, const int height, const int width, const int channels,
    const int* is_real_scales, const int* which_base_scales, const float* rescaling_factors,
    float* top_data, float* trace_data, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_top * height * width * channels;
  cudaError_t err;

  FeatureExtrapolatingForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, num_scale_base, num_scale, channels_trace, height, width, channels, 
      is_real_scales, which_base_scales, rescaling_factors, top_data, trace_data);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
  }

  return d.ok();
}


template <typename Dtype>
__global__ void FeatureExtrapolatingBackward(const int nthreads, const Dtype* top_diff, const Dtype* trace_data, 
    const int num_scale_base, const int num_scale, const int channels_trace,
    const int height, const int width, const int channels, const int* which_base_scales,
    const float* rescaling_factors, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, h, w, c) coords in bottom data
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    int index_image = n / num_scale_base;
    int index_scale_base = n % num_scale_base;

    Dtype val = 0;
    for(int i = 0; i < num_scale; i++)
    {
      if(which_base_scales[i] == index_scale_base)
      {
        int index_batch = index_image * num_scale + i;
        float factor = rescaling_factors[i];
        float xp = w * factor;
        float yp = h * factor;
        int xi = (int)floor(xp); 
        int yi = (int)floor(yp);
        
        for(int dx = -2; dx <= 2; dx++)
        {
          for(int dy = -2; dy <= 2; dy++)
          {
            if(xi+dx >= 0 && xi+dx < width && yi+dy >= 0 && yi+dy < height)
            {
              for(int j = 0; j < channels_trace / 2; j++)
              {
                int index_trace = int(trace_data[index_batch * channels_trace * height * width + 2 * j + ((yi+dy) * width + (xi+dx)) * channels_trace]);
                float weight_trace = trace_data[index_batch * channels_trace * height * width + (2 * j + 1) + ((yi+dy) * width + (xi+dx)) * channels_trace];
                if(index_trace == n * channels * height * width + (h * width + w) * channels)
                  val += weight_trace * top_diff[index_batch * channels * height * width + ((yi+dy) * width + (xi+dx)) * channels + c];
              }
            }
          }
        }
      }
    }
    // assign value
    bottom_diff[index] = val;
  }
}


bool FeatureExtrapolatingBackwardLaucher(const float* top_diff, const int num_scale_base, const int num_scale, const int channels_trace, const int batch_size,
    const int height, const int width, const int channels, const int* which_base_scales, const float* rescaling_factors,
    float* bottom_diff, const float* trace_data, const Eigen::GpuDevice& d)
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  FeatureExtrapolatingBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, trace_data, num_scale_base, num_scale, channels_trace, height, width, channels, 
      which_base_scales, rescaling_factors, bottom_diff);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
