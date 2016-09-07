import tensorflow as tf
from tensorflow.python.framework import ops
import feature_extrapolating_op

@tf.RegisterShape("FeatureExtrapolating")
def _feature_extrapolating_shape(op):
  """Shape function for the FeatureExtrapolating op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  batch_size = dims_data[0]
  height = dims_data[1]
  width = dims_data[2]
  channels = dims_data[3]

  num_scale_base = op.get_attr('num_scale_base')
  num_per_octave = op.get_attr('num_per_octave')

  num_scale = (num_scale_base - 1) * num_per_octave + 1;

  if batch_size is not None:
      num_image = batch_size / num_scale_base;
      num_top = num_image * num_scale;
  else:
      num_top = None

  output_shape = tf.TensorShape([num_top, height, width, channels])
  trace_shape = tf.TensorShape([num_top, height, width, 8])
  return [output_shape, trace_shape]

@ops.RegisterGradient("FeatureExtrapolating")
def _feature_extrapolating_grad(op, grad, _):
  """The gradients for `feature_extrapolating`.
  Args:
    op: The `feature_extrapolating` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `feature_extrapolating` op.
  Returns:
    Gradients with respect to the input of `feature_extrapolating`.
  """
  data = op.inputs[0]
  trace = op.outputs[1]
  scales_base = op.get_attr('scales_base')
  num_scale_base = op.get_attr('num_scale_base')
  num_per_octave = op.get_attr('num_per_octave')

  # compute gradient
  data_grad = feature_extrapolating_op.feature_extrapolating_grad(data, trace, grad, scales_base, num_scale_base, num_per_octave)

  return [data_grad]  # List of one Tensor, since we have one input
