import tensorflow as tf
from tensorflow.python.framework import ops
import roi_pooling_op
import pdb


@ops.RegisterShape("RoiPool")
def _roi_pool_shape(op):
  """Shape function for the RoiPool op.

  """
  dims_data = op.inputs[0].get_shape().as_list()
  channels = dims_data[3]
  dims_rois = op.inputs[1].get_shape().as_list()
  num_rois = dims_rois[0]

  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')

  output_shape = tf.TensorShape([num_rois, pooled_height, pooled_width, channels])
  return [output_shape, output_shape]

@ops.RegisterGradient("RoiPool")
def _roi_pool_grad(op, grad, _):
  """The gradients for `roi_pool`.
  Args:
    op: The `roi_pool` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `roi_pool` op.
  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  data = op.inputs[0]
  rois = op.inputs[1]
  argmax = op.outputs[1]
  pooled_height = op.get_attr('pooled_height')
  pooled_width = op.get_attr('pooled_width')
  spatial_scale = op.get_attr('spatial_scale')

  # compute gradient
  data_grad = roi_pooling_op.roi_pool_grad(data, rois, argmax, grad, pooled_height, pooled_width, spatial_scale)

  return [data_grad, None]  # List of one Tensor, since we have one input
