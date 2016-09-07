import tensorflow as tf
import numpy as np
import feature_extrapolating_op
import feature_extrapolating_op_grad

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

array = np.random.rand(5, 100, 100, 3)
data = tf.convert_to_tensor(array, dtype=tf.float32)

W = weight_variable([3, 3, 3, 1])
h = conv2d(data, W)

scales_base = [0.25, 0.5, 1.0, 2.0, 3.0]
num_scale_base = 5
num_per_octave = 4

[y, trace] = feature_extrapolating_op.feature_extrapolating(h, scales_base, num_scale_base, num_per_octave)

#y_data = tf.convert_to_tensor(np.ones((17, 100, 100, 1)), dtype=tf.float32)
#print y_data, y, trace

# Minimize the mean squared errors.
#loss = tf.reduce_mean(tf.square(y - y_data))
#optimizer = tf.train.GradientDescentOptimizer(0.5)
#train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

#for step in xrange(1):
#    sess.run(train)
#    print(step, sess.run(W))
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print(sess.run(y))
print(sess.run(trace))

#with tf.device('/gpu:0'):
#  result = module.roi_pool(data, rois, 1, 1, 1.0/1)
#  print result.eval()
#with tf.device('/cpu:0'):
#  run(init)
