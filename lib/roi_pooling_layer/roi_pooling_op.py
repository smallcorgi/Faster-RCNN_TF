import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'roi_pooling.so')
_roi_pooling_module = tf.load_op_library(filename)
roi_pool = _roi_pooling_module.roi_pool
roi_pool_grad = _roi_pooling_module.roi_pool_grad
