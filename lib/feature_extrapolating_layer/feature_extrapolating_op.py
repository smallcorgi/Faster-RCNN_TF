import tensorflow as tf
import os.path as osp

filename = osp.join(osp.dirname(__file__), 'feature_extrapolating.so')
_feature_extrapolating_module = tf.load_op_library(filename)
feature_extrapolating = _feature_extrapolating_module.feature_extrapolating
feature_extrapolating_grad = _feature_extrapolating_module.feature_extrapolating_grad
