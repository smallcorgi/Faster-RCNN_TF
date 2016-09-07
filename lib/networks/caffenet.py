import tensorflow as tf
from networks.network import Network

class caffenet(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.rois = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'rois':self.rois})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1', trainable=False)
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .feature_extrapolating([1.0, 2.0, 3.0, 4.0], 4, 4, name='conv5_feature'))

        (self.feed('conv5_feature','im_info')
             .conv(3,3,)

        (self.feed('conv5_feature', 'rois')
             .roi_pool(6, 6, 1.0/16, name='pool5')
             .fc(4096, name='fc6')
             .dropout(self.keep_prob, name='drop6')
             .fc(4096, name='fc7')
             .dropout(self.keep_prob, name='drop7')
             .fc(174, relu=False, name='subcls_score')
             .softmax(name='subcls_prob'))

        (self.feed('subcls_score')
             .fc(4, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('subcls_score')
             .fc(16, relu=False, name='bbox_pred'))
