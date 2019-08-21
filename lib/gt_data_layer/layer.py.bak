# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

GtDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from gt_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue

class GtDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH

        """
        # sample images with gt objects
        db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
        i = 0
        while (i < cfg.TRAIN.IMS_PER_BATCH):
            ind = self._perm[self._cur]
            num_objs = self._roidb[ind]['boxes'].shape[0]
            if num_objs != 0:
                db_inds[i] = ind
                i += 1

            self._cur += 1
            if self._cur >= len(self._roidb):
                self._shuffle_roidb_inds()
        """

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)

    # this function is called in training the net
    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        """Setup the GtDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {
            'data': 0,
            'info_boxes': 1,
            'parameters': 2}

        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        num_scale_base = len(cfg.TRAIN.SCALES_BASE)
        top[0].reshape(num_scale_base, 3, 100, 100)

        # info boxes blob
        top[1].reshape(1, 18)

        # parameters blob
        num_scale = len(cfg.TRAIN.SCALES)
        num_aspect = len(cfg.TRAIN.ASPECTS)
        top[2].reshape(2 + 2*num_scale + 2*num_aspect)
            
    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
