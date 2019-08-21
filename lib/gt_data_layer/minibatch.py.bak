# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob = _get_image_blob(roidb)

    # build the box information blob
    info_boxes_blob = np.zeros((0, 18), dtype=np.float32)
    num_scale = len(cfg.TRAIN.SCALES)
    for i in xrange(num_images):
        info_boxes = roidb[i]['info_boxes']

        # change the batch index
        info_boxes[:,2] += i * num_scale
        info_boxes[:,7] += i * num_scale

        info_boxes_blob = np.vstack((info_boxes_blob, info_boxes))

    # build the parameter blob
    num_aspect = len(cfg.TRAIN.ASPECTS)
    num = 2 + 2 * num_scale + 2 * num_aspect
    parameters_blob = np.zeros((num), dtype=np.float32)
    parameters_blob[0] = num_scale
    parameters_blob[1] = num_aspect
    parameters_blob[2:2+num_scale] = cfg.TRAIN.SCALES
    parameters_blob[2+num_scale:2+2*num_scale] = cfg.TRAIN.SCALE_MAPPING
    parameters_blob[2+2*num_scale:2+2*num_scale+num_aspect] = cfg.TRAIN.ASPECT_HEIGHTS
    parameters_blob[2+2*num_scale+num_aspect:2+2*num_scale+2*num_aspect] = cfg.TRAIN.ASPECT_WIDTHS

    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, sublabels_blob)

    blobs = {'data': im_blob,
             'info_boxes': info_boxes_blob,
             'parameters': parameters_blob}

    return blobs

def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the different scales.
    """
    num_images = len(roidb)
    processed_ims = []

    for i in xrange(num_images):
        # read image
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        # build image pyramid
        for im_scale in cfg.TRAIN.SCALES_BASE:
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

            processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights


def _vis_minibatch(im_blob, rois_blob, labels_blob, sublabels_blob):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[2:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        subcls = sublabels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' subclass: ', subcls
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
