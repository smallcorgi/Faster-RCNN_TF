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
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    if cfg.IS_MULTISCALE:
        im_blob, im_scales = _get_image_blob_multiscale(roidb)
    else:
        # Get the input image blob, formatted for caffe
        # Sample random scales to use for each image in this batch
        random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES_BASE), size=num_images)
        im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls)
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)


    else:
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)

        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights, sublabels \
                    = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image, num_classes)

            # Add to RoIs blob
            if cfg.IS_MULTISCALE:
                if cfg.IS_EXTRAPOLATING:
                    rois, levels = _project_im_rois_multiscale(im_rois, cfg.TRAIN.SCALES)
                    batch_ind = im_i * len(cfg.TRAIN.SCALES) + levels
                else:
                    rois, levels = _project_im_rois_multiscale(im_rois, cfg.TRAIN.SCALES_BASE)
                    batch_ind = im_i * len(cfg.TRAIN.SCALES_BASE) + levels
            else:
                rois = _project_im_rois(im_rois, im_scales[im_i])
                batch_ind = im_i * np.ones((rois.shape[0], 1))

            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))

            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps, sublabels_blob, view_targets_blob, view_inside_blob)
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps, sublabels_blob)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = []
    for i in xrange(1, num_classes):
        fg_inds.extend(np.where((labels == i) & (overlaps >= cfg.TRAIN.FG_THRESH))[0])
    fg_inds = np.array(fg_inds)

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = []
    for i in xrange(1, num_classes):
        bg_inds.extend( np.where((labels == i) & (overlaps < cfg.TRAIN.BG_THRESH_HI) &
                        (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0] )

    if len(bg_inds) < bg_rois_per_this_image:
        for i in xrange(1, num_classes):
            bg_inds.extend( np.where((labels == i) & (overlaps < cfg.TRAIN.BG_THRESH_HI))[0] )

    if len(bg_inds) < bg_rois_per_this_image:
        bg_inds.extend( np.where(overlaps < cfg.TRAIN.BG_THRESH_HI)[0] )
    bg_inds = np.array(bg_inds, dtype=np.int32)

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds).astype(int)
    # print '{} foregrounds and {} backgrounds'.format(fg_inds.size, bg_inds.size)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]
    sublabels = sublabels[keep_inds]
    sublabels[fg_rois_per_this_image:] = 0

    bbox_targets, bbox_loss_weights = \
            _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
                                        num_classes)

    if cfg.TRAIN.VIEWPOINT or cfg.TEST.VIEWPOINT:
        viewpoints = viewpoints[keep_inds]
        view_targets, view_loss_weights = \
                _get_viewpoint_estimation_labels(viewpoints, labels, num_classes)
        return labels, overlaps, rois, bbox_targets, bbox_loss_weights, sublabels, view_targets, view_loss_weights

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights, sublabels

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_scale = cfg.TRAIN.SCALES_BASE[scale_inds[i]]
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales


def _get_image_blob_multiscale(roidb):
    """Builds an input blob from the images in the roidb at multiscales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    scales = cfg.TRAIN.SCALES_BASE
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        for im_scale in scales:
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            im_scales.append(im_scale)
            processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois


def _project_im_rois_multiscale(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


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


def _get_viewpoint_estimation_labels(viewpoint_data, clss, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        view_target_data (ndarray): N x 3K blob of regression targets
        view_loss_weights (ndarray): N x 3K blob of loss weights
    """
    view_targets = np.zeros((clss.size, 3 * num_classes), dtype=np.float32)
    view_loss_weights = np.zeros(view_targets.shape, dtype=np.float32)
    inds = np.where( (clss > 0) & np.isfinite(viewpoint_data[:,0]) & np.isfinite(viewpoint_data[:,1]) & np.isfinite(viewpoint_data[:,2]) )[0]
    for ind in inds:
        cls = clss[ind]
        start = 3 * cls
        end = start + 3
        view_targets[ind, start:end] = viewpoint_data[ind, :]
        view_loss_weights[ind, start:end] = [1., 1., 1.]

    assert not np.isinf(view_targets).any(), 'viewpoint undefined'
    return view_targets, view_loss_weights


def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps, sublabels_blob, view_targets_blob=None, view_inside_blob=None):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    import math
    for i in xrange(min(rois_blob.shape[0], 10)):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        subcls = sublabels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' subclass: ', subcls, ' overlap: ', overlaps[i]

        start = 3 * cls
        end = start + 3
        # print 'view: ', view_targets_blob[i, start:end] * 180 / math.pi
        # print 'view weights: ', view_inside_blob[i, start:end]

        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
