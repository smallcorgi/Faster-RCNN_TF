# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import PIL
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
import datasets
from fast_rcnn.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._num_classes = 0
        self._num_subclasses = 0
        self._subclass_mapping = []
        self._classes = []
        self._image_index = []
        self._obj_proposer = 'selective_search'
        self._roidb = None
        self._roidb_handler = self.default_roidb
        # Use this dict for storing dataset specific config options
        self.config = {}

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def num_subclasses(self):
        return self._num_subclasses

    @property
    def subclass_mapping(self):
        return self._subclass_mapping

    @property
    def classes(self):
        return self._classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def roidb_handler(self):
        return self._roidb_handler

    @roidb_handler.setter
    def roidb_handler(self, val):
        self._roidb_handler = val

    @property
    def roidb(self):
        # A roidb is a list of dictionaries, each with the following keys:
        #   boxes
        #   gt_overlaps
        #   gt_classes
        #   flipped
        if self._roidb is not None:
            return self._roidb
        self._roidb = self.roidb_handler()
        return self._roidb

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def num_images(self):
      return len(self.image_index)

    def image_path_at(self, i):
        raise NotImplementedError

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def evaluate_proposals(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def append_flipped_images(self):
        num_images = self.num_images
        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            assert (self.roidb[i]['gt_subindexes'].shape == self.roidb[i]['gt_subindexes_flipped'].shape), \
                'gt_subindexes {}, gt_subindexes_flip {}'.format(self.roidb[i]['gt_subindexes'].shape, 
                                                                 self.roidb[i]['gt_subindexes_flipped'].shape)

            if cfg.TRAIN.VIEWPOINT == True or cfg.TEST.VIEWPOINT == True:
                entry = {'boxes' : boxes,
                         'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                         'gt_classes' : self.roidb[i]['gt_classes'],
                         'gt_viewpoints' : self.roidb[i]['gt_viewpoints_flipped'],
                         'gt_viewpoints_flipped' : self.roidb[i]['gt_viewpoints'],
                         'gt_viewindexes_azimuth' : self.roidb[i]['gt_viewindexes_azimuth_flipped'],
                         'gt_viewindexes_azimuth_flipped' : self.roidb[i]['gt_viewindexes_azimuth'],
                         'gt_viewindexes_elevation' : self.roidb[i]['gt_viewindexes_elevation_flipped'],
                         'gt_viewindexes_elevation_flipped' : self.roidb[i]['gt_viewindexes_elevation'],
                         'gt_viewindexes_rotation' : self.roidb[i]['gt_viewindexes_rotation_flipped'],
                         'gt_viewindexes_rotation_flipped' : self.roidb[i]['gt_viewindexes_rotation'],
                         'gt_subclasses' : self.roidb[i]['gt_subclasses_flipped'],
                         'gt_subclasses_flipped' : self.roidb[i]['gt_subclasses'],
                         'gt_subindexes' : self.roidb[i]['gt_subindexes_flipped'],
                         'gt_subindexes_flipped' : self.roidb[i]['gt_subindexes'],
                         'flipped' : True}
            else:
                entry = {'boxes' : boxes,
                         'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                         'gt_classes' : self.roidb[i]['gt_classes'],
                         'gt_subclasses' : self.roidb[i]['gt_subclasses_flipped'],
                         'gt_subclasses_flipped' : self.roidb[i]['gt_subclasses'],
                         'gt_subindexes' : self.roidb[i]['gt_subindexes_flipped'],
                         'gt_subindexes_flipped' : self.roidb[i]['gt_subindexes'],
                         'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
        print 'finish appending flipped images'

    def evaluate_recall(self, candidate_boxes, ar_thresh=0.5):
        # Record max overlap value for each gt box
        # Return vector of overlap values
        gt_overlaps = np.zeros(0)
        for i in xrange(self.num_images):
            gt_inds = np.where(self.roidb[i]['gt_classes'] > 0)[0]
            gt_boxes = self.roidb[i]['boxes'][gt_inds, :]

            boxes = candidate_boxes[i]
            if boxes.shape[0] == 0:
                continue
            overlaps = bbox_overlaps(boxes.astype(np.float),
                                     gt_boxes.astype(np.float))

            # gt_overlaps = np.hstack((gt_overlaps, overlaps.max(axis=0)))
            _gt_overlaps = np.zeros((gt_boxes.shape[0]))
            for j in xrange(gt_boxes.shape[0]):
                argmax_overlaps = overlaps.argmax(axis=0)
                max_overlaps = overlaps.max(axis=0)
                gt_ind = max_overlaps.argmax()
                gt_ovr = max_overlaps.max()
                assert(gt_ovr >= 0)
                box_ind = argmax_overlaps[gt_ind]
                _gt_overlaps[j] = overlaps[box_ind, gt_ind]
                assert(_gt_overlaps[j] == gt_ovr)
                overlaps[box_ind, :] = -1
                overlaps[:, gt_ind] = -1

            gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

        num_pos = gt_overlaps.size
        gt_overlaps = np.sort(gt_overlaps)
        step = 0.001
        thresholds = np.minimum(np.arange(0.5, 1.0 + step, step), 1.0)
        recalls = np.zeros_like(thresholds)
        for i, t in enumerate(thresholds):
            recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
        ar = 2 * np.trapz(recalls, thresholds)

        return ar, gt_overlaps, recalls, thresholds

    def create_roidb_from_box_list(self, box_list, gt_roidb):
        assert len(box_list) == self.num_images, \
                'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
            subindexes = np.zeros((num_boxes, self.num_classes), dtype=np.int32)
            subindexes_flipped = np.zeros((num_boxes, self.num_classes), dtype=np.int32)
            if cfg.TRAIN.VIEWPOINT == True or cfg.TEST.VIEWPOINT == True:
                viewindexes_azimuth = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
                viewindexes_azimuth_flipped = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
                viewindexes_elevation = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
                viewindexes_elevation_flipped = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
                viewindexes_rotation = np.zeros((num_boxes, self.num_classes), dtype=np.float32)
                viewindexes_rotation_flipped = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None:
                gt_boxes = gt_roidb[i]['boxes']
                if gt_boxes.shape[0] != 0 and num_boxes != 0:
                    gt_classes = gt_roidb[i]['gt_classes']
                    gt_subclasses = gt_roidb[i]['gt_subclasses']
                    gt_subclasses_flipped = gt_roidb[i]['gt_subclasses_flipped']
                    if cfg.TRAIN.VIEWPOINT == True or cfg.TEST.VIEWPOINT == True:
                        gt_viewpoints = gt_roidb[i]['gt_viewpoints']
                        gt_viewpoints_flipped = gt_roidb[i]['gt_viewpoints_flipped']
                    gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float))
                    argmaxes = gt_overlaps.argmax(axis=1)
                    maxes = gt_overlaps.max(axis=1)
                    I = np.where(maxes > 0)[0]
                    overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
                    subindexes[I, gt_classes[argmaxes[I]]] = gt_subclasses[argmaxes[I]]
                    subindexes_flipped[I, gt_classes[argmaxes[I]]] = gt_subclasses_flipped[argmaxes[I]]
                    if cfg.TRAIN.VIEWPOINT == True or cfg.TEST.VIEWPOINT == True:
                        viewindexes_azimuth[I, gt_classes[argmaxes[I]]] = gt_viewpoints[argmaxes[I], 0]
                        viewindexes_azimuth_flipped[I, gt_classes[argmaxes[I]]] = gt_viewpoints_flipped[argmaxes[I], 0]
                        viewindexes_elevation[I, gt_classes[argmaxes[I]]] = gt_viewpoints[argmaxes[I], 1]
                        viewindexes_elevation_flipped[I, gt_classes[argmaxes[I]]] = gt_viewpoints_flipped[argmaxes[I], 1]
                        viewindexes_rotation[I, gt_classes[argmaxes[I]]] = gt_viewpoints[argmaxes[I], 2]
                        viewindexes_rotation_flipped[I, gt_classes[argmaxes[I]]] = gt_viewpoints_flipped[argmaxes[I], 2]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            subindexes = scipy.sparse.csr_matrix(subindexes)
            subindexes_flipped = scipy.sparse.csr_matrix(subindexes_flipped)

            if cfg.TRAIN.VIEWPOINT == True or cfg.TEST.VIEWPOINT == True:
                viewindexes_azimuth = scipy.sparse.csr_matrix(viewindexes_azimuth)
                viewindexes_azimuth_flipped = scipy.sparse.csr_matrix(viewindexes_azimuth_flipped)
                viewindexes_elevation = scipy.sparse.csr_matrix(viewindexes_elevation)
                viewindexes_elevation_flipped = scipy.sparse.csr_matrix(viewindexes_elevation_flipped)
                viewindexes_rotation = scipy.sparse.csr_matrix(viewindexes_rotation)
                viewindexes_rotation_flipped = scipy.sparse.csr_matrix(viewindexes_rotation_flipped)
                roidb.append({'boxes' : boxes,
                              'gt_classes' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_viewpoints': np.zeros((num_boxes, 3), dtype=np.float32),
                              'gt_viewpoints_flipped': np.zeros((num_boxes, 3), dtype=np.float32),
                              'gt_viewindexes_azimuth': viewindexes_azimuth,
                              'gt_viewindexes_azimuth_flipped': viewindexes_azimuth_flipped,
                              'gt_viewindexes_elevation': viewindexes_elevation,
                              'gt_viewindexes_elevation_flipped': viewindexes_elevation_flipped,
                              'gt_viewindexes_rotation': viewindexes_rotation,
                              'gt_viewindexes_rotation_flipped': viewindexes_rotation_flipped,
                              'gt_subclasses' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_subclasses_flipped' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_overlaps' : overlaps,
                              'gt_subindexes': subindexes,
                              'gt_subindexes_flipped': subindexes_flipped,
                              'flipped' : False})
            else:
                roidb.append({'boxes' : boxes,
                              'gt_classes' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_subclasses' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_subclasses_flipped' : np.zeros((num_boxes,), dtype=np.int32),
                              'gt_overlaps' : overlaps,
                              'gt_subindexes': subindexes,
                              'gt_subindexes_flipped': subindexes_flipped,
                              'flipped' : False})
        return roidb

    @staticmethod
    def merge_roidbs(a, b):
        assert len(a) == len(b)
        for i in xrange(len(a)):
            a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
            a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                            b[i]['gt_classes']))
            if cfg.TRAIN.VIEWPOINT == True or cfg.TEST.VIEWPOINT == True:
                a[i]['gt_viewpoints'] = np.vstack((a[i]['gt_viewpoints'],
                                                b[i]['gt_viewpoints']))
                a[i]['gt_viewpoints_flipped'] = np.vstack((a[i]['gt_viewpoints_flipped'],
                                                b[i]['gt_viewpoints_flipped']))
                a[i]['gt_viewindexes_azimuth'] = scipy.sparse.vstack((a[i]['gt_viewindexes_azimuth'],
                                                b[i]['gt_viewindexes_azimuth']))
                a[i]['gt_viewindexes_azimuth_flipped'] = scipy.sparse.vstack((a[i]['gt_viewindexes_azimuth_flipped'],
                                                b[i]['gt_viewindexes_azimuth_flipped']))
                a[i]['gt_viewindexes_elevation'] = scipy.sparse.vstack((a[i]['gt_viewindexes_elevation'],
                                                b[i]['gt_viewindexes_elevation']))
                a[i]['gt_viewindexes_elevation_flipped'] = scipy.sparse.vstack((a[i]['gt_viewindexes_elevation_flipped'],
                                                b[i]['gt_viewindexes_elevation_flipped']))
                a[i]['gt_viewindexes_rotation'] = scipy.sparse.vstack((a[i]['gt_viewindexes_rotation'],
                                                b[i]['gt_viewindexes_rotation']))
                a[i]['gt_viewindexes_rotation_flipped'] = scipy.sparse.vstack((a[i]['gt_viewindexes_rotation_flipped'],
                                                b[i]['gt_viewindexes_rotation_flipped']))

            a[i]['gt_subclasses'] = np.hstack((a[i]['gt_subclasses'],
                                            b[i]['gt_subclasses']))
            a[i]['gt_subclasses_flipped'] = np.hstack((a[i]['gt_subclasses_flipped'],
                                            b[i]['gt_subclasses_flipped']))
            a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                       b[i]['gt_overlaps']])
            a[i]['gt_subindexes'] = scipy.sparse.vstack((a[i]['gt_subindexes'],
                                            b[i]['gt_subindexes']))
            a[i]['gt_subindexes_flipped'] = scipy.sparse.vstack((a[i]['gt_subindexes_flipped'],
                                            b[i]['gt_subindexes_flipped']))
        return a

    def competition_mode(self, on):
        """Turn competition mode on or off."""
        pass
