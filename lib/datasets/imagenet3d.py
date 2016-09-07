__author__ = 'yuxiang' # derived from honda.py by fyang

import datasets
import datasets.imagenet3d
import os
import PIL
import datasets.imdb
import numpy as np
import scipy.sparse
from utils.cython_bbox import bbox_overlaps
from utils.boxes_grid import get_boxes_grid
import subprocess
import cPickle
from fast_rcnn.config import cfg
import math
from rpn_msr.generate_anchors import generate_anchors
import sys

class imagenet3d(datasets.imdb):
    def __init__(self, image_set, imagenet3d_path=None):
        datasets.imdb.__init__(self, 'imagenet3d_' + image_set)
        self._image_set = image_set
        self._imagenet3d_path = self._get_default_path() if imagenet3d_path is None \
                            else imagenet3d_path
        self._data_path = os.path.join(self._imagenet3d_path, 'Images')
        self._classes = ('__background__', 'aeroplane', 'ashtray', 'backpack', 'basket', \
             'bed', 'bench', 'bicycle', 'blackboard', 'boat', 'bookshelf', 'bottle', 'bucket', \
             'bus', 'cabinet', 'calculator', 'camera', 'can', 'cap', 'car', 'cellphone', 'chair', \
             'clock', 'coffee_maker', 'comb', 'computer', 'cup', 'desk_lamp', 'diningtable', \
             'dishwasher', 'door', 'eraser', 'eyeglasses', 'fan', 'faucet', 'filing_cabinet', \
             'fire_extinguisher', 'fish_tank', 'flashlight', 'fork', 'guitar', 'hair_dryer', \
             'hammer', 'headphone', 'helmet', 'iron', 'jar', 'kettle', 'key', 'keyboard', 'knife', \
             'laptop', 'lighter', 'mailbox', 'microphone', 'microwave', 'motorbike', 'mouse', \
             'paintbrush', 'pan', 'pen', 'pencil', 'piano', 'pillow', 'plate', 'pot', 'printer', \
             'racket', 'refrigerator', 'remote_control', 'rifle', 'road_pole', 'satellite_dish', \
             'scissors', 'screwdriver', 'shoe', 'shovel', 'sign', 'skate', 'skateboard', 'slipper', \
             'sofa', 'speaker', 'spoon', 'stapler', 'stove', 'suitcase', 'teapot', 'telephone', \
             'toaster', 'toilet', 'toothbrush', 'train', 'trash_bin', 'trophy', 'tub', 'tvmonitor', \
             'vending_machine', 'washing_machine', 'watch', 'wheelchair')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        if cfg.IS_RPN:
            self._roidb_handler = self.gt_roidb
        else:
            self._roidb_handler = self.region_proposal_roidb

        self.config = {'top_k': 100000}

        # statistics for computing recall
        self._num_boxes_all = np.zeros(self.num_classes, dtype=np.int)
        self._num_boxes_covered = np.zeros(self.num_classes, dtype=np.int)
        self._num_boxes_proposal = 0

        assert os.path.exists(self._imagenet3d_path), \
                'imagenet3d path does not exist: {}'.format(self._imagenet3d_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._imagenet3d_path, 'Image_sets', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where imagenet3d is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'ImageNet3D')


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        cache_file = os.path.join(self.cache_path, self.name + '_' + cfg.SUBCLS_NAME + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_imagenet3d_annotation(index)
                    for index in self.image_index]

        if cfg.IS_RPN:
            # print out recall
            for i in xrange(1, self.num_classes):
                print '{}: Total number of boxes {:d}'.format(self.classes[i], self._num_boxes_all[i])
                print '{}: Number of boxes covered {:d}'.format(self.classes[i], self._num_boxes_covered[i])
                print '{}: Recall {:f}'.format(self.classes[i], float(self._num_boxes_covered[i]) / float(self._num_boxes_all[i]))

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_imagenet3d_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the imagenet3d format.
        """

        if self._image_set == 'test' or self._image_set == 'test_1' or self._image_set == 'test_2':
            lines = []
        else:
            filename = os.path.join(self._imagenet3d_path, 'Labels', index + '.txt')
            lines = []
            with open(filename) as f:
                for line in f:
                    lines.append(line)

        num_objs = len(lines)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        viewpoints = np.zeros((num_objs, 3), dtype=np.float32)          # azimuth, elevation, in-plane rotation
        viewpoints_flipped = np.zeros((num_objs, 3), dtype=np.float32)  # azimuth, elevation, in-plane rotation
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, line in enumerate(lines):
            words = line.split()
            assert len(words) == 5 or len(words) == 8, 'Wrong label format: {}'.format(index)
            cls = self._class_to_ind[words[0]]
            boxes[ix, :] = [float(n) for n in words[1:5]]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            if len(words) == 8:
                viewpoints[ix, :] = [float(n) for n in words[5:8]]
                # flip the viewpoint
                viewpoints_flipped[ix, 0] = -viewpoints[ix, 0]  # azimuth
                viewpoints_flipped[ix, 1] = viewpoints[ix, 1]   # elevation
                viewpoints_flipped[ix, 2] = -viewpoints[ix, 2]  # in-plane rotation
            else:
                viewpoints[ix, :] = np.inf
                viewpoints_flipped[ix, :] = np.inf

        gt_subclasses = np.zeros((num_objs), dtype=np.int32)
        gt_subclasses_flipped = np.zeros((num_objs), dtype=np.int32)
        subindexes = np.zeros((num_objs, self.num_classes), dtype=np.int32)
        subindexes_flipped = np.zeros((num_objs, self.num_classes), dtype=np.int32)
        viewindexes_azimuth = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        viewindexes_azimuth_flipped = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        viewindexes_elevation = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        viewindexes_elevation_flipped = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        viewindexes_rotation = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        viewindexes_rotation_flipped = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        subindexes = scipy.sparse.csr_matrix(subindexes)
        subindexes_flipped = scipy.sparse.csr_matrix(subindexes_flipped)
        viewindexes_azimuth = scipy.sparse.csr_matrix(viewindexes_azimuth)
        viewindexes_azimuth_flipped = scipy.sparse.csr_matrix(viewindexes_azimuth_flipped)
        viewindexes_elevation = scipy.sparse.csr_matrix(viewindexes_elevation)
        viewindexes_elevation_flipped = scipy.sparse.csr_matrix(viewindexes_elevation_flipped)
        viewindexes_rotation = scipy.sparse.csr_matrix(viewindexes_rotation)
        viewindexes_rotation_flipped = scipy.sparse.csr_matrix(viewindexes_rotation_flipped)

        if cfg.IS_RPN:
            if cfg.IS_MULTISCALE:
                # compute overlaps between grid boxes and gt boxes in multi-scales
                # rescale the gt boxes
                boxes_all = np.zeros((0, 4), dtype=np.float32)
                for scale in cfg.TRAIN.SCALES:
                    boxes_all = np.vstack((boxes_all, boxes * scale))
                gt_classes_all = np.tile(gt_classes, len(cfg.TRAIN.SCALES))

                # compute grid boxes
                s = PIL.Image.open(self.image_path_from_index(index)).size
                image_height = s[1]
                image_width = s[0]
                boxes_grid, _, _ = get_boxes_grid(image_height, image_width)

                # compute overlap
                overlaps_grid = bbox_overlaps(boxes_grid.astype(np.float), boxes_all.astype(np.float))
        
                # check how many gt boxes are covered by grids
                if num_objs != 0:
                    index = np.tile(range(num_objs), len(cfg.TRAIN.SCALES))
                    max_overlaps = overlaps_grid.max(axis = 0)
                    fg_inds = []
                    for k in xrange(1, self.num_classes):
                        fg_inds.extend(np.where((gt_classes_all == k) & (max_overlaps >= cfg.TRAIN.FG_THRESH[k-1]))[0])
                    index_covered = np.unique(index[fg_inds])

                    for i in xrange(self.num_classes):
                        self._num_boxes_all[i] += len(np.where(gt_classes == i)[0])
                        self._num_boxes_covered[i] += len(np.where(gt_classes[index_covered] == i)[0])
            else:
                assert len(cfg.TRAIN.SCALES_BASE) == 1
                scale = cfg.TRAIN.SCALES_BASE[0]
                feat_stride = 16
                # faster rcnn region proposal
                base_size = 16
                ratios = cfg.TRAIN.RPN_ASPECTS
                scales = cfg.TRAIN.RPN_SCALES
                anchors = generate_anchors(base_size, ratios, scales)
                num_anchors = anchors.shape[0]

                # image size
                s = PIL.Image.open(self.image_path_from_index(index)).size
                image_height = s[1]
                image_width = s[0]

                # height and width of the heatmap
                height = np.round((image_height * scale - 1) / 4.0 + 1)
                height = np.floor((height - 1) / 2 + 1 + 0.5)
                height = np.floor((height - 1) / 2 + 1 + 0.5)

                width = np.round((image_width * scale - 1) / 4.0 + 1)
                width = np.floor((width - 1) / 2.0 + 1 + 0.5)
                width = np.floor((width - 1) / 2.0 + 1 + 0.5)

                # gt boxes
                gt_boxes = boxes * scale

                # 1. Generate proposals from bbox deltas and shifted anchors
                shift_x = np.arange(0, width) * feat_stride
                shift_y = np.arange(0, height) * feat_stride
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
                # add A anchors (1, A, 4) to
                # cell K shifts (K, 1, 4) to get
                # shift anchors (K, A, 4)
                # reshape to (K*A, 4) shifted anchors
                A = num_anchors
                K = shifts.shape[0]
                all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
                all_anchors = all_anchors.reshape((K * A, 4))

                # compute overlap
                overlaps_grid = bbox_overlaps(all_anchors.astype(np.float), gt_boxes.astype(np.float))
        
                # check how many gt boxes are covered by anchors
                if num_objs != 0:
                    max_overlaps = overlaps_grid.max(axis = 0)
                    fg_inds = []
                    for k in xrange(1, self.num_classes):
                        fg_inds.extend(np.where((gt_classes == k) & (max_overlaps >= cfg.TRAIN.FG_THRESH[k-1]))[0])

                    for i in xrange(self.num_classes):
                        self._num_boxes_all[i] += len(np.where(gt_classes == i)[0])
                        self._num_boxes_covered[i] += len(np.where(gt_classes[fg_inds] == i)[0])

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_viewpoints': viewpoints,
                'gt_viewpoints_flipped': viewpoints_flipped,
                'gt_viewindexes_azimuth': viewindexes_azimuth,
                'gt_viewindexes_azimuth_flipped': viewindexes_azimuth_flipped,
                'gt_viewindexes_elevation': viewindexes_elevation,
                'gt_viewindexes_elevation_flipped': viewindexes_elevation_flipped,
                'gt_viewindexes_rotation': viewindexes_rotation,
                'gt_viewindexes_rotation_flipped': viewindexes_rotation_flipped,
                'gt_subclasses': gt_subclasses,
                'gt_subclasses_flipped': gt_subclasses_flipped,
                'gt_overlaps' : overlaps,
                'gt_subindexes': subindexes,
                'gt_subindexes_flipped': subindexes_flipped,
                'flipped' : False}


    def region_proposal_roidb(self):
        """
        Return the database of regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_' + cfg.REGION_PROPOSAL + '_region_proposal_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()

            print 'Loading region proposal network boxes...'
            model = cfg.REGION_PROPOSAL
            rpn_roidb = self._load_rpn_roidb(gt_roidb, model)
            print 'Region proposal network boxes loaded'
            roidb = datasets.imdb.merge_roidbs(rpn_roidb, gt_roidb)
        else:
            print 'Loading region proposal network boxes...'
            model = cfg.REGION_PROPOSAL
            roidb = self._load_rpn_roidb(None, model)
            print 'Region proposal network boxes loaded'

        print '{} region proposals per image'.format(self._num_boxes_proposal / len(self.image_index))

        # print out recall
        if self._image_set != 'test':
            for i in xrange(1, self.num_classes):
                print '{}: Total number of boxes {:d}'.format(self.classes[i], self._num_boxes_all[i])
                print '{}: Number of boxes covered {:d}'.format(self.classes[i], self._num_boxes_covered[i])
                if self._num_boxes_all[i] > 0:
                    print '{}: Recall {:f}'.format(self.classes[i], float(self._num_boxes_covered[i]) / float(self._num_boxes_all[i]))

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote roidb to {}'.format(cache_file)

        return roidb

    def _load_rpn_roidb(self, gt_roidb, model):

        box_list = []
        for ix, index in enumerate(self.image_index):
            filename = os.path.join(self._imagenet3d_path, 'region_proposals', model, index + '.txt')
            assert os.path.exists(filename), \
                '{} data not found at: {}'.format(model, filename)
            raw_data = np.loadtxt(filename, dtype=float)
            if len(raw_data.shape) == 1:
                if raw_data.size == 0:
                    raw_data = raw_data.reshape((0, 5))
                else:
                    raw_data = raw_data.reshape((1, 5))

            if model == 'selective_search' or model == 'mcg':
                x1 = raw_data[:, 1].copy()
                y1 = raw_data[:, 0].copy()
                x2 = raw_data[:, 3].copy()
                y2 = raw_data[:, 2].copy()
            elif model == 'edge_boxes':
                x1 = raw_data[:, 0].copy()
                y1 = raw_data[:, 1].copy()
                x2 = raw_data[:, 2].copy() + raw_data[:, 0].copy()
                y2 = raw_data[:, 3].copy() + raw_data[:, 1].copy()
            elif model == 'rpn_caffenet' or model == 'rpn_vgg16':
                x1 = raw_data[:, 0].copy()
                y1 = raw_data[:, 1].copy()
                x2 = raw_data[:, 2].copy()
                y2 = raw_data[:, 3].copy()
            else:
                assert 1, 'region proposal not supported: {}'.format(model)

            inds = np.where((x2 > x1) & (y2 > y1))[0]
            raw_data[:, 0] = x1
            raw_data[:, 1] = y1
            raw_data[:, 2] = x2
            raw_data[:, 3] = y2
            raw_data = raw_data[inds,:4]

            self._num_boxes_proposal += raw_data.shape[0]
            box_list.append(raw_data)
            print 'load {}: {}'.format(model, index)

            if gt_roidb is not None:
                # compute overlaps between region proposals and gt boxes
                boxes = gt_roidb[ix]['boxes'].copy()
                gt_classes = gt_roidb[ix]['gt_classes'].copy()
                # compute overlap
                overlaps = bbox_overlaps(raw_data.astype(np.float), boxes.astype(np.float))
                # check how many gt boxes are covered by anchors
                if raw_data.shape[0] != 0:
                    max_overlaps = overlaps.max(axis = 0)
                    fg_inds = []
                    for k in xrange(1, self.num_classes):
                        fg_inds.extend(np.where((gt_classes == k) & (max_overlaps >= cfg.TRAIN.FG_THRESH[k-1]))[0])

                    for i in xrange(self.num_classes):
                        self._num_boxes_all[i] += len(np.where(gt_classes == i)[0])
                        self._num_boxes_covered[i] += len(np.where(gt_classes[fg_inds] == i)[0])

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def evaluate_detections(self, all_boxes, output_dir):

        # for each image
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(output_dir, index + '.txt')
            print 'Writing imagenet3d results to file ' + filename
            with open(filename, 'wt') as f:
                # for each class
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # detection and viewpoint
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:f} {:f} {:f} {:f} {:.32f} {:f} {:f} {:f}\n'.format(\
                                 cls, dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4], dets[k, 6], dets[k, 7], dets[k, 8]))

    # write detection results into one file
    def evaluate_detections_one_file(self, all_boxes, output_dir):

        # for each class
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # open results file
            filename = os.path.join(output_dir, 'detections_{}.txt'.format(cls))
            print 'Writing imagenet3d results to file ' + filename
            with open(filename, 'wt') as f:
                # for each image
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # detection and viewpoint
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:f} {:f} {:f} {:f} {:.32f} {:f} {:f} {:f}\n'.format(\
                                 index, dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4], dets[k, 6], dets[k, 7], dets[k, 8]))

    def evaluate_proposals(self, all_boxes, output_dir):
        # for each image
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(output_dir, index + '.txt')
            print 'Writing imagenet3d results to file ' + filename
            with open(filename, 'wt') as f:
                # for each class
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:f} {:f} {:f} {:f} {:.32f}\n'.format(\
                                 dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4]))

    def evaluate_proposals_msr(self, all_boxes, output_dir):
        # for each image
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(output_dir, index + '.txt')
            print 'Writing imagenet3d results to file ' + filename
            with open(filename, 'wt') as f:
                dets = all_boxes[im_ind]
                if dets == []:
                    continue
                for k in xrange(dets.shape[0]):
                    f.write('{:f} {:f} {:f} {:f} {:.32f}\n'.format(dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4]))


if __name__ == '__main__':
    d = datasets.imagenet3d('trainval')
    res = d.roidb
    from IPython import embed; embed()
