# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import PIL
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from utils.cython_bbox import bbox_overlaps
from utils.boxes_grid import get_boxes_grid
from fast_rcnn.config import cfg
import math
from rpn_msr.generate_anchors import generate_anchors
import sys

class pascal_voc(datasets.imdb):
    def __init__(self, image_set, year, pascal_path=None):
        datasets.imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._pascal_path = self._get_default_path() if pascal_path is None \
                            else pascal_path
        self._data_path = os.path.join(self._pascal_path, 'VOCdevkit' + self._year, 'VOC' + self._year)
        self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        if cfg.IS_RPN:
            self._roidb_handler = self.gt_roidb
        else:
            self._roidb_handler = self.region_proposal_roidb

        # num of subclasses
        self._num_subclasses = 240 + 1

        # load the mapping for subcalss to class
        filename = os.path.join(self._pascal_path, 'subcategory_exemplars', 'mapping.txt')
        assert os.path.exists(filename), 'Path does not exist: {}'.format(filename)
        
        mapping = np.zeros(self._num_subclasses, dtype=np.int)
        with open(filename) as f:
            for line in f:
                words = line.split()
                subcls = int(words[0])
                mapping[subcls] = self._class_to_ind[words[1]]
        self._subclass_mapping = mapping

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        # statistics for computing recall
        self._num_boxes_all = np.zeros(self.num_classes, dtype=np.int)
        self._num_boxes_covered = np.zeros(self.num_classes, dtype=np.int)
        self._num_boxes_proposal = 0

        assert os.path.exists(self._pascal_path), \
                'PASCAL path does not exist: {}'.format(self._pascal_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._pascal_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'PASCAL')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_subcategory_exemplar_annotation(index)
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


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        # print 'Loading: {}'.format(filename)
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)
        gt_subclasses = np.zeros((num_objs), dtype=np.int32)
        gt_subclasses_flipped = np.zeros((num_objs), dtype=np.int32)
        subindexes = np.zeros((num_objs, self.num_classes), dtype=np.int32)
        subindexes_flipped = np.zeros((num_objs, self.num_classes), dtype=np.int32)

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
                anchors = generate_anchors()
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
                'gt_subclasses': gt_subclasses,
                'gt_subclasses_flipped': gt_subclasses_flipped,
                'gt_overlaps' : overlaps,
                'gt_subindexes': subindexes,
                'gt_subindexes_flipped': subindexes_flipped,
                'flipped' : False}


    def _load_pascal_subcategory_exemplar_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the pascal subcategory exemplar format.
        """
        if self._image_set == 'test':
            return self._load_pascal_annotation(index)

        filename = os.path.join(self._pascal_path, 'subcategory_exemplars', index + '.txt')
        assert os.path.exists(filename), \
                'Path does not exist: {}'.format(filename)

        # the annotation file contains flipped objects    
        lines = []
        lines_flipped = []
        with open(filename) as f:
            for line in f:
                words = line.split()
                subcls = int(words[1])
                is_flip = int(words[2])
                if subcls != -1:
                    if is_flip == 0:
                        lines.append(line)
                    else:
                        lines_flipped.append(line)
        
        num_objs = len(lines)

        # store information of flipped objects
        assert (num_objs == len(lines_flipped)), 'The number of flipped objects is not the same!'
        gt_subclasses_flipped = np.zeros((num_objs), dtype=np.int32)
        
        for ix, line in enumerate(lines_flipped):
            words = line.split()
            subcls = int(words[1])
            gt_subclasses_flipped[ix] = subcls

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        gt_subclasses = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        subindexes = np.zeros((num_objs, self.num_classes), dtype=np.int32)
        subindexes_flipped = np.zeros((num_objs, self.num_classes), dtype=np.int32)

        for ix, line in enumerate(lines):
            words = line.split()
            cls = self._class_to_ind[words[0]]
            subcls = int(words[1])
            # Make pixel indexes 0-based
            boxes[ix, :] = [float(n)-1 for n in words[3:7]]
            gt_classes[ix] = cls
            gt_subclasses[ix] = subcls
            overlaps[ix, cls] = 1.0
            subindexes[ix, cls] = subcls
            subindexes_flipped[ix, cls] = gt_subclasses_flipped[ix]

        overlaps = scipy.sparse.csr_matrix(overlaps)

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
                ratios = [3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
                scales = 2**np.arange(1, 6, 0.5)
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
                'gt_subclasses': gt_subclasses,
                'gt_subclasses_flipped': gt_subclasses_flipped,
                'gt_overlaps': overlaps,
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

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote roidb to {}'.format(cache_file)

        return roidb

    def _load_rpn_roidb(self, gt_roidb, model):
        # set the prefix
        if self._image_set == 'test':
            prefix = model + '/testing'
        else:
            prefix = model + '/training'

        box_list = []
        for index in self.image_index:
            filename = os.path.join(self._pascal_path, 'region_proposals',  prefix, index + '.txt')
            assert os.path.exists(filename), \
                'RPN data not found at: {}'.format(filename)
            raw_data = np.loadtxt(filename, dtype=float)
            if len(raw_data.shape) == 1:
                if raw_data.size == 0:
                    raw_data = raw_data.reshape((0, 5))
                else:
                    raw_data = raw_data.reshape((1, 5))

            x1 = raw_data[:, 0]
            y1 = raw_data[:, 1]
            x2 = raw_data[:, 2]
            y2 = raw_data[:, 3]
            score = raw_data[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1))[0]
            raw_data = raw_data[inds,:4]
            self._num_boxes_proposal += raw_data.shape[0]
            box_list.append(raw_data)

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._pascal_path, 'VOCdevkit' + self._year, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            print filename
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, 4],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._pascal_path + '/VOCdevkit' + self._year, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    # evaluate detection results
    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def evaluate_proposals(self, all_boxes, output_dir):
        # for each image
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(output_dir, index + '.txt')
            print 'Writing PASCAL results to file ' + filename
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
            print 'Writing PASCAL results to file ' + filename
            with open(filename, 'wt') as f:
                dets = all_boxes[im_ind]
                if dets == []:
                    continue
                for k in xrange(dets.shape[0]):
                    f.write('{:f} {:f} {:f} {:f} {:.32f}\n'.format(dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4]))


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed; embed()
