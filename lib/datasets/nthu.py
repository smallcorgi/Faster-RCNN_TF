__author__ = 'yuxiang'

import datasets
import datasets.nthu
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

class nthu(datasets.imdb):
    def __init__(self, image_set, nthu_path=None):
        datasets.imdb.__init__(self, 'nthu_' + image_set)
        self._image_set = image_set
        self._nthu_path = self._get_default_path() if nthu_path is None \
                            else nthu_path
        self._data_path = os.path.join(self._nthu_path, 'data')
        self._classes = ('__background__', 'Car', 'Pedestrian', 'Cyclist')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        if cfg.IS_RPN:
            self._roidb_handler = self.gt_roidb
        else:
            self._roidb_handler = self.region_proposal_roidb

        # num of subclasses
        self._num_subclasses = 227 + 36 + 36 + 1

        # load the mapping for subcalss to class
        filename = os.path.join(self._nthu_path, 'mapping.txt')
        assert os.path.exists(filename), 'Path does not exist: {}'.format(filename)
        
        mapping = np.zeros(self._num_subclasses, dtype=np.int)
        with open(filename) as f:
            for line in f:
                words = line.split()
                subcls = int(words[0])
                mapping[subcls] = self._class_to_ind[words[1]]
        self._subclass_mapping = mapping

        self.config = {'top_k': 100000}

        # statistics for computing recall
        self._num_boxes_all = np.zeros(self.num_classes, dtype=np.int)
        self._num_boxes_covered = np.zeros(self.num_classes, dtype=np.int)
        self._num_boxes_proposal = 0

        assert os.path.exists(self._nthu_path), \
                'NTHU path does not exist: {}'.format(self._nthu_path)
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
        # set the prefix
        prefix = self._image_set

        image_path = os.path.join(self._data_path, prefix, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where nthu is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'NTHU')


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        No implementation.
        """

        gt_roidb = []
        return gt_roidb

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
        prefix = model

        box_list = []
        for index in self.image_index:
            filename = os.path.join(self._nthu_path, 'region_proposals',  prefix, self._image_set, index + '.txt')
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

    def evaluate_detections(self, all_boxes, output_dir):
        # load the mapping for subcalss the alpha (viewpoint)
        filename = os.path.join(self._nthu_path, 'mapping.txt')
        assert os.path.exists(filename), \
                'Path does not exist: {}'.format(filename)

        mapping = np.zeros(self._num_subclasses, dtype=np.float)
        with open(filename) as f:
            for line in f:
                words = line.split()
                subcls = int(words[0])
                mapping[subcls] = float(words[3])

        # for each image
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(output_dir, index + '.txt')
            print 'Writing nthu results to file ' + filename
            with open(filename, 'wt') as f:
                # for each class
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        subcls = int(dets[k, 5])
                        cls_name = self.classes[self.subclass_mapping[subcls]]
                        assert (cls_name == cls), 'subclass not in class'
                        alpha = mapping[subcls]
                        f.write('{:s} -1 -1 {:f} {:f} {:f} {:f} {:f} -1 -1 -1 -1 -1 -1 -1 {:.32f}\n'.format(\
                                 cls, alpha, dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4]))

    # write detection results into one file
    def evaluate_detections_one_file(self, all_boxes, output_dir):
        # open results file
        filename = os.path.join(output_dir, 'detections.txt')
        print 'Writing all nthu results to file ' + filename
        with open(filename, 'wt') as f:
            # for each image
            for im_ind, index in enumerate(self.image_index):
                # for each class
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        subcls = int(dets[k, 5])
                        cls_name = self.classes[self.subclass_mapping[subcls]]
                        assert (cls_name == cls), 'subclass not in class'
                        f.write('{:s} {:s} {:f} {:f} {:f} {:f} {:d} {:f}\n'.format(\
                                 index, cls, dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], subcls, dets[k, 4]))

    def evaluate_proposals(self, all_boxes, output_dir):
        # for each image
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(output_dir, index + '.txt')
            print 'Writing nthu results to file ' + filename
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
            print 'Writing nthu results to file ' + filename
            with open(filename, 'wt') as f:
                dets = all_boxes[im_ind]
                if dets == []:
                    continue
                for k in xrange(dets.shape[0]):
                    f.write('{:f} {:f} {:f} {:f} {:.32f}\n'.format(dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, 4]))


if __name__ == '__main__':
    d = datasets.nthu('71')
    res = d.roidb
    from IPython import embed; embed()
