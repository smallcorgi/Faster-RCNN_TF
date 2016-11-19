#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import time, os, sys
import tensorflow as tf

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--weights', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.model) and args.wait:
        print('Waiting for {} to exist...'.format(args.model))
        time.sleep(10)

    weights_filename = os.path.splitext(os.path.basename(args.model))[0]

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    if args.device == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
    else:
        cfg.USE_GPU_NMS = False

    # start a session
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver.restore(sess, args.model)
    print ('Loading model weights from {:s}').format(args.model)

    test_net(sess, network, imdb, weights_filename)
