# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import networks.VGGnet_train
import networks.VGGnet_test
import pdb
import tensorflow as tf

#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()


def get_network(name, device_name=None):
    """Get a network by name."""
    if name.split('_')[1] == 'test':
       return networks.VGGnet_test(device_name)
    elif name.split('_')[1] == 'train':
       return networks.VGGnet_train(device_name)
    else:
       raise KeyError('Unknown network name: {}'.format(name))
    

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
