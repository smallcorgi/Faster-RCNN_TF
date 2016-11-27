# Faster-RCNN_TF

This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.


### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

1. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/smallcorgi/Faster-RCNN_TF.git
  ```

2. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

Download model training on PASCAL VOC 2007  [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbZ0EzeUlHcGFIVWM) [[Dropbox]](https://www.dropbox.com/s/cfz3blmtmwj6bdh/VGGnet_fast_rcnn_iter_70000.ckpt?dl=0)

To run the demo
```Shell
cd $FRCN_ROOT
python ./tools/demo.py --model model_path
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Training Model
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    
5. Download pre-trained ImageNet models

   Download the pre-trained ImageNet models [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)
   
   	```Shell
    mv VGG_imagenet.npy $FRCN_ROOT/data/pretrain_model/VGG_imagenet.npy
    ```

6. Run script to train and test model
	```Shell
	cd $FRCN_ROOT
	./experiments/scripts/faster_rcnn_end2end.sh $DEVICE $DEVICE_ID VGG16 pascal_voc
	```
  DEVICE is either cpu/gpu

### The result of testing on PASCAL VOC 2007 

| Classes       | AP     |
|-------------|--------|
| aeroplane   | 0.698 |
| bicycle     | 0.788 |
| bird        | 0.657 |
| boat        | 0.565 |
| bottle      | 0.478 |
| bus         | 0.762 |
| car         | 0.797 |
| cat         | 0.793 |
| chair       | 0.479 |
| cow         | 0.724 |
| diningtable | 0.648 |
| dog         | 0.803 |
| horse       | 0.797 |
| motorbike   | 0.732 |
| person      | 0.770 |
| pottedplant | 0.384 |
| sheep       | 0.664 |
| sofa        | 0.650 |
| train       | 0.766 |
| tvmonitor   | 0.666 |
| mAP        | 0.681 |


###References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)

