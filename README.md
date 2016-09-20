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
    sh make.sh
    ```

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded [here](#https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM).

### Training Results

Download model training on PASCAL VOC 2007 [here](#https://drive.google.com/open?id=0ByuDEGFYmWsbZ0EzeUlHcGFIVWM).

| Classes       | AP     |
|-------------|--------|
| aeroplane   | 0.6903 |
| bicycle     | 0.7597 |
| bird        | 0.6423 |
| boat        | 0.5408 |
| bottle      | 0.4688 |
| bus         | 0.7609 |
| car         | 0.7920 |
| cat         | 0.7878 |
| chair       | 0.4696 |
| cow         | 0.7030 |
| diningtable | 0.6218 |
| dog         | 0.7525 |
| horse       | 0.7938 |
| motorbike   | 0.7414 |
| person      | 0.7643 |
| pottedplant | 0.3718 |
| sheep       | 0.6476 |
| sofa        | 0.6146 |
| train       | 0.7660 |
| tvmonitor   | 0.6639 |
| mAP        | 0.6676 |



