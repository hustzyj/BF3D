# BF3D
BF3D: Bi-directional Fusion 3D Detector with Semantic Sampling and
Geometric Mapping
## Install
The Environment：
* Linux (tested on Ubuntu 16.04)
* Python 3.6+
* PyTorch 1.0+

a. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.

b. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
EPNet
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```
## Trained model
The pretrained model can be obtained from [Baidu](https://pan.baidu.com/s/1RQznrCOimCpPUjPGOwgC9Q)(i8cr)
