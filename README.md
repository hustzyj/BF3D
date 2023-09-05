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
BF3D
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
The pre-trained model can be obtained from [Baidu](https://pan.baidu.com/s/1RQznrCOimCpPUjPGOwgC9Q)(i8cr)

## Implementation
### Training
Run BF3D for single gpu:
```shell
CUDA_VISIBLE_DEVICES=0 python train_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss_car.yaml --batch_size 2 --train_mode rcnn_online --epochs 50  --ckpt_save_interval 1 --output_dir ./log/Car/full_epnet_without_iou_branch/   --set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False TRAIN.CE_WEIGHT 5.0

```
Run BF3D for multi gpus:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss_car.yaml --batch_size 8 --train_mode rcnn_online --epochs 50 --mgpus --ckpt_save_interval 1 --output_dir ./log/Car/full_epnet_without_iou_branch/   --set LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2 RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False TRAIN.CE_WEIGHT 5.0

```
### Testing
```shell
CUDA_VICUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss_car.yaml --eval_mode rcnn_online  --eval_all  --output_dir ./log/Car_temp1/full_epnet_without_iou_branch/eval_results/  --ckpt_dir ./log/Car_temp1/full_epnet_without_iou_branch/ckpt --set  LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False
```








