import imp
from statistics import mode
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import _init_path
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
import tools.train_utils.train_utils as train_utils
from lib.utils.bbox_transform import decode_bbox_target
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import argparse
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from datetime import datetime
import logging
import re
import glob
import time
from tensorboardX import SummaryWriter
import scipy
import cv2

from lib.datasets.kitti_dataset import KittiDataset
# from torchvision import transforms

def show_feature_map(feature_map):#
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    feature_map = feature_map.squeeze(0)#1x3x384x1280
    
    #以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map =feature_map.view(1,feature_map.shape[0],feature_map.shape[1],feature_map.shape[2])#
    b, c, H, W = feature_map.shape
    upsample = torch.nn.UpsamplingBilinear2d(size=(feature_map.shape[2],feature_map.shape[3]))#这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3]).detach().cpu().numpy()###3x384x1280
  
    feature_map_num = feature_map.shape[0]#返回通道数
    feature_map_sum = np.expand_dims(feature_map[0,:,:],axis = 2)
    for i in range(0,feature_map_num):
        feature_map_split = feature_map[i,:,:]
        feature_map_split_temp = np.expand_dims(feature_map_split, axis = 2)
        if i>0:
            feature_map_sum +=feature_map_split_temp
        # feature_map_split = np.array(feature_map_split,dtype=np.uint8)
        # feature = cv2.applyColorMap(feature_map_split, cv2.COLORMAP_JET)
        # cv2.imwrite('/home/hust/PIS3D/tools/feature_map_save/' + str(i) + ".png",feature)
        plt.axis('off')
        plt.imshow(feature_map_split)       
        plt.savefig('/home/hust/PIS3D/tools/feature_map_save/' + str(i) + ".png",bbox_inches='tight',pad_inches = -0.01,dpi = 300)
        plt.clf()

    plt.axis('off')
    plt.imshow(feature_map_sum)
    plt.savefig('/home/hust/PIS3D/tools/feature_map_save/'  + "sum.png",bbox_inches='tight',pad_inches = -0.01,dpi = 300)
    # feature_map_sum = feature_map_sum.squeeze()
    # feature_map_sum = np.array(feature_map_sum,dtype=np.uint8)
    # feature_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
    # cv2.imwrite('/home/hust/PIS3D/tools/feature_map_save/' + "sum.png",feature_sum)



######读入图片#################
DATA_PATH = '/home/hust/PIS3D/data'
kitidaset = KittiDataset(root_dir=DATA_PATH)

img = kitidaset.get_image_rgb_with_normal(idx = 253)
image = torch.from_numpy(img).cuda(non_blocking = True).float()
image_in = image.unsqueeze(0).permute(0,3,1,2)#######1x3x384x1280
##############################

model = PointRCNN(num_classes = 2, use_xyz = True, mode = 'TEST')
parmeters = torch.load('/home/hust/PIS3D/tools/log/Car/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_50.pth')
model.load_state_dict(parmeters['model_state'])
model = model.cuda().eval()
model_layer = list(model.children())[0]
layer_res = model_layer.backbone_net.Img_Block
####第一个特征图#################
out1 = layer_res[0](image_in)
# show_feature_map(out1)
#######################
out2 = layer_res[1](out1)

out3 = layer_res[2](out2)
out4 = layer_res[3](out3)
show_feature_map(out4)
















