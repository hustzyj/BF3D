from cv2 import split
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
# from pointnet2_lib.pointnet2.pointnet2_modules import pointnetSAModulefinal
from lib.config import cfg
from torch.nn.functional import grid_sample
from pointnet2_lib.pointnet2.pointnet2_utils import grouping_operation
def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """
    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)
    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)
    return interpolate_feature.squeeze(2) # (B,C,N)

def merge_feature_qurry(sample_index_list,l_xyz,li_xyz,li_xy_cor_temp,image,image_fuse_coordinate_conv1,image_fuse_coordinate_conv2, neibor_feature,add_conv):
    xy_show = []
    for j in range(len(sample_index_list)):
            nsample=sample_index_list[j].size(-1)#16
            npoints=sample_index_list[j].size(-2)#4096
            Batchsize=sample_index_list[j].size(-3)
            li_xyz_trans=l_xyz.transpose(1,2).contiguous()#####1x3x16384
            grouped_xyz = grouping_operation(li_xyz_trans, sample_index_list[j])#2x3x4096x16
            #li_xy_cor_temp=li_xy_cor_temp.unsqueeze(-1).transpose(2,3).contiguous()#2x16384x1x2
            grouped_xyz-=li_xyz.transpose(1,2).unsqueeze(-1)#2x3x4096x16
            index_temp=sample_index_list[j]#2x4096x16
            index_temp=index_temp.long().unsqueeze(-1).repeat(1,1,1,2).view(Batchsize,-1,2)#2x4096x16x2 dao 2x65536x2
            li_xy_neiborhood_cor = torch.gather(li_xy_cor_temp, 1,index_temp)#2x65536x2
            #######
            xy_show.append(li_xy_neiborhood_cor.view(Batchsize,npoints,nsample,-1))###2x4096x16x2
            ######

            image_gather_neibor_feature = Feature_Gather(image,li_xy_neiborhood_cor)#2x64x65536
            image_gather_neibor_feature = image_gather_neibor_feature.view(Batchsize,-1,npoints,nsample)#2x64x4096x16

            image_gather_coordinate=torch.cat((image_gather_neibor_feature,grouped_xyz),dim=1)#2x67x4096x16
            image_gather_coordinate_temp =image_gather_coordinate.permute(0,2,3,1)#2x4096x16x67
            #######先经过MLP后sum
            if j == 0:
                image_gather_coordinate1 = image_fuse_coordinate_conv1(image_gather_coordinate_temp)
                image_gather_coordinate_1_final = torch.sum(image_gather_coordinate1,dim=2)#2x4096x67
                neibor_feature.append(image_gather_coordinate_1_final)
            elif j == 1:
                image_gather_coordinate2 = image_fuse_coordinate_conv2(image_gather_coordinate_temp)
                image_gather_coordinate_2_final = torch.sum(image_gather_coordinate2,dim=2)#2x4096x67
                neibor_feature.append(image_gather_coordinate_2_final)
            # torch.sum(image_gather_coordinate,dim=2)#2x4096x16x67
    # neibor_features_final = torch.add(neibor_feature[0],neibor_feature[1])###这个地方设置为两个半径相加结果
    # neibor_features_final = torch.cat((neibor_feature[0], neibor_feature[1]),dim=2)####concat
    neibor_features_final = torch.add(neibor_feature[0], neibor_feature[1])
    neibor_features_out = add_conv(neibor_features_final)
    neibor_features_out = neibor_features_out.permute(0,2,1)
    return neibor_features_out, xy_show