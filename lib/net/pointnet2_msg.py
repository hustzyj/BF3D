import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModuleMSG_WithSampling
from lib.config import cfg
from torch.nn.functional import grid_sample

import lib.utils.loss_utils as loss_utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
BatchNorm2d = nn.BatchNorm2d
from tools.merge_feature import  merge_feature_qurry
def conv3x3(in_planes, out_planes, stride = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out

class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        #print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features






#================addition attention (add)=======================#


class bi_attention_fusion(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        print('##############bi_attention_fusion#########')
        super(bi_attention_fusion, self).__init__()
        self.ic=inplanes_I
        self.pc=inplanes_P
        self.scale = inplanes_P**-0.5
        self.fc1=nn.Linear(inplanes_P*inplanes_I,1)
        self.fc2=nn.Linear(inplanes_P*inplanes_I,1)
        self.conv1 = nn.Sequential(nn.Conv1d(self.pc, self.ic, 1),
                                   nn.BatchNorm1d(self.ic),
                                   nn.ReLU())
     
        self.conv3 = torch.nn.Conv1d(inplanes_I +inplanes_I , inplanes_I, 1)
        self.bn3 = torch.nn.BatchNorm1d(inplanes_I)
    
    ###############另外平行融合分支###########################
        self.ic1=inplanes_I
        self.pc1=inplanes_P
        self.scale1 = inplanes_I**-0.5
        self.fc1_p = nn.Linear(inplanes_P*inplanes_I,1)
        self.fc2_p = nn.Linear(inplanes_P*inplanes_I,1)
        self.conv1_p = nn.Sequential(nn.Conv1d(self.ic1, self.pc1, 1),
                                   nn.BatchNorm1d(self.pc1),
                                   nn.ReLU())
        self.conv3_p = torch.nn.Conv1d(inplanes_P +inplanes_P, outplanes, 1)
        self.bn3_p = torch.nn.BatchNorm1d(outplanes)

        self.conv_out =  nn.Sequential(nn.Conv1d(inplanes_I + inplanes_P , outplanes , 1),
                                   nn.BatchNorm1d(outplanes),
                                   nn.LeakyReLU())

    def forward(self, point_features, img_features):
        batch = img_features.size(0)
        npoint=img_features.size(2)
        img_feas_f = img_features.transpose(1, 2).contiguous().view(-1, self.ic).transpose(0,1)  # BCN->BNC->(BN)C ##67x4096
        point_feas_f = point_features.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'###4096x96
        QK=torch.matmul(img_feas_f, point_feas_f)*self.scale####67x96  C1xC2
        QK=torch.softmax(QK,dim=1)
        att_in = torch.flatten(QK)
        att_p = F.sigmoid(self.fc1(F.tanh(att_in)))
        att_p=att_p.unsqueeze(0).unsqueeze(-1).repeat(batch,1,npoint)
        point_feas_new=self.conv1(point_features)
        point_fea_out = att_p * point_feas_new###1x67x4096
        fusion_fea1 = torch.cat((point_fea_out, img_features),dim=1)##1x134x4096
        out_img = F.leaky_relu(self.bn3(self.conv3(fusion_fea1)))####1x67x4096

       ################另外融合的分支##################################
        img_feas_f_1 = out_img.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C ##4096x67
        point_feas_f_1 = point_features.transpose(1, 2).contiguous().view(-1, self.pc).transpose(0,1)  # BCN->BNC->(BN)C'###96x4096
        QK_1=torch.matmul(point_feas_f_1, img_feas_f_1) * self.scale1####96x67  C1xC2
        QK_1=torch.softmax(QK_1,dim=1)
        att_in_1 = torch.flatten(QK_1)
        att_img = F.sigmoid(self.fc1_p(F.tanh(att_in_1)))
        att_img = att_img.unsqueeze(0).unsqueeze(-1).repeat(batch,1,npoint)
        img_feas_new=self.conv1_p(out_img)
        img_fea_out = att_img * img_feas_new###1x67x4096
        fusion_fea=torch.cat((img_fea_out,point_features),dim=1)
        out_point = F.leaky_relu(self.bn3_p(self.conv3_p(fusion_fea)))

        # fusion_fea_out = torch.cat((out_img, out_point),dim=1)

        # out = self.conv_out(fusion_fea_out)

        return out_point  




class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                    nn.BatchNorm1d(self.pc),
                                    nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)


    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1,2).contiguous().view(-1, self.ic) #BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1,2).contiguous().view(-1, self.pc) #BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp))) #BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1) #B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels = [inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)


    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features =  self.IA_Layer(img_features, point_features)
        #print("img_features:", img_features.shape)

        #fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features

class image_fuse_coordinate_conv(nn.Module):
    def __init__(self, channels):
        super(image_fuse_coordinate_conv, self).__init__()
        self.cin=channels
        self.cout=2*(self.cin)
        self.fc1=nn.Linear(self.cin, self.cout)
        self.Batchnorm1=nn.BatchNorm1d(self.cout)
  
        self.fc3=nn.Linear(self.cout,self.cout//2)
        self.Batchnorm3=nn.BatchNorm1d(self.cout//2)

    def forward(self,x):#2x4096x67
        batch = x.size(0)
        npoints=x.size(1)
        nsamples=x.size(2)
        x = x.contiguous().view(-1, self.cin)  #
        x= F.relu(self.Batchnorm1(self.fc1(x)))
        
        x = F.relu(self.Batchnorm3(self.fc3(x)))
        x=x.view(batch,npoints,nsamples,-1)
       
        return x
class add_conv(nn.Module):
    def __init__(self, channels):
        super(add_conv, self).__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.fc1=nn.Linear(self.in_channels, self.out_channels)
        self.Batchnorm1=nn.BatchNorm1d(self.out_channels)
    def forward(self,x):#2x4096x67
        batch = x.size(0)
        npoints=x.size(1)
        # C=x.size(2)
        x = x.contiguous().view(-1, self.in_channels)  #
        x= F.relu(self.Batchnorm1(self.fc1(x)))
        x=x.view(batch,npoints,self.out_channels)
        return x 







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


def get_model(input_channels = 6, use_xyz = True):
    return Pointnet2MSG(input_channels = input_channels, use_xyz = use_xyz)


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                    PointnetSAModuleMSG_WithSampling(
                            npoint = cfg.RPN.SA_CONFIG.NPOINTS[k],
                            radii = cfg.RPN.SA_CONFIG.RADIUS[k],
                            sample_method = cfg.RPN.SA_CONFIG.sample_method[k],
                            confidence_mlp = cfg.RPN.SA_CONFIG.CONFIDENCE_MLPS[k],
                            nsamples = cfg.RPN.SA_CONFIG.NSAMPLE[k],
                            nsamples_qurry = cfg.RPN.SA_CONFIG.NSAMPLE_qurry[k],
                            mlps = mlps,
                            use_xyz = use_xyz,
                            bn = cfg.RPN.USE_BN
                    )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        ##################
        if cfg.LI_FUSION.ENABLED:
            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            self.image_fuse_coordinate_conv1 = nn.ModuleList()
            self.image_fuse_coordinate_conv2 = nn.ModuleList()
            self.add_conv = nn.ModuleList()

            self.bi_attention_fusion = nn.ModuleList()
            #######第一个SA的MLP
            first_mlp = []
            first_mlp.extend([
                    nn.Conv1d(3,
                            32, kernel_size=1, bias=False),
                    nn.BatchNorm1d(32),
                    nn.ReLU()
                ])
               
            first_mlp.append(
                nn.Conv1d(32, 1, kernel_size=1, bias=True),
            )
            self.first_SSM = nn.Sequential(*first_mlp)
            
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                self.Img_Block.append(BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i], cfg.LI_FUSION.IMG_CHANNELS[i+1], stride=1))
                if cfg.LI_FUSION.ADD_Image_Attention:
                    # self.Fusion_Conv.append(
                    #     Atten_Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1]+ 3, cfg.LI_FUSION.POINT_CHANNELS[i],
                    #                       cfg.LI_FUSION.POINT_CHANNELS[i]))

                    self.bi_attention_fusion.append(
                        bi_attention_fusion(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + 3, cfg.LI_FUSION.POINT_CHANNELS[i],
                                          cfg.LI_FUSION.POINT_CHANNELS[i]))
                else:
                    self.Fusion_Conv.append(Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                                        cfg.LI_FUSION.POINT_CHANNELS[i]))

                self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.DeConv_Reduce[i],
                                                  kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                                                  stride=cfg.LI_FUSION.DeConv_Kernels[i]))
                self.image_fuse_coordinate_conv1.append(image_fuse_coordinate_conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + 3))
                self.image_fuse_coordinate_conv2.append(image_fuse_coordinate_conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + 3))
                
                self.add_conv.append(add_conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + 3))

            self.image_fusion_conv = nn.Conv2d(sum(cfg.LI_FUSION.DeConv_Reduce), cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, kernel_size = 1)
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4)

            if cfg.LI_FUSION.ADD_Image_Attention:
                # self.final_fusion_img_point = Atten_Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
                self.final_fusion_img_point = bi_attention_fusion(cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL//4, cfg.LI_FUSION.IMG_FEATURES_CHANNEL)


        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                    PointnetFPModule(mlp = [pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )

        self.MSG_sample_loss_func = loss_utils.WeightedClassificationLoss()

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None,rpn_cls_label = None):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]

        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]
            size_range = [1280.0, 384.0]
            xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2.0 - 1.0
            xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
            l_xy_cor = [xy]
            img = [image]
            if rpn_cls_label is not None:
                rpn_cls_lable_list = [rpn_cls_label.unsqueeze(-1)]
        
        # li_cls_pred = self.first_SSM(l_xyz[0].transpose(1,2)).transpose(1,2)
        li_cls_pred = None
        sa_ins_preds = []
        sample_idx = []
        qurry_idx__list_all = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index, li_cls_pred, qurry_idx_list = self.SA_modules[i](l_xyz[i], l_features[i], li_cls_pred)
            qurry_idx__list_all.append(qurry_idx_list)
            sample_idx.append(li_index)
            if li_cls_pred is not None:
                sa_ins_preds.append(li_cls_pred) 
            else:
                sa_ins_preds.append([])
            neibor_feature = []
            if cfg.LI_FUSION.ENABLED:
                li_index_1 = li_index.long().unsqueeze(-1).repeat(1,1,2)
                li_xy_cor_temp = l_xy_cor[i]#2x16384x2
                li_xy_cor = torch.gather(l_xy_cor[i],1,li_index_1)
                image = self.Img_Block[i](img[i])

                img_gather_feature = Feature_Gather(image, li_xy_cor)
                neibor_features_qurry,li_xy_neiborhood_cor = merge_feature_qurry(qurry_idx__list_all[i],l_xyz[i],li_xyz,li_xy_cor_temp, image, self.image_fuse_coordinate_conv1[i],self.image_fuse_coordinate_conv2[i],neibor_feature,self.add_conv[i])
                li_features = self.bi_attention_fusion[i](li_features, neibor_features_qurry)
                l_xy_cor.append(li_xy_cor)
                img.append(image)


############################################作图###################################################################################
                # li_index_label = li_index.clone().long().unsqueeze(-1)
                # li_index_temp = li_index.clone()
                # li_index_temp = li_index_temp.long().unsqueeze(-1).repeat(1,1,3)
                # li_xyz_cor = torch.gather(l_xyz[i],1,li_index_temp)
                # image_show = image[0].clone().permute(1, 2, 0)
                # image_show = image_show.cpu().detach().numpy()
                # image_show = np.uint8(np.mean(((image_show + 1) * 255) / 2, axis=2, keepdims=True))
                # image_show = cv2.applyColorMap(image_show, cv2.COLORMAP_BONE )
                # npoint_temp = li_xy_cor.shape[1]
                # li_label_cor = torch.gather(rpn_cls_lable_list[i], 1, li_index_label) 
                # pos = (li_label_cor.view(-1)>0).sum(dim=0).float()
                # cmap = plt.cm.get_cmap('hsv', 256)
                # cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
                
                # for k in range(npoint_temp):
                #     pointcenter_show=li_xy_cor[0][k]
                #     label = li_label_cor[0][k]
                #     if label.item() == 1:
                #         depth = li_xyz_cor[0][k][2].cpu().detach().numpy()
                #         color = cmap[np.clip(int(700/depth), 0, 255), :]
                #         x= ((pointcenter_show[0] + 1) * image_show.shape[1]) / 2#####192x640
                #         y = ((pointcenter_show[1] + 1) * image_show.shape[0]) / 2
                #         x = np.int32(x.cpu().numpy())
                #         y = np.int32(y.cpu().numpy())
                #         cv2.circle(
                #                 image_show,
                #                 center=(x,
                #                         y),
                #                 radius=1,
                #                 color=[0,0,255],
                #                 thickness=-1,
                #                 )
                #         # for n in range(len(li_xy_neiborhood_cor)):   ######1x4096x2x2
                #         #     xy_temp = li_xy_neiborhood_cor[n][0][k]
                #         #     npoint_temp = xy_temp.shape[0]
                #         #     for m in range (npoint_temp):
                #         #         xy_temp_1 = xy_temp[m]
                #         #         x_temp= ((xy_temp_1[0] + 1) * image_show.shape[1]) / 2#####192x640
                #         #         y_temp  = ((xy_temp_1[1] + 1) * image_show.shape[0]) / 2
                #         #         x_temp = np.int32(x_temp.cpu().numpy())
                #         #         y_temp = np.int32(y_temp.cpu().numpy())
                #         #         color_temp = [255,0,255] if m==1 else [255,255,255]
                #         #         cv2.circle(
                #         #             image_show,
                #         #             center=(x_temp,
                #         #                     y_temp),
                #         #             radius=1,
                #         #             color = color_temp,
                #         #             thickness=-1,
                #         #             )

                #         cv2.imwrite('/home/hust/PIS3D/tools/forgroundpoint/FPS-CTR-CTR-CTR/feature_map_save' + str(i+1) +"/" +str(k)+ ".png",image_show.astype(np.uint8)) 
                # rpn_cls_lable_list.append(li_label_cor)
#################################################################################################################################################################
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            



        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if cfg.LI_FUSION.ENABLED:
            #for i in range(1,len(img))
            DeConv = []
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                DeConv.append(self.DeConv[i](img[i + 1]))
            de_concat = torch.cat(DeConv,dim=1)

            img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)))
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        return l_xyz[0], l_features[0],sa_ins_preds,sample_idx


class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels = 6, use_xyz = True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                    l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs