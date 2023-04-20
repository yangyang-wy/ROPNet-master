import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import batch_quat2mat

class PAM(nn.Module):
    def __init__(self, C):
        super(PAM, self).__init__()
        self.dim = C
        self.conv1 = nn.Conv1d(in_channels = C, out_channels=C // 8, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels = C, out_channels=C // 8, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels = C, out_channels=C, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        b, c, n = x.shape

        out1 = self.conv1(x).view(b, -1, n).permute(0, 2, 1) # b, n, c/latent

        out2 = self.conv2(x).view(b, -1, n) # b,c/latent,n

        attention_matrix = self.softmax(torch.bmm(out1, out2)) # b,n,n

        out3 = self.conv3(x).view(b, -1, n) # b,c,n

        attention = torch.bmm(out3, attention_matrix.permute(0, 2, 1))

        out = self.gamma * attention.view(b, c, n) + x
        return  out

# PointNet模型，(gn,cls两个标志位， 第一个 gn 表示group normalize，就是按某一维进行pointnet开始特征提取；第2个cls是用来控制网络只要不是倒数第一层，就每层都进行激活输出relu。 我看PointNet被调用时，只有在解码器网络中设置了cls=True）
class PointNet(nn.Module):
    def __init__(self, in_dim, gn, out_dims, cls=False):
        super(PointNet, self).__init__()
        self.cls = cls
        l = len(out_dims)
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))
            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                         nn.GroupNorm(8, out_dim))
            if self.cls and i != l - 1:
                self.backbone.add_module(f'pointnet_relu_{i}',
                                         nn.ReLU(inplace=True))        # 用来控制网络只要不是倒数第一层，就每层都进行激活输出relu
            in_dim = out_dim

    def forward(self, x, pooling=True):
        f = self.backbone(x)          # x(8,3,717) x:表示置换过后的源点云   f(8,512,717)
        if not pooling:
            return f
        g, _ = torch.max(f, dim=2)     # 返回输入tensor中所有元素的最大值 dim=2表示对应到第三个维度   g(8,512)
        return f, g                  # f(8,512,717),


class MLPs(nn.Module):
    def __init__(self, in_dim, mlps):
        super(MLPs, self).__init__()
        self.mlps = nn.Sequential()            # nn.Sequential
        l = len(mlps)                          # len()长度计算的函数
        for i, out_dim in enumerate(mlps):    # enumerate将一个可遍历的数据对象（如列表、元组、字典和字符串）组合成一个索引序列，同时列出数据下标和数据（索引 值），一般配合for循环使用
            self.mlps.add_module(f'fc_{i}', nn.Linear(in_dim, out_dim))
            if i != l - 1:
                self.mlps.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.mlps(x)   # [8,7]
        return x

# CGModule(encoder,decoder)：CG模型（编码，解码）
class CGModule(nn.Module):
    def __init__(self, in_dim, gn):
        super(CGModule, self).__init__()
        self.encoder = PointNet(in_dim=in_dim,
                                gn=gn,
                                out_dims=[64, 64, 64, 128, 512])
        self.decoder_ol = PointNet(in_dim=2048,
                                   gn=gn,
                                   out_dims=[512, 512, 256, 2],
                                   cls=True)
        self.decoder_qt = MLPs(in_dim=1024,
                               mlps=[512, 512, 256, 7])





        # self.dim = 1
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)





    def forward(self, src, tgt):
        '''
        Context-Guided Model for initial alignment and overlap score.用于初始对齐和重叠分数的上下文引导模型
        :param src: (B, N, 3)
        :param tgt: (B, M, 3)
        :return: T0: (B, 3, 4), OX: (B, N, 2), OY: (B, M, 2)
        '''
        x = src.permute(0, 2, 1).contiguous()    # （8，3，717） permute:置换，contiguous：相接


        b, c, n = x.shape
        # print(x.shape)
        out1 = x.view(b, c, -1)  # b,c,n
        out2 = x.view(b, c, -1).permute(0, 2, 1)  # b,n,c
        attention_matrix = torch.bmm(out1, out2)  # b,c,c
        attention_matrix = self.softmax(
            torch.max(attention_matrix, -1, keepdim=True)[0].expand_as(attention_matrix) - attention_matrix)  # b,c,c
        out3 = x.view(b, c, -1)  # b,c,n
        out = torch.bmm(attention_matrix, out3)  # b,c,n
        x = self.gamma * out.view(b, c, n) + x



        y = tgt.permute(0, 2, 1).contiguous()
        f_x, g_x = self.encoder(x)               # f_x(8,512,717),g_x(8,512)
        f_y, g_y = self.encoder(y)               # f_y(8,512,717),g_y(8,512)
        # print(g_x.shape)
        concat = torch.cat((g_x, g_y), dim=1)     # (8,1024)torch.cat是将两个张量（tensor）拼接在一起，cat是concatenate的意思，即拼接，联系在一起
        # print(concat.shape)


        # b, c, n = x.shape
        # print(x.shape)
        # out1 = x.view(b, c, -1)  # b,c,n
        # out2 = x.view(b, c, -1).permute(0, 2, 1)  # b,n,c
        # attention_matrix = torch.bmm(out1, out2)  # b,c,c
        # attention_matrix = self.softmax(
        #     torch.max(attention_matrix, -1, keepdim=True)[0].expand_as(attention_matrix) - attention_matrix)  # b,c,c
        # out3 = x.view(b, c, -1)  # b,c,n
        # out = torch.bmm(attention_matrix, out3)  # b,c,n
        # out = self.gamma * out.view(b, c, n) + x







        # regression initial alignment  初始对齐
        out = self.decoder_qt(concat)     # [8,7]
        batch_t, batch_quat = out[:, :3], out[:, 3:] / (
                torch.norm(out[:, 3:], dim=1, keepdim=True) + 1e-8)      # batch_t(8,3)  batch_quat(8,4)  torch.norm求范数函数input:输入tensor类型的数据，p:指定的范数默认为p=‘fro’，计算矩阵的Frobenius norm (Frobenius 范数)，就是矩阵A各项元素的绝对值平方的总和，dim:指定在哪个维度进行，如果不指定，则是在所有维度进行计算，keepdim:True or False，如果True，则保留dim指定的维度，False则不保留
        batch_R = batch_quat2mat(batch_quat)        # batch_quat(8,4),batch_R(8,3,3),batch_quat2mat四元数转旋转矩阵
        batch_T = torch.cat([batch_R, batch_t[..., None]], dim=-1)    # batch_t(8,3),batch_R(8,3,3),batch_T(8,3,4)

        # overlap prediction   重叠点预测
        g_x_expand = torch.unsqueeze(g_x, dim=-1).expand_as(f_x)  #unsqueeze()函数起升维的作用,参数dim表示在哪个地方加一个维度,expand_as（）函数与expand（）函数类似，功能都是用来扩展张量中某维数据的尺寸，区别是它括号内的输入参数是另一个张量，作用是将输入tensor的维度扩展为与指定tensor相同的size
        g_y_expand = torch.unsqueeze(g_y, dim=-1).expand_as(f_y)   # (8,512,717)
        f_x_ensemble = torch.cat([f_x, g_x_expand, g_y_expand,
                                  g_x_expand - g_y_expand], dim=1)  # (8,2048,717)
        f_y_ensemble = torch.cat([f_y, g_y_expand, g_x_expand,
                                  g_y_expand - g_x_expand], dim=1)   # (8,2048,717)
        x_ol = self.decoder_ol(f_x_ensemble, pooling=False)         # (8,2,717) decoder解码PointNet，pooling=False 源x点云的点重叠分数
        y_ol = self.decoder_ol(f_y_ensemble, pooling=False)

        return batch_T, x_ol, y_ol   # (8,3,4),(8,2,717),(8,2,717)
