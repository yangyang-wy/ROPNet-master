import copy
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

from utils import batch_transform
from models import CGModule, TFMRModule, gather_points, weighted_icp

# ROPNet模型，
class ROPNet(nn.Module):
    def __init__(self, args):
        super(ROPNet, self).__init__()
        self.N1 = args.test_N1
        self.use_ppf = args.use_ppf
        self.cg= CGModule(in_dim=3, gn=False)
        self.tfmr = TFMRModule(args)

    def forward(self, src, tgt, num_iter=1, train=False):
        '''

        :param src: (B, N, 3) or (B, N, 6) [normal for the last 3]
        :param tgt: (B, M, 3) or (B, M, 6) [normal for the last 3]
        :param num_iter: int, default 1.
        :param train: bool, default False
        :return: dict={'T': (B, 3, 4),
                        '': }
        '''
        B, N, C = src.size()        # B：8，N：717，C：6
        normal_src, normal_tgt = None, None
        if self.use_ppf and C == 6:
            normal_src = src[..., 3:]   # （8，717，3）
            normal_tgt = tgt[..., 3:]
        src = src[..., :3]
        src_raw = copy.deepcopy(src)
        tgt = tgt[..., :3]

        results = {}
        pred_Ts, pred_src = [], []

        # CG module （CG模型）
        T0, x_ol, y_ol = self.cg(src, tgt)       # (8,3,4),(8,2,717),(8,2,717)，变换矩阵T0
        R, t = T0[:, :3, :3], T0[:, :3, 3]       # (8,3,3),(8,3)，旋转矩阵+平移向量
        src_t = batch_transform(src_raw, R, t)   # transform转换
        normal_src_t = None
        if normal_src is not None:
            normal_src_t = batch_transform(normal_src, R).detach()  # detach分离，用于从计算图中分离出一个Tensor
        pred_Ts.append(T0)    # append进行要素的添加
        pred_src.append(src_t)
        x_ol_score = torch.softmax(x_ol, dim=1)[:, 1, :].detach()  # (B, N)（8，717）先detach在对x_ol进行softmax得到x_ol的分数
        y_ol_score = torch.softmax(y_ol, dim=1)[:, 1, :].detach()  # (B, N)

        for i in range(num_iter):
            src_t = src_t.detach()
            src, tgt_corr, icp_weights, similarity_max_inds = \
                self.tfmr(src=src_t,
                          tgt=tgt,
                          x_ol_score=x_ol_score,
                          y_ol_score=y_ol_score,
                          train=train,
                          iter=i,
                          normal_src=normal_src_t,
                          normal_tgt=normal_tgt)

            R_cur, t_cur, _ = weighted_icp(src=src,
                                           tgt=tgt_corr,
                                           weights=icp_weights)
            R, t = R_cur @ R, R_cur @ t[:, :, None] + t_cur[:, :, None]
            T = torch.cat([R, t], dim=-1)
            pred_Ts.append(T)
            src_t = batch_transform(src_raw, R, torch.squeeze(t, -1))   # (8,717,3)
            pred_src.append(src_t)
            normal_src_t = batch_transform(normal_src, R).detach()    # (8,717,3)
            t = torch.squeeze(t, dim=-1)

        ## for overlapping points in src（源点云中重叠点）
        _, x_ol_inds = torch.sort(x_ol_score, dim=-1, descending=True)   # 根据给定的维度对输入张量进行升值或降值排序,descending=True代表降值排序，False代表升值排序
        x_ol_inds = x_ol_inds[:, :self.N1]          # (8,448)
        src_ol1 = gather_points(src_raw, x_ol_inds)   # (8,448,3)
        src_ol2 = gather_points(src_ol1, similarity_max_inds)    # (8,268,3)

        results['pred_Ts'] = pred_Ts
        results['pred_src'] = pred_src
        results['x_ol'] = x_ol
        results['y_ol'] = y_ol
        results['src_ol1'] = src_ol1
        results['src_ol2'] = src_ol2

        return results
