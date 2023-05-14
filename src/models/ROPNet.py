import copy
import numpy as np
# import open3d as o3d
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
        # self.N1 = args.test_N1
        self.N1 = 448
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
        src_raw = copy.deepcopy(src)     # 复制 raw（原始的）
        tgt = tgt[..., :3]

        results = {}
        pred_Ts, pred_src = [], []

        # CG module （CG模型）
        T0, x_ol, y_ol = self.cg(src, tgt)       # (8,3,4),(8,2,717),(8,2,717)，变换矩阵T0，x_ol，y_ol:源X和目标Y点云的点重叠分数
        R, t = T0[:, :3, :3], T0[:, :3, 3]       # (8,3,3),(8,3)，旋转矩阵+平移向量
        src_t = batch_transform(src_raw, R, t)   # transform转换 src_t转换后的源点云
        normal_src_t = None
        if normal_src is not None:
            normal_src_t = batch_transform(normal_src, R).detach()  # detach分离，用于从计算图中分离出一个Tensor
        pred_Ts.append(T0)    # append进行要素的添加
        pred_src.append(src_t)
        x_ol_score = torch.softmax(x_ol, dim=1)[:, 1, :].detach()  # (B, N)（8，717）先detach在对x_ol进行softmax得到x_ol的分数
        y_ol_score = torch.softmax(y_ol, dim=1)[:, 1, :].detach()  # (B, N)

        # 获取点云切分获得索引
        batch_num, point_num, xyz = src_t.shape
        batch_indices_list = []
        for b in range(batch_num):
            src_t_batch = src_t[b:b+1, :, :]
            src_t_x = src_t_batch[:, :, 0:1]
            src_t_y = src_t_batch[:, :, 1:2]
            src_t_z = src_t_batch[:, :, 2:3]
            x_median = float(torch.median(src_t_x).data)
            y_median = float(torch.median(src_t_y).data)
            z_median = float(torch.median(src_t_z).data)
            split_indices_list = []
            condition_1 = torch.logical_and((src_t_batch[:, :, 0] >= x_median), (src_t_batch[:, :, 1] > y_median))
            split_indices_list.append(torch.nonzero(condition_1)[:, 1])
            condition_2 = torch.logical_and((src_t_batch[:, :, 0] < x_median), (src_t_batch[:, :, 1] >= y_median))
            split_indices_list.append(torch.nonzero(condition_2)[:, 1])
            condition_3 = torch.logical_and((src_t_batch[:, :, 0] > x_median), (src_t_batch[:, :, 1] <= y_median))
            split_indices_list.append(torch.nonzero(condition_3)[:, 1])
            condition_4 = torch.logical_and((src_t_batch[:, :, 0] <= x_median), (src_t_batch[:, :, 1] < y_median))
            split_indices_list.append(torch.nonzero(condition_4)[:, 1])
            batch_indices_list.append(split_indices_list)

        R_batch_list = []
        T_batch_list = []
        similarity_max_inds_batch_list = []
        for b, indices in enumerate(batch_indices_list):
            R_split_list = []
            T_split_list = []
            similarity_max_inds_split_list = []
            for i in indices:
                src_t_split = src_t[b:b+1, i, :].detach()
                src_split, tgt_corr_split, icp_weights_split, similarity_max_inds_split = \
                    self.tfmr(src=src_t_split,
                              tgt=tgt[b:b+1, i, :],
                              x_ol_score=x_ol_score[b:b+1, i],
                              y_ol_score=y_ol_score[b:b+1, i],
                              train=train,
                              iter=0,
                              normal_src=normal_src_t[b:b+1, i, :],
                              normal_tgt=normal_tgt[b:b+1, i, :])
                R_cur_split, t_cur_split, _ = weighted_icp(src=src_split,
                                               tgt=tgt_corr_split,
                                               weights=icp_weights_split)
                R_split, t_split = R_cur_split @ R[b:b+1, :, :], R_cur_split @ t[b:b+1, :, None] + t_cur_split[:, :, None]
                T_split = torch.cat([R_split, t_split], dim=-1)
                R_split_list.append(R_split)
                T_split_list.append(T_split)
                similarity_max_inds_split_list.append(i[similarity_max_inds_split])
            R_batch = torch.mean(torch.stack(R_split_list), dim=0)
            T_batch = torch.mean(torch.stack(T_split_list), dim=0)
            # similarity_max_inds_batch = torch.stack(similarity_max_inds_split_list, dim=0).reshape(1, -1)
            R_batch_list.append(R_batch)
            T_batch_list.append(T_batch)
            # similarity_max_inds_batch_list.append(similarity_max_inds_batch)
        R = torch.stack(R_batch_list, dim=0).squeeze()
        T = torch.stack(T_batch_list, dim=0).squeeze()
        # similarity_max_inds = torch.stack(similarity_max_inds_batch_list, dim=0).squeeze()
        pred_Ts.append(T)
        src_t = batch_transform(src_raw, R, torch.squeeze(t, -1))  # (8,717,3)
        pred_src.append(src_t)
        normal_src_t = batch_transform(normal_src, R).detach()  # (8,717,3)
        t = torch.squeeze(t, dim=-1)


        # for i in range(num_iter):
        #     src_t = src_t.detach()
        #     src, tgt_corr, icp_weights, similarity_max_inds = \
        #         self.tfmr(src=src_t,
        #                   tgt=tgt,
        #                   x_ol_score=x_ol_score,
        #                   y_ol_score=y_ol_score,
        #                   train=train,
        #                   iter=i,
        #                   normal_src=normal_src_t,
        #                   normal_tgt=normal_tgt)
        #
        #     R_cur, t_cur, _ = weighted_icp(src=src,
        #                                    tgt=tgt_corr,
        #                                    weights=icp_weights)
        #     R, t = R_cur @ R, R_cur @ t[:, :, None] + t_cur[:, :, None]
        #     T = torch.cat([R, t], dim=-1)
        #     pred_Ts.append(T)
        #     src_t = batch_transform(src_raw, R, torch.squeeze(t, -1))   # (8,717,3)
        #     pred_src.append(src_t)
        #     normal_src_t = batch_transform(normal_src, R).detach()    # (8,717,3)
        #     t = torch.squeeze(t, dim=-1)

        ## for overlapping points in src（源点云中重叠点）
        _, x_ol_inds = torch.sort(x_ol_score, dim=-1, descending=True)   # 根据给定的维度对输入张量进行升值或降值排序,descending=True代表降值排序，False代表升值排序
        # x_ol_inds = x_ol_inds[:, :self.N1]          # (8,448)
        # src_ol1 = gather_points(src_raw, x_ol_inds)   # (8,448,3)
        # src_ol2 = gather_points(src_ol1, similarity_max_inds)    # (8,268,3)
        # src_ol2 = gather_points(src_raw, similarity_max_inds)

        results['pred_Ts'] = pred_Ts
        results['pred_src'] = pred_src
        results['x_ol'] = x_ol
        results['y_ol'] = y_ol
        # results['src_ol1'] = src_ol1
        # results['src_ol2'] = src_ol2

        return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Configuration Parameters',
                                     add_help=False)
    parser.add_argument('--use_ppf', default=True,
                        help='whether to use_ppf as input feature')       #是否使用_ppf作为输入特征
    # parser.add_argument('--train_N1', type=int, default=448, help='')
    parser.add_argument('--train_N1', type=int, default=112, help='')
    # parser.add_argument('--train_M1', type=int, default=717, help='')
    parser.add_argument('--train_M1', type=int, default=180, help='')
    parser.add_argument('--train_similarity_topk', type=int, default=3, help='')
    # parser.add_argument('--test_N1', type=int, default=448, help='')
    parser.add_argument('--test_N1', type=int, default=112, help='')
    # parser.add_argument('--test_M1', type=int, default=717, help='')
    parser.add_argument('--test_M1', type=int, default=180, help='')
    parser.add_argument('--test_similarity_topk', type=int, default=1,
                        help='')
    parser.add_argument('--train_top_prob', type=float, default=0.6,
                        help='')
    parser.add_argument('--test_top_prob', type=float, default=0.4,
                        help='')
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Neighborhood radius for computing pointnet features')       # 计算pointnet特征的邻域半径
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N',
                        help='Max num of neighbors to use')                               # 要使用的最大邻居数
    parser.add_argument('--feat_dim', type=int, default=192,
                        help='Feature dimension (to compute distances on)')
    args = parser.parse_args()
    model = ROPNet(args)
    src = torch.randn((8, 717, 6), dtype=torch.float32)
    tgt = torch.randn((8, 717, 6), dtype=torch.float32)
    results = model(src, tgt)

