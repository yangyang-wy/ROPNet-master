import torch
import torch.nn as nn
from utils import batch_transform, square_dists
# batch_transform：批处理转换；square_dists：平方距离,主要用来在ball query过程中确定每一个点距离采样点的距离
# fps:farthest_point_sample最远点采样（从点中采样M个点）
def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.根据最远点采样（FPS）算法从点中采样M个点
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)        # centroids:质心，初始化
    dists = torch.ones(B, N).to(device) * 1e10                               # 定义距离
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)     # batch里每个样本随机初始化一个最远点的索引（inds:farthest最远）
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds        # 第一个采样点选随机初始化的索引
        cur_point = xyz[batchlists, inds, :] # (B, 3)：得到当前采样点的坐标B*3
        cur_dist = torch.squeeze(square_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1) # 计算当前采样点与其他点的距离
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]    # 选择距离最近的来更新距离（更新维护这个表）
        inds = torch.max(dists, dim=1)[1]  # inds=farthest重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids

# gather_points:聚集点；def index_points(points,idx)
def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C) ：采样点
    '''
    device = points.device
    B, N, C = points.shape                      # (8,448,3)
    inds_shape = list(inds.shape)               # inds_shape(8,1),(8,268)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)             # (1,268)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)   # (8,268),torch.arange().reshape()：改变维度
    return points[batchlists, inds, :]

#
def ball_query(xyz, new_xyz, radius, K, rt_density=False):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)  batchsize，M，nsample,8,512,16
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = square_dists(new_xyz, xyz)  # 得到B N M （就是N个点中每一个和M中每一个的欧氏距离）
    grouped_inds[dists > radius ** 2] = N  # 找到距离大于给定半径的设置成一个N值（1024）索引
    if rt_density:
        density = torch.sum(grouped_inds < N, dim=-1)
        density = density / N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]  # sort排序操作（做升序排序，后面的都是大的值（1024））
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)  # 如果半径内的点没那么多，就直接用最小的距离的那个点（第一个点来代替）来复制补全
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    if rt_density:
        return grouped_inds, density
    return grouped_inds


def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False): # 使用sample_and_group以达到选取中心点分局部区域的目的
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    if M < 0:
        new_xyz = xyz
    else:
        new_xyz = gather_points(xyz, fps(xyz, M))
    if rt_density:
        grouped_inds, density = ball_query(xyz, new_xyz, radius, K,
                                           rt_density=True)   # 遍历半径
    else:
        grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)   # new_xyz:采样的质心点，xyz:原始的点；返回的是索引
    grouped_xyz = gather_points(xyz, grouped_inds) # 得到各个组中实际点
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)  # 去mean new_xyz相当于簇的中心点 (去均值的操作)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    if rt_density:
        return new_xyz, new_points, grouped_inds, grouped_xyz, density
    return new_xyz, new_points, grouped_inds, grouped_xyz


def weighted_icp(src, tgt, weights, _EPS = 1e-8):
    """Compute rigid transforms between two point sets：计算两个点集之间的刚性变换
    参数
    Args:
        src (torch.Tensor): (B, M, 3) points ：源点云
        tgt (torch.Tensor): (B, M, 3) points ：目标点云
        weights (torch.Tensor): (B, M) ：权重

    Returns:
        R, t, transformed_src: (B, 3, 3), (B, 3), (B, M, 3) ：返回值：变换的源点云，R,t

    Modified from open source code:
        https://github.com/yewzijian/RPMNet/blob/master/src/models/rpmnet.py
    """
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_src = torch.sum(src * weights_normalized, dim=1)
    centroid_tgt = torch.sum(tgt * weights_normalized, dim=1)
    src_centered = src - centroid_src[:, None, :]
    tgt_centered = tgt - centroid_tgt[:, None, :]
    cov = src_centered.transpose(-2, -1) @ (tgt_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    #u, s, v = torch.svd(cov, some=False, compute_uv=True)
    # print(cov.shape)
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    # print(u)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid) 计算平移（非中心 质心）
    translation = -rot_mat @ centroid_src[:, :, None] + centroid_tgt[:, :, None]
    translation = torch.squeeze(translation, -1)
    transformed_src = batch_transform(src, rot_mat, translation)
    return rot_mat, translation, transformed_src


def update_split_indices(split_indices, N1, condition):
    pad_num =  N1 - split_indices.shape[0] + 5
    non_split_indices = torch.nonzero(~condition)[:, 1]
    pad_indices = torch.randint(0, non_split_indices.shape[0], (pad_num,))
    pad_split_indices = torch.index_select(non_split_indices, 0, pad_indices)
    new_split_indices = torch.cat((split_indices, pad_split_indices), dim=0)
    return new_split_indices