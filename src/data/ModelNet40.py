import copy
import h5py
import math
import numpy as np
import os
import torch

from torch.utils.data import Dataset
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import  random_select_points, shift_point_cloud, jitter_point_cloud, \
    generate_random_rotation_matrix, generate_random_tranlation_vector, \
    transform, random_crop, shuffle_pc, random_scale_point_cloud, flip_pc

# 训练
half1 = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
         'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
         'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp']
half1_symmetric = ['bottle', 'bowl', 'cone', 'cup', 'flower_pot', 'lamp']

half2 = ['laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano',
         'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
         'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
half2_symmetric = ['tent', 'vase']

# 导入ModelNet40数据集的类
class ModelNet40(Dataset):
    def __init__(self, root, split, npts, p_keep, noise, unseen, ao=False,
                 normal=False):
        super(ModelNet40, self).__init__()
        self.single = False                         # for specific-class visualization （用于特定类可视化）
        assert split in ['train', 'val', 'test']
        self.split = split                         # split：分裂
        self.npts = npts
        self.p_keep = p_keep
        self.noise = noise
        self.unseen = unseen
        self.ao = ao                         # Asymmetric Objects  (不对称对象)
        self.normal = normal
        self.half = half1 if split in 'train' else half2        # 训练 half1
        self.symmetric = half1_symmetric + half2_symmetric
        self.label2cat, self.cat2label = self.label2category(
            os.path.join(root, 'shape_names.txt'))
        self.half_labels = [self.cat2label[cat] for cat in self.half]
        self.symmetric_labels = [self.cat2label[cat] for cat in self.symmetric]
        files = [os.path.join(root, 'ply_data_train{}.h5'.format(i))
                 for i in range(5)]        # 训练文件1-5
        if split == 'test':
            files = [os.path.join(root, 'ply_data_test{}.h5'.format(i))
                     for i in range(2)]
        self.data, self.labels = self.decode_h5(files)
        print(f'split: {self.split}, unique_ids: {len(np.unique(self.labels))}')

        if self.split == 'train':
            self.Rs = [generate_random_rotation_matrix() for _ in range(len(self.data))]  # len（）获取字符串长度或字节数
            self.ts = [generate_random_tranlation_vector() for _ in range(len(self.data))]
# 标签2类别
    def label2category(self, file):            # 标签2类别
        with open(file, 'r') as f:             # r:文件以只读方式打开。文件的指针将会放在文件的开头。
            label2cat = [category.strip() for category in f.readlines()]  # 读标签2类别共40个种类
            cat2label = {label2cat[i]: i for i in range(len(label2cat))}  # 给40个种类贴上索引（cat2label={aieplane:0,......}）
        return label2cat, cat2label

    def decode_h5(self, files):
        points, normal, label = [], [], []           # 点，法线，标签
        for file in files:
            f = h5py.File(file, 'r')
            cur_points = f['data'][:].astype(np.float32)    # （2048，2048，3）
            cur_normal = f['normal'][:].astype(np.float32)   # （2048，2048，3）
            cur_label = f['label'][:].flatten().astype(np.int32)  # （2048）
            if self.unseen:
                idx = np.isin(cur_label, self.half_labels)   # np.isin判断数组元素在另一数组中是否存在
                cur_points = cur_points[idx]   # （1038，2048，3）
                cur_normal = cur_normal[idx]   # （1038，2048，3）
                cur_label = cur_label[idx]     # （1038）
            if self.ao and self.split in ['val', 'test']:
                idx = ~np.isin(cur_label, self.symmetric_labels)
                cur_points = cur_points[idx]
                cur_normal = cur_normal[idx]
                cur_label = cur_label[idx]
            if self.single:
                idx = np.isin(cur_label, [8])
                cur_points = cur_points[idx]
                cur_normal = cur_normal[idx]
                cur_label = cur_label[idx]
            points.append(cur_points)          # append(添加，增加)对点进行要素的添加
            normal.append(cur_normal)
            label.append(cur_label)
        points = np.concatenate(points, axis=0)       # （5190，2048，3）   # 实现点云拼接（concatenate:连接）
        normal = np.concatenate(normal, axis=0)
        data = np.concatenate([points, normal], axis=-1).astype(np.float32)   # （5190，2048，6）
        label = np.concatenate(label, axis=0)
        return data, label
# compose：组成
    def compose(self, item, p_keep):
        tgt_cloud = self.data[item, ...]
        if self.split != 'train':
            np.random.seed(item)
            R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        else:
            tgt_cloud = flip_pc(tgt_cloud)
            R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()

        src_cloud = random_crop(copy.deepcopy(tgt_cloud), p_keep=p_keep[0])  # （1433，6）copy.deepcopy（复制数据），random_crop（随机裁剪）
        src_size = math.ceil(self.npts * p_keep[0])     # 717   # math.ceil返回大于等于参数x的最小整数,即对浮点数向上取整
        tgt_size = self.npts   # 1024
        if len(p_keep) > 1:      # len()函数是Python中的库函数，用于获取对象的长度(对象可以是字符串，列表，元组等)。 它接受一个对象并返回其长度(如果是字符串，则为字符总数，如果是可迭代的则为元素总数)。
            tgt_cloud = random_crop(copy.deepcopy(tgt_cloud),
                                    p_keep=p_keep[1])
            tgt_size = math.ceil(self.npts * p_keep[1])

        src_cloud_points = transform(src_cloud[:, :3], R, t)   # （1433，3）
        src_cloud_normal = transform(src_cloud[:, 3:], R)     # （1433，3）
        src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],     # np.concatenate对array进行拼接 (717,6)
                                   axis=-1)
        src_cloud = random_select_points(src_cloud, m=src_size)   # (717,6)
        tgt_cloud = random_select_points(tgt_cloud, m=tgt_size)

        if self.split == 'train' or self.noise:
            src_cloud[:, :3] = jitter_point_cloud(src_cloud[:, :3])  # 在原始点云数据集上通过标准正太分布（np.random.randn()）添加噪声，作为数据集增强的一种
            tgt_cloud[:, :3] = jitter_point_cloud(tgt_cloud[:, :3])
        tgt_cloud, src_cloud = shuffle_pc(tgt_cloud), shuffle_pc(
            src_cloud)          # 打乱
        return src_cloud, tgt_cloud, R, t
# 实际帮我们取数据的(源点云、目标点云位置信息-normal:标准化)
    def __getitem__(self, item):
        src_cloud, tgt_cloud, R, t = self.compose(item=item,
                                                  p_keep=self.p_keep)
        if not self.normal:
            tgt_cloud, src_cloud = tgt_cloud[:, :3], src_cloud[:, :3]
        return tgt_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data)
