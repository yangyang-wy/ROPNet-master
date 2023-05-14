import argparse


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters',
                                     add_help=False)
    ## dataset：数据集
    # parser.add_argument('--root', required=True, help='the data path')    # 数据路径
    parser.add_argument('--root', default='./modelnet40_ply_hdf5_2048', help='the data path')
    parser.add_argument('--npts', type=int, default=1024,
                        help='the points number of each pc for training')  # 每个pc（epoch）的训练点数：1024
    # parser.add_argument('--unseen', action='store_true',
    #                     help='whether to use unseen mode')                  # 是否使用不可见模式
    parser.add_argument('--unseen', default=True,
                        help='whether to use unseen mode')
    parser.add_argument('--p_keep', type=list, default=[0.7, 0.7],
                        help='the keep ratio for partial registration')     # 部分配准的保留比率
    parser.add_argument('--ao', action='store_true',
                        help='whether to use asymmetric objects')          # 是否使用非对称对象
    parser.add_argument('--normal', default=True,
                        help='whether to use normal data')                 # 是否使用normal数据
    # parser.add_argument('--noise', action='store_true',
    #                     help='whether to add noise when test')            # 测试时是否添加噪声
    parser.add_argument('--noise', default=True,
                        help='whether to add noise when test')
    parser.add_argument('--use_ppf', default=True,
                        help='whether to use_ppf as input feature')       #是否使用_ppf作为输入特征
    ## model 模型
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
    parser.add_argument('--train_top_prob', type=float, default=0.4,
                        help='')
    parser.add_argument('--test_top_prob', type=float, default=0.4,
                        help='')
    # logs 日志
    parser.add_argument('--resume', default='',
                        help='the path to save training logs and checkpoints')    # 保存训练日志和检查点的路径
    parser.add_argument('--saved_path', default='work_dirs/models',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--log_freq', type=int, default=8,
                        help='the frequency[steps] to save the summary')          # 保存summary的频率[步骤]
    parser.add_argument('--eval_freq', type=int, default=4,
                        help='the frequency[steps] to eval the val set')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='the frequency[epoch] to save the checkpoints')
    return parser


def train_config_params():                                                 # 训练配置参数
    parser = argparse.ArgumentParser(parents=[config_params()])
    parser.add_argument('--seed', type=int, default=1234)                  # 随机种子
    parser.add_argument('--epoches', type=int, default=600)                # 训练次数：600
    parser.add_argument('--batchsize', type=int, default=8)                # 一次训练所选取的样本数，每一次输入的4个元素，训练一遍需要输入Size/batchS
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Neighborhood radius for computing pointnet features')        # 计算pintnet特征的邻域半径：0.3
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N',
                        help='Max num of neighbors to use')
    parser.add_argument('--feat_dim', type=int, default=192,
                        help='Feature dimension (to compute distances on). '                       
                             'Other numbers will be scaled accordingly')            # 特征尺寸（用于计算距离）其他数字将相应缩放
    args = parser.parse_args()
    return args


def eval_config_params():
    parser = argparse.ArgumentParser(parents=[config_params()])
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Neighborhood radius for computing pointnet features')       # 计算pointnet特征的邻域半径
    parser.add_argument('--num_neighbors', type=int, default=16, metavar='N',
                        help='Max num of neighbors to use')                               # 要使用的最大邻居数
    parser.add_argument('--feat_dim', type=int, default=192,
                        help='Feature dimension (to compute distances on)')               # 特征尺寸（用于计算距离）
    parser.add_argument('--checkpoint', default='',
                        help='the path to the trained checkpoint')                        # 通往训练checkpoint的路径
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use the cuda')                                   # 是否使用cuda
    parser.add_argument('--show', action='store_true',
                        help='whether to visualize')                                      # 是否可视化
    args = parser.parse_args()
    return args
