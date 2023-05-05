import argparse                     # argparse是一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
import json                         # JSON(JavaScript Object Notation) 是一种轻量级的数据交换格式，易于人阅读和编写。json是最常用的数据交换格式，在python编码中需要将json字符串加载为python可以识别的python对象。
import numpy as np                  # numpy是使用C语言实现的一个数据计算库，它用来处理相同类型，固定长度的元素。使用numpy操作数据时，系统运行的速度比使用python代码快很多。numpy中还提供了很多的数据处理函数，例如傅里叶变化，矩阵操作，数据拟合等操作。
import open3d                       # 点云数据可视化
import os                           # os.path模块主要用于文件的属性获取
import torch
import torch.nn as nn              # torch.nn是pytorch中自带的一个函数库，里面包含了神经网络中使用的一些常用函数
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter   # 将条目直接写入log_dir中的事件文件以供TensorBoard使用
from tensorboardX import SummaryWriter
from tqdm import tqdm                # 训练进度条提示信息

import sys                     # 先把系统文件夹调到Root下，防止找不到文件

ROOT = os.path.abspath(__file__)     # 返回脚本的绝对路径
sys.path.append(os.path.dirname(os.path.dirname(ROOT)))

from data import ModelNet40
from models import ROPNet
from loss import cal_loss
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc, inv_R_t, batch_transform, setup_seed, square_dists
from configs import train_config_params as config_params
# os.environ["CUDA_VISIBLE_DEVICES"] ="0"

test_min_loss, test_min_r_mse_error, test_min_rot_error = \
        float('inf'), float('inf'), float('inf')

# 保存在summary文件夹下（）
def save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse, global_step, tag,
                 lr=None):
    for k, v in loss_all.items():
        loss = np.mean(v.item())
        writer.add_scalar(f'{k}/{tag}', loss, global_step)
    cur_r_mse = np.mean(cur_r_mse)
    writer.add_scalar(f'RError/{tag}', cur_r_mse, global_step)
    cur_r_isotropic = np.mean(cur_r_isotropic)
    writer.add_scalar(f'rotError/{tag}', cur_r_isotropic, global_step)
    if lr is not None:
        writer.add_scalar('Lr', lr, global_step)

# 训练迭代Epoch轮，每一轮会调用train_one_epoch来进行训练（数据加载、模型设置、计算损失、优化器、轮数、记录日志、保存）
# @time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer, epoch, log_freq, writer):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    global test_min_loss, test_min_r_mse_error, test_min_rot_error
    # 在进程0中打印训练进度，模型构建好之后的取数据迭代训练
    for step, (tgt_cloud, src_cloud, gtR, gtt) in enumerate(tqdm(train_loader)):
        np.random.seed((epoch + 1) * (step + 1))
        # tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
        #                                  gtR.cuda(), gtt.cuda()
        tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cpu(), src_cloud.cpu(), \
                                         gtR.cpu(), gtt.cpu()

        optimizer.zero_grad()
        results = model(src=src_cloud,
                        tgt=tgt_cloud,
                        num_iter=1,
                        train=True)
        pred_Ts = results['pred_Ts']
        pred_src = results['pred_src']
        x_ol = results['x_ol']
        y_ol = results['y_ol']
        inv_R, inv_t = inv_R_t(gtR, gtt)
        gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                             inv_t)   # （8，717，3）transformed_src
        dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])   # (8,717,717)  square_dists:用来在ball query过程中确定每一个点距离采样点的距离,返回的是两组点之间两两的欧几里德距离，即N×M的矩阵
        loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                           pred_transformed_src=pred_src,
                           dists=dists,
                           x_ol=x_ol,
                           y_ol=y_ol)

        loss = loss_all['total']
        loss.backward()
        optimizer.step()

        R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        global_step = epoch * len(train_loader) + step + 1

        if global_step % log_freq == 0:
            save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                         global_step, tag='train',
                         lr=optimizer.param_groups[0]['lr'])

        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic)
        t_isotropic.append(cur_t_isotropic)
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


# @time_calc
def test_one_epoch(test_loader, model, loss_fn, epoch, log_freq, writer):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for step, (tgt_cloud, src_cloud, gtR, gtt) in enumerate(
                tqdm(test_loader)):
            # tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cuda(), src_cloud.cuda(), \
            #                                  gtR.cuda(), gtt.cuda()
            tgt_cloud, src_cloud, gtR, gtt = tgt_cloud.cpu(), src_cloud.cpu(), \
                                             gtR.cpu(), gtt.cpu()

            results = model(src=src_cloud,
                            tgt=tgt_cloud,
                            num_iter=1)
            pred_Ts = results['pred_Ts']
            pred_src = results['pred_src']
            x_ol = results['x_ol']
            y_ol = results['y_ol']
            inv_R, inv_t = inv_R_t(gtR, gtt)
            gt_transformed_src = batch_transform(src_cloud[..., :3], inv_R,
                                                 inv_t)
            dists = square_dists(gt_transformed_src, tgt_cloud[..., :3])
            loss_all = loss_fn(gt_transformed_src=gt_transformed_src,
                               pred_transformed_src=pred_src,
                               dists=dists,
                               x_ol=x_ol,
                               y_ol=y_ol)
            loss = loss_all['total']

            R, t = pred_Ts[-1][:, :3, :3], pred_Ts[-1][:, :3, 3]
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            global_step = epoch * len(test_loader) + step + 1
            if global_step % log_freq == 0:
                save_summary(writer, loss_all, cur_r_isotropic, cur_r_mse,
                             global_step, tag='test')

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic)
            t_isotropic.append(cur_t_isotropic)
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


def main():
    args = config_params()
    print(args)
# 训练前的一些设置（建立文件夹（summary，checkpoints）还有他们的path）
    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)                                     # os.makedirs(path) 方法用于递归创建目录，即支持创建多层目录。
        with open(os.path.join(args.saved_path, 'args.json'), 'w') as f:  # os.path.join()函数用于路径拼接文件路径，可以传入多个路径
            json.dump(args.__dict__, f, ensure_ascii=False, indent=2)
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
# 训练数据集设置，数据加载dataloader
    train_set = ModelNet40(root=args.root,
                           split='train',
                           npts=args.npts,
                           p_keep=args.p_keep,
                           noise=args.noise,
                           unseen=args.unseen,
                           ao=args.ao,
                           normal=args.normal
                           )
    test_set = ModelNet40(root=args.root,
                          split='val',
                          npts=args.npts,
                          p_keep=args.p_keep,
                          noise=args.noise,
                          unseen=args.unseen,
                          ao=args.ao,
                          normal=args.normal
                          )
    train_loader = DataLoader(train_set, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)                  # shuffle=True，用于打乱数据集，每次都会以不同的顺序返回。
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)

    model = ROPNet(args)
    # model = model.cuda()
    model = model.cpu()
    loss_fn = cal_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=40,
                                                                     T_mult=2,
                                                                     eta_min=1e-6,
                                                                     last_epoch=-1)
# 训练参数的可视化 （test_min_loss, test_min_r_mse_error, test_min_rot_error）
    writer = SummaryWriter(summary_path)

    for i in tqdm(range(epoch)):                  # tqdm:python进度条器
        for _ in train_loader:
            pass
        for _ in test_loader:
            pass
        scheduler.step()
    global test_min_loss, test_min_r_mse_error, test_min_rot_error
    for epoch in range(epoch, args.epoches):
        print('=' * 20, epoch + 1, '=' * 20)
        train_results = train_one_epoch(train_loader=train_loader,
                                        model=model,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        epoch=epoch,
                                        log_freq=args.log_freq,
                                        writer=writer)
        print_train_info(train_results)

        test_results = test_one_epoch(test_loader=test_loader,
                                      model=model,
                                      loss_fn=loss_fn,
                                      epoch=epoch,
                                      log_freq=args.log_freq,
                                      writer=writer)
        print_train_info(test_results)
        test_loss, test_r_error, test_rot_error = \
            test_results['loss'], test_results['r_mse'], \
            test_results['r_isotropic']
        if test_loss < test_min_loss:
            saved_path = os.path.join(checkpoints_path,
                                      "min_loss.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_loss = test_loss
        if test_rot_error < test_min_rot_error:
            saved_path = os.path.join(checkpoints_path,
                                      "min_rot_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_rot_error = test_rot_error

        scheduler.step()


if __name__ == '__main__':
    main()
