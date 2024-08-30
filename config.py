# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2024/5/2 14:51
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : config
# @IDE     : PyCharm
# -----------------------------------------------------------------
import warnings, sys
from tqdm import tqdm


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    log_dir = './log'
    # 设置数据信息
    ## 解释因子路径
    data_sources = {
        'S01': r"./data/S01_DEM_ASTGTMV003/mosaic_US_rect_3k",
        'S02': r"./data/S02_basin_set_full_res",
        'QS01': r"./data/QS02_ET_monthly/QS02_ET_monthly_US-low size",
        'QS02': r"./data/QS03_SWC_monthly/QS03_SWC_monthly_US_low size",
        'F01': r"./data/F01_ERA5_7_daily",
    }
    ## 目标因子路径
    labels = {
        'steamflow': r"./data/T01_camels_obsflow/usgs_streamflow_cood_small",
    }
    ## 位置信息
    cood = ((47.23739, -68.58264),)    # 目标点位经纬度
    region_coords = ((25, -125), (50, -66))    # 研究区范围
    ## 数据配置（与解释因子匹配）
    image_S_sizes = [(900, 2120), (900, 2120)]
    image_QS_sizes = [(900, 2120), (900, 2120)]
    image_F_sizes = (101, 237)
    # image_S_sizes = (212401, 90001)
    # image_QS_sizes = [(43200, 21600), (7081, 3001)]
    # image_F_sizes = (101, 237)
    met_variables = ['u10', 'v10', 'd2m', 't2m', 'sp', 'fdir', 'tp']      # 读取所需的变量
    historytime = 30   # 输入驱动的历史序列长度
    leadtime = 5   # 预测提前期的序列长度
    datalen = 50  # 数据总个数
    gauge = 'merged_01013500_streamflow_qc.csv'
    target_column = 'value'
    target_lat = 'gauge_lat'
    target_lon = 'gauge_lon'

    # model信息
    model = 'HydroNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    load_model_path = 'None'  # 加载预训练的模型的路径，为None代表不加载
    ## 构建3种ViFT
    target_size=(512, 256)
    patch_size=(32, 32)
    dim=256
    depth=6
    heads=16
    mlp_dim=256
    emb_dropout=0.1
    channelsS=1
    channelsQS=12
    channelsF=168
    dropoutS = dropoutQS = dropoutF = 0.1
    ## 构建2种ConvLSTM2D(CLF, CLR)
    CLin_channels, CLhd_channels, kernel_size = 1, 1, 3
    ## 构建1种MLP
    MLPhd_size = 128

    # 训练配置（超参数设置）
    batch_size = 2  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch
    max_epoch = 100
    lr = 0.01  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

    
    def parse(self, kwargs) -> object:
        '''
        根据字典kwargs 更新 config参数
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))



