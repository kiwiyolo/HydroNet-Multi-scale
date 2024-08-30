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
    model = 'HydroNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    data_sources = {
        'S01': r".\data\S01_DEM_ASTGTMV003\mosaic_US_rect",
        'QS01': r".\data\QS02_ET_monthly\QS02_ET_monthly_US",
        'QS02': r".\data\QS03_SWC_monthly\QS03_SWC_monthly_US_small",
        'F01': r".\data\F01_ERA5_daily",
    }

    labels = {
        'steamflow': r".\data\T01_camels_obsflow\usgs_streamflow_cood_small",
    }

    load_model_path = 'None'  # 加载预训练的模型的路径，为None代表不加载

    image_S_sizes = (300, 700)
    image_QS_sizes = [(300, 700), (300, 700)]
    image_F_sizes = (101, 237)


    CLin_channels, CLhd_channels, kernel_size = 1, 1, 3

    pe_num = 129
    Seq = 10

    batch_size = 2  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    max_epoch = 10
    lr = 0.1  # initial learning rate
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



