# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2024/5/6 23:20
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : model_utils
# @IDE     : PyCharm
# -----------------------------------------------------------------
# -*- coding: UTF-8 -*-
#
# Copyright (C) 2023 - 2023 QunLuo, Inc. All Rights Reserved
#
# @Time    : 2023/11/12 9:52
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : model_utils
# @IDE     : PyCharm
# -----------------------------------------------------------------
import numpy as np
import os
import torch
import torch.nn.functional as F



# 获取当前脚本文件所在的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 构建上一级文件夹的路径
parent_path = os.path.dirname(current_path)


def LoadConstantMask():
    # 指定.npy文件的路径
    file_path = parent_path + '\kw_01_data\constant_masks/'
    file_names = ['land_mask', 'soil_type', 'topography']

    # 使用numpy.load()加载.npy文件
    land_mask =  np.load(file_path + 'land_mask.npy')
    soil_type = np.load(file_path + 'land_mask.npy')
    topography = np.load(file_path + 'topography.npy')
    return torch.tensor(land_mask), torch.tensor(soil_type), torch.tensor(topography)


def pad2d(input_tensor):
    # 在二维张量的上、下、左、右分别填充1个元素,0值填充
    padding = (1, 1, 1, 1)

    # 使用F.pad进行填充
    output_tensor = F.pad(input_tensor, padding)

    return output_tensor


def pad3d(input_tensor_3d):
    # 在三维张量的前后、上下、左右分别填充1个元素,0值填充
    padding_3d = (1, 1, 1, 1, 1, 1)

    # 使用F.pad进行三维张量的填充
    output_tensor_3d = F.pad(input_tensor_3d, padding_3d)
    return output_tensor_3d


def gen_mask(atte_shape, masked):
    # 生成一个遮掩矩阵，将非相邻元素之间的位置设置为 -float('inf')，以便在后续的 softmax 操作中被忽略
    mask = torch.zeros(atte_shape)
    batch_size, _, sequence_length, _ = mask.size()
    if masked == True:
        # 遮掩非相邻元素
        for i in range(sequence_length):
            for j in range(sequence_length):
                if abs(i - j) > 1:
                    mask[:, :, i, j] = -float('inf')
    return mask


def truncated_normal_init(p_tensor, std):
        # Calculate the truncation range (you can adjust this based on your requirements)
        a, b = -2.0 * std, 2.0 * std

        # Generate truncated normal distribution values
        values = torch.empty(p_tensor.size()).normal_(mean=0, std=std)
        values = torch.clamp(values, min=a, max=b)

        # Update the earth_specific_bias with the initialized values
        p_tensor.data.copy_(values)