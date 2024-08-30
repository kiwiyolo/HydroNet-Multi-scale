# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved 
#
# @Time    : 2024/5/6 23:10
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : HydroNet.py
# @IDE     : PyCharm
# -----------------------------------------------------------------

import sys
from .BasicModule import *
import psutil

class HydroNet(nn.Module):
    def __init__(self, configs: object):
        super(HydroNet, self).__init__()
        print("构建HydroNet模型实例中......")
        self.configs = configs
        self.batch_size = configs.batch_size
        self.Seq = configs.historytime

        self.Svift = ViFT(
            target_size=configs.target_size,
            imgs_size=configs.image_S_sizes,
            patch_size=configs.patch_size,
            num_classes=4,
            dim=configs.dim,
            depth=configs.depth,
            heads=configs.heads,
            mlp_dim=configs.mlp_dim,
            channels=configs.channelsS,
            dropout=configs.dropoutS,
            emb_dropout=configs.emb_dropout
        )

        self.QSvift = ViFT(
            target_size=configs.target_size,
            imgs_size=configs.image_QS_sizes,
            patch_size=configs.patch_size,
            num_classes=4,
            dim=configs.dim,
            depth=configs.depth,
            heads=configs.heads,
            mlp_dim=configs.mlp_dim,
            channels=configs.channelsQS,
            dropout=configs.dropoutQS,
            emb_dropout=configs.emb_dropout
        )

        self.Fvift = ViFT(
            target_size=configs.target_size,
            imgs_size=configs.image_F_sizes,
            patch_size=configs.patch_size,
            num_classes=4,
            dim=configs.dim,
            depth=configs.depth,
            heads=configs.heads,
            mlp_dim=configs.mlp_dim,
            channels=configs.channelsF,
            dropout=configs.dropoutF,
            emb_dropout=configs.emb_dropout
        )
        
        pe_num = self.Svift.tar_pos_dim
        self.GeoPoint = GeoPointEncoding(2, pe_num)

        MLPin_size = (configs.historytime+2) * 1 * (pe_num) * (configs.dim+1)  # S * C * P * D
        MLPhd_size = configs.MLPhd_size  # You can adjust the hidden layer size
        MLPot_size = configs.leadtime  # R
        self.mlp = MLP(MLPin_size, MLPhd_size, MLPot_size)

    def forward(self, Ss, QSs, Fs, rel_pos):
        ## 1、构建模型，进行静态和准静态特征的提取
        # Initialize and forward pass through the model
        sf, sf_class = self.Svift(Ss)
        qsf, qsf_class = self.QSvift(QSs)

        ## 2、构建模型，进行强迫因子特征的提取
        Seq = self.Seq
        f_list = []
        for s in range(Seq):
            f, f_class = self.Fvift(Fs[:, s, :, :, :])  # x(b, tar_pos_dim, dim)
            f_list.append(f)
        f_seq = torch.stack(f_list, dim=1).unsqueeze(2)

        ## 3、构建模型，进行关注点位的相对地理位置编码
        point_embed = self.GeoPoint(rel_pos, self.batch_size).unsqueeze(1)  # 形状为 (B, 1, PE, 1)
        sf = torch.cat((sf.unsqueeze(1), point_embed), -1)  # 形状为 (B, 1, PE, D+1)
        qsf = torch.cat((qsf.unsqueeze(1), point_embed), -1)  # 形状为 (B, 1, PE, D+1)
        point_embed = point_embed.expand(-1, self.Seq, -1, -1).unsqueeze(2)
        f_seq = torch.cat((f_seq, point_embed), -1)  # 形状为 (B, S, 1, PE, D+1)
        output = self.mlp(
            torch.cat(
                (sf.unsqueeze(1), qsf.unsqueeze(1), f_seq), 1))
        return output

def monitor_memory():
    # 获取当前Python进程的内存占用情况
    process = psutil.Process()
    memory_info = process.memory_info()
    # 打印内存占用情况
    
    return memory_info.rss / 1024 / 1024

