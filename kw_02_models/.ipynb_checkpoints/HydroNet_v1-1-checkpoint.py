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

        self.CLF = ConvLSTM2D(configs.CLin_channels, configs.CLhd_channels, configs.kernel_size)
        self.CLR = ConvLSTM2D(configs.CLin_channels, configs.CLhd_channels, configs.kernel_size)
        
        MLPin_size = configs.historytime * 2 * (pe_num) * (configs.dim+1)  # S * C * P * D
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
        sf = (sf.unsqueeze(1)  # 形状为 (B, 1, PE, D)
        qsf = (qsf.unsqueeze(1)  # 形状为 (B, 1, PE, D)
 
        ## 3、构建模型，进行CLencoder、CLdecoder得到的双向包含静态、准静态、强迫信息的deep feature
        B, S, C, P, D = f_seq.size()  # 输入序列的形状为  (B, S, 1, PE, D), B为批次大小，S为序列长度，P为位置编码的维度，D+1为对应位置的信息维度
        initial_states = (qsf, sf)  # initial_states 2个均为(B, PE, D) h:qsf, c:sf
        ff, rf = [], []
        hidden_state, cell_state = None, None
        for fi in range(S):
            input_ff = f_seq[:, fi, :, :, :]  # 输入序列的第t个时间步，形状为 (B, PE, D)
            if fi == 0:
                hidden_state, cell_state = self.CLF(input_ff, initial_states)
            else:
                hidden_state, cell_state = self.CLF(input_ff, (hidden_state, cell_state))
            ff.append(hidden_state)
        ff = torch.stack(ff, dim=1)
        for ri in range(S-1, -1, -1):
            input_rf = ff[:, ri, :, :, :]
            if ri == S-1:
                hidden_state, cell_state = self.CLR(input_rf, initial_states)
            else:
                hidden_state, cell_state = self.CLR(input_rf, (hidden_state, cell_state))
            rf.append(hidden_state)
        rf = torch.stack(rf, dim=1)
        x = torch.concat([ff, rf], 2)
        output = self.mlp(x)
        return output

def monitor_memory():
    # 获取当前Python进程的内存占用情况
    process = psutil.Process()
    memory_info = process.memory_info()
    # 打印内存占用情况
    mem = memory_info.rss / 1024 / 1024
    return mem


if __name__ == "__main__":
    from config import DefaultConfig
    import psutil
    import netron

    batch_size = 4
    configs = DefaultConfig
    model = HydroNet(configs)
    print(f"Memory usage after 1 : {monitor_memory()} MB")
    # Dummy inputs for visualization
    S01 = torch.randn(300, 700)
    QS01 = torch.randn(2, 300, 700)
    QS02 = torch.randn(2, 300, 700)
    Fs = torch.randn(batch_size, 10, 168, 101, 237)
    cood = (32, -105)

    print(f"Memory usage after 2 : {monitor_memory()} MB")
    # Adjust tensor sizes to match batch size
    S01 = S01.expand(batch_size, 1, -1, -1)
    QS01 = QS01.expand(batch_size, -1, -1, -1)
    QS02 = QS02.expand(batch_size, -1, -1, -1)

    print(f"Memory usage after 3 : {monitor_memory()} MB")
    # Generate the model visualization
    output = model(S01, [QS01, QS02], Fs, cood)
    print(f"Memory usage after 4 : {monitor_memory()} MB")
    modelData = "./demo.pth"  # 定义模型数据保存的路径
    # modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的
    netron.start(modelData)  # 输出网络结构