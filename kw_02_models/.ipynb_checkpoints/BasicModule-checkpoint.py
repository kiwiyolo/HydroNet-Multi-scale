# -*- coding: UTF-8 -*-
#
# Copyright (C) 2024 - 2024 QunLuo, Inc. All Rights Reserved
#
# @Time    : 2024/5/6 23:10
# @Author  : QunLuo
# @Email   : 18098503978@163.com
# @File    : BasicModule.py
# @IDE     : PyCharm
# -----------------------------------------------------------------
import random
import sys
from tqdm import tqdm
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViFT(nn.Module):
    def __init__(self, *, target_size, imgs_size: list, patch_size, num_classes=0,  dim, depth, heads, mlp_dim,
                 pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.target_size = target_size
        self.target_height, self.target_width = pair(target_size)
        self.patch_height, self.patch_width = pair(patch_size)
        # 额外编码数，在 ViT 中，为 1，指 class embedding；在 DeiT 中为 2
        self.num_extra_tokens = 1
        if not isinstance(imgs_size, list):
            imgs_size = [imgs_size]

        num_patches = (self.target_height // self.patch_height) * (self.target_width // self.patch_width)
        patch_dim = channels * self.patch_height * self.patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_height, p2=self.patch_width),
            nn.Linear(patch_dim, dim)
        )
        self.tar_pos_dim = num_patches + self.num_extra_tokens
        self.tar_pos_embedding = nn.Parameter(torch.randn(1, self.tar_pos_dim, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # nn.Parameter()定义可学习参数

        # 初始化 self.mlps 为 nn.ModuleDict
        self.mlps = nn.ModuleDict()
        self.vision_norm = nn.LayerNorm(dim)
        for i, img_size in enumerate(imgs_size):
            img_height, img_width = pair(img_size)
            img_dim = (math.ceil(img_height / self.patch_height)) * (
                math.ceil(img_width / self.patch_width)) + self.num_extra_tokens
            # 为每个 i 创建对应的 MLP 实例，并添加到 self.mlps 中, 用于img_dim-->tar_pos_dim
            self.mlps[f'mlp_{i}'] = nn.Sequential(
                nn.Linear(img_dim, self.tar_pos_dim)
            )
        self.img_dim = img_dim
        self.mlp_fusion = nn.Sequential(
            nn.LayerNorm(int(len(imgs_size)) * dim),
            nn.Linear(int(len(imgs_size)) * dim, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        # 如果num_classes == 0，则冻结mlp_head参数
        if num_classes == 0 and self.mlp_head is not None:
            for param in self.mlp_head.parameters():
                param.requires_grad = False

    def forward(self, imgs: list, texts_size=None, texts=None):
        """
         # imgs_size = [(256, 256), (512, 256), (512, 525)],
         output = model([image_inputs1, image_inputs2, image_inputs3], text_inputs)
        """
        # 如果输入不是list，就将其转为list
        if not isinstance(imgs, list):
            imgs = [imgs]
        fusion_x = []
        for i, img in enumerate(imgs):
            _, _, img_height, img_width = img.shape
            # 对输入的images进行填充处理，防止无法被整除的情况
            padding_h = (self.patch_height - img_height % self.patch_height) if img_height % self.patch_height != 0 else 0
            padding_w = (self.patch_height - img_width % self.patch_width) if img_width % self.patch_width != 0 else 0
            self.pad = nn.ZeroPad2d(
                (0, padding_w, 0, padding_h))       # pad左右上下
            img = self.pad(img)
            x = self.to_patch_embedding(img)            # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim

            if tuple(img.shape) != self.target_size:
                # 目标图像尺寸下，长和宽方向的 patch 数
                self.target_shape = (self.target_height // self.patch_height, self.target_width // self.patch_width)
                # 输入图像尺寸下，长和宽方向的 patch 数
                input_shape = (math.ceil(img_height / self.patch_height)) * (
                    math.ceil(img_width / self.patch_width))
                new_pos_embedding = self.interp_pos_encoding(input_shape)
                b, n, _ = x.shape                       # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
                cls_tokens = repeat(self.cls_token, '() n d -> b n d',
                                    b=b)                # self.cls_token: (1, 1, dim) -> cls_tokens: (b, 1, dim)
                x = torch.cat((cls_tokens, x), dim=1)   # 将cls_token拼接到patch token中去       (b, 65, dim)
                x += new_pos_embedding[:, :(n + self.num_extra_tokens)]         # 加位置嵌入（直接加）      (b, 65, dim)
                x = self.vision_norm(x)
                x = x.permute(0, 2, 1)
                x = self.mlps[f'mlp_{i}'](x).permute(0, 2, 1)
            else:
                b, n, _ = x.shape                       # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
                cls_tokens = repeat(self.cls_token, '() n d -> b n d',
                                    b=b)                # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
                x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)
                x += self.pos_embedding[:, :(n + self.num_extra_tokens)]        # 加位置嵌入（直接加）      (b, 65, dim)
            fusion_x.append(x)

        x = torch.cat(fusion_x, dim=2)              # 融合n个输入-->(b, 65, n*dim)
        x = self.mlp_fusion(x)                          # n * dim --> dim

        x = self.dropout(x)
        x = self.transformer(x)                                                 # (b, tar_pos_dim, dim)
        x_cls = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]               # (b, dim)
        x_class = self.mlp_head(self.to_latent(x_cls))                          # Identity (b, dim) --> (b, num_classes)


        return x, x_class                           # x(b, tar_pos_dim, dim), x_class(b, num_classes)每个类型的概率

    def interp_pos_encoding(self, input_shape):
        _, L, D = self.tar_pos_embedding.shape
        target_h, target_w = self.target_shape
        # 位置编码第二个维度大小应当等于 patch 数 + 额外编码数
        assert L == target_h * target_w + self.num_extra_tokens

        # 拆分额外编码和纯位置编码
        extra_tokens = self.tar_pos_embedding[:, :self.num_extra_tokens]
        tar_pos = self.tar_pos_embedding[:, self.num_extra_tokens:]

        # 将位置编码组织成 (1, D, H, W) 形式，其中 D 为通道数
        tar_pos = tar_pos.reshape(1, target_h, target_w, D).permute(0, 3, 1, 2)
        # 进行双三次插值
        input_weight = F.interpolate(tar_pos, size=input_shape, mode='bicubic')
        # 重组位置编码为（1，H*W, D）形式，再拼接上额外编码，即获得新的位置编码
        input_weight = torch.flatten(input_weight, 2).transpose(1, 2)
        new_pos_embed = torch.cat((extra_tokens, input_weight), dim=1)

        return new_pos_embed


class ConvLSTM2D(nn.Module):
    # input size: (B, patch_num + extra_num, D)
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTM2D, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              4 * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)

        # Xavier Initialization
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)


    def forward(self, input, cur_state):
        hidden_state, cell_state = cur_state
        input_hidden_state = torch.cat((input, hidden_state), dim=1)
        conv_outputs = self.conv(input_hidden_state)

        f, i, c, o = torch.split(conv_outputs, self.hidden_channels, dim=1)
        # 避免就地操作（inplace）
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c = torch.tanh(c)

        new_cell_state = cell_state * f + i * c
        hidden_state = o * torch.tanh(new_cell_state)

        return hidden_state, new_cell_state


class GeoPointEncoding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(GeoPointEncoding, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, rel_pos, batch_size):
        """
        经纬度位置-->相对位置-->相对位置编码(B, PE, 1)
        :param coords: 关注点位的经纬度
        :param region_coords: 研究区域的左上/下角和右下/上角的经纬度 [(lat_min, lon_min), (lat_max, lon_max)]【！注意南北半球的问题！】
        :param batch_size: 单次训练参数更新的批次大小
        :return:
        """
        # 将 (a, b) 坐标输入线性层，得到嵌入
        embedded_coords = self.linear(rel_pos)  # 形状为 (1, PE)
        embedded_coords = embedded_coords.unsqueeze(-1)  # 形状为 (1, PE, 1)
        embedded_coords = embedded_coords.expand(batch_size, -1, -1)  # 形状为 (B, PE, 1)
        return embedded_coords




class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, S, 2, P, D)
        B, S, C, P, D = x.shape

        # Reshape to (B, S * 2 * P * D)
        x = x.view(B, S * C * P * D)

        # Pass through MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def main():
    # Example usage
    h1 = random.randint(1, 1000)
    w1 = random.randint(1, 1000)
    h2 = random.randint(1, 1000)
    w2 = random.randint(1, 1000)
    h3 = random.randint(1, 1000)
    w3 = random.randint(1, 1000)
    image_inputs1 = torch.randn(5, 1, h1, w1)  # Batch size, channels, height, width
    image_inputs2 = torch.randn(5, 1, h2, w2)   # Batch size, channels, height, width
    image_inputs3 = torch.randn(5, 1, h3, w3)  # Batch size, channels, height, width
    text_inputs = torch.randn(1, 512)  # Batch size, text embedding size

    # Define image input sizes and text input size
    image_input_sizes = [(h1, w1), (h2, w2), (h3, w3)]
    text_input_size = text_inputs.size(1)

    ## 1、构建模型，进行静态和准静态特征的提取
    # Initialize and forward pass through the model
    model = ViFT(
        target_size=(512, 256),
        imgs_size=image_input_sizes,
        patch_size=32,
        num_classes=4,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    pe_num = model.tar_pos_dim

    x, x_class = model([image_inputs1, image_inputs2, image_inputs3], text_inputs)

    print(f"经过ViFT提取的feature：{tuple(x.shape)}")        # x(b, tar_pos_dim, dim)
    print(f"经过ViFT分类的probability：{tuple(x_class.shape)}")      # x_class(b, num_classes)每个类型的概率

    S = 10

    images_f1 = torch.randn(5, S, 1, 256, 256)
    images_f2 = torch.randn(5, S, 1, 512, 256)
    image_force_sizes = [(256, 256), (512, 256)]
    model_f = ViFT(
        target_size=(512, 256),
        imgs_size=image_force_sizes,
        patch_size=32,
        num_classes=4,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    f_list = []
    for s in range(S):
        f, f_class = model_f([images_f1[:, s, :, :, :], images_f2[:, s, :, :, :]])        # x(b, tar_pos_dim, dim)
        f_list.append(f)
    f_seq = torch.stack(f_list, dim=1).unsqueeze(2)
    print(f"经过ViFT提取的force feature：{tuple(f_seq.shape)}")

    ## 2、构建模型，进行关注点位的相对地理位置编码
    GeoPoint = GeoPointEncoding(2, pe_num)
    point_embed = GeoPoint(torch.tensor(((48, 127),)), ((40, 120), (50, 140)), 5).unsqueeze(1)  # 形状为 (B, PE, 1)
    print(f"经过GeoPoint编码的position feature：{tuple(point_embed.shape)}")
    x = torch.cat((x.unsqueeze(1), point_embed), -1)    # 形状为 (B, 1, PE, D+1)
    point_embed = point_embed.expand(-1, S, -1, -1).unsqueeze(2)
    f = torch.cat((f_seq, point_embed), -1)      # 形状为 (B, S, 1, PE, D+1)

    ## 3、构建模型，进行ConvLSTM得到的包含静态、准静态、强迫信息的deep feature
    input_channels, hidden_channels, kernel_size = 1, 1, 3
    model2 = ConvLSTM2D(input_channels, hidden_channels, kernel_size)
    outputs, (hidden_state, cell_state) = model2(f, (x, x))
    print(f"经过ConvLSTM编码的deep feature：{tuple(outputs.shape)}")      # x_class(b, num_classes)每个类型的概率
    def forward(self, input_seq, initial_states):
        B, S, C, P, D = input_seq.size()   # 输入序列的形状为 (B, S, PE, D+1), B为批次大小，S为序列长度，P为位置编码的维度，D+1为对应位置的信息维度
        hidden_state, cell_state = initial_states  # initial_states 2个均为(B, PE, D+1)

        outputs = []
        for t in range(S):
            input_t = input_seq[:, t, :, :, :]  # 输入序列的第t个时间步，形状为 (B, PE, D+1)
            hidden_state, cell_state = self.step(input_t, (hidden_state, cell_state))
            outputs.append(hidden_state)

        outputs = torch.stack(outputs, dim=1)  # 将所有时间步的输出堆叠在一起，形状为 (B, S, P, D)
        return outputs, (hidden_state, cell_state)


if __name__ == '__main__':
    for t in tqdm(range(1000), desc="利用随机size的输入验证模型数据流上的可行性..."):
        main()
    sys.exit()

