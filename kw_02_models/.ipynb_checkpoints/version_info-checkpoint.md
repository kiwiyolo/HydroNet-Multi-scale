# 不同版本的HydroNet模型的情况
## 版本 v1
- 变量整理：
    - 路径字符串整合到config.py文件中
    - 模型构建时所需变量整合到config.py，便于在不修改模型框架时训练不同参数下的model
- 优化：
    - 删去了不必要的shape的print（最开始只是为了检查整个model数据流合理而设置）
    - 删去了__main__当其为主文件可以正常运行的检测
    
## 版本 v11
- 删去cood的输入以及编码：
    - 减少内存占用
    - 降低负荷
    - 结果：cood编码并没有占用大量内存，删去之后并没有明显改善
    
## 版本 v2
- 保留cood encoding过程，删去ConvLSTM进行temporal encoding：
    - 减少内存占用
    - 降低负荷
    - 仅依靠transform的attention机制实现预测