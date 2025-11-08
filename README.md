# 从零构建 Transformer：Multi30k 机器翻译实践

本项目基于 PyTorch 从零实现经典 Transformer 架构（Vaswani et al., 2017），在 Multi30k 数据集（德英跨语言翻译）上完成端到端训练与测试，并集成 Weights & Biases (W&B) 实现训练过程可视化与结果追踪。

## ✨ 项目亮点

### 🔧 从零实现核心组件
- 手动构建多头自注意力、位置编码、前馈网络等关键模块
- 拒绝黑箱调用，适配论文学习与二次开发

### 🚀 全流程工程化
- 覆盖「数据预处理→模型训练→性能评估→可视化分析」闭环
- 提供 Shell 脚本一键启动

### 📊 专业训练策略
- 集成 Label Smoothing 正则化、梯度累积
- 学习率预热与线性衰减
- 提升模型泛化能力与训练稳定性

### 📈 实时可视化跟踪
- 通过 W&B 记录损失曲线、BLEU 分数、学习率变化
- 支持超参数对比与实验结果复现

## 🛠️ 环境准备

### 创建并激活 Conda 环境

```bash
conda create -n transformer python=3.8
conda activate transformer  # 激活环境（后续所有操作需在此环境下执行）
