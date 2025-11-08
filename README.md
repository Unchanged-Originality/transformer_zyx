# 从零构建 Transformer：Multi30k 机器翻译实践

本项目基于 PyTorch 从零实现经典 Transformer 架构（Vaswani et al., 2017），在 Multi30k 数据集（德英跨语言翻译）上完成端到端训练与测试，并集成 Weights & Biases (W&B) 实现训练过程可视化与结果追踪。

### 🔧 从零实现核心组件
- 手动构建多头自注意力、位置编码、前馈网络等关键模块，适配论文学习与二次开发

## 🛠️ 环境准备

### 创建并激活 Conda 环境

```bash
conda create -n transformer python=3.8
conda activate transformer  # 激活环境（后续所有操作需在此环境下执行）

### 安装依赖包
```bash
pip install -r requirements.txt

###配置 Weights & Biases（可选，推荐）
若需可视化训练过程，需先注册 W&B 账号，获取 API 密钥后执行：

```bash
wandb login

## 🚀 快速开始
### 训练模型
```bash
bash train.sh
###测试模型
```bash
bash test.sh
