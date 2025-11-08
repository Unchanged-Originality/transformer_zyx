从零构建 Transformer：Multi30k 机器翻译实践
本项目基于 PyTorch 从零实现经典 Transformer 架构（Vaswani et al., 2017），在 Multi30k 数据集（德英跨语言翻译）上完成端到端训练与测试，并集成 Weights & Biases (W&B) 实现训练过程可视化与结果追踪。
项目亮点
从零实现核心组件：手动构建多头自注意力、位置编码、前馈网络等关键模块，拒绝黑箱调用，适配论文学习与二次开发。
全流程工程化：覆盖「数据预处理→模型训练→性能评估→可视化分析」闭环，提供 Shell 脚本一键启动。
专业训练策略：集成 Label Smoothing 正则化、梯度累积、学习率预热与线性衰减，提升模型泛化能力与训练稳定性。
实时可视化跟踪：通过 W&B 记录损失曲线、BLEU 分数、学习率变化，支持超参数对比与实验结果复现。
环境准备
1. 创建并激活 Conda 环境
bash
conda create -n transformer python=3.8
conda activate transformer  # 激活环境（后续所有操作需在此环境下执行）
2. 安装依赖包
bash
pip install -r requirements.txt
依赖说明：torch==1.12.1（PyTorch 核心框架）、wandb==0.15.8（可视化工具）、spacy==3.5.3（文本分词）、sacrebleu==2.3.1（BLEU 评估指标）等。
3. 配置 Weights & Biases（可选，推荐）
若需可视化训练过程，需先注册 W&B 账号（官网链接），获取 API 密钥后执行：
bash
wandb login  # 输入 API 密钥完成登录
快速开始
1. 训练模型
bash
bash train.sh
训练过程说明：
自动下载 Multi30k 数据集（德英翻译，包含 train/val/test 子集），并通过 SpaCy 完成分词与词汇表构建。
初始化 Transformer 模型（6 层 Encoder + 6 层 Decoder，d_model=512，8 头注意力）。
训练参数：批量大小 32（梯度累积 2 步模拟批量 64）、学习率 1e-4、预热步数 4000、Dropout 0.1。
每隔 100 步在验证集评估 BLEU 分数，保存最优模型权重至 checkpoints/best_model.pt。
实时在 W&B Dashboard 显示训练损失、验证损失、BLEU 分数、学习率变化曲线。
2. 测试模型
bash
bash test.sh
测试过程说明：
加载 checkpoints/best_model.pt 最优权重，在测试集上执行贪心解码生成译文。
计算并输出最终 BLEU 分数（机器翻译标准评估指标）。
打印 10 组翻译示例（格式：源文（德语）→ 模型译文（英语）→ 参考译文（英语））。
项目结构
