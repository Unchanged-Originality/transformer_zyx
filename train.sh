#!/bin/bash

# Transformer 训练脚本
# 使用方法: ./train.sh [配置参数]

echo "开始 Transformer 训练..."

# 设置默认参数
DATA_DIR="./multi30k"
SAVE_DIR="./save_pt"
LOG_DIR="./training_logs"
BATCH_SIZE=32
NUM_EPOCHS=8
RUN_NAME="transformer_train_$(date +%Y%m%d_%H%M%S)"
FILE_PREFIX="multi30k_1108"
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --file_prefix)
            FILE_PREFIX="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建目录
mkdir -p $DATA_DIR
mkdir -p $SAVE_DIR
mkdir -p $LOG_DIR

echo "训练配置:"
echo "数据目录: $DATA_DIR"
echo "保存目录: $SAVE_DIR"
echo "日志目录: $LOG_DIR"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $NUM_EPOCHS"
echo "实验名称: $RUN_NAME"
echo "文件前缀: $FILE_PREFIX"

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH

# 运行训练脚本
#python transformer.py \
python tr.py \
    --mode train \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --run_name $RUN_NAME \
    --file_prefix $FILE_PREFIX \
    --gpu 1

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "训练完成!"
    echo "模型保存在: $SAVE_DIR"
    echo "日志保存在: $LOG_DIR"
else
    echo "训练失败!"
    exit 1
fi