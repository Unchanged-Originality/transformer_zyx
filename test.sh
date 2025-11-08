#!/bin/bash

# Transformer 测试脚本
# 使用方法: ./test.sh [配置参数]

echo "开始 Transformer 测试..."

# 设置默认参数
DATA_DIR="./multi30k"
MODEL_PATH="./multi30k_model_final.pt"
LOG_DIR="./test_logs"
BATCH_SIZE=32
RUN_NAME="transformer_test_$(date +%Y%m%d_%H%M%S)"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
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
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件不存在: $MODEL_PATH"
    echo "请先运行训练脚本或指定正确的模型路径"
    exit 1
fi

# 创建日志目录
mkdir -p $LOG_DIR

echo "测试配置:"
echo "数据目录: $DATA_DIR"
echo "模型路径: $MODEL_PATH"
echo "日志目录: $LOG_DIR"
echo "批次大小: $BATCH_SIZE"
echo "实验名称: $RUN_NAME"

# 设置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH

# 运行测试脚本
python transformer.py \
    --mode test \
    --data_dir $DATA_DIR \
    --model_path $MODEL_PATH \
    --log_dir $LOG_DIR \
    --batch_size $BATCH_SIZE \
    --run_name $RUN_NAME \
    --gpu 1

# 检查测试是否成功
if [ $? -eq 0 ]; then
    echo "测试完成!"
    echo "测试日志保存在: $LOG_DIR"
else
    echo "测试失败!"
    exit 1
fi