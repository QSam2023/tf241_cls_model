#!/bin/bash

# TensorFlow 2.4.1 债券分类训练脚本
echo "=========================================="
echo "TensorFlow 2.4.1 债券分类训练脚本"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python 未安装"
    exit 1
fi

# 检查配置文件
CONFIG_FILE=${1:-"bond_cls_tf.config"}
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

echo "使用配置文件: $CONFIG_FILE"

# 显示配置信息
echo "配置信息:"
python -c "
from config_helper import ConfigParse
config = ConfigParse('$CONFIG_FILE')
print(f'  模型路径: {config.model_path}')
print(f'  数据目录: {config.data_dir}')
print(f'  保存目录: {config.save_dir}')
print(f'  训练轮次: {config.train_epochs}')
print(f'  批次大小: {config.train_batch_size}')
print(f'  学习率: {config.learning_rate}')
print(f'  GPU: {config.gpus}')
"

echo "=========================================="
echo "开始训练..."
python train_model_tf.py --config_file "$CONFIG_FILE"

echo "=========================================="
echo "训练完成！"
echo "=========================================="
