#!/usr/bin/env python
# coding: utf8

"""
TensorFlow 2.4.1 债券分类使用示例
演示如何使用训练脚本和预测脚本
"""

import os
import json
import subprocess
import sys

def check_environment():
    """检查环境依赖"""
    print("=== 检查环境 ===")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow 版本: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow 未安装")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers 版本: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers 未安装")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn 版本: {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn 未安装")
        return False
    
    return True

def check_data():
    """检查数据文件"""
    print("\n=== 检查数据 ===")
    
    data_dir = "../data/bond"
    files = ["train.jsonl", "test.jsonl", "validation.jsonl"]
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = sum(1 for _ in f)
            print(f"✓ {file}: {lines} 条数据")
        else:
            print(f"✗ {file}: 文件不存在")
            return False
    
    return True

def show_sample_data():
    """显示样本数据"""
    print("\n=== 样本数据 ===")
    
    data_file = "../data/bond/validation.jsonl"
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 只显示前3条
                    break
                data = json.loads(line.strip())
                print(f"样本 {i+1}:")
                print(f"  ID: {data.get('id', 'N/A')}")
                print(f"  文本: {data['text'][:50]}...")
                print(f"  标签: {data['label']}")
                print()

def show_config():
    """显示配置信息"""
    print("\n=== 配置信息 ===")
    
    try:
        from config_helper import ConfigParse
        config = ConfigParse('bond_cls_tf.config')
        
        print(f"模型路径: {config.model_path}")
        print(f"数据目录: {config.data_dir}")
        print(f"保存目录: {config.save_dir}")
        print(f"训练轮次: {config.train_epochs}")
        print(f"批次大小: {config.train_batch_size}")
        print(f"学习率: {config.learning_rate}")
        print(f"标签数量: {config.num_labels}")
        print(f"最大长度: {config.max_length}")
        print(f"GPU设备: {config.gpus}")
        
    except Exception as e:
        print(f"配置加载失败: {e}")

def train_model():
    """训练模型"""
    print("\n=== 开始训练 ===")
    print("注意：这只是演示，实际训练需要预训练模型")
    print("运行命令：python train_model_tf.py")
    print("或者：./run_training.sh")

def predict_example():
    """预测示例"""
    print("\n=== 预测示例 ===")
    print("训练完成后，可以使用以下命令进行预测：")
    print()
    print("1. 预测单个文本：")
    print("   python predict_tf.py --model_path output/bond_cls_tf --text '债券交易相关文本'")
    print()
    print("2. 预测文件：")
    print("   python predict_tf.py --model_path output/bond_cls_tf --input_file ../data/bond/test.jsonl --output_file predictions.jsonl")

def main():
    """主函数"""
    print("TensorFlow 2.4.1 债券分类使用示例")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("\n请先安装依赖:")
        print("pip install -r requirements.txt")
        return
    
    # 检查数据
    if not check_data():
        print("\n请确保数据文件存在于 ../data/bond/ 目录下")
        return
    
    # 显示样本数据
    show_sample_data()
    
    # 显示配置
    show_config()
    
    # 训练说明
    train_model()
    
    # 预测说明
    predict_example()
    
    print("\n" + "=" * 50)
    print("使用说明完成！")
    print("开始训练：./run_training.sh")
    print("查看文档：cat README.md")

if __name__ == "__main__":
    main()
