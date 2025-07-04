#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境初始化脚本
用于设置tf241_cls_model项目的开发环境
"""

import os
import sys
import json
import platform
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("正在检查Python版本...")
    current_version = sys.version_info
    required_version = (3, 7)
    
    if current_version[:2] >= required_version:
        print(f"✅ Python版本检查通过: {current_version.major}.{current_version.minor}.{current_version.micro}")
        return True
    else:
        print(f"❌ Python版本不符合要求!")
        print(f"   当前版本: {current_version.major}.{current_version.minor}.{current_version.micro}")
        print(f"   要求版本: {required_version[0]}.{required_version[1]}+")
        return False


def create_directory_structure():
    """创建项目目录结构"""
    print("正在创建项目目录结构...")
    
    directories = [
        "data",
        "models",
        "scripts",
        "config",
        "logs",
        "results",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 创建目录: {directory}/")
    
    print("✅ 目录结构创建完成")


def create_default_config():
    """创建默认配置文件"""
    print("正在创建默认配置文件...")
    
    config = {
        "model_config": {
            "model_type": "text_classifier",
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout_rate": 0.3,
            "max_sequence_length": 256
        },
        "training_config": {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 3,
            "save_best_only": True
        },
        "data_config": {
            "text_column": "text",
            "label_column": "label",
            "encoding": "utf-8",
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True
        },
        "paths": {
            "data_dir": "data",
            "model_dir": "models",
            "log_dir": "logs",
            "result_dir": "results"
        }
    }
    
    config_path = Path("config") / "default_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 默认配置文件创建完成: {config_path}")


def create_gitignore():
    """创建.gitignore文件"""
    print("正在创建.gitignore文件...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
*.pyc
*.pyo
*.pyd
.env
.env.local
.env.dev
.env.prod

# TensorFlow
*.pb
*.pbtxt
*.ckpt*
*.index
*.meta
*.data*
saved_model/
checkpoints/

# 模型文件
*.h5
*.hdf5
*.model
*.weights

# 数据文件
*.csv
*.json
*.txt
*.xlsx
*.xls
data/raw/
data/processed/

# 日志文件
*.log
logs/
tensorboard_logs/

# 结果文件
results/
output/
predictions/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# macOS
.DS_Store
.AppleDouble
.LSOverride

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
"""
    
    with open(".gitignore", 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore文件创建完成")


def create_sample_data():
    """创建示例数据文件"""
    print("正在创建示例数据文件...")
    
    sample_data = [
        {"text": "这是一个正面的评论，产品质量很好", "label": 1},
        {"text": "这个产品不太好，质量有问题", "label": 0},
        {"text": "服务态度非常好，值得推荐", "label": 1},
        {"text": "价格太贵了，性价比不高", "label": 0},
        {"text": "物流速度很快，包装也很好", "label": 1}
    ]
    
    sample_path = Path("data") / "sample_data.json"
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 示例数据文件创建完成: {sample_path}")


def print_system_info():
    """打印系统信息"""
    print("\n" + "="*50)
    print("系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    print("="*50)


def print_next_steps():
    """打印后续步骤"""
    print("\n" + "="*50)
    print("🎉 环境初始化完成!")
    print("\n后续步骤:")
    print("1. 安装依赖包:")
    print("   pip install -r requirements.txt")
    print("\n2. 准备训练数据:")
    print("   将数据文件放入 data/ 目录")
    print("\n3. 开始训练:")
    print("   python train.py")
    print("\n4. 进行预测:")
    print("   python predict.py --text '您的文本内容'")
    print("="*50)


def main():
    """主函数"""
    print("🚀 开始初始化tf241_cls_model项目环境...")
    print()
    
    # 打印系统信息
    print_system_info()
    
    # 检查Python版本
    if not check_python_version():
        print("❌ 环境初始化失败: Python版本不符合要求")
        sys.exit(1)
    
    print()
    
    # 创建目录结构
    create_directory_structure()
    print()
    
    # 创建配置文件
    create_default_config()
    print()
    
    # 创建.gitignore
    create_gitignore()
    print()
    
    # 创建示例数据
    create_sample_data()
    print()
    
    # 打印后续步骤
    print_next_steps()


if __name__ == "__main__":
    main() 