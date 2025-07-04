# tf241_cls_model

基于TensorFlow 2.4.1的单一分类文本标签预测模型训练项目

## 项目描述
本项目专注于训练一个单一分类的文本标签预测模型，使用TensorFlow 2.4.1框架。项目旨在处理文本分类任务，能够对输入的文本进行标签预测。

## 环境要求
- **Python版本**: 3.7
- **TensorFlow版本**: 2.4.1
- **操作系统**: 支持macOS、Linux、Windows

## 项目结构
```
tf241_cls_model/
├── data/                    # 数据集存储目录
├── pytorch_pipeline/        # 参考代码目录
├── models/                  # 训练好的模型存储目录
├── scripts/                 # 训练和预测脚本
├── config/                  # 配置文件目录
├── logs/                    # 训练日志目录
├── requirements.txt         # 依赖包列表
├── setup.py                 # 环境初始化脚本
├── train.py                 # 训练脚本
├── predict.py               # 预测脚本
└── README.md               # 项目说明文档
```

## 快速开始

### 1. 环境初始化
```bash
# 运行环境初始化脚本
python setup.py
```

### 2. 安装依赖
```bash
# 安装所需依赖包
pip install -r requirements.txt
```

### 3. 数据准备
将训练数据放入 `data/` 目录下，支持的数据格式：
- CSV格式：包含文本列和标签列
- JSON格式：包含text和label字段

### 4. 模型训练
```bash
# 使用默认配置训练模型
python train.py

# 使用自定义配置训练模型
python train.py --config config/custom_config.json
```

### 5. 模型预测
```bash
# 单个文本预测
python predict.py --text "您的文本内容"

# 批量预测
python predict.py --input_file data/test_data.csv --output_file results/predictions.csv
```

## 配置说明
项目使用JSON格式的配置文件，主要配置项包括：
- `model_config`: 模型结构配置
- `training_config`: 训练参数配置
- `data_config`: 数据处理配置

## 参考代码
- `pytorch_pipeline/`: 包含PyTorch实现的参考代码，可用于对比和学习

## 注意事项
- 确保Python版本为3.7
- 建议使用虚拟环境进行开发
- 训练前请确保数据格式正确
- 大型数据集建议使用GPU加速训练

## 贡献
欢迎提交Issue和Pull Request来改进项目。
