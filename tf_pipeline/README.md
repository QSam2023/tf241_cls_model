# TensorFlow 2.4.1 债券分类训练脚本

基于 TensorFlow 2.4.1 的债券分类训练脚本，从 PyTorch 版本迁移而来。

## 快速开始

### 1. 安装依赖
```bash
pip install tensorflow==2.4.1
pip install transformers==4.21.3
pip install scikit-learn numpy pandas
```

### 2. 准备数据
确保数据在 `data/bond/` 目录下，包含：
- `train.jsonl` - 训练数据
- `test.jsonl` - 测试数据  
- `validation.jsonl` - 验证数据

数据格式（JSONL）：
```json
{"id": 1, "text": "债券交易文本", "label": 0}
```

### 3. 训练模型
```bash
cd tf_pipeline
python train_model_tf.py
```

### 4. 预测
```bash
# 预测单个文本
python predict_tf.py --model_path output/bond_cls_tf --text "你的文本"

# 预测文件
python predict_tf.py --model_path output/bond_cls_tf --input_file data/bond/test.jsonl --output_file predictions.jsonl
```

## 配置文件

编辑 `bond_cls_tf.config` 修改训练参数：
```json
{
    "model_path": "./albert_chinese_tiny",
    "data_dir": "data/bond",
    "save_dir": "output/bond_cls_tf",
    "train_epochs": 10,
    "train_batch_size": 256,
    "eval_batch_size": 128,
    "learning_rate": 5e-4,
    "num_labels": 20,
    "max_length": 100,
    "gpus": "0"
}
```

## 主要特性

1. **完整的训练流程**：数据加载、预处理、训练、评估
2. **灵活的配置**：JSON配置文件，支持GPU训练
3. **评估指标**：准确率、F1分数
4. **模型保存**：完整模型和分词器保存
5. **预测功能**：单文本和批量预测

## 文件结构

```
tf_pipeline/
├── train_model_tf.py       # 主训练脚本
├── predict_tf.py           # 预测脚本
├── config_helper.py        # 配置解析器
├── bond_cls_tf.config      # 配置文件
└── README.md              # 说明文档
```

## 与 PyTorch 版本对比

- **数据处理**：自定义Dataset类替代Datasets库
- **训练循环**：tf.keras.Model.fit 替代 Trainer
- **评估**：自定义回调替代compute_metrics
- **模型保存**：save_pretrained + h5格式

## 注意事项

1. 确保预训练模型路径正确
2. 根据GPU内存调整batch_size
3. 数据必须是JSONL格式
4. 支持20个分类标签（0-19）

## 故障排除

- **内存不足**：减少batch_size或max_length
- **模型加载失败**：检查model_path
- **数据格式错误**：确保JSONL格式正确
