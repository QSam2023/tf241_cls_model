# coding: utf8

import os
import json
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFAlbertForSequenceClassification, BertTokenizer

from config_helper import ConfigParse


class BondClassificationDataset:
    """债券分类数据集处理类"""
    
    def __init__(self, data_path, tokenizer, max_length=100, num_labels=20):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        self.texts = []
        self.labels = []
        self.load_data()
    
    def load_data(self):
        """加载JSONL格式数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    self.texts.append(data['text'])
                    self.labels.append(data['label'])
                except Exception as e:
                    print(f"Error parsing line: {line[:50]}..., Error: {e}")
                    continue
    
    def tokenize_and_encode(self):
        """对文本进行分词和编码"""
        input_ids = []
        attention_masks = []
        
        for text in self.texts:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='tf'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        
        return {
            'input_ids': tf.concat(input_ids, axis=0),
            'attention_mask': tf.concat(attention_masks, axis=0),
            'labels': tf.constant(self.labels, dtype=tf.int32)
        }


def create_tf_dataset(data_dict, batch_size, shuffle=True):
    """创建TensorFlow数据集"""
    dataset = tf.data.Dataset.from_tensor_slices({
        'input_ids': data_dict['input_ids'],
        'attention_mask': data_dict['attention_mask'],
        'labels': data_dict['labels']
    })
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    predictions = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true, predictions, average="micro")
    acc = accuracy_score(y_true, predictions)
    return {
        "accuracy": acc,
        "f1": f1
    }


class MetricsCallback(tf.keras.callbacks.Callback):
    """自定义指标回调"""
    
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时计算指标"""
        if self.validation_data:
            val_predictions = self.model.predict(self.validation_data)
            val_labels = []
            
            for batch in self.validation_data:
                val_labels.extend(batch['labels'].numpy())
            
            val_labels = np.array(val_labels)
            metrics = compute_metrics(val_labels, val_predictions)
            
            print(f"\nEpoch {epoch + 1} - Val Accuracy: {metrics['accuracy']:.4f}, Val F1: {metrics['f1']:.4f}")
            logs = logs or {}
            logs['val_accuracy'] = metrics['accuracy']
            logs['val_f1'] = metrics['f1']


def main():
    """主训练函数"""
    parser = ArgumentParser(description="TensorFlow 2.4.1 Bond Classification Training Script")
    parser.add_argument("--config_file", type=str, default="bond_cls_tf.config", 
                       help="config file for training")
    
    args = parser.parse_args()
    
    # 加载配置
    config = ConfigParse(args.config_file)
    
    model_path = config.model_path
    input_data_dir = config.data_dir
    save_dir = config.save_dir
    num_labels = config.num_labels
    max_length = config.max_length
    train_epochs = config.train_epochs
    train_batch_size = config.train_batch_size
    eval_batch_size = config.eval_batch_size
    learning_rate = config.learning_rate
    gpu_list = config.gpus
    
    # 设置GPU
    if gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 数据路径
    train_data_path = os.path.join(input_data_dir, "train.jsonl")
    test_data_path = os.path.join(input_data_dir, "test.jsonl")
    val_data_path = os.path.join(input_data_dir, "validation.jsonl")
    
    # 加载分词器
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = BondClassificationDataset(train_data_path, tokenizer, max_length, num_labels)
    test_dataset = BondClassificationDataset(test_data_path, tokenizer, max_length, num_labels)
    val_dataset = BondClassificationDataset(val_data_path, tokenizer, max_length, num_labels)
    
    print(f"Train samples: {len(train_dataset.texts)}")
    print(f"Test samples: {len(test_dataset.texts)}")
    print(f"Validation samples: {len(val_dataset.texts)}")
    
    # 编码数据
    print("Tokenizing and encoding data...")
    train_data = train_dataset.tokenize_and_encode()
    test_data = test_dataset.tokenize_and_encode()
    val_data = val_dataset.tokenize_and_encode()
    
    # 创建TensorFlow数据集
    print("Creating TensorFlow datasets...")
    train_tf_dataset = create_tf_dataset(train_data, train_batch_size, shuffle=True)
    test_tf_dataset = create_tf_dataset(test_data, eval_batch_size, shuffle=False)
    val_tf_dataset = create_tf_dataset(val_data, eval_batch_size, shuffle=False)
    
    # 加载模型
    print("Loading model...")
    model = TFAlbertForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    
    # 编译模型
    print("Compiling model...")
    optimizer = Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    # 设置回调
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        MetricsCallback(validation_data=test_tf_dataset)
    ]
    
    # 训练模型
    print("Starting training...")
    history = model.fit(
        train_tf_dataset,
        validation_data=test_tf_dataset,
        epochs=train_epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # 在验证集上进行最终评估
    print("Final evaluation on validation set...")
    val_predictions = model.predict(val_tf_dataset)
    val_labels = []
    
    for batch in val_tf_dataset:
        val_labels.extend(batch['labels'].numpy())
    
    val_labels = np.array(val_labels)
    final_metrics = compute_metrics(val_labels, val_predictions)
    
    print(f"\nFinal Validation Results:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    
    # 保存模型
    print("Saving model...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 保存训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history.history, f, indent=2)
    
    print(f"Training completed! Model saved to {save_dir}")


if __name__ == "__main__":
    main()
