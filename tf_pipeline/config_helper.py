# coding: utf8

import json
import os

class ConfigParse:
    """配置文件解析器"""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    @property
    def model_path(self):
        return self.config.get('model_path', './albert_chinese_tiny')
    
    @property
    def data_dir(self):
        return self.config.get('data_dir', 'data/bond')
    
    @property
    def save_dir(self):
        return self.config.get('save_dir', 'output/bond_cls_tf')
    
    @property
    def num_labels(self):
        return self.config.get('num_labels', 20)
    
    @property
    def max_length(self):
        return self.config.get('max_length', 100)
    
    @property
    def train_epochs(self):
        return self.config.get('train_epochs', 10)
    
    @property
    def train_batch_size(self):
        return self.config.get('train_batch_size', 256)
    
    @property
    def eval_batch_size(self):
        return self.config.get('eval_batch_size', 128)
    
    @property
    def learning_rate(self):
        return self.config.get('learning_rate', 5e-4)
    
    @property
    def gpus(self):
        return self.config.get('gpus', '0')
    
    @property
    def gpu_list(self):
        return self.gpus
