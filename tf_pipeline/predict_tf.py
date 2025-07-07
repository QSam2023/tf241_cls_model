# coding: utf8

import os
import json
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf
from transformers import TFAlbertForSequenceClassification, BertTokenizer


class BondClassificationPredictor:
    """债券分类预测器"""
    
    def __init__(self, model_path, max_length=100):
        self.model_path = model_path
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        print(f"Loading model from {self.model_path}")
        self.model = TFAlbertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        print("Model loaded successfully!")
    
    def preprocess_text(self, text):
        """预处理文本"""
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        return encoded
    
    def predict(self, text):
        """预测单个文本"""
        inputs = self.preprocess_text(text)
        
        outputs = self.model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        
        predicted_class = tf.argmax(predictions, axis=-1).numpy()[0]
        confidence = tf.reduce_max(predictions, axis=-1).numpy()[0]
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'probabilities': predictions.numpy()[0].tolist()
        }
    
    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def predict_file(self, input_file, output_file=None):
        """预测文件中的数据"""
        results = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = data['text']
                    
                    prediction = self.predict(text)
                    
                    result = {
                        'id': data.get('id', line_num),
                        'text': text,
                        'true_label': data.get('label', -1),
                        'predicted_label': prediction['predicted_class'],
                        'confidence': prediction['confidence']
                    }
                    results.append(result)
                    
                    if line_num % 100 == 0:
                        print(f"Processed {line_num} samples...")
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Results saved to {output_file}")
        
        return results


def main():
    """主预测函数"""
    parser = ArgumentParser(description="TensorFlow Bond Classification Prediction")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the trained model")
    parser.add_argument("--input_file", type=str, 
                       help="Input file with texts to predict")
    parser.add_argument("--output_file", type=str, 
                       help="Output file for predictions")
    parser.add_argument("--text", type=str, 
                       help="Single text to predict")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # 初始化预测器
    predictor = BondClassificationPredictor(args.model_path, args.max_length)
    
    if args.text:
        # 预测单个文本
        result = predictor.predict(args.text)
        print(f"Input text: {args.text}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
    elif args.input_file:
        # 预测文件
        if not os.path.exists(args.input_file):
            print(f"Input file not found: {args.input_file}")
            return
        
        results = predictor.predict_file(args.input_file, args.output_file)
        
        # 计算准确率（如果有真实标签）
        if results and results[0]['true_label'] != -1:
            correct = sum(1 for r in results if r['true_label'] == r['predicted_label'])
            accuracy = correct / len(results)
            print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
        
        print(f"Total predictions: {len(results)}")
    
    else:
        print("Please provide either --text or --input_file")


if __name__ == "__main__":
    main()
