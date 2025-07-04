#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装测试脚本
验证 tf241_cls_model 项目的环境安装是否成功
"""

import sys
import importlib
import warnings

# 忽略一些不重要的警告
warnings.filterwarnings('ignore')

def test_package_import(package_name, display_name=None):
    """测试包的导入"""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name} {version}")
        return True
    except ImportError as e:
        print(f"❌ {display_name} 导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {display_name} 导入时出现警告: {e}")
        return True

def test_tensorflow():
    """测试 TensorFlow 特定功能"""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        # 测试基本计算
        x = tf.constant([1, 2, 3, 4])
        y = tf.constant([2, 3, 4, 5])
        result = tf.add(x, y)
        print(f"✅ TensorFlow 基本计算测试通过: {result.numpy()}")
        
        # 检查 GPU 支持
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ 检测到 {len(gpus)} 个 GPU 设备")
        else:
            print("ℹ️  未检测到 GPU 设备（CPU 模式）")
        
        return True
    except ImportError as e:
        print(f"❌ TensorFlow 导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️  TensorFlow 测试时出现问题: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试 tf241_cls_model 环境安装...")
    print("=" * 50)
    
    # 测试 Python 版本
    print(f"Python 版本: {sys.version}")
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor == 7:
        print("✅ Python 版本符合要求")
    else:
        print("⚠️  Python 版本建议使用 3.7.x")
    
    print("\n" + "=" * 50)
    print("测试核心依赖包...")
    
    # 要测试的包列表
    core_packages = [
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'TQDM'),
        ('yaml', 'PyYAML'),
        ('jieba', 'Jieba'),
        ('nltk', 'NLTK'),
    ]
    
    success_count = 0
    total_count = len(core_packages)
    
    for package, display_name in core_packages:
        if test_package_import(package, display_name):
            success_count += 1
    
    print("\n" + "=" * 50)
    print("测试 TensorFlow 功能...")
    
    if test_tensorflow():
        success_count += 1
    total_count += 1
    
    print("\n" + "=" * 50)
    print(f"测试完成！")
    print(f"成功: {success_count}/{total_count} 项测试通过")
    
    if success_count == total_count:
        print("🎉 所有测试通过！环境安装成功！")
        return 0
    elif success_count >= total_count * 0.8:
        print("⚠️  大部分测试通过，环境基本可用")
        return 0
    else:
        print("❌ 多个测试失败，请检查依赖安装")
        return 1

if __name__ == "__main__":
    sys.exit(main())