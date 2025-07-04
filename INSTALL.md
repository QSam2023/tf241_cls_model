# tf241_cls_model 环境安装指南

## 环境要求

### 系统要求
- 操作系统：macOS、Linux 或 Windows
- Python 版本：3.7.x
- 内存：建议 8GB 以上
- 硬盘空间：建议 5GB 以上

### GPU 支持（可选）
如果您需要 GPU 加速训练，请确保：
- NVIDIA GPU（计算能力 3.5 及以上）
- CUDA 11.0
- cuDNN 8.0

## 安装步骤

### 1. 创建虚拟环境（强烈推荐）

#### 使用 conda
```bash
# 创建虚拟环境
conda create -n tf241_cls python=3.7
conda activate tf241_cls
```

#### 使用 virtualenv
```bash
# 创建虚拟环境
python -m venv tf241_cls_env
source tf241_cls_env/bin/activate  # Linux/macOS
# tf241_cls_env\Scripts\activate  # Windows
```

### 2. 运行环境初始化脚本
```bash
# 初始化项目环境
python setup.py
```

### 3. 安装依赖包

#### 基础安装（推荐）
```bash
# 安装基础依赖
pip install -r requirements.txt
```

#### 最小化安装（避免依赖冲突）
```bash
# 如果遇到依赖冲突，使用最小化依赖
pip install -r requirements-minimal.txt
```

#### 开发环境安装
```bash
# 安装开发依赖（包含所有基础依赖）
pip install -r requirements-dev.txt
```

## 依赖包说明

### 核心依赖
- **tensorflow==2.4.1**: 深度学习框架
- **numpy==1.19.5**: 数值计算
- **pandas==1.3.5**: 数据处理
- **scikit-learn==0.24.2**: 机器学习工具

### 文本处理
- **nltk==3.6.7**: 英文文本处理
- **jieba==0.42.1**: 中文分词
- **gensim==4.1.2**: 主题建模和词向量
- **transformers==4.5.1**: 预训练模型

### 可视化
- **matplotlib==3.3.4**: 基础绘图
- **seaborn==0.11.2**: 统计图表
- **plotly==5.3.1**: 交互式图表

## 验证安装
```bash
# 验证 TensorFlow 安装
python -c "import tensorflow as tf; print(tf.__version__)"

# 验证 GPU 支持（如果有GPU）
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# 运行测试
python -c "import numpy, pandas, sklearn, nltk, jieba; print('所有包安装成功')"
```

## 常见问题解决

### 1. 依赖冲突问题
如果遇到类似 `typing-extensions` 版本冲突的错误：

```bash
# 方案1：使用最小化依赖
pip install -r requirements-minimal.txt

# 方案2：逐步安装
pip install tensorflow==2.4.1
pip install numpy==1.19.5 pandas==1.3.5
pip install nltk==3.6.7 jieba==0.42.1
pip install matplotlib==3.3.4 tqdm==4.62.3

# 方案3：忽略依赖冲突（不推荐）
pip install -r requirements.txt --force-reinstall --no-deps
```

### 2. TensorFlow 安装失败
```bash
# 清除缓存重新安装
pip cache purge
pip install --upgrade pip
pip install tensorflow==2.4.1
```

### 3. 中文包安装问题
```bash
# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jieba
```

### 4. 内存不足
```bash
# 减少并行进程数
pip install --no-cache-dir -r requirements.txt
```

### 5. 版本兼容性说明
- **TensorFlow 2.4.1** 已包含 GPU 支持，无需单独安装 tensorflow-gpu
- **black** 版本已降低到 20.8b1 以避免 typing-extensions 冲突
- **transformers** 版本已降低到 4.2.2 以确保与 TensorFlow 2.4.1 兼容
- **gensim** 版本已降低到 3.8.3 以避免依赖冲突