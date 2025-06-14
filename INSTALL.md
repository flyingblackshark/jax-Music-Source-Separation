# 安装指南

## 从源码安装

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/jax-Music-Source-Separation.git
cd jax-Music-Source-Separation
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -e .
```

### 4. 开发模式安装（包含开发工具）
```bash
pip install -e ".[dev]"
```

### 5. 训练模式安装（包含训练工具）
```bash
pip install -e ".[train]"
```

## 使用 pip 安装（发布后）
```bash
pip install jaxmsst
```

## 验证安装
```bash
# 检查模块是否可用
python -m src.jaxmsst.infer --help
python -m src.jaxmsst.train --help
python -m src.jaxmsst.webui --help
```

## 使用示例

### 推理
```bash
python -m src.jaxmsst.infer --config_path configs/bs_roformer_logic.yaml --input_folder ./input --store_dir ./output
```

### 启动Web界面
```bash
python -m src.jaxmsst.webui --config_path configs/webui/model_options.yaml
```

### 训练
```bash
python -m src.jaxmsst.train --config configs/bs_roformer_base.yaml --hardware gpu
```

## 系统要求

- Python 3.8+
- JAX 和 JAXlib
- CUDA（GPU 支持）或 TPU（可选）
- 足够的内存和存储空间

## 故障排除

如果遇到安装问题，请检查：
1. Python 版本是否符合要求
2. 是否正确安装了 JAX 和相关依赖
3. GPU 驱动和 CUDA 版本是否兼容