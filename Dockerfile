# 使用官方 Python 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制文件
COPY . /app/

# 安装所需的Python依赖
RUN pip install -r requirements.gke.txt

# 设置启动命令
CMD ["python", "worker.py"]