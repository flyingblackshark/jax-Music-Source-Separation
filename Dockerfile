# 使用官方 Python 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制文件
COPY gcs_file_handler.py /app/

# 安装所需的Python依赖
RUN pip install --no-cache-dir google-cloud-storage 
RUN pip install -r requirements.txt

# 设置启动命令
CMD ["python", "infer.py"]