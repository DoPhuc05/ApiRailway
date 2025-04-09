FROM python:3.10-slim

# Cài thư viện hệ thống cần cho OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Tạo và kích hoạt virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy mã nguồn
COPY . /app
WORKDIR /app

# Cài thư viện Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Mở cổng
EXPOSE 5050

