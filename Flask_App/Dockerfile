# Sử dụng hình ảnh Python chính thức làm base
FROM python:3.9-slim

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt các dependencies hệ thống
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sao chép các tệp requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
COPY . .

# Tạo thư mục templates và model nếu chưa tồn tại
RUN mkdir -p templates /app/models

# Tạo script để tải model một cách an toàn
RUN echo '#!/bin/bash\n\
if [ ! -d "/app/models/minhtoan_t5_translate_lao_vietnamese" ]; then\n\
    python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; \n\
model_name = \"minhtoan/t5-translate-lao-vietnamese\"; \n\
T5Tokenizer.from_pretrained(model_name, cache_dir=\"/app/models\"); \n\
T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=\"/app/models\")"\n\
fi\n\
exec "$@"' > /download_model.sh

# Cấp quyền thực thi cho script
RUN chmod +x /download_model.sh

# Expose cổng ứng dụng
EXPOSE 5000

# Cài đặt biến môi trường
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/models

# Sử dụng entrypoint để tải model trước khi chạy ứng dụng
ENTRYPOINT ["/download_model.sh"]

# Chạy ứng dụng với gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "app:app"] 