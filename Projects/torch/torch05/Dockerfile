FROM python:3.12-slim

LABEL authors="skandan-c-y"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pillow \
    python-multipart

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]