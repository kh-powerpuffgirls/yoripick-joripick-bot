FROM python:3.11-slim

WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libaio-dev wget unzip && \
    pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

EXPOSE 8000

# FastAPI 실행
CMD ["uvicorn", "faqService:app", "--host", "0.0.0.0", "--port", "8000"]
