# syntax=docker/dockerfile:1
FROM python:3.10-bookworm

# 作業ディレクトリの設定
WORKDIR /app

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# 日本ミラーに変更してシステムパッケージのインストール
RUN echo "deb http://ftp.jp.debian.org/debian bookworm main" > /etc/apt/sources.list && \
    echo "deb http://ftp.jp.debian.org/debian bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb http://security.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    curl \
    netcat-traditional \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Poetryのインストール
# RUN curl -sSL https://install.python-poetry.org | python3 - \
#     && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# 依存関係ファイルのコピー
# COPY pyproject.toml poetry.lock* ./
COPY requirements.txt ./

# 依存関係のインストール（BuildKitキャッシュマウント使用）
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -v -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# ポートの公開
EXPOSE 8005

# アプリケーションの起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005", "--reload"]
