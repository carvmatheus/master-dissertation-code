FROM python:3.11-slim

# Basic runtime env
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers

WORKDIR /app/master-dissertation-code

# System deps (minimal; add build-essential if you later need to compile something)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . ./

# Default command (interactive; override with compose if desired)
CMD ["python", "01-context-extension-comparison/chat.py"]
