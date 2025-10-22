FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        git \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Create virtual env and install llmperf
RUN python3.10 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install -e .

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "token_benchmark_ray.py"]



