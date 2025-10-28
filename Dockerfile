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
RUN python3.10 -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel
RUN /opt/venv/bin/pip install -e .

# --- Default environment variables ---
# Default OpenAI-compatible endpoint (can be overridden with -e)
ENV OPENAI_API_BASE=http://localhost:8000/v1

# The API key should be provided at runtime for security
# Providing stub default value to avoid errors during installation
ENV OPENAI_API_KEY="stub_key"

# Ensure venv and environment behave like a clean Python install
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "token_benchmark_ray.py", "--log-level", "DEBUG"]



