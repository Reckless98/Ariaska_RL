# ARIASKA_RL Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    sqlite3 \
    libsqlite3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py | python3

# Create non-root user
RUN useradd -ms /bin/bash ariaska
WORKDIR /home/ariaska/ARIASKA_RL

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install shell-gpt for LLM interaction
RUN pip install --no-cache-dir shell-gpt

# Copy ARIASKA_RL code
COPY . .
RUN chown -R ariaska:ariaska /home/ariaska/ARIASKA_RL

# Switch to non-root user
USER ariaska

# Create necessary directories
RUN mkdir -p logs data models core/memories/shared

# Create environment context detection config
RUN mkdir -p config && \
    echo '{"default_mode": "simulated", "safety_level": "strict"}' > config/environment.json

# Fix import paths
RUN bash fix_imports.sh

# Set entry point and default command
ENTRYPOINT ["python", "-m"]
CMD ["main", "--verbosity", "standard"]

# Expose ports for Streamlit UI (if enabled)
EXPOSE 8501

# Add image metadata and labels
LABEL maintainer="Filip Volf"
LABEL version="2.1-apex"
LABEL description="ARIASKA_RL - GPT-augmented Multi-Agent RL Cybersecurity Platform"
