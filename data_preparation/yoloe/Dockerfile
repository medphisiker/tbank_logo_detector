FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install uv binary
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Install Python with uv, create venv, install dependencies
RUN uv python install 3.13
RUN uv venv --python 3.13
ENV PATH="/app/.venv/bin:$PATH"
RUN uv sync --frozen

# Copy the yoloe project
COPY . /app/yoloe

# Change to yoloe dir
WORKDIR /app/yoloe

# Run the main script (assumes data volume mounted at /app/data)
CMD ["python", "yoloe_package/main.py"]