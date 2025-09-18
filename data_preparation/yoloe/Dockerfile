FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Install Python with uv, create venv, install dependencies
RUN uv python install 3.12
RUN uv venv --python 3.12
ENV PATH="/app/.venv/bin:$PATH"
RUN uv lock
RUN uv sync --frozen

# Copy the yoloe project
COPY . /app/yoloe

# Change to yoloe dir
WORKDIR /app/yoloe

# Run the main script (assumes data volume mounted at /app/data)
CMD ["python", "tbank_yoloe_bulk_inference.py"]