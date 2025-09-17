FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy project file
COPY pyproject.toml .

# Install dependencies with uv (creates .venv)
RUN uv sync

# Copy the script
COPY gen_synth.py .

# Run the script (assumes volume mount for crops, backgrounds, and output)
CMD [".venv/bin/python", "gen_synth.py"]