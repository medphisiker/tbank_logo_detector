FROM python:3.10-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# Install dependencies with uv (creates .venv)
RUN uv lock && uv sync --frozen

# Activate virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the project
COPY . /app/synthesis

# Change to synthesis dir
WORKDIR /app/synthesis

# Run the script (assumes volume mount for crops, backgrounds, and output)
CMD ["python", "gen_synth.py"]