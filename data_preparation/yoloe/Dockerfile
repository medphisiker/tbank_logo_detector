FROM ultralytics/ultralytics:latest

# Install SAHI for tiled inference using conda
RUN conda install -c conda-forge sahi -y

# Set working directory
WORKDIR /app

# Run the main script
CMD ["sh"]