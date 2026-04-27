FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY *.py ./
COPY README.md ./
COPY LICENSE ./

# Install the package and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[dev,docs,scripts]

# Install additional useful packages for genomic analysis
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipykernel \
    biopython \
    pysam \
    plotly \
    dash

# Create a non-root user
RUN useradd -m -s /bin/bash genomics && \
    chown -R genomics:genomics /app

USER genomics

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["bash"]