# Use Python 
FROM python:3.12

# Set working directory
WORKDIR /app

# Install curl for healthcheck (slim images don't include it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security (HF Spaces best practice)
RUN useradd -m -u 1000 user

# Set user environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user's app directory
WORKDIR $HOME/app

# Copy application files with correct ownership
COPY --chown=user . $HOME/app

# Switch to non-root user
USER user

# Expose Hugging Face Spaces standard port
EXPOSE 7860

# Health check for container monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit on port 7860 with production settings
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
