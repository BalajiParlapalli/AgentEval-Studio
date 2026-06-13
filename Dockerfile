FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK data (for tokenization)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy app
COPY . .

# Create persistent dirs
RUN mkdir -p results datasets

# Expose ports
EXPOSE 7860 8000

# Startup script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
