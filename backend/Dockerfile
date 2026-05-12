# Use slim python image for a smaller footprint and better security
FROM python:3.10-slim

# Prevent Python from writing .pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr, ensuring logs are printed immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for compiling C-extensions (like NumPy/SciPy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Run the API with uvicorn on all interfaces (0.0.0.0)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
