FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System dependencies can be added here if needed.

COPY requirements.txt /app/requirements.txt
# Use uv instead of pip for faster, cached installs.
RUN pip install --no-cache-dir uv && uv pip install --system -r /app/requirements.txt

COPY . /app

EXPOSE 8501

# Default command (can be overridden by `docker run ... python ...`)
CMD ["python", "scripts/train.py", "--help"]

