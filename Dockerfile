FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System dependencies can be added here if needed.

COPY requirements.lock /app/requirements.lock
# Use uv instead of pip for faster, cached installs.
RUN pip install --no-cache-dir uv && uv pip install --system -r /app/requirements.lock

COPY . /app

EXPOSE 8000 8501

# Default command (can be overridden by `docker run ... python ...`)
CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

