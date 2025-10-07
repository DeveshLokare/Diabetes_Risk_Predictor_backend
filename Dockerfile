# ===== Stage 1: Build dependencies on a matching OS =====
FROM cgr.dev/chainguard/python:latest-dev AS builder

WORKDIR /app

# Switch to the root user to install packages
USER root

# Install the build-base package needed to compile C extensions
RUN apk update && apk add build-base

# Switch back to the secure, default non-root user
USER nonroot

COPY requirements.txt .

# Install packages to a target directory
RUN pip install --no-cache-dir -r requirements.txt --target /app/packages


# ===== Stage 2: Secure runtime =====
FROM cgr.dev/chainguard/python:latest

WORKDIR /app

# Copy installed dependencies
COPY --from=builder /app/packages /packages

# Copy your app and model files
COPY app.py .
COPY xgb_diabetes_model.pkl .
COPY background_sample.pkl .

# Tell Python where to find the installed packages
ENV PYTHONPATH="/packages"
ENV PORT=5000
EXPOSE 5000

# Use the full path to the gunicorn executable
CMD ["/packages/bin/gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2"]