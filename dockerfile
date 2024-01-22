# Base image (replace with the appropriate image for your environment)
FROM python:3.9

# Copy application code and dependencies
COPY . .
RUN pip install -r requirements.txt  # Install dependencies

# Working directory
WORKDIR /app

# Expose port (if applicable)
EXPOSE 5000

ENTRYPOINT ["python", "app.py"]
