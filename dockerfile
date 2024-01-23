# Base image
FROM python:3.10

# Copy application code and dependencies
COPY . .
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

WORKDIR /app

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]
