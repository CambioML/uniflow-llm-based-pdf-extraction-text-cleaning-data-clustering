
#!/bin/bash

# Set the variables
DOCKERFILE_PATH="dockerfile"
IMAGE_NAME="uniflow-expand-reduce"
REGISTRY_URL="sherry0506"

TAG="1.2"

# Build the Docker image
docker build --no-cache -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .

# Tag the Docker image
docker tag "$IMAGE_NAME" "$REGISTRY_URL/$IMAGE_NAME:$TAG"
docker tag "$IMAGE_NAME" "$REGISTRY_URL/$IMAGE_NAME:latest"

# Push the Docker image to the registry
docker push "$REGISTRY_URL/$IMAGE_NAME:$TAG"
docker push "$REGISTRY_URL/$IMAGE_NAME:latest"
