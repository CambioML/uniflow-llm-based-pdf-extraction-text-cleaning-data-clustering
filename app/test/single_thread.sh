#!/bin/bash

# Local port
BASE_URL="http://localhost:5000"
# Docker port
# BASE_URL="http://localhost:8080"
# Kubernetes port
BASE_URL="http://192.168.49.2:30080"
DATA="[{\"How are you?\": \"Fine\", \"Who are you?\": \"I am Bob\"}]"

# Start a flow and get job ID
RESPONSE=$(curl -X POST "$BASE_URL/flows/expand_reduce" \
    -H 'Content-Type: application/json' \
    -d "$DATA")

if [ -z "$BASE_URL" ]; then
    echo "BASE_URL is not set. Please set it to the base URL of your service."
    exit 1
fi

echo "Response: $RESPONSE"

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')

# Get job status
RESPONSE=$(curl "$BASE_URL/flows/status/$JOB_ID")

echo "Response: $RESPONSE"

STATUS=$(echo "$RESPONSE" | jq -r '.status')

# Print the status
echo "Job status: $STATUS"

# Get results (page 1 with 15 results)
RESPONSE=$(curl "$BASE_URL/flows/results?page=1&limit=15")

echo "Response: $RESPONSE"

RESULTS=$(echo "$RESPONSE" | jq -r '.results')
