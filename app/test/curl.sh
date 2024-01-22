#!/bin/bash

BASE_URL="http://localhost:5000"
DATA="{\"How are you?\": \"Fine\", \"Who are you?\": \"I am Bob\"}"

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

# # Wait for the job to finish (replace sleep with actual logic)
# sleep 5

# # Get job status
# RESPONSE=$(curl "$BASE_URL/flows/status/$JOB_ID")

# echo "Response: $RESPONSE"

# STATUS=$(echo "$RESPONSE" | jq -r '.status')

# # Print the status
# echo "Job status: $STATUS"

# # Get results (page 2 with 15 results)
# RESPONSE=$(curl "$BASE_URL/flows/results?page=2&limit=15")

# echo "Response: $RESPONSE"

# RESULTS=$(echo "$RESPONSE" | jq -r '.results')
