#!/bin/bash

BASE_URL="http://localhost:5000"
DATA_OPTIONS=(
    "[{}]"
    "[{\"How are you?\": \"Fine\", \"Who are you?\": \"I am Bob\"}]"
    "[{\"0?\": \"0!\", \"1?\": \"1!\"}, {\"2?\": \"2!\", \"3?\": \"3!\"}]"
)
THREADS=10

for i in $(seq 1 $THREADS)
do
    {
        # Select a random DATA option
        DATA=${DATA_OPTIONS[$RANDOM % ${#DATA_OPTIONS[@]}]}
        echo "Thread $i Data: $DATA"
        # Start a flow and get job ID
        RESPONSE=$(curl -X POST "$BASE_URL/flows/expand_reduce" \
            -H 'Content-Type: application/json' \
            -d "$DATA")

        JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
        echo "echo/$JOB_ID"
        # Get job status
        STATUS_RESPONSE=$(curl "$BASE_URL/flows/status/$JOB_ID")
        echo "Thread $i Status Response: $STATUS_RESPONSE"
    } &
done

# Wait for all background jobs to finish
wait

# Get job results
RESULTS_RESPONSE=$(curl "$BASE_URL/flows/results?page=1&limit=15")
echo "Results Response: $RESULTS_RESPONSE"