#!/bin/bash

BASE_URL="http://localhost:5001"
DATA_OPTIONS=(
    '{}'
    # '{\"0?\": \"0!\", \"1?\": \"1!\"}'
    # '{\"0?\": \"0!\", \"1?\": \"1!\", \"0?\": \"0!\", \"1?\": \"1!\"}'
)
THREADS=2

for i in $(seq 1 $THREADS)
do
    {
        # Select a random DATA option
        DATA=${DATA_OPTIONS[$RANDOM % ${#DATA_OPTIONS[@]}]}
        # Start a flow and get job ID
        JOB_ID=$(curl -X POST "$BASE_URL/flows/expand_reduce" \
            -H 'Content-Type: application/json' \
            -d "$DATA")

        echo "echo/$JOB_ID"
        # Get job status
        STATUS_RESPONSE=$(curl "$BASE_URL/flows/status/$JOB_ID")
        # echo "Thread $i Status Response: $STATUS_RESPONSE"

        # # Get job results
        # RESULTS_RESPONSE=$(curl "$BASE_URL/flows/results?page=2&limit=15")
        # echo "Thread $i Results Response: $RESULTS_RESPONSE"
    } &
done

# Wait for all background jobs to finish
wait