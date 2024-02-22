#!/bin/bash

# Find all notebook files in the current directory and subdirectories
find . -type f -name "*.ipynb" | while read notebook; do
    # Check if the file contains the string "langchain"
    if grep -q "langchain" "$notebook"; then
        # echo "Found 'langchain' in $notebook"
        echo "$notebook"
    fi
done
