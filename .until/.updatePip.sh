#!/bin/bash

# Path to the 'example' folder (change this to the full or relative path of your 'example' folder)
example_folder="../example"

# Function to update pip install commands in a .ipynb file
update_pip_install() {
    local file=$1
    local tempfile="temp_$(basename "$file")"

    # Check if the file contains 'pip install' command and update it
    if grep -q "!{sys.executable} -m pip install" "$file"; then
        # Attempt to add the -q flag to pip install commands
        cat "$file" | sed 's/!{sys.executable} -m pip install/!{sys.executable} -m pip install -q/g' > "$tempfile"
        # Verify if the update is successful
        if grep -q "!{sys.executable} -m pip install -q" "$tempfile"; then
            echo "'pip install' command updated in $file"
            mv "$tempfile" "$file"
        else
            echo "Failed to update 'pip install' command in $file"
            rm "$tempfile"
        fi
    else
        echo "No 'pip install' command found in $file"
    fi
}

# Iterate over the directories and find all .ipynb files
find "$example_folder" -name '*.ipynb' | while read -r notebook; do
    update_pip_install "$notebook"
done

echo "All notebooks in the 'example' folder have been processed."
