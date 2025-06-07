#!/bin/bash
# This script generates symbolic links for data, logs, and outputs
# It ensures the directories exist and links them to the specified locations in the .env file
# It assumes the script is run from the root project directory

set -euo pipefail

# Ensure we are in the root project folder
cd "$(dirname "$0")/.."

# Source the .env file to get the environment variables
source .env

# Function to create the directory if it doesn't exist and link it
create_and_link_dir() {
    local dir_name="$1"
    local target_dir="$2"

    # If the target directory doesn't exist, create it
    if [ ! -d "$target_dir" ]; then
        echo "Target directory $target_dir does not exist. Creating it."
        mkdir -p "$target_dir"
    fi

    # If the directory already exists as a symbolic link, unlink it
    if [ -L "$dir_name" ]; then
        echo "Found existing symbolic link $dir_name. It will be replaced."
        unlink "$dir_name"
    elif [ -d "$dir_name" ]; then
        # If a non-empty directory exists, remove it and its contents
        echo "Removing existing directory $dir_name."
        rm -rf "$dir_name"
    fi

    # Create the symbolic link
    ln -s "$target_dir" "$dir_name"
}

# Link the data, logs, and outputs directories based on the .env variables
create_and_link_dir "$PROJECT_ROOT/data" "$DATA_DIR"
create_and_link_dir "$PROJECT_ROOT/logs" "$LOG_DIR"
create_and_link_dir "$PROJECT_ROOT/outputs" "$OUTPUT_DIR"

echo "Symbolic links for data, logs, and outputs created successfully."