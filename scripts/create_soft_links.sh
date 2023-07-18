#!/bin/bash
# Generates symbolic links for data, logs and outputs
# Useful to link these folders to some remote location
set -euo pipefail 

cd "$(dirname "$0")/.."
source .env # read environment variables

if [ -d "data" ]; then
    if [ -L "data" ]; then
        echo "Found existing data link. It will be replaced with a new link"
        unlink "data"
    elif [ -f "data/.gitkeep" ]; then
        rm data/.gitkeep
        rmdir data  # raises an error if the directory is not empty
    fi
fi
ln -s $DATA_DIR data

if [ -d "logs" ]; then
    if [ -L "logs" ]; then
        echo "Found existing logs link. It will be replaced with a new link"
        unlink "logs"
    elif [ -f "logs/.gitkeep" ]; then
        rm logs/.gitkeep
        rmdir logs  # raises an error if the directory is not empty
    fi
fi
ln -s $LOG_DIR logs

if [ -d "outputs" ]; then
    if [ -L "outputs" ]; then
        echo "Found existing outputs link. It will be replaced with a new link"
        unlink "outputs"
    else
        rmdir outputs # raises an error if the directory is not empty
    fi
fi
ln -s $OUTPUT_DIR outputs
