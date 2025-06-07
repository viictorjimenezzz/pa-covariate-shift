#!/bin/bash

# Enable color output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'  # No Color

REQUIREMENTS_FILE="requirements.txt"

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: $REQUIREMENTS_FILE not found in the current directory.${NC}"
    exit 1
fi

# Read requirements file line-by-line, ignoring comments and empty lines
echo -e "\nReading requirements from ${REQUIREMENTS_FILE}...\n"
failed_packages=()

while IFS= read -r package || [ -n "$package" ]; do
    # Skip empty lines and comments
    if [[ -z "$package" || "$package" == \#* ]]; then
        continue
    fi

    echo -e "Installing package: ${package}..."

    # Attempt to install the package
    python -m pip install "$package"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed: ${package}${NC}"
    else
        echo -e "${RED}Failed to install: ${package}${NC}"
        failed_packages+=("$package")
    fi
    echo
done < "$REQUIREMENTS_FILE"

# Display a summary of failed packages
if [ ${#failed_packages[@]} -ne 0 ]; then
    echo -e "\n${RED}The following packages failed to install:${NC}"
    for pkg in "${failed_packages[@]}"; do
        echo -e "${RED}- $pkg${NC}"
    done
    exit 1
else
    echo -e "\n${GREEN}All packages installed successfully!${NC}"
    exit 0
fi
