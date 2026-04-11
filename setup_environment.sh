#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Ensure environment.yml exists
if [ ! -f environment.yml ]; then
  echo "environment.yml not found. Please create it and try again."
  exit 1
fi

# Extract environment name from the environment.yml
env_name=$(grep "name:" environment.yml | cut -d' ' -f2)

# Create or update the conda environment
if conda env list | grep -q "$env_name"; then
  echo "Updating existing environment: $env_name"
  conda env update -f environment.yml --prune
else
  echo "Creating new environment: $env_name"
  conda env create -f environment.yml
fi

# Activate the environment
conda activate "$env_name"

# Install ipykernel for Jupyter support
pip install ipykernel

# Add the environment to Jupyter as a kernel
python -m ipykernel install --user --name="$env_name" --display-name "Python ($env_name)"

echo "Environment setup complete. Restart VSCode to see the new kernel."
