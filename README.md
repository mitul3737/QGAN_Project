Run these once you are within the cloned repository.

#!/bin/bash

# Create fresh virtual environment
python -m venv qgan_env
source qgan_env/bin/activate

# Force clean installation of critical packages
pip install --upgrade pip setuptools wheel
pip uninstall -y h5py numpy  # Clean any existing installations

# Install numpy first with specific version
pip install "numpy>=2.0.0"

# Install h5py with specific version and no dependencies
pip install "h5py==3.11.0" --no-build-isolation --no-deps

# Install core packages
pip install torch pennylane
pip install pennylane-qchem --no-deps

# Install openfermion separately
pip install "openfermion>=1.0" --no-deps
pip install "openfermionpsi4>=0.5" --no-deps

# Set environment variables
export PYTHONPATH=$(pwd)

# Install remaining requirements
[ -f "temp_requirements.txt" ] && pip install -r temp_requirements.txt

# Run the training process
python qgan/train.py
