
#!/bin/bash

# Install required dependencies
pip install -r requirements.txt

# Set environment variables if needed
export PYTHONPATH=$(pwd)

# Run the training process
python QGAN_Project/qgan/train.py
