#!/bin/bash

pip install -U uv

# Create a virtual environment in .venv if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install PyTorch nightly with CUDA 12.8
uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
uv pip install pandas numpy requests pyarrow matplotlib tqdm