#!/bin/bash

# Create virtual env if not exists
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate it
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment activated and requirements installed."
