#!/bin/bash

# Activate your Python virtual environment
source venv/bin/activate

# Navigate to the tests directory
cd tests

# Run pytest
pytest

# Optionally, you can deactivate the virtual environment
deactivate