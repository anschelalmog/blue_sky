#!/bin/bash

source venv/bin/activate


cd tests

# Run pytest
pytest
 
deactivate