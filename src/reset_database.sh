#!/bin/bash

# This script will run the code for removing the training images and annotation database

echo "--- Deleting annotation database and removing training images ---"
python code/clean_data.py

# Check for errors
if [[ $? -ne 0 ]]; then
  echo "Error running Python script."
  exit 1
fi

echo "Data cleaning operation completed successfully."