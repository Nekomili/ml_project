#!/bin/bash

# Run the scripts

# Import Label Studio annotations to database
echo "--- Importing annotations to database ---"
python code/import_annotations.py

# Transform data from bronze to silver and to gold in database
echo "--- Transforming data in database ---"
python code/from_silver_to_gold.py

# Create training files for YOLO
echo "--- Creating training images for YOLO training ---"
python code/create_training_files.py

# Check for errors
if [[ $? -ne 0 ]]; then
  echo "Error running Python scripts."
  exit 1
fi

echo "Database and training files created successfully."