#!/bin/bash

# Check for CUDA support using nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA is available. GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "⚠️ CUDA not available. Training will proceed on CPU, which may be very slow."
    read -p "Do you want to continue training on CPU? (y/n): " proceed
    if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
        echo "❌ Training aborted by user."
        exit 1
    fi
fi

# Start model training
python code/run_hybrid_pipeline.py