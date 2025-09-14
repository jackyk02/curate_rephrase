#!/bin/bash

# Array of GPU output files to process
gpu_files=(
    "instruction_output_ids_gpu0_20250914_054300.json"
    "instruction_output_ids_gpu1_20250912_180744.json"
    "instruction_output_ids_gpu2_20250912_172652.json"
    "instruction_output_ids_gpu3_20250912_035021.json"
    "instruction_output_ids_gpu4_20250912_141854.json"
    "instruction_output_ids_gpu5_20250912_160613.json"
    "instruction_output_ids_gpu6_20250912_133855.json"
)

# Process each file sequentially
for file in "${gpu_files[@]}"; do
    echo "Processing: $file"
    echo "----------------------------------------"
    
    # Run the python command with the current file
    python curate_hard_negatives.py --gpu-output "gpus/$file" --max-positives 30
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $file"
    else
        echo "Error processing: $file"
        echo "Continuing with next file..."
    fi
    
    echo "----------------------------------------"
    echo ""
done

echo "All files processed."