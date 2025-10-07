#!/bin/bash

# Array of GPU output files to process
gpu_files=(
    "instruction_output_ids_gpu0_20250917_090006.json"
    "instruction_output_ids_gpu1_20250917_101852.json"
    "instruction_output_ids_gpu2_20250918_083210.json"
    "instruction_output_ids_gpu3_20250917_094723.json"
    "instruction_output_ids_gpu4_20250917_092637.json"
    "instruction_output_ids_gpu5_20250917_082739.json"
    "instruction_output_ids_gpu6_20250917_102826.json"
    "instruction_output_ids_gpu7_20250917_092733.json"
    "instruction_output_ids_gpu8_20250917_082152.json"
    "instruction_output_ids_gpu9_20250918_133037.json"
    "instruction_output_ids_gpu10_20250917_073216.json"
    "instruction_output_ids_gpu11_20250917_093238.json"
)

# Process each file sequentially
for file in "${gpu_files[@]}"; do
    echo "Processing: $file"
    echo "----------------------------------------"
    
    # Run the python command with the current file
    python curate_hard_negative_actions.py --gpu-output "gpus/$file"
    
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
echo ""
echo "Output files generated:"
for file in "${gpu_files[@]}"; do
    output_file="hard_negative_actions_${file}"
    if [ -f "$output_file" ]; then
        echo "  ✓ $output_file"
    else
        echo "  ✗ $output_file (not found)"
    fi
done
