#!/bin/bash

S3_BASE="s3://bridge-data-bucket/rephrase/new_gpus"
FILES=(
"instruction_output_ids_gpu0_20250917_090006.json"
"instruction_output_ids_gpu1_20250917_101852.json"
"instruction_output_ids_gpu10_20250917_073216.json"
"instruction_output_ids_gpu11_20250917_093238.json"
"instruction_output_ids_gpu3_20250917_094723.json"
"instruction_output_ids_gpu4_20250917_092637.json"
"instruction_output_ids_gpu5_20250917_082739.json"
"instruction_output_ids_gpu6_20250917_102826.json"
"instruction_output_ids_gpu7_20250917_092733.json"
"instruction_output_ids_gpu8_20250917_082152.json"
"instruction_output_ids_gpu2_20250918_083210.json"
"instruction_output_ids_gpu9_20250918_133037.json"
)

# FILES=(
# instruction_output_ids_gpu2_20250918_083210.json
# instruction_output_ids_gpu9_20250918_133037.json
# )

for file in "${FILES[@]}"; do
    echo "Downloading $file..."
    aws s3 cp "${S3_BASE}/${file}" "${file}"
done
