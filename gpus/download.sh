#!/bin/bash

# Array of files to download
files=(
    "instruction_output_ids_gpu0_20250914_054300.json"
    "instruction_output_ids_gpu1_20250912_180744.json"
    "instruction_output_ids_gpu2_20250912_172652.json"
    "instruction_output_ids_gpu3_20250912_035021.json"
    "instruction_output_ids_gpu4_20250912_141854.json"
    "instruction_output_ids_gpu5_20250912_160613.json"
    "instruction_output_ids_gpu6_20250912_133855.json"
)

# S3 bucket path
S3_BUCKET="s3://bridge-data-bucket/rephrase/gpus/"

# Maximum number of parallel downloads
MAX_PARALLEL=8

# Function to download a file
download_file() {
    local file=$1
    echo "Downloading: $file"
    aws s3 cp "${S3_BUCKET}${file}" "${file}"
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $file"
    else
        echo "✗ Failed: $file"
    fi
}

# Export the function so it can be used by parallel processes
export -f download_file
export S3_BUCKET

echo "Starting parallel download of ${#files[@]} files..."
echo "========================================="

# Method 1: Using GNU parallel (if installed)
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel..."
    printf '%s\n' "${files[@]}" | parallel -j $MAX_PARALLEL download_file {}
else
    # Method 2: Using background processes
    echo "Using background processes..."
    
    # Counter for managing parallel downloads
    count=0
    
    for file in "${files[@]}"; do
        # Start download in background
        download_file "$file" &
        
        # Increment counter
        ((count++))
        
        # Wait if we've reached the maximum parallel downloads
        if [ $((count % MAX_PARALLEL)) -eq 0 ]; then
            wait
        fi
    done
    
    # Wait for any remaining background jobs
    wait
fi

echo "========================================="
echo "All downloads completed!"

# Optional: List downloaded files with sizes
echo ""
echo "Downloaded files:"
ls -lh *.json 2>/dev/null | awk '{print $9, $5}'