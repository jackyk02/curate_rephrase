import json
import glob
import os
from pathlib import Path

def merge_json_files(file_pattern="hard_instruction_output_ids_gpu*.json", output_file="merged_instructions.json"):
    """
    Merge multiple JSON files with the same structure and sort by sample_id.
    
    Args:
        file_pattern (str): Glob pattern to match input files
        output_file (str): Name of the output merged file
    """
    
    # Find all matching files
    files = glob.glob(file_pattern)
    files.sort()  # Sort filenames for consistent processing order
    
    print(f"Found {len(files)} files to merge:")
    for file in files:
        print(f"  - {file}")
    
    # Collect all data from files
    all_data = []
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle both single dict and list formats
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
                
            print(f"Loaded {len(data) if isinstance(data, list) else 1} samples from {file_path}")
            
        except json.JSONDecodeError as e:
            print(f"Error reading {file_path}: Invalid JSON - {e}")
            continue
        except FileNotFoundError:
            print(f"Error: File {file_path} not found")
            continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not all_data:
        print("No data found to merge!")
        return
    
    # Sort by sample_id
    try:
        all_data.sort(key=lambda x: x.get('sample_id', float('inf')))
        print(f"\nSorted {len(all_data)} total samples by sample_id")
    except Exception as e:
        print(f"Warning: Could not sort data - {e}")
    
    # Check for duplicate sample_ids
    sample_ids = [item.get('sample_id') for item in all_data if 'sample_id' in item]
    duplicates = set([x for x in sample_ids if sample_ids.count(x) > 1])
    if duplicates:
        print(f"Warning: Found duplicate sample_ids: {sorted(duplicates)}")
    
    # Save merged data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully merged data saved to: {output_file}")
        print(f"Total samples: {len(all_data)}")
        
        # Print summary statistics
        if sample_ids:
            print(f"Sample ID range: {min(sample_ids)} to {max(sample_ids)}")
            
        # Show first few sample_ids as verification
        first_few_ids = [item.get('sample_id') for item in all_data[:5]]
        print(f"First 5 sample_ids: {first_few_ids}")
        
    except Exception as e:
        print(f"Error saving merged file: {e}")

def main():
    """Main function to run the merger with user-friendly options."""
    
    print("JSON File Merger")
    print("=" * 40)
    
    # Check if files exist in current directory
    default_pattern = "hard_instruction_output_ids_gpu*.json"
    matching_files = glob.glob(default_pattern)
    
    if matching_files:
        print(f"\nFound {len(matching_files)} files matching default pattern:")
        for f in sorted(matching_files)[:5]:  # Show first 5
            print(f"  - {f}")
        if len(matching_files) > 5:
            print(f"  ... and {len(matching_files) - 5} more")
        
        # Use default pattern
        file_pattern = default_pattern
        output_file = "merged_instructions.json"
    else:
        # Ask user for pattern
        print("\nNo files found with default pattern.")
        file_pattern = input("Enter file pattern (e.g., 'data_*.json'): ").strip()
        if not file_pattern:
            file_pattern = "*.json"
        
        output_file = input("Enter output filename (default: merged_data.json): ").strip()
        if not output_file:
            output_file = "merged_data.json"
    
    # Run the merger
    merge_json_files(file_pattern, output_file)

if __name__ == "__main__":
    main()