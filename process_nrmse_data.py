#!/usr/bin/env python3
"""
Process the three JSON files to create a new JSON file with sample_id, instruction, 
and NRMSE list for original + rephrased instructions.
"""

import json
import numpy as np
from token2action import TokenActionConverter
from datetime import datetime
import os

# Define the ranges for NRMSE calculation (from generate_vla_actions.py)
min_values = np.array([-0.02872725307941437,
          -0.04170349963009357,
          -0.026093858778476715,
          -0.08092105075716972,
          -0.09288699507713317,
          -0.20718276381492615,
          0.0])
max_values = np.array([0.028309678435325586,
          0.040855254605412394,
          0.040161586627364146,
          0.08192047759890528,
          0.07792850524187081,
          0.20382574498653397,
          1.0])
ranges = max_values - min_values

def calculate_nrmse(action0, action1):
    """
    Calculate normalized root mean squared error between two actions
    """
    # Normalize the difference by the range
    normalized_diff = (action0 - action1) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2))
    return nrmse

def load_json_files():
    """Load all three JSON files"""
    # Load groundtruth actions
    with open('groundtruth_actions.json', 'r') as f:
        groundtruth_data = json.load(f)
    
    # Load GPU output actions
    with open('gpu_output_actions.json', 'r') as f:
        gpu_output_data = json.load(f)
    
    # Load augmented instructions
    with open('augmented_instructions.json', 'r') as f:
        augmented_data = json.load(f)
    
    return groundtruth_data, gpu_output_data, augmented_data

def main():
    print("Loading JSON files...")
    groundtruth_data, gpu_output_data, augmented_data = load_json_files()
    
    # Initialize token-to-action converter
    print("Initializing token-to-action converter...")
    converter = TokenActionConverter()
    
    # Create dictionaries for quick lookup
    groundtruth_dict = {item['sample_id']: item for item in groundtruth_data}
    augmented_dict = {item['sample_id']: item for item in augmented_data}
    
    results = []
    
    print("Processing samples...")
    for sample_result in gpu_output_data['results']:
        sample_id = sample_result['sample_id']
        original_instruction = sample_result['original_instruction']
        output_ids_list = sample_result['output_ids_list']
        
        # Get ground truth action for this sample
        if sample_id not in groundtruth_dict:
            print(f"Warning: No ground truth found for sample_id {sample_id}")
            continue
            
        ground_truth_action = np.array(groundtruth_dict[sample_id]['current_groundtruth_action'])
        
        # Get rephrases for this sample
        if sample_id not in augmented_dict:
            print(f"Warning: No augmented instructions found for sample_id {sample_id}")
            continue
            
        rephrases = augmented_dict[sample_id]['rephrases']
        
        # Verify that the original instruction matches
        if original_instruction != augmented_dict[sample_id]['instruction']:
            print(f"Warning: Original instruction mismatch for sample_id {sample_id}")
            print(f"  GPU output: '{original_instruction}'")
            print(f"  Augmented:  '{augmented_dict[sample_id]['instruction']}'")
        
        # Process all output_ids (first is original, rest are rephrases)
        all_instructions = [original_instruction] + rephrases
        nrmse_values = []
        
        # Check if we have the right number of output_ids
        if len(output_ids_list) != len(all_instructions):
            print(f"Warning: Mismatch in number of output_ids ({len(output_ids_list)}) and instructions ({len(all_instructions)}) for sample_id {sample_id}")
            # Take the minimum to avoid index errors
            num_to_process = min(len(output_ids_list), len(all_instructions))
            all_instructions = all_instructions[:num_to_process]
            output_ids_list = output_ids_list[:num_to_process]
        
        for i, (instruction, output_ids) in enumerate(zip(all_instructions, output_ids_list)):
            try:
                # Convert tokens to actions
                generated_action = converter.token_to_action(output_ids)
                
                # Calculate NRMSE
                nrmse = calculate_nrmse(ground_truth_action, generated_action)
                nrmse_values.append({
                    'instruction': instruction,
                    'is_original': i == 0,
                    'nrmse': float(nrmse),
                    'generated_action': generated_action.tolist(),
                    'output_ids': output_ids
                })
                
            except Exception as e:
                print(f"Error processing instruction {i} for sample_id {sample_id}: {e}")
                nrmse_values.append({
                    'instruction': instruction,
                    'is_original': i == 0,
                    'nrmse': None,
                    'generated_action': None,
                    'output_ids': output_ids,
                    'error': str(e)
                })
        
        # Create result entry for this sample
        result_entry = {
            'sample_id': sample_id,
            'original_instruction': original_instruction,
            'ground_truth_action': ground_truth_action.tolist(),
            'nrmse_data': nrmse_values,
            'num_rephrases': len(rephrases),
            'total_instructions': len(all_instructions)
        }
        
        results.append(result_entry)
        
        # Print progress
        if len(results) % 10 == 0:
            print(f"Processed {len(results)} samples...")
    
    # Calculate summary statistics
    print(f"\nProcessed {len(results)} samples total")
    
    # Collect all NRMSE values for statistics
    all_nrmse = []
    original_nrmse = []
    rephrase_nrmse = []
    
    for result in results:
        for nrmse_item in result['nrmse_data']:
            if nrmse_item['nrmse'] is not None:
                all_nrmse.append(nrmse_item['nrmse'])
                if nrmse_item['is_original']:
                    original_nrmse.append(nrmse_item['nrmse'])
                else:
                    rephrase_nrmse.append(nrmse_item['nrmse'])
    
    # Create output data structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_samples': len(results),
            'total_instructions': len(all_nrmse),
            'original_instructions': len(original_nrmse),
            'rephrased_instructions': len(rephrase_nrmse),
            'source_files': {
                'groundtruth': 'groundtruth_actions.json',
                'gpu_output': 'gpu_output_actions.json', 
                'augmented': 'augmented_instructions.json'
            },
            'nrmse_ranges': {
                'min_values': min_values.tolist(),
                'max_values': max_values.tolist(),
                'ranges': ranges.tolist()
            },
            'summary_stats': {
                'all_nrmse': {
                    'mean': float(np.mean(all_nrmse)) if all_nrmse else None,
                    'std': float(np.std(all_nrmse)) if all_nrmse else None,
                    'min': float(np.min(all_nrmse)) if all_nrmse else None,
                    'max': float(np.max(all_nrmse)) if all_nrmse else None,
                    'count': len(all_nrmse)
                },
                'original_nrmse': {
                    'mean': float(np.mean(original_nrmse)) if original_nrmse else None,
                    'std': float(np.std(original_nrmse)) if original_nrmse else None,
                    'min': float(np.min(original_nrmse)) if original_nrmse else None,
                    'max': float(np.max(original_nrmse)) if original_nrmse else None,
                    'count': len(original_nrmse)
                },
                'rephrase_nrmse': {
                    'mean': float(np.mean(rephrase_nrmse)) if rephrase_nrmse else None,
                    'std': float(np.std(rephrase_nrmse)) if rephrase_nrmse else None,
                    'min': float(np.min(rephrase_nrmse)) if rephrase_nrmse else None,
                    'max': float(np.max(rephrase_nrmse)) if rephrase_nrmse else None,
                    'count': len(rephrase_nrmse)
                }
            }
        },
        'results': results
    }
    
    # Save to file
    output_filename = f"nrmse_analysis_{timestamp}.json"
    print(f"\nSaving results to {output_filename}...")
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Results saved successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total samples: {len(results)}")
    print(f"Total instructions processed: {len(all_nrmse)}")
    print(f"Original instructions: {len(original_nrmse)}")
    print(f"Rephrased instructions: {len(rephrase_nrmse)}")
    
    if all_nrmse:
        print(f"\nOverall NRMSE:")
        print(f"  Mean: {np.mean(all_nrmse):.6f}")
        print(f"  Std:  {np.std(all_nrmse):.6f}")
        print(f"  Min:  {np.min(all_nrmse):.6f}")
        print(f"  Max:  {np.max(all_nrmse):.6f}")
    
    if original_nrmse:
        print(f"\nOriginal Instruction NRMSE:")
        print(f"  Mean: {np.mean(original_nrmse):.6f}")
        print(f"  Std:  {np.std(original_nrmse):.6f}")
        print(f"  Min:  {np.min(original_nrmse):.6f}")
        print(f"  Max:  {np.max(original_nrmse):.6f}")
    
    if rephrase_nrmse:
        print(f"\nRephrased Instruction NRMSE:")
        print(f"  Mean: {np.mean(rephrase_nrmse):.6f}")
        print(f"  Std:  {np.std(rephrase_nrmse):.6f}")
        print(f"  Min:  {np.min(rephrase_nrmse):.6f}")
        print(f"  Max:  {np.max(rephrase_nrmse):.6f}")
    
    print(f"\nOutput saved to: {output_filename}")

if __name__ == "__main__":
    main()
