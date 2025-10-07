#!/usr/bin/env python3
"""
Process augmented_instructions.json, gpu_output_actions.json, and groundtruth_actions.json
to identify hard negative actions.

Hard negative actions are defined as:
1. Actions in Q4 (worst 25% of NRMSE values across all samples)
2. Worse than the NRMSE of the original instruction for that sample
"""

import json
import numpy as np
from token2action import TokenActionConverter
from datetime import datetime
import argparse
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple

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
    """Calculate normalized root mean squared error between two actions"""
    normalized_diff = (action0 - action1) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2))
    return nrmse

def process_sample_nrmse(sample_id: int, groundtruth_dict: Dict, gpu_output_data: Dict, 
                         augmented_dict: Dict, converter: TokenActionConverter) -> Optional[Dict]:
    """Process a single sample to compute NRMSE values for all instructions."""
    
    # Find the sample in GPU output data
    sample_result = None
    for result in gpu_output_data['results']:
        if result['sample_id'] == sample_id:
            sample_result = result
            break
    
    if sample_result is None:
        return None
    
    original_instruction = sample_result['original_instruction']
    output_ids_list = sample_result['output_ids_list']
    
    # Get ground truth action for this sample
    if sample_id not in groundtruth_dict:
        return None
        
    ground_truth_action = np.array(groundtruth_dict[sample_id]['current_groundtruth_action'])
    
    # Get rephrases for this sample
    if sample_id not in augmented_dict:
        return None
        
    rephrases = augmented_dict[sample_id]['rephrases']
    
    # Process all instructions (original + rephrases)
    all_instructions = [original_instruction] + rephrases
    
    # Check if we have the right number of output_ids
    if len(output_ids_list) != len(all_instructions):
        num_to_process = min(len(output_ids_list), len(all_instructions))
        all_instructions = all_instructions[:num_to_process]
        output_ids_list = output_ids_list[:num_to_process]
    
    # Compute NRMSE for all instructions using batch inference
    try:
        # Convert all output_ids to actions in batch
        generated_actions = converter.token_to_action(output_ids_list)
        
        # Compute NRMSE for each generated action
        nrmse_values = []
        actions = []
        for generated_action in generated_actions:
            nrmse = calculate_nrmse(ground_truth_action, generated_action)
            nrmse_values.append(float(nrmse))
            actions.append(generated_action)
    except Exception as e:
        return None
    
    if len(nrmse_values) == 0:
        return None
    
    return {
        "sample_id": sample_id,
        "original_instruction": original_instruction,
        "groundtruth_action": ground_truth_action.tolist(),
        "instructions": all_instructions,
        "actions": [action.tolist() for action in actions],
        "nrmse_values": nrmse_values
    }

def curate_hard_negative_actions(all_sample_results: List[Dict]) -> List[Dict]:
    """
    Curate hard negative actions based on per-sample Q4 threshold and comparison to original instruction.
    
    Args:
        all_sample_results: List of processed sample results with NRMSE values
    
    Returns:
        List of curated results with hard negative actions
    """
    curated_results = []
    
    for sample_result in all_sample_results:
        sample_id = sample_result['sample_id']
        original_instruction = sample_result['original_instruction']
        groundtruth_action = sample_result['groundtruth_action']
        instructions = sample_result['instructions']
        actions = sample_result['actions']
        nrmse_values = sample_result['nrmse_values']
        
        # Original instruction is always at index 0
        original_nrmse = nrmse_values[0]
        original_action = actions[0]
        
        # Calculate Q4 threshold (75th percentile) for THIS sample
        sample_q4_threshold = np.percentile(nrmse_values, 75)
        
        # Find hard negative actions:
        # 1. In Q4 for this sample (NRMSE >= sample_q4_threshold)
        # 2. Worse than original instruction (NRMSE > original_nrmse)
        hard_negative_actions = []
        
        for idx, (instruction, action, nrmse) in enumerate(zip(instructions, actions, nrmse_values)):
            # Skip original instruction
            if idx == 0:
                continue
            
            # Check if in Q4 for this sample and worse than original
            if nrmse >= sample_q4_threshold and nrmse > original_nrmse:
                hard_negative_actions.append(action)
        
        # Only include samples that have hard negative actions
        if len(hard_negative_actions) > 0:
            curated_results.append({
                "sample_id": sample_id,
                "hard_negative_actions": hard_negative_actions
            })
    
    return curated_results

def main():
    parser = argparse.ArgumentParser(description="Curate hard negative actions based on Q4 NRMSE threshold")
    parser.add_argument("--gpu-output", required=True, help="Path to GPU output data JSON file")
    parser.add_argument("--groundtruth", default="actions_instructions.json", 
                       help="Path to groundtruth actions JSON file")
    parser.add_argument("--augmented", default="augmented_instructions.json",
                       help="Path to augmented instructions JSON file")
    args = parser.parse_args()
    
    print("Loading JSON files...")
    
    # Load all data files
    with open(args.groundtruth, 'r') as f:
        groundtruth_data = json.load(f)
    
    with open(args.gpu_output, 'r') as f:
        gpu_output_data = json.load(f)
    
    with open(args.augmented, 'r') as f:
        augmented_data = json.load(f)
    
    print("Initializing token-to-action converter...")
    converter = TokenActionConverter()
    
    # Pre-process data into dictionaries for faster lookup
    print("Pre-processing data for faster lookup...")
    groundtruth_dict = {item['sample_id']: item for item in groundtruth_data}
    augmented_dict = {item['sample_id']: item for item in augmented_data}
    
    # Compute NRMSE for all samples and curate hard negatives
    print("Computing NRMSE values and curating hard negative actions...")
    all_sample_results = []
    
    # Get list of sample IDs from GPU output data
    sample_ids = [result['sample_id'] for result in gpu_output_data['results']]
    
    for sample_id in tqdm(sample_ids, desc="Processing samples"):
        result = process_sample_nrmse(sample_id, groundtruth_dict, gpu_output_data, 
                                     augmented_dict, converter)
        
        if result is not None:
            all_sample_results.append(result)
    
    # Curate hard negative actions (Q4 threshold calculated per sample)
    print("\nCurating hard negative actions (Q4 per sample)...")
    curated_results = curate_hard_negative_actions(all_sample_results)
    
    # Save final results
    import os
    gpu_output_basename = os.path.basename(args.gpu_output)
    output_filename = f"hard_negative_actions_{gpu_output_basename}"
    
    with open(output_filename, 'w') as f:
        json.dump(curated_results, f, indent=2)
    
    print(f"\nResults saved to: {output_filename}")
    
    # Print summary statistics
    total_hard_negatives = sum(len(result['hard_negative_actions']) for result in curated_results)
    samples_with_hard_negatives = len(curated_results)
    
    print(f"\nSummary Statistics:")
    print(f"  Total samples processed: {len(all_sample_results)}")
    print(f"  Samples with hard negative actions: {samples_with_hard_negatives}")
    print(f"  Total hard negative actions: {total_hard_negatives}")
    if samples_with_hard_negatives > 0:
        print(f"  Avg hard negatives per sample: {total_hard_negatives/samples_with_hard_negatives:.1f}")
    
    # Distribution of hard negatives per sample
    hard_neg_counts = [len(result['hard_negative_actions']) for result in curated_results]
    if len(hard_neg_counts) > 0:
        print(f"\nHard Negative Distribution:")
        print(f"  Min per sample: {min(hard_neg_counts)}")
        print(f"  Max per sample: {max(hard_neg_counts)}")
        print(f"  Median per sample: {np.median(hard_neg_counts):.1f}")

if __name__ == "__main__":
    main()
