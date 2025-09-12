#!/usr/bin/env python3
"""
Extract optimal rephrases from NRMSE analysis data.
Optimal rephrases are those with lower NRMSE than the original instruction.
"""

import json
import glob
from datetime import datetime

def load_nrmse_analysis():
    """Find and load the most recent NRMSE analysis file"""
    files = glob.glob('nrmse_analysis_*.json')
    if not files:
        raise FileNotFoundError("No NRMSE analysis files found!")
    
    # Get the most recent file
    latest_file = sorted(files)[-1]
    print(f"Loading data from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f), latest_file

def load_augmented_instructions():
    """Load augmented instructions to get the actual rephrase text"""
    with open('augmented_instructions.json', 'r') as f:
        augmented_data = json.load(f)
    
    # Create lookup dictionary
    augmented_dict = {item['sample_id']: item for item in augmented_data}
    return augmented_dict

def main():
    print("Loading NRMSE analysis and augmented instructions...")
    nrmse_data, source_file = load_nrmse_analysis()
    augmented_dict = load_augmented_instructions()
    
    results = []
    
    print("Processing samples to find optimal rephrases...")
    
    for sample in nrmse_data:
        sample_id = sample['sample_id']
        original_instruction = sample['original_instruction']
        nrmse_list = sample['nrmse_list']
        
        if not nrmse_list:
            print(f"Warning: No NRMSE values found for sample_id {sample_id}")
            continue
        
        # First NRMSE value is always the original instruction
        original_nrmse = nrmse_list[0]
        rephrase_nrmse_values = nrmse_list[1:]  # Rest are rephrases
        
        # Get the actual rephrase texts
        if sample_id not in augmented_dict:
            print(f"Warning: No augmented instructions found for sample_id {sample_id}")
            continue
        
        rephrases = augmented_dict[sample_id]['rephrases']
        
        # Check if we have matching number of rephrases and NRMSE values
        if len(rephrases) != len(rephrase_nrmse_values):
            print(f"Warning: Mismatch in number of rephrases ({len(rephrases)}) and NRMSE values ({len(rephrase_nrmse_values)}) for sample_id {sample_id}")
            # Take the minimum to avoid index errors
            min_length = min(len(rephrases), len(rephrase_nrmse_values))
            rephrases = rephrases[:min_length]
            rephrase_nrmse_values = rephrase_nrmse_values[:min_length]
        
        # Find optimal rephrases (those with lower NRMSE than original)
        optimal_rephrases = []
        
        for rephrase, rephrase_nrmse in zip(rephrases, rephrase_nrmse_values):
            if rephrase_nrmse < original_nrmse:
                optimal_rephrases.append({
                    'rephrase': rephrase,
                    'nrmse': rephrase_nrmse,
                    'improvement': original_nrmse - rephrase_nrmse
                })
        
        # Sort optimal rephrases by NRMSE (best first)
        optimal_rephrases.sort(key=lambda x: x['nrmse'])
        
        # Create result entry
        result_entry = {
            'sample_id': sample_id,
            'original_instruction': original_instruction,
            'original_nrmse': original_nrmse,
            'optimal_rephrases': optimal_rephrases,
            'num_optimal_rephrases': len(optimal_rephrases),
            'total_rephrases': len(rephrases),
            'improvement_rate': len(optimal_rephrases) / len(rephrases) if rephrases else 0
        }
        
        results.append(result_entry)
        
        # Print progress
        print(f"Sample {sample_id}: {len(optimal_rephrases)}/{len(rephrases)} optimal rephrases (original NRMSE: {original_nrmse:.6f})")
    
    # Create simplified output (only required fields)
    simplified_results = []
    for result in results:
        simplified_entry = {
            'sample_id': result['sample_id'],
            'original_instruction': result['original_instruction'],
            'optimal_rephrases': [item['rephrase'] for item in result['optimal_rephrases']]
        }
        simplified_results.append(simplified_entry)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"optimal_rephrases_{timestamp}.json"
    
    print(f"\nSaving results to {output_filename}...")
    with open(output_filename, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print("Results saved successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    total_samples = len(results)
    total_rephrases = sum(r['total_rephrases'] for r in results)
    total_optimal = sum(r['num_optimal_rephrases'] for r in results)
    
    print(f"Total samples processed: {total_samples}")
    print(f"Total rephrases: {total_rephrases}")
    print(f"Total optimal rephrases: {total_optimal}")
    print(f"Overall improvement rate: {total_optimal/total_rephrases:.2%}" if total_rephrases > 0 else "N/A")
    
    print(f"\nPer-sample breakdown:")
    for result in results:
        print(f"  Sample {result['sample_id']}: {result['num_optimal_rephrases']}/{result['total_rephrases']} optimal ({result['improvement_rate']:.1%})")
    
    # Show some examples of best improvements
    print(f"\nTop improvements:")
    all_optimal = []
    for result in results:
        for optimal in result['optimal_rephrases']:
            all_optimal.append({
                'sample_id': result['sample_id'],
                'original_instruction': result['original_instruction'],
                'original_nrmse': result['original_nrmse'],
                'rephrase': optimal['rephrase'],
                'rephrase_nrmse': optimal['nrmse'],
                'improvement': optimal['improvement']
            })
    
    # Sort by improvement and show top 5
    all_optimal.sort(key=lambda x: x['improvement'], reverse=True)
    for i, opt in enumerate(all_optimal[:5]):
        print(f"  {i+1}. Sample {opt['sample_id']}: {opt['improvement']:.6f} improvement")
        print(f"     Original: '{opt['original_instruction']}' (NRMSE: {opt['original_nrmse']:.6f})")
        print(f"     Optimal:  '{opt['rephrase']}' (NRMSE: {opt['rephrase_nrmse']:.6f})")
        print()
    
    print(f"Output saved to: {output_filename}")
    print(f"Source file: {source_file}")

if __name__ == "__main__":
    main()
