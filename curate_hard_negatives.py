#!/usr/bin/env python3
"""
Process augmented_instructions.json, gpu_output_actions.json, and groundtruth_actions.json
to compute NRMSE values and mine hard negatives, then output in the specified format.
Uses precomputed Qwen3 embeddings for efficiency.
"""

import json
import numpy as np
import pickle
from token2action import TokenActionConverter
from datetime import datetime
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import argparse
from tqdm import tqdm

def load_precomputed_embeddings(embeddings_file: str, metadata_file: str) -> Tuple[np.ndarray, Dict]:
    """Load precomputed Qwen3 embeddings and metadata."""
    print(f"Loading precomputed embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    print(f"Loading metadata from {metadata_file}")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total instructions: {len(metadata['all_instructions'])}")
    
    # Create instruction-to-index mapping once for efficiency
    print("Creating instruction-to-index mapping...")
    metadata['instruction_to_idx'] = {inst: idx for idx, inst in enumerate(metadata['all_instructions'])}
    
    return embeddings, metadata

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

def mine_hard_negatives_for_anchor(
    anchor_idx: int,
    e_lang,            # (N, D) language embeddings for all rephrases
    errs,              # (N,) action errors aligned with e_lang
    original_idx: int = 0,  # Index of original instruction
    tau=0.98,          # sim threshold to be "close" to anchor
    delta=0.02,        # error margin beyond original to count as worse
    topk=5,
    diversity_clusters=3
):
    """Mine hard negatives for a specific anchor rephrase."""
    e_anchor = F.normalize(e_lang[anchor_idx:anchor_idx+1], dim=-1)
    E = F.normalize(e_lang, dim=-1)
    sim = (E @ e_anchor.T).squeeze(-1)

    # Hard negatives should be worse than the ORIGINAL instruction, not the anchor
    err_original = errs[original_idx]
    worse_than_original = (errs >= err_original + delta)
    close = (sim >= tau)
    not_anchor = torch.ones_like(worse_than_original, dtype=torch.bool); not_anchor[anchor_idx] = False

    # Candidates are close to anchor but worse than original
    candidates = worse_than_original & close & not_anchor
    if candidates.sum() == 0:
        return torch.tensor([], dtype=torch.long)

    # Hardness score based on similarity to anchor and how much worse than original
    hardness = sim * torch.clamp(errs - err_original, min=0.0)
    hardness = torch.where(candidates, hardness, torch.tensor(-1.0, device=hardness.device))

    # Diversity: pick best per cluster
    idx = torch.topk(hardness, k=min(topk*3, candidates.sum().item())).indices
    chosen = []

    if diversity_clusters > 1 and idx.numel() > diversity_clusters:
        # Bucket by similarity bands
        q_tensor = torch.linspace(0, 1, diversity_clusters+1).to(sim.device)
        bands = torch.quantile(sim[idx], q=q_tensor)
        for b in range(diversity_clusters):
            in_band = idx[(sim[idx] >= bands[b]) & (sim[idx] <= bands[b+1])]
            if in_band.numel() > 0:
                best = in_band[torch.argmax(hardness[in_band])]
                chosen.append(best.item())
                if len(chosen) >= topk: break
        if len(chosen) < topk:
            for j in idx:
                j = j.item()
                if j not in chosen:
                    chosen.append(j)
                    if len(chosen) >= topk: break
        return torch.tensor(chosen[:topk], dtype=torch.long)
    else:
        return idx[:topk]

def identify_synthetic_positives(errs: torch.Tensor, original_idx: int = 0, delta: float = 0.02) -> torch.Tensor:
    """Identify synthetic positives (rephrases that perform better than original)."""
    original_err = errs[original_idx]
    return errs < (original_err - delta)

def process_sample(sample_id: int, groundtruth_dict: Dict, gpu_output_data: Dict, 
                  augmented_dict: Dict, converter: TokenActionConverter, 
                  embeddings: np.ndarray, metadata: Dict, max_positives: int = 40) -> Optional[Dict]:
    """Process a single sample to compute NRMSE and mine hard negatives."""
    
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
    nrmse_values = []
    
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
        for generated_action in generated_actions:
            nrmse = calculate_nrmse(ground_truth_action, generated_action)
            nrmse_values.append(float(nrmse))
    except Exception as e:
        return None
    
    if len(nrmse_values) == 0:
        return None
    
    # Get embeddings for this sample's instructions using precomputed embeddings
    sample_embedding_indices = []
    for instruction in all_instructions:
        if instruction in metadata['instruction_to_idx']:
            sample_embedding_indices.append(metadata['instruction_to_idx'][instruction])
        else:
            return None
    
    # Extract embeddings for this sample and move to device efficiently
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_embeddings = embeddings[sample_embedding_indices]
    e_lang = torch.tensor(sample_embeddings, dtype=torch.float32, device=device)
    errs = torch.tensor(nrmse_values, dtype=torch.float32, device=device)
    
    # Identify synthetic positives (including original instruction at index 0)
    synthetic_positives = identify_synthetic_positives(errs, original_idx=0, delta=0.02)
    positive_indices = torch.where(synthetic_positives)[0]
    
    # Always include original instruction as a positive (even if it's not the best)
    if 0 not in positive_indices:
        positive_indices = torch.cat([torch.tensor([0], device=positive_indices.device), positive_indices])
    
    # Cap at max_positives, selecting original + (max_positives-1) with lowest NRMSE
    if len(positive_indices) > max_positives:
        # Sort positive indices by their NRMSE values (ascending)
        sorted_indices = positive_indices[torch.argsort(errs[positive_indices])]
        
        # Always keep the original instruction (index 0) if it's in the list
        if 0 in positive_indices:
            # Remove original from sorted list temporarily
            sorted_indices_no_orig = sorted_indices[sorted_indices != 0]
            # Take top (max_positives-1) excluding original + original = max_positives total
            top_indices = sorted_indices_no_orig[:max_positives-1]
            positive_indices = torch.cat([torch.tensor([0], device=positive_indices.device), top_indices])
        else:
            # Original not in positives, just take top max_positives
            positive_indices = sorted_indices[:max_positives]
        
        pass
    
    # Continue processing with selected positives
    
    # Pre-compute normalized embeddings once for efficiency
    E_norm = F.normalize(e_lang, dim=-1)
    
    # Mine hard negatives for each positive
    positive_instructions = {}
    
    for pos_idx in positive_indices:
        pos_idx_val = pos_idx.item()
        positive_text = all_instructions[pos_idx_val]
        positive_error = errs[pos_idx_val].item()
        
        # Mine hard negatives for this positive
        hard_neg_indices = mine_hard_negatives_for_anchor(
            anchor_idx=pos_idx_val,
            e_lang=e_lang,
            errs=errs,
            original_idx=0,
            tau=0.98,
            delta=0.02,
            topk=3,
            diversity_clusters=2
        )
        
        negative_instructions = {}
        if len(hard_neg_indices) > 0:
            # Use pre-computed normalized embeddings for similarity calculation
            e_pos_norm = E_norm[pos_idx_val:pos_idx_val+1]
            similarities = (E_norm @ e_pos_norm.T).squeeze(-1)
            
            for neg_idx in hard_neg_indices:
                neg_idx_val = neg_idx.item()
                neg_text = all_instructions[neg_idx_val]
                neg_sim = similarities[neg_idx_val].item()
                neg_err = errs[neg_idx_val].item()
                
                negative_instructions[neg_text] = {
                    "similarity": float(neg_sim),
                    "error": float(neg_err)
                }
                
        
        positive_instructions[positive_text] = {
            "negative_instructions": negative_instructions
        }
    
    return {
        "sample_id": sample_id,
        "groundtruth_action": ground_truth_action.tolist(),
        "positive_instructions": positive_instructions
    }

def main():
    parser = argparse.ArgumentParser(description="Curate hard negatives using precomputed embeddings")
    parser.add_argument("--gpu-output", required=True, help="Path to GPU output data JSON file")
    parser.add_argument("--max-positives", type=int, default=30, 
                       help="Maximum number of positive instructions per sample (original + N-1 best rephrases)")
    args = parser.parse_args()
    
    print("Loading JSON files...")
    
    # Load all data files
    with open('actions_instructions.json', 'r') as f:
        groundtruth_data = json.load(f)
    
    with open(args.gpu_output, 'r') as f:
        gpu_output_data = json.load(f)
    
    with open('augmented_instructions.json', 'r') as f:
        augmented_data = json.load(f)
    
    print("Initializing token-to-action converter...")
    converter = TokenActionConverter()
    
    print("Loading precomputed embeddings...")
    embeddings, metadata = load_precomputed_embeddings(
        "qwen_embeddings_20250914_003853.npy", 
        "qwen_embeddings_metadata_20250914_003853.pkl"
    )
    
    # Pre-process data into dictionaries for faster lookup
    print("Pre-processing data for faster lookup...")
    groundtruth_dict = {item['sample_id']: item for item in groundtruth_data}
    augmented_dict = {item['sample_id']: item for item in augmented_data}
    
    # Process each sample
    curated_results = []
    
    # Get list of sample IDs from GPU output data
    sample_ids = [result['sample_id'] for result in gpu_output_data['results']]
    
    for sample_id in tqdm(sample_ids, desc="Curating hard negatives"):
        result = process_sample(sample_id, groundtruth_dict, gpu_output_data, 
                              augmented_dict, converter, embeddings, metadata, args.max_positives)
        
        if result is not None:
            curated_results.append(result)
    
    # Save final results
    output_filename = "curated_hard_negatives.json"
    
    with open(output_filename, 'w') as f:
        json.dump(curated_results, f, indent=2)
    
    print(f"Results saved to: {output_filename}")
    
    # Print summary statistics
    total_positives = sum(len(result['positive_instructions']) for result in curated_results)
    total_negatives = sum(
        sum(len(pos_data['negative_instructions']) 
            for pos_data in result['positive_instructions'].values())
        for result in curated_results
    )
    
    print(f"Total samples: {len(curated_results)}")
    print(f"Total positives: {total_positives}")
    print(f"Total negatives: {total_negatives}")
    print(f"Avg positives/sample: {total_positives/len(curated_results):.1f}")
    print(f"Avg negatives/positive: {total_negatives/max(total_positives, 1):.1f}")

if __name__ == "__main__":
    main()
