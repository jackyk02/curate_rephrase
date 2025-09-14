#!/usr/bin/env python3
"""
Precompute Qwen3 embeddings for all instructions in augmented_instructions.json
"""

import json
import numpy as np
import pickle
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch

class QwenEmbedder:
    """Qwen3-based text embedder for computing instruction embeddings."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "auto", use_flash_attention: bool = True):
        """Initialize Qwen3 embedder."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Initialize with flash attention and optimized settings if available
        if use_flash_attention:
            try:
                self.model = SentenceTransformer(
                    model_name,
                    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                    tokenizer_kwargs={"padding_side": "left"},
                )
                print(f"Initialized Qwen3 embedder with flash attention: {model_name}")
            except Exception as e:
                print(f"Flash attention failed, falling back to standard: {e}")
                self.model = SentenceTransformer(model_name, device=device)
                print(f"Initialized Qwen3 embedder (standard): {model_name} on {device}")
        else:
            self.model = SentenceTransformer(model_name, device=device)
            print(f"Initialized Qwen3 embedder: {model_name} on {device}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, use_query_prompt: bool = False) -> np.ndarray:
        """Compute Qwen3 embeddings for a list of texts."""
        if use_query_prompt:
            try:
                embeddings = self.model.encode(
                    texts, 
                    prompt_name="query",
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
            except Exception as e:
                print(f"Query prompt failed, using default: {e}")
                embeddings = self.model.encode(
                    texts, 
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
        else:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        
        return embeddings

def collect_all_instructions(augmented_data: List[Dict]) -> List[str]:
    """Collect all unique instructions from augmented data."""
    all_instructions = set()
    
    for sample in augmented_data:
        # Add original instruction
        all_instructions.add(sample['instruction'])
        
        # Add all rephrases
        for rephrase in sample['rephrases']:
            all_instructions.add(rephrase)
    
    return sorted(list(all_instructions))

def create_instruction_mapping(augmented_data: List[Dict], all_instructions: List[str]) -> Dict:
    """Create mapping from sample_id to instruction indices."""
    # Create index mapping
    instruction_to_idx = {inst: idx for idx, inst in enumerate(all_instructions)}
    
    sample_mappings = {}
    for sample in augmented_data:
        sample_id = sample['sample_id']
        
        # Map original instruction + rephrases to indices
        original_idx = instruction_to_idx[sample['instruction']]
        rephrase_indices = [instruction_to_idx[rephrase] for rephrase in sample['rephrases']]
        
        sample_mappings[sample_id] = {
            'original_instruction': sample['instruction'],
            'original_idx': original_idx,
            'rephrase_indices': rephrase_indices,
            'all_indices': [original_idx] + rephrase_indices,
            'all_instructions': [sample['instruction']] + sample['rephrases']
        }
    
    return sample_mappings

def main():
    print("Precomputing Qwen3 embeddings for all instructions")
    print("=" * 60)
    
    # Load augmented instructions
    print("Loading augmented instructions...")
    with open('augmented_instructions.json', 'r') as f:
        augmented_data = json.load(f)
    
    print(f"Loaded {len(augmented_data)} samples")
    
    # Collect all unique instructions
    print("Collecting all unique instructions...")
    all_instructions = collect_all_instructions(augmented_data)
    print(f"Total unique instructions to embed: {len(all_instructions)}")
    
    # Create instruction mapping
    print("Creating instruction mappings...")
    sample_mappings = create_instruction_mapping(augmented_data, all_instructions)
    
    # Initialize embedder
    print("Initializing Qwen3 embedder...")
    import torch
    embedder = QwenEmbedder()
    
    # Compute embeddings for all instructions
    print("Computing Qwen3 embeddings...")
    embeddings = embedder.embed_texts(all_instructions, batch_size=32, use_query_prompt=True)
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Save embeddings and metadata
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("Saving embeddings...")
    embeddings_file = f"qwen_embeddings_{timestamp}.npy"
    np.save(embeddings_file, embeddings)
    
    print("Saving metadata...")
    metadata_file = f"qwen_embeddings_metadata_{timestamp}.pkl"
    metadata = {
        'all_instructions': all_instructions,
        'sample_mappings': sample_mappings,
        'embedding_shape': embeddings.shape,
        'embedding_dim': embeddings.shape[1],
        'timestamp': timestamp,
        'model_name': "Qwen/Qwen3-Embedding-0.6B"
    }
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save summary
    summary_file = f"qwen_embeddings_summary_{timestamp}.json"
    summary = {
        'embeddings_file': embeddings_file,
        'metadata_file': metadata_file,
        'total_instructions': len(all_instructions),
        'total_samples': len(augmented_data),
        'embedding_shape': list(embeddings.shape),
        'embedding_dimension': int(embeddings.shape[1]),
        'timestamp': timestamp
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Summary saved to: {summary_file}")
    print("Precomputation complete!")

if __name__ == "__main__":
    main()
