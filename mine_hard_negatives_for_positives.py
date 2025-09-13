import torch
import torch.nn.functional as F
import json
import numpy as np
from typing import Optional, Tuple, List, Dict
from sentence_transformers import SentenceTransformer

class BERTEmbedder:
    """Sentence transformer-based text embedder for computing instruction embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "auto"):
        """
        Initialize sentence transformer embedder.
        
        Args:
            model_name: SentenceTransformer model name
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
        print(f"Initialized SentenceTransformer embedder with {model_name} on {device}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Compute sentence embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape (len(texts), embedding_dim)
        """
        # SentenceTransformer handles batching internally
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        return embeddings

def load_data_for_sample(sample_id: int, data_dir: str = ".", embedder: Optional[BERTEmbedder] = None) -> Tuple[torch.Tensor, torch.Tensor, List[str], str]:
    """
    Load embeddings and error data for a specific sample using BERT embeddings computed on-the-fly.
    
    Args:
        sample_id: The sample ID to load data for
        data_dir: Directory containing the data files
        embedder: BERTEmbedder instance. If None, will create a new one.
        
    Returns:
        e_lang: (N, D) language embeddings for the sample's rephrases (including original at idx 0)
        errs: (N,) action errors aligned with e_lang
        all_rephrases: List of all rephrase texts
        original_instruction: The original instruction text
    """
    # Initialize embedder if not provided
    if embedder is None:
        embedder = BERTEmbedder()
    
    # Load the NRMSE data
    with open(f"{data_dir}/nrmse_analysis_20250912_102735.json", 'r') as f:
        nrmse_data = json.load(f)
    
    # Load the augmented instructions
    with open(f"{data_dir}/augmented_instructions.json", 'r') as f:
        aug_data = json.load(f)
    
    # Find the sample data
    sample_found = False
    for nrmse_sample in nrmse_data:
        if nrmse_sample['sample_id'] == sample_id:
            original_instruction = nrmse_sample['original_instruction']
            nrmse_list = nrmse_sample['nrmse_list']
            sample_found = True
            break
    
    if not sample_found:
        raise ValueError(f"Sample ID {sample_id} not found in NRMSE data")
    
    # Find the corresponding augmented data
    aug_sample = None
    for aug in aug_data:
        if aug['sample_id'] == sample_id:
            aug_sample = aug
            break
    
    if aug_sample is None:
        raise ValueError(f"Sample ID {sample_id} not found in augmented instructions data")
    
    # Verify instruction match
    if aug_sample['instruction'] != original_instruction:
        raise ValueError(f"Instruction mismatch for sample {sample_id}")
    
    # Build the rephrase list: original instruction + rephrases
    # The NRMSE list has one more entry (154) than the rephrases (153), 
    # suggesting the first NRMSE is for the original instruction
    all_rephrases = [original_instruction] + aug_sample['rephrases']
    
    # Verify lengths match
    if len(all_rephrases) != len(nrmse_list):
        print(f"Warning: Length mismatch for sample {sample_id}: {len(all_rephrases)} rephrases vs {len(nrmse_list)} NRMSE values")
        # Truncate to shorter length
        min_len = min(len(all_rephrases), len(nrmse_list))
        all_rephrases = all_rephrases[:min_len]
        nrmse_list = nrmse_list[:min_len]
    
    print(f"Computing BERT embeddings for {len(all_rephrases)} rephrases...")
    
    # Compute embeddings using BERT
    e_lang = embedder.embed_texts(all_rephrases)
    errs = torch.tensor(nrmse_list, dtype=torch.float32)
    
    # Ensure both tensors are on the same device
    if e_lang.device != errs.device:
        errs = errs.to(e_lang.device)
    
    print(f"Generated embeddings shape: {e_lang.shape}")
    print(f"Embeddings device: {e_lang.device}, Errors device: {errs.device}")
    
    return e_lang, errs, all_rephrases, original_instruction

def mine_hard_negatives_for_anchor(
    anchor_idx: int,
    e_lang,            # (N, D) language embeddings for all rephrases
    errs,              # (N,) action errors aligned with e_lang
    original_idx: int = 0,  # Index of original instruction
    e_action=None,     # (N, A) optional action embeddings
    e_star=None,       # (A,) optional target/good action embedding
    tau=0.95,          # sim threshold to be "close" to anchor
    delta=0.02,        # error margin beyond original to count as worse
    topk=8,
    lambda_action=0.0, # set >0 if using action mismatch
    diversity_clusters=4
):
    """
    Mine hard negatives for a specific anchor rephrase.
    Hard negatives are rephrases that are semantically close to the anchor
    but have worse performance than the ORIGINAL instruction.
    
    Args:
        anchor_idx: Index of the anchor rephrase
        e_lang: (N, D) language embeddings for all rephrases
        errs: (N,) action errors for each rephrase
        original_idx: Index of the original instruction (default: 0)
        e_action: (N, A) optional action embeddings for each rephrase
        e_star: (A,) optional target action embedding
        tau: similarity threshold to be considered "close" to anchor
        delta: error margin beyond original instruction to count as worse
        topk: number of hard negatives to return
        lambda_action: weight for action mismatch term
        diversity_clusters: number of clusters for diversity
        
    Returns:
        indices of hard negatives relative to anchor
    """
    e_anchor = F.normalize(e_lang[anchor_idx:anchor_idx+1], dim=-1)
    E = F.normalize(e_lang, dim=-1)
    sim = (E @ e_anchor.T).squeeze(-1)

    # Hard negatives should be worse than the ORIGINAL instruction, not the anchor
    err_original = errs[original_idx]
    worse_than_original = (errs >= err_original + delta)
    close = (sim >= tau)
    not_anchor = torch.ones_like(worse_than_original, dtype=torch.bool); not_anchor[anchor_idx] = False

    # Optionally compute distance from "good" action
    if e_action is not None and e_star is not None and lambda_action > 0:
        a = F.normalize(e_action, dim=-1)
        a_star = F.normalize(e_star.unsqueeze(0), dim=-1)
        action_mismatch = 1.0 + lambda_action * (1.0 - (a @ a_star.T).squeeze(-1))
    else:
        action_mismatch = torch.ones_like(sim)

    # Candidates are close to anchor but worse than original
    candidates = worse_than_original & close & not_anchor
    if candidates.sum() == 0:
        return torch.tensor([], dtype=torch.long)

    # Hardness score based on similarity to anchor and how much worse than original
    hardness = sim * torch.clamp(errs - err_original, min=0.0) * action_mismatch
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
    """
    Identify synthetic positives (rephrases that perform better than original).
    
    Args:
        errs: (N,) action errors
        original_idx: Index of original instruction
        delta: Error margin for being considered "better"
        
    Returns:
        Boolean tensor indicating which rephrases are synthetic positives
    """
    original_err = errs[original_idx]
    return errs < (original_err - delta)

def mine_hard_negatives_for_all_positives(
    sample_id: int, 
    embedder: Optional[BERTEmbedder] = None,
    tau: float = 0.95,
    delta: float = 0.02,
    topk: int = 8,
    diversity_clusters: int = 4,
    data_dir: str = "."
) -> Dict:
    """
    Mine hard negatives for all synthetic positives in a sample.
    
    Args:
        sample_id: Sample ID to analyze
        embedder: BERTEmbedder instance
        tau: Similarity threshold for hard negatives
        delta: Error margin for positive/negative classification
        topk: Number of hard negatives per positive
        diversity_clusters: Number of diversity clusters
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with results for all positives
    """
    # Load data
    e_lang, errs, all_rephrases, original_instruction = load_data_for_sample(
        sample_id, data_dir, embedder
    )
    
    print(f"\nSample {sample_id}: '{original_instruction}'")
    print(f"Loaded {len(e_lang)} rephrases with embeddings shape {e_lang.shape}")
    print(f"Error range: {errs.min():.4f} - {errs.max():.4f}, original error: {errs[0]:.4f}")
    
    # Identify synthetic positives
    synthetic_positives = identify_synthetic_positives(errs, original_idx=0, delta=delta)
    positive_indices = torch.where(synthetic_positives)[0]
    
    print(f"Found {len(positive_indices)} synthetic positives")
    
    if len(positive_indices) == 0:
        return {
            'sample_id': sample_id,
            'original_instruction': original_instruction,
            'synthetic_positives': [],
            'hard_negatives_per_positive': {},
            'total_rephrases': len(e_lang)
        }
    
    # Mine hard negatives for each positive
    hard_negatives_per_positive = {}
    
    print(f"\nMining hard negatives for each positive...")
    
    for pos_idx in positive_indices:
        pos_idx_val = pos_idx.item()
        positive_text = all_rephrases[pos_idx_val]
        positive_error = errs[pos_idx_val].item()
        
        print(f"\nPositive [{pos_idx_val}] (err={positive_error:.4f} vs orig={errs[0]:.4f}): '{positive_text[:80]}{'...' if len(positive_text) > 80 else ''}'")
        
        # Mine hard negatives for this positive
        # Hard negatives must be: 1) similar to positive, 2) worse than ORIGINAL instruction
        hard_neg_indices = mine_hard_negatives_for_anchor(
            anchor_idx=pos_idx_val,
            e_lang=e_lang,
            errs=errs,
            original_idx=0,  # Original instruction is always at index 0
            tau=tau,
            delta=delta,
            topk=topk,
            diversity_clusters=diversity_clusters
        )
        
        if len(hard_neg_indices) == 0:
            print(f"  No hard negatives found (need: similar to positive, worse than original {errs[0]:.4f})")
            hard_negatives_per_positive[pos_idx_val] = {
                'positive_text': positive_text,
                'positive_error': positive_error,
                'hard_negatives': []
            }
            continue
        
        # Calculate similarities for analysis
        e_pos_norm = F.normalize(e_lang[pos_idx_val:pos_idx_val+1], dim=-1)
        E_norm = F.normalize(e_lang, dim=-1)
        similarities = (E_norm @ e_pos_norm.T).squeeze(-1)
        
        # Collect hard negative info
        hard_negatives_info = []
        print(f"  Found {len(hard_neg_indices)} hard negatives:")
        
        for i, neg_idx in enumerate(hard_neg_indices):
            neg_idx_val = neg_idx.item()
            neg_text = all_rephrases[neg_idx_val]
            neg_sim = similarities[neg_idx_val].item()
            neg_err = errs[neg_idx_val].item()
            
            print(f"    {i+1}. [{neg_idx_val}] sim={neg_sim:.3f}, err={neg_err:.4f} (vs orig {errs[0]:.4f}): '{neg_text[:60]}{'...' if len(neg_text) > 60 else ''}'")
            
            hard_negatives_info.append({
                'index': neg_idx_val,
                'text': neg_text,
                'similarity': neg_sim,
                'error': neg_err,
                'original_error': errs[0].item()
            })
        
        hard_negatives_per_positive[pos_idx_val] = {
            'positive_text': positive_text,
            'positive_error': positive_error,
            'hard_negatives': hard_negatives_info
        }
    
    return {
        'sample_id': sample_id,
        'original_instruction': original_instruction,
        'synthetic_positives': [
            {
                'index': idx.item(),
                'text': all_rephrases[idx.item()],
                'error': errs[idx.item()].item()
            }
            for idx in positive_indices
        ],
        'hard_negatives_per_positive': hard_negatives_per_positive,
        'total_rephrases': len(e_lang),
        'mining_params': {
            'tau': tau,
            'delta': delta,
            'topk': topk,
            'diversity_clusters': diversity_clusters
        }
    }

def save_results(results: Dict, output_file: str):
    """Save mining results to JSON file."""
    # Convert tensors to regular Python types for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    print("Mining hard negatives for each synthetic positive")
    print("=" * 60)
    
    # Initialize BERT embedder once for efficiency
    print("Initializing BERT embedder...")
    embedder = BERTEmbedder()
    
    # Mine hard negatives for all available samples
    all_results = []
    
    try:
        for sample_id in [0, 1]:  # We have 2 samples
            print(f"\n{'='*80}")
            result = mine_hard_negatives_for_all_positives(
                sample_id=sample_id,
                embedder=embedder,
                tau=0.98,  # High similarity threshold for hard negatives
                delta=0.02,
                topk=5,    # Top 5 hard negatives per positive
                diversity_clusters=3
            )
            all_results.append(result)
            
            # Print summary
            num_positives = len(result['synthetic_positives'])
            total_hard_negs = sum(len(pos_data['hard_negatives']) 
                                for pos_data in result['hard_negatives_per_positive'].values())
            print(f"\nSample {sample_id} Summary:")
            print(f"  Synthetic positives: {num_positives}")
            print(f"  Total hard negatives mined: {total_hard_negs}")
            print(f"  Average hard negatives per positive: {total_hard_negs/max(num_positives, 1):.1f}")
        
        # Save all results
        from datetime import datetime
        output_file = f"hard_negatives_for_positives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results({
            'results': all_results,
            'summary': {
                'total_samples': len(all_results),
                'total_positives': sum(len(r['synthetic_positives']) for r in all_results),
                'total_hard_negatives': sum(
                    sum(len(pos_data['hard_negatives']) 
                        for pos_data in r['hard_negatives_per_positive'].values())
                    for r in all_results
                )
            }
        }, output_file)
            
    except Exception as e:
        print(f"Error mining hard negatives: {e}")
        import traceback
        traceback.print_exc()
